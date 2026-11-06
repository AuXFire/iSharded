//! Running a pass through llama.cpp, by the same steps `LlamaStage` takes in llamacpp.py.
//!
//! Python loads the library and the model, and hands over the function pointers it bound.
//! So there is one copy of the library in the process, and this code needs no linking to a
//! particular build.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::node::{Compute, Failure, Output};
use crate::sample::{self, Draws};
use crate::wire::{Pass, Refusal};

/// `llama_batch`, passed by value.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Batch {
    pub n_tokens: i32,
    pub token: *const i32,
    pub embd: *const f32,
    pub pos: *const i32,
    pub n_seq_id: *const i32,
    pub seq_id: *const *const i32,
    pub logits: *const i8,
}

pub type Decode = unsafe extern "C" fn(*mut c_void, Batch) -> i32;
pub type LogitsIth = unsafe extern "C" fn(*mut c_void, i32) -> *mut f32;
pub type SeqRm = unsafe extern "C" fn(*mut c_void, i32, i32, i32) -> bool;
pub type TensorName = unsafe extern "C" fn(*const c_void) -> *const c_char;
pub type TensorBytes = unsafe extern "C" fn(*const c_void) -> usize;
pub type TensorGet = unsafe extern "C" fn(*const c_void, *mut c_void, usize, usize);
pub type ToFloat = unsafe extern "C" fn(*const c_void, *mut f32, i64);

/// What the output capture needs. Set before the model's context exists.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TensorApi {
    pub name: TensorName,
    pub bytes: TensorBytes,
    pub get: TensorGet,
}

/// What a stage's passes need, once the context exists.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct StageApi {
    pub decode: Decode,
    pub logits_ith: LogitsIth,
    pub seq_rm: SeqRm,
    pub context: *mut c_void,
    pub memory: *mut c_void,
    pub width: i32,
    pub vocab: i32,
    pub last: bool,
    /// Rows of more than one token cross as BF16, as they do over HTTP.
    pub wire_bf16: bool,
    pub table: *const u8,
    pub table_rows: i64,
    pub row_bytes: i64,
    pub to_float: Option<ToFloat>,
}

/// Holds what the stage's last layer produced. llama.cpp calls `on_eval` for every node of
/// every pass, which is thousands of calls: far too many to cross into Python.
pub struct Capture {
    target: CString,
    tensors: TensorApi,
    rows: Mutex<Vec<f32>>,
}

impl Capture {
    pub fn new(target: CString, tensors: TensorApi) -> Capture {
        Capture { target, tensors, rows: Mutex::new(Vec::new()) }
    }

    pub fn begin(&self) {
        self.rows.lock().clear();
    }

    pub fn copy_out(&self, out: *mut f32, capacity: usize) -> usize {
        let rows = self.rows.lock();
        let count = rows.len().min(capacity);
        // Safety: the caller gave room for `capacity` floats.
        unsafe { std::ptr::copy_nonoverlapping(rows.as_ptr(), out, count) };
        rows.len()
    }

    /// # Safety
    /// `tensor` must be a live ggml tensor, as llama.cpp passes to its eval callback.
    pub unsafe fn on_eval(&self, tensor: *const c_void, ask: bool) -> bool {
        if ask {
            let name = (self.tensors.name)(tensor);
            return !name.is_null() && CStr::from_ptr(name) == self.target.as_c_str();
        }
        let size = (self.tensors.bytes)(tensor);
        let mut rows = self.rows.lock();
        let start = rows.len();
        rows.resize(start + size / 4, 0.0);
        (self.tensors.get)(tensor, rows[start..].as_mut_ptr().cast(), 0, size);
        false // stop here: the output norm and head are not needed
    }
}

pub struct LlamaCompute {
    api: StageApi,
    capture: Arc<Capture>,
    embedded: Vec<f32>,
    input: Vec<f32>,
    positions: Vec<i32>,
    sequence: i32,
    counts: Vec<i32>,
    owners: Vec<*const i32>,
    outputs: Vec<i8>,
}

// Safety: the pointers belong to the model, and only the gate's holder uses them.
unsafe impl Send for LlamaCompute {}

fn stage_error(message: String) -> Failure {
    Failure { refusal: Refusal::StageError, message }
}

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let round = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

impl LlamaCompute {
    pub fn new(api: StageApi, capture: Arc<Capture>) -> LlamaCompute {
        LlamaCompute {
            api,
            capture,
            embedded: Vec::new(),
            input: Vec::new(),
            positions: Vec::new(),
            sequence: 0,
            counts: Vec::new(),
            owners: Vec::new(),
            outputs: Vec::new(),
        }
    }

    fn embed(&mut self, tokens: &[u32]) -> Result<(), Failure> {
        let width = self.api.width as usize;
        let row_bytes = self.api.row_bytes as usize;
        if self.api.table.is_null() {
            return Err(stage_error("this stage has no embedding table".into()));
        }
        self.embedded.resize(tokens.len() * width, 0.0);
        for (index, token) in tokens.iter().enumerate() {
            if i64::from(*token) >= self.api.table_rows {
                return Err(stage_error(format!("token {token} is outside the embedding table")));
            }
            // Safety: the row is inside the table, which Python keeps mapped.
            let row = unsafe { self.api.table.add(*token as usize * row_bytes) };
            let out = &mut self.embedded[index * width..(index + 1) * width];
            match self.api.to_float {
                // Safety: `to_float` reads one row and writes `width` floats.
                Some(to_float) => unsafe { to_float(row.cast(), out.as_mut_ptr(), width as i64) },
                // Safety: a table with no conversion holds FP32 rows.
                None => unsafe { std::ptr::copy_nonoverlapping(row.cast::<f32>(), out.as_mut_ptr(), width) },
            }
        }
        Ok(())
    }

    fn read_rows(&mut self, pass: &Pass) -> Result<(), Failure> {
        if pass.width != self.api.width as u32 {
            return Err(stage_error(format!("rows are {} wide and this stage takes {}", pass.width, self.api.width)));
        }
        self.input.clear();
        if pass.is_bf16() {
            self.input.extend(pass.data.chunks_exact(2).map(|item| f32::from_bits(u32::from(u16::from_le_bytes([item[0], item[1]])) << 16)));
        } else {
            self.input.extend(pass.data.chunks_exact(4).map(|item| f32::from_le_bytes([item[0], item[1], item[2], item[3]])));
        }
        Ok(())
    }

    fn decode(&mut self, hidden: *const f32, count: usize, position: u32, slot: u32, want_all: bool) -> Result<(), Failure> {
        // Embedding input reads every position section itself.
        self.positions.clear();
        for _ in 0..3 {
            self.positions.extend((0..count as i32).map(|step| position as i32 + step));
        }
        self.positions.resize(count * 4, 0);
        self.sequence = slot as i32;
        self.counts.clear();
        self.counts.resize(count, 1);
        self.owners.clear();
        self.owners.resize(count, &self.sequence);
        self.outputs.clear();
        self.outputs.resize(count, i8::from(want_all));
        self.outputs[count - 1] = 1;
        let batch = Batch {
            n_tokens: count as i32,
            token: std::ptr::null(),
            embd: hidden,
            pos: self.positions.as_ptr(),
            n_seq_id: self.counts.as_ptr(),
            seq_id: self.owners.as_ptr(),
            logits: self.outputs.as_ptr(),
        };
        self.capture.begin();
        // Safety: every array in the batch lives until this returns.
        let status = unsafe { (self.api.decode)(self.api.context, batch) };
        if status != 0 {
            return Err(stage_error(format!("llama_decode failed with status {status}")));
        }
        Ok(())
    }

    fn logits(&self, row: i32) -> Result<&'static [f32], Failure> {
        // Safety: the context outlives this pass, which the gate keeps to one at a time.
        let pointer = unsafe { (self.api.logits_ith)(self.api.context, row) };
        if pointer.is_null() {
            return Err(stage_error(format!("no logits for output {row}")));
        }
        // Safety: llama.cpp gives one row of the vocabulary's size.
        Ok(unsafe { std::slice::from_raw_parts(pointer, self.api.vocab as usize) })
    }
}

impl LlamaCompute {
    fn rows_out(&self, rows: &[f32]) -> Output {
        let bf16 = false; // one row per pass crosses exactly, as it does over HTTP
        let mut data = Vec::with_capacity(rows.len() * 4);
        rows.iter().for_each(|value| data.extend_from_slice(&value.to_le_bytes()));
        Output::Rows { rows: 1, width: self.api.width as u32, bf16, data }
    }

    fn merged(&mut self, passes: &[Pass]) -> Result<Vec<Result<Output, Failure>>, Failure> {
        let width = self.api.width as usize;
        // llama.cpp gathers sequences in order, and its rows come back in that order.
        let mut order: Vec<usize> = (0..passes.len()).collect();
        order.sort_unstable_by_key(|index| passes[*index].slot);
        let mut hidden = Vec::with_capacity(passes.len() * width);
        for index in &order {
            let pass = &passes[*index];
            if let Some(keep) = pass.rollback_to {
                // Safety: only the gate's holder touches the memory.
                if !unsafe { (self.api.seq_rm)(self.api.memory, pass.slot as i32, keep as i32, -1) } {
                    return Err(Failure { refusal: Refusal::Retry, message: "this model cannot rewind; replay instead".into() });
                }
            }
            if pass.carries_tokens() {
                self.embed(&pass.tokens)?;
                hidden.extend_from_slice(&self.embedded);
            } else {
                self.read_rows(pass)?;
                hidden.extend_from_slice(&self.input);
            }
        }
        let count = passes.len();
        self.positions.clear();
        for _ in 0..3 {
            self.positions.extend(order.iter().map(|index| passes[*index].position as i32));
        }
        self.positions.resize(count * 4, 0);
        let slots: Vec<i32> = order.iter().map(|index| passes[*index].slot as i32).collect();
        self.counts.clear();
        self.counts.resize(count, 1);
        self.owners.clear();
        self.owners.extend(slots.iter().map(std::ptr::from_ref));
        self.outputs.clear();
        self.outputs.resize(count, 1);
        let batch = Batch {
            n_tokens: count as i32,
            token: std::ptr::null(),
            embd: hidden.as_ptr(),
            pos: self.positions.as_ptr(),
            n_seq_id: self.counts.as_ptr(),
            seq_id: self.owners.as_ptr(),
            logits: self.outputs.as_ptr(),
        };
        self.capture.begin();
        // Safety: every array in the batch lives until this returns.
        let status = unsafe { (self.api.decode)(self.api.context, batch) };
        if status != 0 {
            return Err(stage_error(format!("llama_decode failed with status {status}")));
        }
        let mut results: Vec<Option<Result<Output, Failure>>> = (0..count).map(|_| None).collect();
        if self.api.last {
            for (row, index) in order.iter().enumerate() {
                let pass = &passes[*index];
                let mut draws = Draws::new(pass.sampling.seed, pass.position);
                let logits = self.logits(row as i32)?;
                results[*index] = Some(Ok(Output::Tokens(vec![sample::choose(logits, &pass.sampling, &mut draws)])));
            }
        } else {
            let rows = self.capture.rows.lock().clone();
            if rows.len() != count * width {
                return Err(stage_error("a merged pass did not produce one row per reply".into()));
            }
            for (row, index) in order.iter().enumerate() {
                results[*index] = Some(Ok(self.rows_out(&rows[row * width..(row + 1) * width])));
            }
        }
        Ok(results.into_iter().flatten().collect())
    }
}

impl Compute for LlamaCompute {
    fn run_merged(&mut self, passes: &[Pass]) -> Vec<Result<Output, Failure>> {
        match self.merged(passes) {
            Ok(results) => results,
            Err(failure) => passes.iter().map(|_| Err(Failure { refusal: failure.refusal, message: failure.message.clone() })).collect(),
        }
    }

    fn run(&mut self, pass: &Pass) -> Result<Output, Failure> {
        if let Some(keep) = pass.rollback_to {
            // Safety: only the gate's holder touches the memory.
            if !unsafe { (self.api.seq_rm)(self.api.memory, pass.slot as i32, keep as i32, -1) } {
                return Err(Failure { refusal: Refusal::Retry, message: "this model cannot rewind; replay instead".into() });
            }
        }
        let hidden = if pass.carries_tokens() {
            self.embed(&pass.tokens)?;
            self.embedded.as_ptr()
        } else {
            self.read_rows(pass)?;
            self.input.as_ptr()
        };
        let count = pass.row_count();
        let verify = self.api.last && pass.verifying();
        self.decode(hidden, count, pass.position, pass.slot, !self.api.last || verify)?;
        if !self.api.last {
            let rows = self.capture.rows.lock();
            let width = self.api.width as usize;
            if rows.len() != count * width {
                return Err(stage_error("the stage output tensor was never produced".into()));
            }
            let bf16 = count > 1 && self.api.wire_bf16;
            let mut data = Vec::with_capacity(rows.len() * if bf16 { 2 } else { 4 });
            if bf16 {
                rows.iter().for_each(|value| data.extend_from_slice(&to_bf16(*value).to_le_bytes()));
            } else {
                rows.iter().for_each(|value| data.extend_from_slice(&value.to_le_bytes()));
            }
            return Ok(Output::Rows { rows: count as u32, width: width as u32, bf16, data });
        }
        let mut draws = Draws::new(pass.sampling.seed, pass.position);
        if verify {
            let mut rows = Vec::with_capacity(count);
            for row in 0..count {
                rows.push(self.logits(row as i32)?);
            }
            let tokens = sample::speculative(|row| rows[row], &pass.guesses, &pass.sampling, &mut draws);
            return Ok(Output::Tokens(tokens));
        }
        let logits = self.logits(-1)?;
        Ok(Output::Tokens(vec![sample::choose(logits, &pass.sampling, &mut draws)]))
    }
}

#[cfg(test)]
mod tests {
    use super::to_bf16;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(to_bf16(1.0), 0x3F80);
        // Halfway between two values goes to the even one, as torch rounds.
        assert_eq!(to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert_eq!(to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(to_bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
        assert_eq!(to_bf16(f32::NAN) & 0x7FC0, 0x7FC0);
    }
}
