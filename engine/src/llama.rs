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
    /// The rows of the coming batch.
    hidden: Vec<f32>,
    positions: Vec<i32>,
    slots: Vec<i32>,
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

/// Rows as they cross the wire.
fn to_wire(rows: &[f32], bf16: bool) -> Vec<u8> {
    let mut data = vec![0; rows.len() * if bf16 { 2 } else { 4 }];
    if bf16 {
        data.chunks_exact_mut(2).zip(rows).for_each(|(out, value)| out.copy_from_slice(&to_bf16(*value).to_le_bytes()));
    } else {
        data.chunks_exact_mut(4).zip(rows).for_each(|(out, value)| out.copy_from_slice(&value.to_le_bytes()));
    }
    data
}

impl LlamaCompute {
    pub fn new(api: StageApi, capture: Arc<Capture>) -> LlamaCompute {
        LlamaCompute {
            api,
            capture,
            hidden: Vec::new(),
            positions: Vec::new(),
            slots: Vec::new(),
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
        let start = self.hidden.len();
        self.hidden.resize(start + tokens.len() * width, 0.0);
        for (token, out) in tokens.iter().zip(self.hidden[start..].chunks_exact_mut(width)) {
            if i64::from(*token) >= self.api.table_rows {
                return Err(stage_error(format!("token {token} is outside the embedding table")));
            }
            // Safety: the row is inside the table, which Python keeps mapped.
            let row = unsafe { self.api.table.add(*token as usize * row_bytes) };
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
        if pass.is_bf16() {
            self.hidden.extend(pass.data.chunks_exact(2).map(|item| f32::from_bits(u32::from(u16::from_le_bytes([item[0], item[1]])) << 16)));
        } else {
            self.hidden.extend(pass.data.chunks_exact(4).map(|item| f32::from_le_bytes([item[0], item[1], item[2], item[3]])));
        }
        Ok(())
    }

    /// Does a pass's rewind, and adds its rows to the coming batch.
    fn load(&mut self, pass: &Pass) -> Result<(), Failure> {
        if let Some(keep) = pass.rollback_to {
            // Safety: only the gate's holder touches the memory.
            if !unsafe { (self.api.seq_rm)(self.api.memory, pass.slot as i32, keep as i32, -1) } {
                return Err(Failure { refusal: Refusal::Retry, message: "this model cannot rewind; replay instead".into() });
            }
        }
        if pass.carries_tokens() {
            self.embed(&pass.tokens)
        } else {
            self.read_rows(pass)
        }
    }

    /// Runs the batch. `rows` gives each row's position and slot.
    fn decode(&mut self, rows: impl ExactSizeIterator<Item = (i32, i32)> + Clone, want_all: bool) -> Result<(), Failure> {
        let count = rows.len();
        // Embedding input reads every position section itself.
        self.positions.clear();
        for _ in 0..3 {
            self.positions.extend(rows.clone().map(|(position, _)| position));
        }
        self.positions.resize(count * 4, 0);
        self.slots.clear();
        self.slots.extend(rows.map(|(_, slot)| slot));
        self.counts.clear();
        self.counts.resize(count, 1);
        self.owners.clear();
        self.owners.extend(self.slots.iter().map(std::ptr::from_ref));
        self.outputs.clear();
        self.outputs.resize(count, i8::from(want_all));
        self.outputs[count - 1] = 1;
        let batch = Batch {
            n_tokens: count as i32,
            token: std::ptr::null(),
            embd: self.hidden.as_ptr(),
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

    fn merged(&mut self, passes: &[Pass]) -> Result<Vec<Result<Output, Failure>>, Failure> {
        let width = self.api.width as usize;
        // llama.cpp gathers sequences in order, and its rows come back in that order.
        let mut order: Vec<usize> = (0..passes.len()).collect();
        order.sort_unstable_by_key(|index| passes[*index].slot);
        self.hidden.clear();
        for index in &order {
            self.load(&passes[*index])?;
        }
        let count = passes.len();
        self.decode(order.iter().map(|index| (passes[*index].position as i32, passes[*index].slot as i32)), true)?;
        let mut results: Vec<Option<Result<Output, Failure>>> = (0..count).map(|_| None).collect();
        if self.api.last {
            for (row, index) in order.iter().enumerate() {
                let pass = &passes[*index];
                let mut draws = Draws::new(pass.sampling.seed, pass.position);
                let logits = self.logits(row as i32)?;
                results[*index] = Some(Ok(Output::Tokens(vec![sample::choose(logits, &pass.sampling, &mut draws)])));
            }
        } else {
            let rows = self.capture.rows.lock();
            if rows.len() != count * width {
                return Err(stage_error("a merged pass did not produce one row per reply".into()));
            }
            for (row, index) in order.iter().enumerate() {
                // One row per pass crosses exactly, as it does over HTTP.
                let data = to_wire(&rows[row * width..(row + 1) * width], false);
                results[*index] = Some(Ok(Output::Rows { rows: 1, width: width as u32, bf16: false, data }));
            }
        }
        Ok(results.into_iter().flatten().collect())
    }
}

impl Compute for LlamaCompute {
    fn run_merged(&mut self, passes: &[Pass]) -> Vec<Result<Output, Failure>> {
        match self.merged(passes) {
            Ok(results) => results,
            Err(failure) => passes.iter().map(|_| Err(failure.clone())).collect(),
        }
    }

    fn run(&mut self, pass: &Pass) -> Result<Output, Failure> {
        self.hidden.clear();
        self.load(pass)?;
        let count = pass.row_count();
        let verify = self.api.last && pass.verifying();
        let rows = (0..count as i32).map(|step| (pass.position as i32 + step, pass.slot as i32));
        self.decode(rows, !self.api.last || verify)?;
        if !self.api.last {
            let rows = self.capture.rows.lock();
            let width = self.api.width as usize;
            if rows.len() != count * width {
                return Err(stage_error("the stage output tensor was never produced".into()));
            }
            let bf16 = count > 1 && self.api.wire_bf16;
            return Ok(Output::Rows { rows: count as u32, width: width as u32, bf16, data: to_wire(&rows, bf16) });
        }
        let mut draws = Draws::new(pass.sampling.seed, pass.position);
        if verify {
            let rows = (0..count).map(|row| self.logits(row as i32)).collect::<Result<Vec<_>, _>>()?;
            let tokens = sample::speculative(|row| rows[row], &pass.guesses, &pass.sampling, &mut draws);
            return Ok(Output::Tokens(tokens));
        }
        let logits = self.logits(-1)?;
        Ok(Output::Tokens(vec![sample::choose(logits, &pass.sampling, &mut draws)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{Sampling, FLAG_TOKENS};

    const WIDTH: usize = 2;

    /// Stands in for llama.cpp: writes down each batch, and gives each row back doubled.
    #[derive(Default)]
    struct Fake {
        capture: Option<Arc<Capture>>,
        rewinds: Vec<(i32, i32)>,
        batches: Vec<Seen>,
        logits: Vec<f32>,
    }

    #[derive(Debug, PartialEq)]
    struct Seen {
        hidden: Vec<f32>,
        positions: Vec<i32>,
        slots: Vec<i32>,
        outputs: Vec<i8>,
    }

    unsafe extern "C" fn fake_decode(context: *mut c_void, batch: Batch) -> i32 {
        let fake = &mut *context.cast::<Fake>();
        let count = batch.n_tokens as usize;
        let hidden = std::slice::from_raw_parts(batch.embd, count * WIDTH).to_vec();
        fake.batches.push(Seen {
            positions: std::slice::from_raw_parts(batch.pos, count * 4).to_vec(),
            slots: (0..count).map(|row| **batch.seq_id.add(row)).collect(),
            outputs: std::slice::from_raw_parts(batch.logits, count).to_vec(),
            hidden: hidden.clone(),
        });
        assert!(std::slice::from_raw_parts(batch.n_seq_id, count).iter().all(|item| *item == 1));
        let doubled = hidden.iter().map(|value| value * 2.0);
        fake.capture.as_ref().unwrap().rows.lock().extend(doubled);
        0
    }

    unsafe extern "C" fn fake_logits(context: *mut c_void, _row: i32) -> *mut f32 {
        (*context.cast::<Fake>()).logits.as_mut_ptr()
    }

    unsafe extern "C" fn fake_rewind(memory: *mut c_void, slot: i32, keep: i32, _end: i32) -> bool {
        (*memory.cast::<Fake>()).rewinds.push((slot, keep));
        true
    }

    unsafe extern "C" fn no_name(_tensor: *const c_void) -> *const c_char {
        std::ptr::null()
    }

    unsafe extern "C" fn no_bytes(_tensor: *const c_void) -> usize {
        0
    }

    unsafe extern "C" fn no_get(_tensor: *const c_void, _out: *mut c_void, _offset: usize, _size: usize) {}

    fn stage(fake: &mut Fake, last: bool, table: &[f32]) -> LlamaCompute {
        let tensors = TensorApi { name: no_name, bytes: no_bytes, get: no_get };
        let capture = Arc::new(Capture::new(CString::default(), tensors));
        fake.capture = Some(capture.clone());
        fake.logits = vec![0.0, 5.0, 1.0];
        let pointer = std::ptr::from_mut(fake).cast();
        let api = StageApi {
            decode: fake_decode,
            logits_ith: fake_logits,
            seq_rm: fake_rewind,
            context: pointer,
            memory: pointer,
            width: WIDTH as i32,
            vocab: 3,
            last,
            wire_bf16: true,
            table: table.as_ptr().cast(),
            table_rows: (table.len() / WIDTH) as i64,
            row_bytes: (WIDTH * 4) as i64,
            to_float: None,
        };
        LlamaCompute::new(api, capture)
    }

    fn rows_pass(slot: u32, position: u32, values: &[f32]) -> Pass {
        Pass {
            id: u64::from(slot),
            epoch: 1,
            slot,
            position,
            rollback_to: None,
            flags: 0,
            sampling: Sampling::GREEDY,
            guesses: vec![],
            tokens: vec![],
            rows: (values.len() / WIDTH) as u32,
            width: WIDTH as u32,
            data: bytes(values),
        }
    }

    fn floats(output: Result<Output, Failure>) -> (u32, bool, Vec<u8>) {
        match output {
            Ok(Output::Rows { rows, bf16, data, .. }) => (rows, bf16, data),
            _ => panic!("expected rows"),
        }
    }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    #[test]
    fn one_pass_runs_its_rows_in_place_and_several_rows_cross_as_bf16() {
        let mut fake = Fake::default();
        let mut compute = stage(&mut fake, false, &[]);
        let got = floats(compute.run(&rows_pass(3, 10, &[1.0, 2.0, 3.0, 4.0])));
        let data: Vec<u8> = [2.0f32, 4.0, 6.0, 8.0].iter().flat_map(|value| to_bf16(*value).to_le_bytes()).collect();
        assert_eq!(got, (2, true, data));
        let got = floats(compute.run(&rows_pass(3, 12, &[1.0, 2.0])));
        assert_eq!(got, (1, false, bytes(&[2.0, 4.0])));
        assert_eq!(
            fake.batches[0],
            Seen { hidden: vec![1.0, 2.0, 3.0, 4.0], positions: vec![10, 11, 10, 11, 10, 11, 0, 0], slots: vec![3, 3], outputs: vec![1, 1] }
        );
    }

    #[test]
    fn a_token_pass_is_embedded_and_the_last_stage_only_asks_for_its_last_row() {
        let table = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let mut fake = Fake::default();
        let mut compute = stage(&mut fake, true, &table);
        let pass = Pass { flags: FLAG_TOKENS, tokens: vec![2, 0], rows: 0, width: 0, data: vec![], rollback_to: Some(4), ..rows_pass(1, 4, &[]) };
        assert!(matches!(compute.run(&pass), Ok(Output::Tokens(tokens)) if tokens == vec![1]));
        assert_eq!(fake.rewinds, vec![(1, 4)]);
        assert_eq!(fake.batches[0].hidden, vec![3.0, 3.5, 1.0, 1.5]);
        assert_eq!(fake.batches[0].outputs, vec![0, 1]);
        let outside = Pass { tokens: vec![3], ..pass };
        assert!(matches!(compute.run(&outside), Err(failure) if failure.message == "token 3 is outside the embedding table"));
    }

    #[test]
    fn merged_passes_run_in_slot_order_and_answer_in_their_own_order() {
        let mut fake = Fake::default();
        let mut compute = stage(&mut fake, false, &[]);
        let passes = [rows_pass(5, 20, &[1.0, 2.0]), Pass { rollback_to: Some(7), ..rows_pass(2, 7, &[3.0, 4.0]) }];
        let got: Vec<_> = compute.run_merged(&passes).into_iter().map(floats).collect();
        assert_eq!(got, vec![(1, false, bytes(&[2.0, 4.0])), (1, false, bytes(&[6.0, 8.0]))]);
        assert_eq!(fake.rewinds, vec![(2, 7)]);
        assert_eq!(
            fake.batches[0],
            Seen { hidden: vec![3.0, 4.0, 1.0, 2.0], positions: vec![7, 20, 7, 20, 7, 20, 0, 0], slots: vec![2, 5], outputs: vec![1, 1] }
        );
    }

    #[test]
    fn a_refused_merge_refuses_every_pass_in_it() {
        let mut fake = Fake::default();
        let mut compute = stage(&mut fake, false, &[]);
        let wrong = Pass { width: 1, rows: 2, ..rows_pass(2, 7, &[3.0, 4.0]) };
        let got = compute.run_merged(&[rows_pass(5, 20, &[1.0, 2.0]), wrong]);
        assert!(got.iter().all(|item| matches!(item, Err(failure) if failure.message == "rows are 1 wide and this stage takes 2")));
        assert!(fake.batches.is_empty());
    }

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
