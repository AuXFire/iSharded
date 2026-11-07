//! Running a pass through llama.cpp, by the same steps `LlamaStage` takes in llamacpp.py.
//!
//! Python loads the library and the model, and hands over the function pointers it bound.
//! So there is one copy of the library in the process, and this code needs no linking to a
//! particular build.

use std::collections::{HashMap, VecDeque};
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::Serialize;

use crate::node::{Compute, Failure, Output};
use crate::sample::{self, Draws};
use crate::watchdog::Watchdog;
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
pub type StateSize = unsafe extern "C" fn(*mut c_void, i32, u32) -> usize;
pub type StateGet = unsafe extern "C" fn(*mut c_void, *mut u8, usize, i32, u32) -> usize;
pub type StateSet = unsafe extern "C" fn(*mut c_void, *const u8, usize, i32, u32) -> usize;
pub type TensorName = unsafe extern "C" fn(*const c_void) -> *const c_char;
pub type TensorBytes = unsafe extern "C" fn(*const c_void) -> usize;
pub type TensorGet = unsafe extern "C" fn(*const c_void, *mut c_void, usize, usize);
pub type ToFloat = unsafe extern "C" fn(*const c_void, *mut f32, i64);

/// `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`. Attention can rewind by itself, so a snapshot leaves
/// it out, which keeps its size the same however long the reply gets.
const PARTIAL_ONLY: u32 = 1;

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
    pub state_size: StateSize,
    pub state_get: StateGet,
    pub state_set: StateSet,
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
    /// How many snapshots each slot holds, which must cover every pass a miss can undo.
    pub keep_limit: u32,
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

/// A slot's state just after a pass, named by the position the next pass runs at.
struct Snapshot {
    end: u32,
    bytes: Vec<u8>,
}

/// One slot's snapshots, oldest first.
#[derive(Default)]
struct SlotStates {
    saved: VecDeque<Snapshot>,
    spare: Vec<Vec<u8>>,
}

impl SlotStates {
    fn drop_past(&mut self, end: u32) {
        while self.saved.back().is_some_and(|item| item.end > end) {
            self.spare.extend(self.saved.pop_back().map(|item| item.bytes));
        }
    }

    /// A buffer for the next snapshot. A state can be many megabytes, so they are reused.
    fn buffer(&mut self, limit: usize) -> Vec<u8> {
        if self.saved.len() >= limit {
            if let Some(oldest) = self.saved.pop_front() {
                return oldest.bytes;
            }
        }
        self.spare.pop().unwrap_or_default()
    }
}

/// What recurrent state can go back to. A rewind cannot undo a finished pass by itself, so a
/// reply that may need one asks each stage to hold on to where it has been.
///
/// Python changes slots too, and nothing here would notice: a snapshot from an earlier reply
/// restores without complaint and gives wrong answers. So it is shared, as `Capture` is, and
/// Python says when a slot's history stops being the one these came from.
#[derive(Default)]
pub struct Snapshots {
    slots: Mutex<HashMap<i32, SlotStates>>,
}

impl Snapshots {
    /// A negative slot means all of them.
    pub fn forget(&self, slot: i32) {
        let mut slots = self.slots.lock();
        if slot < 0 {
            slots.clear();
        } else {
            slots.remove(&slot);
        }
    }
}

pub struct LlamaCompute {
    api: StageApi,
    capture: Arc<Capture>,
    snapshots: Arc<Snapshots>,
    watchdog: Arc<Watchdog>,
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
    pub fn new(api: StageApi, capture: Arc<Capture>, snapshots: Arc<Snapshots>, watchdog: Arc<Watchdog>) -> LlamaCompute {
        LlamaCompute {
            api,
            capture,
            snapshots,
            watchdog,
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

    /// Forgets positions from `keep` on, going through a snapshot if the model refuses.
    fn rewind(&self, slot: i32, keep: u32) -> Result<(), Failure> {
        let refused = || Failure { refusal: Refusal::Retry, message: "this model cannot rewind; replay instead".into() };
        // Safety: only the gate's holder touches the memory.
        let cut = || unsafe { (self.api.seq_rm)(self.api.memory, slot, keep as i32, -1) };
        let mut slots = self.snapshots.slots.lock();
        if !cut() {
            let found = slots.get(&slot).and_then(|states| states.saved.iter().find(|item| item.end == keep));
            let state = found.ok_or_else(refused)?;
            // Safety: the bytes came from `state_get`, for this slot of this context.
            let read = unsafe { (self.api.state_set)(self.api.context, state.bytes.as_ptr(), state.bytes.len(), slot, PARTIAL_ONLY) };
            if read == 0 || !cut() {
                // llama.cpp empties a slot whose restore fails, so nothing here fits it now.
                slots.remove(&slot);
                return Err(refused());
            }
        }
        if let Some(states) = slots.get_mut(&slot) {
            states.drop_past(keep);
        }
        Ok(())
    }

    /// Saves where a pass left its slot, if the pass asked. Failing to is no failure of the
    /// pass: a rewind that needs the snapshot refuses, and the reply replays.
    fn keep_state(&self, pass: &Pass) {
        if !pass.keeps_state() {
            return;
        }
        let slot = pass.slot as i32;
        let end = pass.position + pass.row_count() as u32;
        let mut slots = self.snapshots.slots.lock();
        let states = slots.entry(slot).or_default();
        states.drop_past(end - 1);
        let mut bytes = states.buffer(self.api.keep_limit as usize);
        // Safety: the buffer is resized to what llama.cpp asks for, and the gate is held.
        let wrote = unsafe {
            let size = (self.api.state_size)(self.api.context, slot, PARTIAL_ONLY);
            bytes.resize(size, 0);
            (self.api.state_get)(self.api.context, bytes.as_mut_ptr(), size, slot, PARTIAL_ONLY)
        };
        if wrote == 0 {
            states.spare.push(bytes);
            return;
        }
        bytes.truncate(wrote);
        states.saved.push_back(Snapshot { end, bytes });
    }

    /// Does a pass's rewind, and adds its rows to the coming batch.
    fn load(&mut self, pass: &Pass) -> Result<(), Failure> {
        if let Some(keep) = pass.rollback_to {
            self.rewind(pass.slot as i32, keep)?;
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

    /// Runs loaded passes in one batch, and answers in the order given.
    fn merged(&mut self, passes: &[Pass], order: &[usize]) -> Vec<Result<Output, Failure>> {
        let rows = order.iter().map(|index| (passes[*index].position as i32, passes[*index].slot as i32));
        if let Err(failure) = self.decode(rows, true) {
            // llama.cpp refuses a whole batch over one bad row. Alone, only that row is refused.
            return match order {
                [_] => vec![Err(failure)],
                _ => order.iter().map(|index| self.run(&passes[*index])).collect(),
            };
        }
        order.iter().for_each(|index| self.keep_state(&passes[*index]));
        if self.api.last {
            let choose = |(row, index): (usize, &usize)| {
                let pass = &passes[*index];
                let mut draws = Draws::new(pass.sampling.seed, pass.position);
                self.logits(row as i32).map(|logits| Output::Tokens(vec![sample::choose(logits, &pass.sampling, &mut draws)]))
            };
            return order.iter().enumerate().map(choose).collect();
        }
        let width = self.api.width as usize;
        let rows = self.capture.rows.lock();
        if rows.len() != order.len() * width {
            let failure = stage_error("a merged pass did not produce one row per reply".into());
            return order.iter().map(|_| Err(failure.clone())).collect();
        }
        // One row per pass crosses exactly, as it does over HTTP.
        let output = |row: &[f32]| Ok(Output::Rows { rows: 1, width: width as u32, bf16: false, data: to_wire(row, false) });
        rows.chunks_exact(width).map(output).collect()
    }
}

/// What a stage found out at load about how it rounds, before any reply could see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Exactness {
    /// The most companions a one-row pass shared a trip with and still came out byte for byte
    /// as it does alone, every time and at every smaller count.
    pub exact_merge_limit: u32,
    /// Whether one reply's rows in one pass came out as the same rows run one at a time.
    pub verify_exact: bool,
}

/// Companion sets tried at each count. The target row moves through the batch, and each slot
/// starts at its own place in its history, so one lucky layout cannot pass for all of them.
const PROBE_TRIALS: usize = 3;
const PROBE_STARTS: [u32; 4] = [1, 6, 13, 3];
/// Rows in a probed verify pass: as many as a guess usually checks.
const PROBE_VERIFY_ROWS: usize = 4;

/// A small fixed generator, so a probe that finds a difference finds it again.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

impl LlamaCompute {
    /// Runs rows of other replies beside one row, and that row alone, and compares the bytes
    /// that come out, for 1 to `most` companions. Every slot it used is emptied afterwards,
    /// snapshots included, so it must run before any reply is admitted.
    pub fn probe(&mut self, slots: u32, most: usize) -> Result<Exactness, Failure> {
        let slots = slots.max(1) as i32;
        let found = self.probe_slots(slots, most.min(slots as usize - 1));
        for slot in 0..slots {
            self.empty(slot);
        }
        found
    }

    fn probe_slots(&mut self, slots: i32, most: usize) -> Result<Exactness, Failure> {
        let mut noise = Noise(0x9E37_79B9_7F4A_7C15);
        let mut limit = 0;
        'counts: for companions in 1..=most {
            for trial in 0..PROBE_TRIALS {
                if !self.merge_matches(slots, companions, trial, &mut noise)? {
                    break 'counts;
                }
            }
            limit = companions;
        }
        let mut verify_exact = true;
        for trial in 0..2 {
            verify_exact &= self.verify_matches(PROBE_STARTS[trial], &mut noise)?;
        }
        Ok(Exactness { exact_merge_limit: limit as u32, verify_exact })
    }

    fn empty(&self, slot: i32) {
        // Safety: the probe's caller holds the gate. A whole slot always goes, recurrent or not.
        unsafe { (self.api.seq_rm)(self.api.memory, slot, -1, -1) };
        self.snapshots.forget(slot);
    }

    /// A row as this stage sees one: a token's embedding on the first stage, numbers elsewhere.
    fn probe_row(&mut self, noise: &mut Noise) -> Result<Vec<f32>, Failure> {
        self.hidden.clear();
        if self.api.table.is_null() || self.api.table_rows < 1 {
            let width = self.api.width as usize;
            return Ok((0..width).map(|_| (noise.next() % 2001) as f32 / 1000.0 - 1.0).collect());
        }
        self.embed(&[(noise.next() % self.api.table_rows as u64) as u32])?;
        Ok(std::mem::take(&mut self.hidden))
    }

    fn probe_rows(&mut self, count: u32, noise: &mut Noise) -> Result<Vec<Vec<f32>>, Failure> {
        (0..count).map(|_| self.probe_row(noise)).collect()
    }

    /// Runs rows of (position, slot, values) in one batch, and gives each row's output bytes:
    /// its logits on the last stage, the rows it sends on elsewhere.
    fn probe_decode(&mut self, rows: &[(u32, i32, &[f32])]) -> Result<Vec<Vec<u8>>, Failure> {
        self.hidden.clear();
        rows.iter().for_each(|(_, _, values)| self.hidden.extend_from_slice(values));
        self.decode(rows.iter().map(|(position, slot, _)| (*position as i32, *slot)), true)?;
        if self.api.last {
            let logits = |row| self.logits(row as i32).map(|values| values.iter().flat_map(|value| value.to_le_bytes()).collect());
            return (0..rows.len()).map(logits).collect();
        }
        let width = self.api.width as usize;
        let captured = self.capture.rows.lock();
        if captured.len() != rows.len() * width {
            return Err(stage_error("the stage output tensor was never produced".into()));
        }
        Ok(captured.chunks_exact(width).map(|row| row.iter().flat_map(|value| value.to_le_bytes()).collect()).collect())
    }

    /// Gives a slot its history again from nothing, alone, as it was first built.
    fn build(&mut self, slot: i32, history: &[Vec<f32>]) -> Result<(), Failure> {
        self.empty(slot);
        let rows: Vec<_> = history.iter().enumerate().map(|(position, row)| (position as u32, slot, row.as_slice())).collect();
        self.probe_decode(&rows).map(drop)
    }

    /// Takes a slot back to `keep`, through a rebuild if the model cannot cut it there.
    fn cut(&mut self, slot: i32, keep: u32, history: &[Vec<f32>]) -> Result<(), Failure> {
        // Safety: the probe's caller holds the gate.
        if unsafe { (self.api.seq_rm)(self.api.memory, slot, keep as i32, -1) } {
            return Ok(());
        }
        self.build(slot, history)
    }

    fn merge_matches(&mut self, slots: i32, companions: usize, trial: usize, noise: &mut Noise) -> Result<bool, Failure> {
        let first = (trial * 7) as i32 % slots;
        let mut members: Vec<i32> = (0..=companions as i32).map(|step| (first + step) % slots).collect();
        members.sort_unstable();
        let target = [0, companions, companions / 2][trial % 3];
        let starts: Vec<u32> = (0..members.len()).map(|index| PROBE_STARTS[(trial + index) % PROBE_STARTS.len()]).collect();
        let mut histories = Vec::with_capacity(members.len());
        for start in &starts {
            histories.push(self.probe_rows(*start, noise)?);
        }
        let next = self.probe_rows(members.len() as u32, noise)?;
        // The target is built alone, as its rebuild will be. The others share one batch.
        self.build(members[target], &histories[target])?;
        let mut others = Vec::new();
        for (index, slot) in members.iter().enumerate().filter(|(index, _)| *index != target) {
            self.empty(*slot);
            others.extend(histories[index].iter().enumerate().map(|(position, row)| (position as u32, *slot, row.as_slice())));
        }
        self.probe_decode(&others)?;
        let alone = self.probe_decode(&[(starts[target], members[target], &next[target])])?;
        self.cut(members[target], starts[target], &histories[target])?;
        // Slot order, as a merged pass runs them.
        let together: Vec<_> = members.iter().enumerate().map(|(index, slot)| (starts[index], *slot, next[index].as_slice())).collect();
        let merged = self.probe_decode(&together)?;
        members.iter().for_each(|slot| self.empty(*slot));
        Ok(alone[0] == merged[target])
    }

    fn verify_matches(&mut self, start: u32, noise: &mut Noise) -> Result<bool, Failure> {
        let history = self.probe_rows(start, noise)?;
        let rows = self.probe_rows(PROBE_VERIFY_ROWS as u32, noise)?;
        self.build(0, &history)?;
        let mut alone = Vec::with_capacity(rows.len());
        for (step, row) in rows.iter().enumerate() {
            alone.extend(self.probe_decode(&[(start + step as u32, 0, row)])?);
        }
        // Every size of verify pass up to the probed one, since a kernel picked for one size of
        // batch can round differently from the next. Guesses past it go out under exact replies
        // only that many at a time.
        let together: Vec<_> = rows.iter().enumerate().map(|(step, row)| (start + step as u32, 0, row.as_slice())).collect();
        let mut matched = true;
        for size in 2..=together.len() {
            self.cut(0, start, &history)?;
            matched &= self.probe_decode(&together[..size])? == alone[..size];
        }
        self.empty(0);
        Ok(matched)
    }
}

impl Compute for LlamaCompute {
    fn run_merged(&mut self, passes: &[Pass]) -> Vec<Result<Output, Failure>> {
        let watchdog = self.watchdog.clone();
        let _timed = watchdog.time(passes.len() as u64);
        let mut results: Vec<Option<Result<Output, Failure>>> = (0..passes.len()).map(|_| None).collect();
        // llama.cpp gathers sequences in order, and its rows come back in that order.
        let mut order: Vec<usize> = (0..passes.len()).collect();
        order.sort_unstable_by_key(|index| passes[*index].slot);
        self.hidden.clear();
        // A pass that cannot load is refused here. The others still share the trip.
        order.retain(|index| {
            let filled = self.hidden.len();
            let Err(failure) = self.load(&passes[*index]) else {
                return true;
            };
            self.hidden.truncate(filled);
            results[*index] = Some(Err(failure));
            false
        });
        if !order.is_empty() {
            let answers = self.merged(passes, &order);
            order.iter().zip(answers).for_each(|(index, answer)| results[*index] = Some(answer));
        }
        results.into_iter().flatten().collect()
    }

    fn run(&mut self, pass: &Pass) -> Result<Output, Failure> {
        // Around the whole pass: llama_decode can return with the work still queued, so a hung
        // GPU may only show when the output is read.
        let watchdog = self.watchdog.clone();
        let _timed = watchdog.time(pass.row_count() as u64);
        self.hidden.clear();
        self.load(pass)?;
        let count = pass.row_count();
        let verify = self.api.last && pass.verifying();
        let rows = (0..count as i32).map(|step| (pass.position as i32 + step, pass.slot as i32));
        self.decode(rows, !self.api.last || verify)?;
        self.keep_state(pass);
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
    use crate::watchdog::Limits;
    use crate::wire::{Sampling, FLAG_KEEP, FLAG_TOKENS};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    const WIDTH: usize = 2;

    /// Stands in for llama.cpp: writes down each batch, and gives each row back doubled.
    #[derive(Default)]
    struct Fake {
        capture: Option<Arc<Capture>>,
        rewinds: Vec<(i32, i32)>,
        batches: Vec<Seen>,
        logits: Vec<f32>,
        /// Refuses a rewind across finished passes, as recurrent memory does.
        recurrent: bool,
        /// Each slot's state: the position that comes next, and every position it has run.
        slots: HashMap<i32, (i32, Vec<i32>)>,
        restores: Vec<(i32, i32)>,
        saves: usize,
        refuse_saves: bool,
        refuse_restores: bool,
        /// How long a decode takes.
        stall: Duration,
        /// Refuses any batch with a row of this slot, as llama.cpp does a row it cannot place.
        refused_slot: Option<i32>,
        refusals: usize,
        /// Rows in a batch of more than this many come out a little different, as a GPU
        /// kernel picked for a bigger batch can round differently.
        rounds_above: Option<usize>,
        /// Rows in a batch of exactly this many come out a little different, as a kernel picked
        /// for one size can round differently from the sizes either side of it.
        rounds_at: Option<usize>,
        /// Holds each slot to positions that follow on, as llama.cpp does.
        strict: bool,
    }

    #[derive(Debug, PartialEq)]
    struct Seen {
        hidden: Vec<f32>,
        positions: Vec<i32>,
        slots: Vec<i32>,
        outputs: Vec<i8>,
    }

    unsafe extern "C" fn fake_decode(context: *mut c_void, batch: Batch) -> i32 {
        std::thread::sleep((*context.cast::<Fake>()).stall);
        let fake = &mut *context.cast::<Fake>();
        let count = batch.n_tokens as usize;
        if (0..count).any(|row| Some(**batch.seq_id.add(row)) == fake.refused_slot) {
            fake.refusals += 1;
            return 1;
        }
        let hidden = std::slice::from_raw_parts(batch.embd, count * WIDTH).to_vec();
        fake.batches.push(Seen {
            positions: std::slice::from_raw_parts(batch.pos, count * 4).to_vec(),
            slots: (0..count).map(|row| **batch.seq_id.add(row)).collect(),
            outputs: std::slice::from_raw_parts(batch.logits, count).to_vec(),
            hidden: hidden.clone(),
        });
        assert!(std::slice::from_raw_parts(batch.n_seq_id, count).iter().all(|item| *item == 1));
        for row in 0..count {
            let state = fake.slots.entry(**batch.seq_id.add(row)).or_default();
            assert!(!fake.strict || *batch.pos.add(row) == state.0, "a slot's positions must follow on");
            state.0 = *batch.pos.add(row) + 1;
            state.1.push(*batch.pos.add(row));
        }
        let shift = if fake.rounds_above.is_some_and(|most| count > most) || fake.rounds_at == Some(count) { 0.5 } else { 0.0 };
        let doubled = hidden.iter().map(|value| value * 2.0 + shift);
        fake.capture.as_ref().unwrap().rows.lock().extend(doubled);
        fake.logits[0] = shift;
        0
    }

    unsafe extern "C" fn fake_logits(context: *mut c_void, _row: i32) -> *mut f32 {
        (*context.cast::<Fake>()).logits.as_mut_ptr()
    }

    unsafe extern "C" fn fake_rewind(memory: *mut c_void, slot: i32, keep: i32, _end: i32) -> bool {
        let fake = &mut *memory.cast::<Fake>();
        fake.rewinds.push((slot, keep));
        if keep < 0 {
            fake.slots.remove(&slot);
            return true;
        }
        let allowed = !fake.recurrent || keep >= fake.slots.get(&slot).map_or(0, |state| state.0);
        if let Some(state) = fake.slots.get_mut(&slot).filter(|_| allowed && fake.strict) {
            state.0 = state.0.min(keep);
            state.1.retain(|position| *position < keep);
        }
        allowed
    }

    fn state_bytes(state: &(i32, Vec<i32>)) -> Vec<u8> {
        std::iter::once(state.0).chain(state.1.iter().copied()).flat_map(i32::to_le_bytes).collect()
    }

    unsafe extern "C" fn fake_state_size(context: *mut c_void, slot: i32, flags: u32) -> usize {
        assert_eq!(flags, PARTIAL_ONLY);
        state_bytes(&(&*context.cast::<Fake>()).slots[&slot]).len()
    }

    unsafe extern "C" fn fake_state_get(context: *mut c_void, out: *mut u8, size: usize, slot: i32, flags: u32) -> usize {
        assert_eq!(flags, PARTIAL_ONLY);
        let fake = &mut *context.cast::<Fake>();
        if fake.refuse_saves {
            return 0;
        }
        fake.saves += 1;
        let bytes = state_bytes(&fake.slots[&slot]);
        assert_eq!(size, bytes.len());
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, size);
        size
    }

    unsafe extern "C" fn fake_state_set(context: *mut c_void, data: *const u8, size: usize, slot: i32, flags: u32) -> usize {
        assert_eq!(flags, PARTIAL_ONLY);
        let fake = &mut *context.cast::<Fake>();
        if fake.refuse_restores {
            fake.slots.remove(&slot);
            return 0;
        }
        let bytes = std::slice::from_raw_parts(data, size);
        let mut items = bytes.chunks_exact(4).map(|item| i32::from_le_bytes([item[0], item[1], item[2], item[3]]));
        let next = items.next().unwrap();
        fake.restores.push((slot, next));
        fake.slots.insert(slot, (next, items.collect()));
        size
    }

    unsafe extern "C" fn no_name(_tensor: *const c_void) -> *const c_char {
        std::ptr::null()
    }

    unsafe extern "C" fn no_bytes(_tensor: *const c_void) -> usize {
        0
    }

    unsafe extern "C" fn no_get(_tensor: *const c_void, _out: *mut c_void, _offset: usize, _size: usize) {}

    fn watching() -> Arc<Watchdog> {
        Arc::new(Watchdog::new(|_| {}, || {}))
    }

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
            state_size: fake_state_size,
            state_get: fake_state_get,
            state_set: fake_state_set,
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
            keep_limit: 3,
        };
        LlamaCompute::new(api, capture, Arc::default(), watching())
    }

    fn rows_pass(slot: u32, position: u32, values: &[f32]) -> Pass {
        Pass {
            generation: 1,
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
    fn a_pass_that_cannot_load_is_refused_alone_and_the_rest_of_its_merge_runs() {
        let mut fake = Fake::default();
        let mut compute = stage(&mut fake, false, &[]);
        let wrong = Pass { width: 1, rows: 2, ..rows_pass(2, 7, &[3.0, 4.0]) };
        let mut got = compute.run_merged(&[rows_pass(5, 20, &[1.0, 2.0]), wrong, rows_pass(1, 9, &[5.0, 6.0])]);
        assert_eq!(floats(got.pop().unwrap()), (1, false, bytes(&[10.0, 12.0])));
        assert!(matches!(got.pop().unwrap(), Err(failure) if failure.message == "rows are 1 wide and this stage takes 2"));
        assert_eq!(floats(got.pop().unwrap()), (1, false, bytes(&[2.0, 4.0])));
        assert_eq!(fake.batches.len(), 1);
        assert_eq!(fake.batches[0].slots, vec![1, 5]);
        assert_eq!(fake.batches[0].hidden, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn a_token_outside_the_table_takes_its_half_built_row_with_it() {
        let table = [1.0, 1.5, 2.0, 2.5];
        let mut fake = Fake::default();
        let mut compute = stage(&mut fake, true, &table);
        let token = |slot, id| Pass { flags: FLAG_TOKENS, tokens: vec![id], rows: 0, width: 0, data: vec![], ..rows_pass(slot, 4, &[]) };
        let got = compute.run_merged(&[token(0, 9), token(1, 1)]);
        assert!(matches!(&got[0], Err(failure) if failure.message == "token 9 is outside the embedding table"));
        assert!(matches!(&got[1], Ok(Output::Tokens(tokens)) if *tokens == vec![1]));
        assert_eq!(fake.batches[0].hidden, vec![2.0, 2.5]);
    }

    #[test]
    fn a_batch_llama_cpp_refuses_runs_again_row_by_row_so_only_the_bad_row_is_refused() {
        let mut fake = Fake { refused_slot: Some(2), ..Fake::default() };
        let mut compute = stage(&mut fake, false, &[]);
        let mut got = compute.run_merged(&[rows_pass(5, 20, &[1.0, 2.0]), rows_pass(2, 7, &[3.0, 4.0]), rows_pass(1, 9, &[5.0, 6.0])]);
        assert_eq!(floats(got.pop().unwrap()), (1, false, bytes(&[10.0, 12.0])));
        assert!(matches!(got.pop().unwrap(), Err(failure) if failure.message.contains("llama_decode failed")));
        assert_eq!(floats(got.pop().unwrap()), (1, false, bytes(&[2.0, 4.0])));
        // Nothing of the refused batch ran, and each good row ran once.
        assert_eq!(fake.slots[&1].1, vec![9]);
        assert_eq!(fake.slots[&5].1, vec![20]);
    }

    #[test]
    fn a_lone_row_llama_cpp_refuses_is_refused_once() {
        let mut fake = Fake { refused_slot: Some(2), ..Fake::default() };
        let mut compute = stage(&mut fake, false, &[]);
        let wrong = Pass { width: 1, rows: 2, ..rows_pass(5, 20, &[1.0, 2.0]) };
        let got = compute.run_merged(&[wrong, rows_pass(2, 7, &[3.0, 4.0])]);
        assert!(got.iter().all(Result::is_err));
        assert_eq!(fake.refusals, 1);
    }

    fn plain(slot: u32, position: u32) -> Pass {
        rows_pass(slot, position, &[1.0, 2.0])
    }

    fn kept(slot: u32, position: u32) -> Pass {
        Pass { flags: FLAG_KEEP, ..plain(slot, position) }
    }

    fn rewound(slot: u32, position: u32) -> Pass {
        Pass { rollback_to: Some(position), ..kept(slot, position) }
    }

    fn recurrent(fake: &mut Fake, start: i32) -> LlamaCompute {
        fake.recurrent = true;
        fake.slots.insert(0, (start, vec![]));
        fake.slots.insert(1, (start, vec![]));
        stage(fake, false, &[])
    }

    fn runs(compute: &mut LlamaCompute, slot: u32, positions: std::ops::Range<u32>) {
        for position in positions {
            assert!(compute.run(&kept(slot, position)).is_ok());
        }
    }

    fn refused(output: Result<Output, Failure>) -> bool {
        matches!(output, Err(failure) if failure.refusal == Refusal::Retry && failure.message == "this model cannot rewind; replay instead")
    }

    #[test]
    fn a_rewind_the_model_refuses_goes_through_the_snapshot_of_that_position() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        runs(&mut compute, 0, 10..13);
        assert!(compute.run(&rewound(0, 11)).is_ok());
        assert_eq!(fake.restores, vec![(0, 11)]);
        // Refused, restored, and then the same cut goes through.
        assert_eq!(fake.rewinds, vec![(0, 11), (0, 11)]);
        // The wrong passes are gone from the state, and the new one follows the right one.
        assert_eq!(fake.slots[&0], (12, vec![10, 11]));
    }

    #[test]
    fn a_rewind_with_nothing_to_remove_needs_no_snapshot() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        runs(&mut compute, 0, 10..12);
        assert!(compute.run(&rewound(0, 12)).is_ok());
        assert!(fake.restores.is_empty());
    }

    #[test]
    fn a_pass_that_did_not_ask_keeps_nothing() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        assert!(compute.run(&plain(0, 10)).is_ok());
        assert!(compute.run(&plain(0, 11)).is_ok());
        assert_eq!(fake.saves, 0);
        assert!(refused(compute.run(&rewound(0, 11))));
    }

    #[test]
    fn a_rewind_past_every_snapshot_refuses_and_the_ring_stays_within_its_limit() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        runs(&mut compute, 0, 10..16); // the limit is 3, so 14, 15, and 16 are left
        assert!(refused(compute.run(&rewound(0, 13))));
        assert!(fake.restores.is_empty());
        assert!(compute.run(&rewound(0, 14)).is_ok());
        assert_eq!(fake.restores, vec![(0, 14)]);
    }

    #[test]
    fn a_snapshot_from_an_abandoned_path_never_answers_for_the_new_one() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        runs(&mut compute, 0, 10..13);
        assert!(compute.run(&rewound(0, 11)).is_ok());
        assert!(compute.run(&plain(0, 12)).is_ok());
        assert!(compute.run(&plain(0, 13)).is_ok());
        // 13 was saved on the wrong path, and the right path has not saved one.
        assert!(refused(compute.run(&rewound(0, 13))));
        assert_eq!(fake.restores, vec![(0, 11)]);
    }

    #[test]
    fn a_rewind_the_model_allows_still_drops_what_it_went_past() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        runs(&mut compute, 0, 10..13);
        fake.recurrent = false; // inside the last pass, llama.cpp can rewind by itself
        assert!(compute.run(&rewound(0, 11)).is_ok());
        fake.recurrent = true;
        assert!(compute.run(&plain(0, 12)).is_ok());
        assert!(compute.run(&plain(0, 13)).is_ok());
        assert!(refused(compute.run(&rewound(0, 13))));
        assert!(fake.restores.is_empty());
    }

    #[test]
    fn a_slot_only_ever_goes_back_to_its_own_snapshots() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        runs(&mut compute, 0, 10..13);
        runs(&mut compute, 1, 12..14);
        assert!(refused(compute.run(&rewound(1, 12)))); // slot 0 holds a 12, and slot 1 does not
        assert!(compute.run(&rewound(0, 12)).is_ok());
        assert_eq!(fake.restores, vec![(0, 12)]);
    }

    #[test]
    fn forgetting_a_slot_leaves_the_others_and_a_negative_slot_forgets_them_all() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        let snapshots = compute.snapshots.clone();
        runs(&mut compute, 0, 10..13);
        runs(&mut compute, 1, 10..13);
        snapshots.forget(0);
        assert!(refused(compute.run(&rewound(0, 12))));
        assert!(compute.run(&rewound(1, 12)).is_ok());
        assert_eq!(fake.restores, vec![(1, 12)]);
        snapshots.forget(-1);
        assert!(refused(compute.run(&rewound(1, 11))));
    }

    #[test]
    fn merged_passes_rewind_and_keep_each_slot_for_itself() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        assert!(compute.run_merged(&[kept(1, 10), kept(0, 10)]).iter().all(Result::is_ok));
        assert!(compute.run_merged(&[kept(1, 11), kept(0, 11)]).iter().all(Result::is_ok));
        assert_eq!(fake.saves, 4);
        assert!(compute.run_merged(&[rewound(1, 11), kept(0, 12)]).iter().all(Result::is_ok));
        assert_eq!(fake.restores, vec![(1, 11)]);
        assert_eq!(fake.slots[&1], (12, vec![10, 11]));
        assert_eq!(fake.slots[&0], (13, vec![10, 11, 12]));
    }

    #[test]
    fn a_rewind_nothing_reaches_is_refused_alone_and_the_other_reply_goes_on() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        runs(&mut compute, 1, 10..12);
        let mut got = compute.run_merged(&[kept(0, 10), rewound(1, 10)]);
        assert!(refused(got.pop().unwrap()));
        assert!(got.pop().unwrap().is_ok());
        assert_eq!(fake.slots[&0], (11, vec![10]));
    }

    #[test]
    fn a_snapshot_that_could_not_be_saved_only_costs_the_rewind_that_needed_it() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        fake.refuse_saves = true;
        runs(&mut compute, 0, 10..12);
        assert!(refused(compute.run(&rewound(0, 11))));
    }

    #[test]
    fn a_restore_that_fails_leaves_nothing_to_go_back_to() {
        let mut fake = Fake::default();
        let mut compute = recurrent(&mut fake, 10);
        runs(&mut compute, 0, 10..13);
        fake.refuse_restores = true;
        assert!(refused(compute.run(&rewound(0, 12))));
        assert!(compute.snapshots.slots.lock().get(&0).is_none());
    }

    #[test]
    fn a_ring_reuses_its_buffers_once_it_is_full() {
        let mut states = SlotStates::default();
        for end in 1..=3 {
            let mut bytes = states.buffer(3);
            assert!(bytes.is_empty());
            bytes.resize(8, 0);
            states.saved.push_back(Snapshot { end, bytes });
        }
        assert_eq!(states.buffer(3).capacity(), 8);
        assert_eq!(states.saved.front().map(|item| item.end), Some(2));
        states.drop_past(2);
        assert_eq!(states.spare.len(), 1);
        assert_eq!(states.buffer(3).capacity(), 8);
    }

    /// Runs a pass that stalls, under a limit, and says whether the watchdog saw it.
    fn overran(stall_ms: u64, limits: Limits, go: impl FnOnce(&mut LlamaCompute)) -> bool {
        let ended = Arc::new(AtomicBool::new(false));
        let end = ended.clone();
        let mut fake = Fake { stall: Duration::from_millis(stall_ms), ..Fake::default() };
        let mut compute = stage(&mut fake, false, &[]);
        compute.watchdog = Arc::new(Watchdog::new(|_| {}, move || end.store(true, Ordering::Release)));
        compute.watchdog.set_limits(limits);
        go(&mut compute);
        std::thread::sleep(Duration::from_millis(250));
        ended.load(Ordering::Acquire)
    }

    #[test]
    fn a_pass_that_stalls_past_its_limit_is_seen() {
        let limits = Limits { pass_ms: 200, row_ms: 0 };
        assert!(overran(700, limits, |compute| drop(compute.run(&rows_pass(0, 0, &[1.0, 2.0])))));
    }

    #[test]
    fn a_merged_pass_has_the_time_of_all_its_rows() {
        let limits = Limits { pass_ms: 200, row_ms: 300 };
        let passes = [rows_pass(0, 0, &[1.0, 2.0]), rows_pass(1, 0, &[1.0, 2.0])];
        assert!(!overran(700, limits, |compute| drop(compute.run_merged(&passes))), "two rows have 800 ms");
        assert!(overran(700, Limits { pass_ms: 200, row_ms: 50 }, |compute| drop(compute.run_merged(&passes))), "two rows have 300 ms");
    }

    #[test]
    fn a_pass_that_finishes_in_time_is_left_alone() {
        let limits = Limits { pass_ms: 500, row_ms: 0 };
        assert!(!overran(50, limits, |compute| drop(compute.run(&rows_pass(0, 0, &[1.0, 2.0])))));
    }

    fn probed(fake: &mut Fake, last: bool, slots: u32) -> (Result<Exactness, Failure>, LlamaCompute) {
        fake.strict = true;
        let mut compute = stage(fake, last, &[]);
        (compute.probe(slots, 63), compute)
    }

    fn emptied(fake: &Fake, slots: i32) -> bool {
        fake.slots.is_empty() && (0..slots).all(|slot| fake.rewinds.contains(&(slot, -1)))
    }

    #[test]
    fn a_probe_where_every_batch_rounds_alike_finds_every_count_exact() {
        let mut fake = Fake::default();
        let (found, _) = probed(&mut fake, false, 6);
        assert_eq!(found.map_err(|failure| failure.message), Ok(Exactness { exact_merge_limit: 5, verify_exact: true }));
        assert!(emptied(&fake, 6));
        // Each count tried several companion sets, and the target ran at several positions.
        let merged: Vec<_> = fake.batches.iter().filter(|batch| batch.slots.len() == 6 && batch.slots.windows(2).all(|pair| pair[0] < pair[1])).collect();
        assert!(merged.len() >= PROBE_TRIALS);
    }

    #[test]
    fn a_probe_stops_at_the_first_count_that_rounds_differently() {
        let mut fake = Fake { rounds_above: Some(3), ..Fake::default() };
        let (found, _) = probed(&mut fake, false, 8);
        assert_eq!(found.map_err(|failure| failure.message), Ok(Exactness { exact_merge_limit: 2, verify_exact: false }));
        // It stopped there: no trip of one row a slot ever held more than four.
        let one_each = |batch: &&Seen| batch.slots.windows(2).all(|pair| pair[0] < pair[1]);
        assert_eq!(fake.batches.iter().filter(one_each).map(|batch| batch.slots.len()).max(), Some(4));
        assert!(emptied(&fake, 8));
    }

    #[test]
    fn the_last_stage_compares_its_logits() {
        let mut fake = Fake { rounds_above: Some(2), ..Fake::default() };
        let (found, _) = probed(&mut fake, true, 4);
        assert_eq!(found.map_err(|failure| failure.message), Ok(Exactness { exact_merge_limit: 1, verify_exact: false }));
        let mut fake = Fake::default();
        let (found, _) = probed(&mut fake, true, 4);
        assert_eq!(found.map_err(|failure| failure.message), Ok(Exactness { exact_merge_limit: 3, verify_exact: true }));
    }

    #[test]
    fn a_verify_pass_must_match_at_every_size_up_to_the_probed_one() {
        // Four rows in one pass round as alone, three do not: a guess check of two guesses differs.
        let mut fake = Fake { rounds_at: Some(3), ..Fake::default() };
        let (found, _) = probed(&mut fake, false, 2);
        assert_eq!(found.map_err(|failure| failure.message), Ok(Exactness { exact_merge_limit: 1, verify_exact: false }));
    }

    #[test]
    fn one_slot_has_nothing_to_merge_with() {
        let mut fake = Fake::default();
        let (found, _) = probed(&mut fake, false, 1);
        assert_eq!(found.map_err(|failure| failure.message), Ok(Exactness { exact_merge_limit: 0, verify_exact: true }));
    }

    #[test]
    fn a_model_that_cannot_rewind_is_built_again_from_nothing() {
        let mut fake = Fake { recurrent: true, rounds_above: Some(4), ..Fake::default() };
        let (found, _) = probed(&mut fake, false, 6);
        assert_eq!(found.map_err(|failure| failure.message), Ok(Exactness { exact_merge_limit: 3, verify_exact: true }));
        assert!(emptied(&fake, 6));
    }

    #[test]
    fn a_probe_leaves_no_snapshot_behind() {
        let mut fake = Fake { strict: true, ..Fake::default() };
        let mut compute = stage(&mut fake, false, &[]);
        runs(&mut compute, 2, 0..3);
        assert!(!compute.snapshots.slots.lock().is_empty());
        assert!(compute.probe(4, 63).is_ok());
        assert!(compute.snapshots.slots.lock().is_empty());
        assert!(fake.slots.is_empty());
    }

    #[test]
    fn a_probe_the_model_refuses_says_so_and_still_empties_every_slot() {
        let mut fake = Fake { refused_slot: Some(2), ..Fake::default() };
        let (found, _) = probed(&mut fake, false, 4);
        assert!(matches!(found, Err(failure) if failure.message.contains("llama_decode failed")));
        assert!(emptied(&fake, 4));
    }

    #[test]
    fn a_first_stage_probes_with_rows_from_its_table() {
        let mut fake = Fake { strict: true, ..Fake::default() };
        let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut compute = stage(&mut fake, false, &table);
        assert!(compute.probe(3, 63).is_ok());
        assert!(fake.batches.iter().all(|batch| batch.hidden.chunks_exact(WIDTH).all(|row| table.chunks_exact(WIDTH).any(|known| known == row))));
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
