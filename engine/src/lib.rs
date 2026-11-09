//! The native token path for iSharded stages.
//!
//! It runs inside the stage process, beside the Python that loaded the model. Python keeps
//! everything that happens rarely, such as loading, saved states and images. This takes the
//! part that happens for every token: reading rows off a link, running them through
//! llama.cpp, and sending them on, with no interpreter, no HTTP, and no waiting on the way
//! back. Anything it cannot do, it refuses, and the reply carries on along the Python path.

pub mod link;
pub mod llama;
pub mod node;
pub mod sample;
pub mod watchdog;
pub mod wire;

use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use llama::{Capture, LlamaCompute, Snapshots, StageApi, TensorApi};
use node::{Links, Node, Plan, Stats};
use serde::de::DeserializeOwned;
use watchdog::{Limits, Watchdog};

/// Bumped when the functions below change, so a stale library is refused and not misused.
/// Three: a plan carries a key for each link, and answers carry a seal.
/// Four: a stage can probe at load whether merged passes round as rows run alone.
/// Five: the stage's functions include one that waits for its device, so kept state is timed alone.
/// Six: Python says when a slot begins a new reply, so passes of the one before are dropped.
pub const ABI_VERSION: u32 = 6;

/// One stage's engine, as Python holds it.
pub struct Engine {
    capture: Arc<Capture>,
    snapshots: Arc<Snapshots>,
    watchdog: Arc<Watchdog>,
    node: Node,
    /// What `attach` was given, which the probe runs a compute of its own on.
    api: Mutex<Option<StageApi>>,
    /// The staged route's pass limits, which the watchdog takes at the flip.
    staged_limits: Mutex<Option<Limits>>,
}

/// Folds a sample into a running average that leans on recent ones, so a figure follows the
/// route as it is now and not as it was an hour ago.
pub(crate) fn average(slot: &AtomicU64, sample: u64) {
    let current = slot.load(Ordering::Relaxed);
    let value = if current == 0 { sample } else { (current * 7 + sample) / 8 };
    slot.store(value, Ordering::Relaxed);
}

fn write_text(text: &[u8], out: *mut c_char, capacity: usize) -> i32 {
    if out.is_null() || text.len() + 1 > capacity {
        return -1;
    }
    // Safety: the caller gave `capacity` bytes, and this checked they are enough.
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr().cast(), out, text.len());
        *out.add(text.len()) = 0;
    }
    text.len() as i32
}

/// # Safety
/// `text` must be a NUL-terminated string.
unsafe fn read_text<'a>(text: *const c_char) -> Option<&'a str> {
    if text.is_null() {
        return None;
    }
    CStr::from_ptr(text).to_str().ok()
}

/// # Safety
/// `text` must be a NUL-terminated string.
unsafe fn read_json<T: DeserializeOwned>(text: *const c_char) -> Option<T> {
    serde_json::from_str(read_text(text)?).ok()
}

#[no_mangle]
pub extern "C" fn isharded_engine_abi() -> u32 {
    ABI_VERSION
}

/// Makes an engine, before the model's context exists, because the context needs its callback.
///
/// # Safety
/// `target` must be a NUL-terminated tensor name and `tensors` must point to valid functions.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_new(target: *const c_char, tensors: *const TensorApi) -> *mut Engine {
    if target.is_null() || tensors.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(node) = Node::new() else {
        return std::ptr::null_mut();
    };
    let capture = Arc::new(Capture::new(CStr::from_ptr(target).to_owned(), *tensors));
    let watchdog = Arc::new(Watchdog::new(node.refuser(), || std::process::abort()));
    Box::into_raw(Box::new(Engine { capture, snapshots: Arc::default(), watchdog, node, api: Mutex::new(None), staged_limits: Mutex::new(None) }))
}

/// The eval callback, with the engine as its user data.
///
/// # Safety
/// Called by llama.cpp with a live tensor, and the engine given at context creation.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_eval(tensor: *const c_void, ask: bool, user: *mut c_void) -> bool {
    (*(user.cast::<Engine>())).capture.on_eval(tensor, ask)
}

/// # Safety
/// `engine` must come from `isharded_engine_new`.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_capture_begin(engine: *mut Engine) {
    (*engine).capture.begin();
}

/// Copies what was captured and says how many floats there were, which may exceed `capacity`.
///
/// # Safety
/// `out` must have room for `capacity` floats.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_capture_read(engine: *mut Engine, out: *mut f32, capacity: usize) -> usize {
    (*engine).capture.copy_out(out, capacity)
}

/// Hands the model over, once the context exists.
///
/// # Safety
/// `api` must point to valid pointers that outlive the engine.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_attach(engine: *mut Engine, api: *const StageApi) {
    let engine = &*engine;
    *engine.api.lock() = Some(*api);
    engine.node.set_compute(Box::new(LlamaCompute::new(*api, engine.capture.clone(), engine.snapshots.clone(), engine.watchdog.clone())));
}

/// Finds out how far merged passes round as rows run alone, over `slots` slots, and writes
/// it as JSON. Every slot it touched is empty again afterwards. Negative on failure, with
/// the reason written instead.
///
/// # Safety
/// `engine` must come from `isharded_engine_new`, after `attach`, and `out` must have room
/// for `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_probe(engine: *mut Engine, slots: u32, out: *mut c_char, capacity: usize) -> i32 {
    let engine = &*engine;
    let Some(api) = *engine.api.lock() else {
        write_text(b"this stage has nothing loaded", out, capacity);
        return -1;
    };
    let mut compute = LlamaCompute::new(api, engine.capture.clone(), engine.snapshots.clone(), engine.watchdog.clone());
    engine.node.lock();
    let found = compute.probe(slots, node::MAX_MERGED - 1);
    engine.node.unlock();
    write_found(found, out, capacity)
}

/// Times passes of several rows on empty slots: one row a slot up to `slots`, then the same
/// rows in every slot up to `most` rows, for as long as `budget_ms` allows. Writes the rows
/// and the ms each took as JSON. Every slot it touched is empty again afterwards. Negative
/// on failure, with the reason written instead.
///
/// # Safety
/// `engine` must come from `isharded_engine_new`, after `attach`, and `out` must have room
/// for `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_time_passes(engine: *mut Engine, slots: u32, most: u32, budget_ms: u32, out: *mut c_char, capacity: usize) -> i32 {
    let engine = &*engine;
    let Some(api) = *engine.api.lock() else {
        write_text(b"this stage has nothing loaded", out, capacity);
        return -1;
    };
    let mut compute = LlamaCompute::new(api, engine.capture.clone(), engine.snapshots.clone(), engine.watchdog.clone());
    engine.node.lock();
    let times = compute.time_passes(slots, most, std::time::Duration::from_millis(u64::from(budget_ms)));
    engine.node.unlock();
    write_found(times, out, capacity)
}

/// Writes what a probe found as JSON, or the reason it failed, which is negative.
unsafe fn write_found<T: serde::Serialize>(found: Result<T, node::Failure>, out: *mut c_char, capacity: usize) -> i32 {
    match found.map(|found| serde_json::to_vec(&found)) {
        Ok(Ok(text)) => write_text(&text, out, capacity),
        Ok(Err(error)) => {
            write_text(error.to_string().as_bytes(), out, capacity);
            -1
        }
        Err(failure) => {
            write_text(failure.message.as_bytes(), out, capacity);
            -1
        }
    }
}

/// Forgets what a rewind could go back to, because Python changed what a slot holds. A
/// negative slot means all of them.
///
/// # Safety
/// `engine` must come from `isharded_engine_new`.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_forget(engine: *mut Engine, slot: i32) {
    (*engine).snapshots.forget(slot);
}

/// Says a slot now holds a new reply, which Python began with a restore or a clear. Passes
/// of the reply before it that are still queued, or still on their way, are dropped rather
/// than run on the new reply's state. The caller holds the lock from `isharded_engine_lock`.
///
/// # Safety
/// `engine` must come from `isharded_engine_new`.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_begin(engine: *mut Engine, slot: u32, generation: u64) {
    (*engine).node.begin(slot, generation);
}

/// # Safety
/// `engine` must come from `isharded_engine_new`.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_lock(engine: *mut Engine) {
    (*engine).node.lock();
}

/// # Safety
/// The calling thread must hold the lock.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_unlock(engine: *mut Engine) {
    (*engine).node.unlock();
}

/// Starts listening and writes JSON with the ports and fingerprint. Negative on failure,
/// with the reason written instead.
///
/// # Safety
/// `host` must be a NUL-terminated string, and `out` must have room for `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_listen(engine: *mut Engine, host: *const c_char, port: u16, out: *mut c_char, capacity: usize) -> i32 {
    let Some(host) = read_text(host) else {
        return -1;
    };
    match (*engine).node.listen(host, port) {
        Ok(text) => write_text(&text, out, capacity),
        Err(error) => {
            write_text(error.as_bytes(), out, capacity);
            -1
        }
    }
}

/// # Safety
/// `plan` must be a NUL-terminated JSON string.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_open(engine: *mut Engine, plan: *const c_char) -> i32 {
    let (Some(limits), Some(plan)) = (read_json::<Limits>(plan), read_json::<Plan>(plan)) else {
        return -1;
    };
    if (*engine).node.open(plan).is_err() {
        return -1;
    }
    (*engine).watchdog.set_limits(limits);
    0
}

/// Starts dialling the neighbours, and returns at once. The stats say how it went.
///
/// # Safety
/// `links` must be a NUL-terminated JSON string.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_connect(engine: *mut Engine, links: *const c_char) -> i32 {
    let Some(links) = read_json::<Links>(links) else {
        return -1;
    };
    (*engine).node.connect_in_background(links);
    0
}

/// # Safety
/// `engine` must come from `isharded_engine_new`.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_close_ring(engine: *mut Engine) {
    (*engine).node.close_ring();
}

// The calls below hold the next route beside the current one. They were added without a new
// ABI version: nothing above changed, and Python looks each of them up before it uses them.

/// Opens the next route's ring beside the current one, which goes on carrying replies.
///
/// # Safety
/// `plan` must be a NUL-terminated JSON string.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_stage(engine: *mut Engine, plan: *const c_char) -> i32 {
    let (Some(limits), Some(plan)) = (read_json::<Limits>(plan), read_json::<Plan>(plan)) else {
        return -1;
    };
    if (*engine).node.stage(plan).is_err() {
        return -1;
    }
    *(*engine).staged_limits.lock() = Some(limits);
    0
}

/// Starts dialling the staged ring's neighbours, and returns at once. The stats say how it went.
///
/// # Safety
/// `links` must be a NUL-terminated JSON string.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_connect_staged(engine: *mut Engine, links: *const c_char) -> i32 {
    let Some(links) = read_json::<Links>(links) else {
        return -1;
    };
    (*engine).node.connect_staged_in_background(links);
    0
}

/// Gives slots, a JSON list, to one of the routes held here while two overlap.
///
/// # Safety
/// `slots` must be a NUL-terminated JSON string.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_give_slots(engine: *mut Engine, epoch: u64, slots: *const c_char) -> i32 {
    let Some(slots) = read_json::<Vec<u32>>(slots) else {
        return -1;
    };
    match (*engine).node.give_slots(epoch, &slots) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// The flip: the staged route takes every slot listed and becomes the current one. One if
/// nothing of the old route is left queued or running here within `wait_ms`, zero if
/// something is, and negative if that route is not staged here.
///
/// # Safety
/// `slots` must be a NUL-terminated JSON string.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_flip(engine: *mut Engine, epoch: u64, slots: *const c_char, wait_ms: u32) -> i32 {
    let Some(slots) = read_json::<Vec<u32>>(slots) else {
        return -1;
    };
    match (*engine).node.flip(epoch, &slots, std::time::Duration::from_millis(u64::from(wait_ms))) {
        Ok(quiet) => {
            if let Some(limits) = (*engine).staged_limits.lock().take() {
                (*engine).watchdog.set_limits(limits);
            }
            i32::from(quiet)
        }
        Err(_) => -1,
    }
}

/// Closes one route's ring and leaves any other alone. One if it was held here.
///
/// # Safety
/// `engine` must come from `isharded_engine_new`.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_close_epoch(engine: *mut Engine, epoch: u64) -> i32 {
    i32::from((*engine).node.close_epoch(epoch))
}

/// # Safety
/// `out` must have room for `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_stats(engine: *mut Engine, out: *mut c_char, capacity: usize) -> i32 {
    let engine = &*engine;
    let keeps = engine.snapshots.keep_stats();
    let stats = Stats {
        keep_us: keeps.keep_us,
        keep_bytes: keeps.keep_bytes,
        keeps_total: keeps.keeps_total,
        keep_us_total: keeps.keep_us_total,
        ..engine.node.stats()
    };
    match serde_json::to_vec(&stats) {
        Ok(text) => write_text(&text, out, capacity),
        Err(_) => -1,
    }
}
