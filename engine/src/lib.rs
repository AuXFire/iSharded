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
pub mod wire;

use std::ffi::{c_char, c_void, CStr};
use std::sync::Arc;

use llama::{Capture, LlamaCompute, StageApi, TensorApi};
use node::{Links, Node, Plan};
use serde::de::DeserializeOwned;

/// Bumped when the functions below change, so a stale library is refused and not misused.
pub const ABI_VERSION: u32 = 1;

/// One stage's engine, as Python holds it.
pub struct Engine {
    capture: Arc<Capture>,
    node: Node,
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
    Box::into_raw(Box::new(Engine { capture, node }))
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
    engine.node.set_compute(Box::new(LlamaCompute::new(*api, engine.capture.clone())));
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
    let Some(plan) = read_json::<Plan>(plan) else {
        return -1;
    };
    (*engine).node.open(plan);
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

/// # Safety
/// `out` must have room for `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn isharded_engine_stats(engine: *mut Engine, out: *mut c_char, capacity: usize) -> i32 {
    match serde_json::to_vec(&(*engine).node.stats()) {
        Ok(text) => write_text(&text, out, capacity),
        Err(_) => -1,
    }
}
