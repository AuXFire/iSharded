//! One stage's place in a ring.
//!
//! A pass moves forward from stage to stage, and nobody waits for it: each stage hands it on
//! and is free again. The last stage sends its answer straight to the entry over a closing
//! link. If that link cannot exist, the answer walks back along the route, which is slower
//! and still right, so the closing link is only ever an optimization.
//!
//! This code never decides what a failure means. It says what it saw, and the entry raises
//! the same errors the HTTP path raises, so every failure lands on recovery that is already
//! proven: a reply replays, and a lost worker is replaced.

use std::collections::VecDeque;
use std::io;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use parking_lot::lock_api::RawMutex as _;
use parking_lot::{Condvar, Mutex, RawMutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::link::{self, Frames, Identity, Link, LinkError};
use crate::wire::{self, Ack, Answer, Hello, Kind, Nack, Pass, Refusal, Role, FLAG_BF16, FLAG_TOKENS};

/// How many connections may be mid-handshake at once, so strangers cannot pile up threads.
const MAX_STRANGERS: usize = 16;
const PING_EVERY: Duration = Duration::from_millis(1000);
const WATCH_EVERY: Duration = Duration::from_millis(50);
const ACCEPT_EVERY: Duration = Duration::from_millis(20);
const DIAL_ATTEMPT: Duration = Duration::from_secs(5);
const DIAL_RETRY: Duration = Duration::from_millis(250);
/// The most replies that share one trip through the model.
const MAX_MERGED: usize = 64;

/// What a stage does with a pass. `Rows` go on to the next stage, and `Tokens` are the answer.
pub enum Output {
    Rows { rows: u32, width: u32, bf16: bool, data: Vec<u8> },
    Tokens(Vec<u32>),
}

#[derive(Clone)]
pub struct Failure {
    pub refusal: Refusal,
    pub message: String,
}

pub trait Compute: Send {
    fn run(&mut self, pass: &Pass) -> Result<Output, Failure>;

    /// Several one-row passes from different replies, in one trip through the model.
    /// Reading the weights is what a pass costs, so sharing it costs little more than one.
    fn run_merged(&mut self, passes: &[Pass]) -> Vec<Result<Output, Failure>> {
        passes.iter().map(|pass| self.run(pass)).collect()
    }
}

/// Whether a pass can share a trip through the model with others. A verify pass has rows
/// that belong together.
fn can_merge(pass: &Pass) -> bool {
    pass.row_count() == 1 && !pass.verifying()
}

#[derive(Debug, Deserialize)]
pub struct Plan {
    pub epoch: u64,
    pub index: u16,
    pub count: u16,
    /// Shared by the route, and given out over the authenticated control plane.
    pub key: Vec<u8>,
    #[serde(default = "default_timeout_ms")]
    pub ack_timeout_ms: u64,
    #[serde(default = "default_timeout_ms")]
    pub silence_timeout_ms: u64,
}

fn default_timeout_ms() -> u64 {
    5000
}

#[derive(Debug, Clone, Deserialize)]
pub struct Address {
    pub address: String,
    pub fingerprint: String,
}

#[derive(Debug, Default, Deserialize)]
pub struct Links {
    pub next: Option<Address>,
    /// For the last stage: where the entry listens.
    pub entry: Option<Address>,
    #[serde(default = "default_connect_ms")]
    pub timeout_ms: u64,
}

fn default_connect_ms() -> u64 {
    10_000
}

#[derive(Debug, Serialize)]
pub struct Listening {
    pub port: u16,
    pub local_port: u16,
    pub local_key: Vec<u8>,
    pub fingerprint: String,
}

#[derive(Debug, Default, Serialize)]
pub struct Stats {
    pub passes: u64,
    pub rows: u64,
    pub merged_passes: u64,
    pub merged_rows: u64,
    pub answers: u64,
    pub refusals: u64,
    pub dropped: u64,
    pub compute_us: u64,
    pub wait_us: u64,
    pub link_rtt_us: u64,
    /// A pass's trip to the next stage and its acknowledgement's trip back.
    pub hop_us: u64,
    pub next_up: bool,
    pub closing_up: bool,
    pub epoch: Option<u64>,
    /// "idle", "dialling", "up", or why it failed.
    pub connect: String,
}

#[derive(Default)]
struct Counters {
    passes: AtomicU64,
    rows: AtomicU64,
    merged_passes: AtomicU64,
    merged_rows: AtomicU64,
    answers: AtomicU64,
    refusals: AtomicU64,
    dropped: AtomicU64,
    compute_us: AtomicU64,
    wait_us: AtomicU64,
    link_rtt_us: AtomicU64,
    hop_us: AtomicU64,
}

fn average(slot: &AtomicU64, sample: u64) {
    let current = slot.load(Ordering::Relaxed);
    let value = if current == 0 { sample } else { (current * 7 + sample) / 8 };
    slot.store(value, Ordering::Relaxed);
}

struct Job {
    pass: Pass,
    arrived: Instant,
}

struct Peer {
    link: Arc<Link>,
    stage: u16,
    closing: bool,
    unacked: Mutex<VecDeque<(u64, Instant)>>,
    heard: Mutex<Instant>,
    lost: AtomicBool,
    /// The clock heartbeats are stamped against, so a round trip needs no shared time.
    born: Instant,
}

impl Peer {
    fn new(link: Arc<Link>, stage: u16, closing: bool) -> Arc<Peer> {
        Arc::new(Peer { link, stage, closing, unacked: Mutex::new(VecDeque::new()), heard: Mutex::new(Instant::now()), lost: AtomicBool::new(false), born: Instant::now() })
    }
}

struct Ring {
    epoch: u64,
    index: u16,
    count: u16,
    key: Vec<u8>,
    ack_timeout: Duration,
    silence_timeout: Duration,
    next: Mutex<Option<Arc<Peer>>>,
    closing: Mutex<Option<Arc<Peer>>>,
    back: Mutex<Option<Arc<Link>>>,
    entry_closing: Mutex<Option<Arc<Link>>>,
}

impl Ring {
    fn is_entry(&self) -> bool {
        self.index == 0
    }

    fn is_last(&self) -> bool {
        self.index + 1 == self.count
    }

    fn shut(&self) {
        for peer in [self.next.lock().take(), self.closing.lock().take()].into_iter().flatten() {
            peer.lost.store(true, Ordering::Release);
            peer.link.close();
        }
        for link in [self.back.lock().take(), self.entry_closing.lock().take()].into_iter().flatten() {
            link.close();
        }
    }
}

struct Inner {
    identity: Identity,
    /// Whoever holds this may touch the model. Python takes it for what it does rarely.
    gate: RawMutex,
    compute: Mutex<Option<Box<dyn Compute>>>,
    ring: RwLock<Option<Arc<Ring>>>,
    local: Mutex<Option<Arc<Link>>>,
    local_key: [u8; 32],
    jobs: Mutex<VecDeque<Job>>,
    ready: Condvar,
    stopping: AtomicBool,
    strangers: AtomicUsize,
    counters: Counters,
    listening: Mutex<Option<Listening>>,
    accepting: Mutex<Vec<thread::JoinHandle<()>>>,
    connect: Mutex<String>,
}

pub struct Node {
    inner: Arc<Inner>,
}

impl Node {
    pub fn new() -> Result<Node, String> {
        let inner = Arc::new(Inner {
            identity: Identity::generate()?,
            gate: RawMutex::INIT,
            compute: Mutex::new(None),
            ring: RwLock::new(None),
            local: Mutex::new(None),
            local_key: link::random_bytes(),
            jobs: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
            stopping: AtomicBool::new(false),
            strangers: AtomicUsize::new(0),
            counters: Counters::default(),
            listening: Mutex::new(None),
            accepting: Mutex::new(Vec::new()),
            connect: Mutex::new("idle".to_string()),
        });
        let worker = inner.clone();
        thread::Builder::new().name("isharded-compute".into()).spawn(move || compute_loop(&worker)).map_err(|error| error.to_string())?;
        Ok(Node { inner })
    }

    pub fn set_compute(&self, compute: Box<dyn Compute>) {
        *self.inner.compute.lock() = Some(compute);
    }

    /// Holds the model for Python until `unlock`. Blocks while a pass is running.
    pub fn lock(&self) {
        self.inner.gate.lock();
    }

    /// # Safety
    /// The caller must hold the gate from `lock`.
    pub unsafe fn unlock(&self) {
        self.inner.gate.unlock();
    }

    /// Starts taking connections. Port zero asks the system for one.
    pub fn listen(&self, host: &str, port: u16) -> Result<Vec<u8>, String> {
        if let Some(listening) = &*self.inner.listening.lock() {
            return serde_json::to_vec(listening).map_err(|error| error.to_string());
        }
        let data = TcpListener::bind((host, port)).map_err(|error| format!("could not listen on {host}:{port}: {error}"))?;
        // The worker on this machine is the only one who should ever reach this port.
        let local = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
        let answer = Listening {
            port: data.local_addr().map_err(|error| error.to_string())?.port(),
            local_port: local.local_addr().map_err(|error| error.to_string())?.port(),
            local_key: self.inner.local_key.to_vec(),
            fingerprint: self.inner.identity.fingerprint.clone(),
        };
        let text = serde_json::to_vec(&answer).map_err(|error| error.to_string())?;
        *self.inner.listening.lock() = Some(answer);
        for (listener, encrypted) in [(data, true), (local, false)] {
            // Nothing wakes a blocking accept, and this loop must see the node stopping.
            listener.set_nonblocking(true).map_err(|error| error.to_string())?;
            let inner = self.inner.clone();
            let thread = spawn("isharded-accept", move || accept_loop(&inner, &listener, encrypted)).map_err(|error| error.to_string())?;
            self.inner.accepting.lock().push(thread);
        }
        Ok(text)
    }

    /// Takes a place in a route. Anything from an earlier route ends here.
    pub fn open(&self, plan: Plan) {
        let ring = Arc::new(Ring {
            epoch: plan.epoch,
            index: plan.index,
            count: plan.count,
            key: plan.key,
            ack_timeout: Duration::from_millis(plan.ack_timeout_ms),
            silence_timeout: Duration::from_millis(plan.silence_timeout_ms),
            next: Mutex::new(None),
            closing: Mutex::new(None),
            back: Mutex::new(None),
            entry_closing: Mutex::new(None),
        });
        if let Some(old) = self.inner.ring.write().replace(ring) {
            old.shut();
        }
    }

    /// Dials the next stage and, at the last stage, the entry. The closing link may fail.
    pub fn connect(&self, links: Links) -> Result<bool, String> {
        connect(&self.inner, &links)
    }

    /// The same, off this thread, so the stage goes on serving meanwhile.
    pub fn connect_in_background(&self, links: Links) {
        *self.inner.connect.lock() = "dialling".to_string();
        let inner = self.inner.clone();
        let spawned = thread::Builder::new().name("isharded-dial".into()).spawn(move || {
            let outcome = connect(&inner, &links);
            *inner.connect.lock() = match outcome {
                Ok(_) => "up".to_string(),
                Err(error) => error,
            };
        });
        if spawned.is_err() {
            *self.inner.connect.lock() = "could not start a thread".to_string();
        }
    }

    pub fn close_ring(&self) {
        if let Some(old) = self.inner.ring.write().take() {
            old.shut();
        }
    }

    pub fn stats(&self) -> Stats {
        let counters = &self.inner.counters;
        let ring = self.inner.ring.read().clone();
        Stats {
            passes: counters.passes.load(Ordering::Relaxed),
            rows: counters.rows.load(Ordering::Relaxed),
            merged_passes: counters.merged_passes.load(Ordering::Relaxed),
            merged_rows: counters.merged_rows.load(Ordering::Relaxed),
            answers: counters.answers.load(Ordering::Relaxed),
            refusals: counters.refusals.load(Ordering::Relaxed),
            dropped: counters.dropped.load(Ordering::Relaxed),
            compute_us: counters.compute_us.load(Ordering::Relaxed),
            wait_us: counters.wait_us.load(Ordering::Relaxed),
            link_rtt_us: counters.link_rtt_us.load(Ordering::Relaxed),
            hop_us: counters.hop_us.load(Ordering::Relaxed),
            next_up: ring.as_ref().is_some_and(|item| item.next.lock().is_some()),
            closing_up: ring.as_ref().is_some_and(|item| item.closing.lock().is_some() || item.entry_closing.lock().is_some()),
            epoch: ring.as_ref().map(|item| item.epoch),
            connect: self.inner.connect.lock().clone(),
        }
    }

    pub fn stop(&self) {
        self.inner.stopping.store(true, Ordering::Release);
        self.inner.ready.notify_all();
        self.close_ring();
        if let Some(local) = self.inner.local.lock().take() {
            local.close();
        }
        // The ports are free once these end, and whoever stopped this may want them back.
        for thread in std::mem::take(&mut *self.inner.accepting.lock()) {
            let _ = thread.join();
        }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        self.stop();
    }
}

fn spawn(name: &str, work: impl FnOnce() + Send + 'static) -> io::Result<thread::JoinHandle<()>> {
    #[cfg(test)]
    tests::spend_a_spawn()?;
    thread::Builder::new().name(name.into()).spawn(work)
}

fn connect(inner: &Arc<Inner>, links: &Links) -> Result<bool, String> {
    let ring = inner.ring.read().clone().ok_or("no route has been opened")?;
    if let Some(next) = &links.next {
        let link = dial_with_retry(next, &ring, Role::Hop, Duration::from_millis(links.timeout_ms)).map_err(|error| error.to_string())?;
        let peer = Peer::new(link, ring.index + 1, false);
        *ring.next.lock() = Some(peer.clone());
        start_peer(inner, &ring, &peer);
        if peer.lost.load(Ordering::Acquire) {
            return Err("the link to the next stage could not start".to_string());
        }
    }
    let mut closing = false;
    if let Some(entry) = &links.entry {
        if let Ok(link) = dial_with_retry(entry, &ring, Role::Closing, Duration::from_millis(links.timeout_ms)) {
            let peer = Peer::new(link, 0, true);
            *ring.closing.lock() = Some(peer.clone());
            start_peer(inner, &ring, &peer);
            closing = !peer.lost.load(Ordering::Acquire);
        }
    }
    Ok(closing)
}

fn dial_with_retry(target: &Address, ring: &Ring, role: Role, patience: Duration) -> Result<Arc<Link>, LinkError> {
    // A neighbour may still be starting, so the first refusal means little.
    let deadline = Instant::now() + patience;
    loop {
        match link::dial(&target.address, Some(&target.fingerprint), &ring.key, role, ring.epoch, DIAL_ATTEMPT) {
            Ok(link) => return Ok(link),
            Err(error) if Instant::now() >= deadline => return Err(error),
            Err(_) => thread::sleep(DIAL_RETRY),
        }
    }
}

fn accept_loop(inner: &Arc<Inner>, listener: &TcpListener, encrypted: bool) {
    while !inner.stopping.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((socket, _)) => admit(inner, socket, encrypted),
            Err(_) => thread::sleep(ACCEPT_EVERY),
        }
    }
}

/// Gives a connection a thread, if there is room for another stranger.
fn admit(inner: &Arc<Inner>, socket: TcpStream, encrypted: bool) {
    // On some systems a socket takes this from the listener, and links block.
    if socket.set_nonblocking(false).is_err() {
        return;
    }
    if inner.strangers.fetch_add(1, Ordering::AcqRel) >= MAX_STRANGERS {
        inner.strangers.fetch_sub(1, Ordering::AcqRel);
        return;
    }
    let serving = inner.clone();
    if spawn("isharded-link", move || serve(&serving, socket, encrypted)).is_err() {
        inner.strangers.fetch_sub(1, Ordering::AcqRel);
    }
}

fn key_for(inner: &Inner, role: Role, epoch: u64, encrypted: bool) -> Option<Vec<u8>> {
    match (role, encrypted) {
        (Role::Local, false) => Some(inner.local_key.to_vec()),
        (Role::Hop | Role::Closing, true) => inner.ring.read().as_ref().filter(|ring| ring.epoch == epoch).map(|ring| ring.key.clone()),
        _ => None,
    }
}

fn serve(inner: &Arc<Inner>, socket: TcpStream, encrypted: bool) {
    let identity = encrypted.then_some(&inner.identity);
    let accepted = link::accept(socket, identity, |role, epoch| key_for(inner, role, epoch, encrypted));
    inner.strangers.fetch_sub(1, Ordering::AcqRel);
    let Ok((link, frames, hello)) = accepted else {
        return;
    };
    match hello.role {
        Role::Local => {
            if let Some(old) = inner.local.lock().replace(link.clone()) {
                old.close();
            }
            serve_passes(inner, &link, frames, false);
        }
        Role::Hop => {
            let Some(ring) = attach(inner, &hello, |ring| ring.back.lock().replace(link.clone())) else {
                return;
            };
            serve_passes(inner, &link, frames, true);
            ring.back.lock().take();
        }
        Role::Closing => {
            if attach(inner, &hello, |ring| ring.entry_closing.lock().replace(link.clone())).is_some() {
                serve_answers(inner, &link, frames);
            }
        }
    }
    link.close();
}

fn attach(inner: &Inner, hello: &Hello, put: impl Fn(&Ring) -> Option<Arc<Link>>) -> Option<Arc<Ring>> {
    let ring = inner.ring.read().clone().filter(|ring| ring.epoch == hello.epoch)?;
    if let Some(old) = put(&ring) {
        old.close();
    }
    Some(ring)
}

/// Answers a heartbeat. False means the link is finished.
fn answer_ping(link: &Link, body: &[u8]) -> bool {
    wire::read_beat(body).is_ok_and(|stamp| link.send(&wire::beat(Kind::Pong, stamp)).is_ok())
}

/// Reads passes off a link. Taking one off is what the acknowledgement means, so a stage
/// busy with a long pass still answers its link and is not taken for dead.
fn serve_passes(inner: &Inner, link: &Link, mut frames: Frames, acknowledge: bool) {
    while let Ok((kind, body)) = frames.next() {
        match kind {
            Kind::Pass => {
                let Ok(pass) = Pass::decode(body) else {
                    return;
                };
                if acknowledge && link.send(&Ack { id: pass.id }.encode()).is_err() {
                    return;
                }
                enqueue(inner, pass);
            }
            Kind::Ping if answer_ping(link, body) => {}
            _ => return,
        }
    }
}

/// Queues a pass. A rewind says that whatever waits past it has already been thrown away,
/// so it goes now and the rewind does not sit behind it. Nobody expects an answer to those:
/// passes run in order, so once the newest pass of a reply is back, nothing of it is left.
fn enqueue(inner: &Inner, pass: Pass) {
    let mut jobs = inner.jobs.lock();
    if let Some(keep) = pass.rollback_to {
        let before = jobs.len();
        jobs.retain(|job| job.pass.slot != pass.slot || job.pass.epoch != pass.epoch || job.pass.position < keep);
        inner.counters.dropped.fetch_add((before - jobs.len()) as u64, Ordering::Relaxed);
    }
    jobs.push_back(Job { pass, arrived: Instant::now() });
    drop(jobs);
    inner.ready.notify_one();
}

/// At the entry: what the last stage sends straight back.
fn serve_answers(inner: &Inner, link: &Link, mut frames: Frames) {
    while let Ok((kind, body)) = frames.next() {
        match kind {
            Kind::Result => {
                inner.counters.answers.fetch_add(1, Ordering::Relaxed);
                to_local(inner, frames.whole());
            }
            Kind::Ping if answer_ping(link, body) => {}
            _ => return,
        }
    }
}

fn to_local(inner: &Inner, frame: &[u8]) {
    let local = inner.local.lock().clone();
    if let Some(local) = local {
        let _ = local.send(frame);
    }
}

/// Towards the entry, one stage at a time.
fn send_back(inner: &Inner, ring: &Ring, frame: &[u8]) {
    if ring.is_entry() {
        to_local(inner, frame);
        return;
    }
    let back = ring.back.lock().clone();
    if let Some(back) = back {
        // A dead link here is seen by the stage on the other end, which says so itself.
        let _ = back.send(frame);
    }
}

fn start_peer(inner: &Arc<Inner>, ring: &Arc<Ring>, peer: &Arc<Peer>) {
    // The watcher waits behind a stuck write, so that write must fail inside the silence
    // limit. Half leaves room for a piece that only just finished before the one that stuck.
    let limited = peer.link.limit_writes(ring.silence_timeout / 2);
    let reading = (inner.clone(), ring.clone(), peer.clone());
    let watching = (inner.clone(), ring.clone(), peer.clone());
    let started = limited
        .and_then(|()| {
            spawn("isharded-peer", move || {
                let (inner, ring, peer) = reading;
                read_peer(&inner, &ring, &peer);
                peer_lost(&inner, &ring, &peer, "the link closed");
            })
        })
        .and_then(|_| {
            spawn("isharded-watch", move || {
                let (inner, ring, peer) = watching;
                watch_peer(&inner, &ring, &peer);
            })
        });
    // A peer nobody reads or watches would fail in silence.
    if let Err(error) = started {
        peer_lost(inner, ring, peer, &error.to_string());
    }
}

fn read_peer(inner: &Inner, ring: &Ring, peer: &Peer) {
    let mut frames = peer.link.reader(wire::MAX_BODY_BYTES);
    while let Ok((kind, body)) = frames.next() {
        *peer.heard.lock() = Instant::now();
        match kind {
            Kind::Ack => {
                let Ok(ack) = Ack::decode(body) else {
                    return;
                };
                let mut unacked = peer.unacked.lock();
                let at = unacked.iter().position(|(id, _)| *id == ack.id);
                // A round trip that carried a real row, so unlike a heartbeat it shows
                // what the row's size costs. It adds no traffic.
                if let Some((_, sent)) = at.and_then(|at| unacked.remove(at)) {
                    average(&inner.counters.hop_us, sent.elapsed().as_micros() as u64);
                }
            }
            Kind::Pong => {
                let Ok(stamp) = wire::read_beat(body) else {
                    return;
                };
                let now = peer.born.elapsed().as_micros() as u64;
                if !peer.closing {
                    average(&inner.counters.link_rtt_us, now.saturating_sub(stamp));
                }
            }
            // Coming back along the route, towards the entry.
            Kind::Result | Kind::Nack => send_back(inner, ring, frames.whole()),
            _ => return,
        }
    }
}

fn watch_peer(inner: &Inner, ring: &Ring, peer: &Peer) {
    let mut pinged = Instant::now();
    // A link with nothing to carry must still be heard from, well inside its limit.
    let ping_every = PING_EVERY.min(ring.silence_timeout / 3);
    while !peer.lost.load(Ordering::Acquire) && !peer.link.is_closed() {
        thread::sleep(WATCH_EVERY);
        let stale = peer.unacked.lock().front().is_some_and(|(_, sent)| sent.elapsed() > ring.ack_timeout);
        if stale {
            peer_lost(inner, ring, peer, "a pass was not acknowledged");
            return;
        }
        if peer.heard.lock().elapsed() > ring.silence_timeout {
            peer_lost(inner, ring, peer, "the link went silent");
            return;
        }
        if pinged.elapsed() >= ping_every {
            pinged = Instant::now();
            let stamp = peer.born.elapsed().as_micros() as u64;
            if peer.link.send(&wire::beat(Kind::Ping, stamp)).is_err() {
                peer_lost(inner, ring, peer, "the link failed");
                return;
            }
        }
    }
}

fn peer_lost(inner: &Inner, ring: &Ring, peer: &Peer, why: &str) {
    if peer.lost.swap(true, Ordering::AcqRel) {
        return;
    }
    peer.link.close();
    if peer.closing {
        // Answers walk back along the route instead. Nothing has failed.
        ring.closing.lock().take();
        return;
    }
    ring.next.lock().take();
    // Id zero means every pass this stage has sent on: nothing can say which ones were lost.
    let nack = Nack { id: 0, epoch: ring.epoch, refusal: Refusal::FailedStage, stage: peer.stage, message: why.to_string() };
    inner.counters.refusals.fetch_add(1, Ordering::Relaxed);
    send_back(inner, ring, &nack.encode());
}

fn compute_loop(inner: &Inner) {
    while let Some(batch) = next_batch(inner) {
        handle(inner, batch);
    }
}

/// Waits for work. Nothing comes back once the node is stopping.
fn next_batch(inner: &Inner) -> Option<Vec<Job>> {
    let mut jobs = inner.jobs.lock();
    loop {
        if inner.stopping.load(Ordering::Acquire) {
            return None;
        }
        if let Some(job) = jobs.pop_front() {
            return Some(take_batch(job, &mut jobs));
        }
        inner.ready.wait(&mut jobs);
    }
}

/// A job, and whatever waiting behind it can share its trip. There is no window to wait
/// out: passes gather while the stage is busy, so a lone reply goes straight through.
fn take_batch(first: Job, jobs: &mut VecDeque<Job>) -> Vec<Job> {
    let mut batch = vec![first];
    if !can_merge(&batch[0].pass) {
        return batch;
    }
    let fits = |batch: &[Job], job: &Job| {
        can_merge(&job.pass) && job.pass.epoch == batch[0].pass.epoch && batch.iter().all(|item| item.pass.slot != job.pass.slot)
    };
    while batch.len() < MAX_MERGED && jobs.front().is_some_and(|job| fits(&batch, job)) {
        batch.extend(jobs.pop_front());
    }
    batch
}

fn refuse(inner: &Inner, ring: &Ring, pass: &Pass, refusal: Refusal, message: String) {
    let nack = Nack { id: pass.id, epoch: pass.epoch, refusal, stage: ring.index, message };
    inner.counters.refusals.fetch_add(1, Ordering::Relaxed);
    send_back(inner, ring, &nack.encode());
}

fn handle(inner: &Inner, jobs: Vec<Job>) {
    let Some(ring) = inner.ring.read().clone() else {
        // No route: the link that carried this has nothing to do any more. Say so at the entry.
        for job in &jobs {
            let nack = Nack { id: job.pass.id, epoch: job.pass.epoch, refusal: Refusal::Epoch, stage: 0, message: "no route here".into() };
            to_local(inner, &nack.encode());
        }
        return;
    };
    if jobs[0].pass.epoch != ring.epoch {
        for job in &jobs {
            refuse(inner, &ring, &job.pass, Refusal::Epoch, format!("this stage is on route {}", ring.epoch));
        }
        return;
    }
    average(&inner.counters.wait_us, jobs[0].arrived.elapsed().as_micros() as u64);
    let passes: Vec<Pass> = jobs.into_iter().map(|job| job.pass).collect();
    inner.gate.lock();
    let started = Instant::now();
    let results = match inner.compute.lock().as_mut() {
        Some(compute) if passes.len() > 1 => compute.run_merged(&passes),
        Some(compute) => vec![compute.run(&passes[0])],
        None => passes.iter().map(|_| Err(Failure { refusal: Refusal::Retry, message: "this stage has nothing loaded".into() })).collect(),
    };
    average(&inner.counters.compute_us, started.elapsed().as_micros() as u64);
    // Safety: taken just above, on this thread.
    unsafe { inner.gate.unlock() };
    let counters = &inner.counters;
    counters.passes.fetch_add(passes.len() as u64, Ordering::Relaxed);
    counters.rows.fetch_add(passes.iter().map(|pass| pass.row_count() as u64).sum(), Ordering::Relaxed);
    if passes.len() > 1 {
        counters.merged_passes.fetch_add(1, Ordering::Relaxed);
        counters.merged_rows.fetch_add(passes.len() as u64, Ordering::Relaxed);
    }
    for (pass, result) in passes.into_iter().zip(results) {
        finish(inner, &ring, pass, result);
    }
}

fn finish(inner: &Inner, ring: &Ring, pass: Pass, result: Result<Output, Failure>) {
    match result {
        Err(failure) => refuse(inner, ring, &pass, failure.refusal, failure.message),
        Ok(Output::Tokens(tokens)) => {
            inner.counters.answers.fetch_add(1, Ordering::Relaxed);
            let frame = Answer { id: pass.id, epoch: pass.epoch, tokens }.encode();
            answer(inner, ring, &frame);
        }
        Ok(Output::Rows { rows, width, bf16, data }) => {
            let mut flags = pass.flags & !(FLAG_TOKENS | FLAG_BF16);
            if bf16 {
                flags |= FLAG_BF16;
            }
            let onward = Pass { flags, tokens: Vec::new(), rows, width, data, ..pass };
            forward(inner, ring, &onward);
        }
    }
}

fn answer(inner: &Inner, ring: &Ring, frame: &[u8]) {
    if ring.is_entry() && ring.is_last() {
        to_local(inner, frame);
        return;
    }
    let closing = ring.closing.lock().clone();
    if let Some(peer) = closing {
        if peer.link.send(frame).is_ok() {
            return;
        }
        peer_lost(inner, ring, &peer, "the closing link failed");
    }
    send_back(inner, ring, frame);
}

fn forward(inner: &Inner, ring: &Ring, pass: &Pass) {
    let peer = ring.next.lock().clone();
    let Some(peer) = peer else {
        refuse(inner, ring, pass, Refusal::FailedStage, "no link to the next stage".into());
        return;
    };
    peer.unacked.lock().push_back((pass.id, Instant::now()));
    if let Err(error) = peer.link.send(&pass.encode()) {
        peer_lost(inner, ring, &peer, &error.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        static SPAWNS_LEFT: Cell<u32> = const { Cell::new(u32::MAX) };
    }

    /// A test cannot really run the machine out of threads, so it says when that happens.
    pub(super) fn spend_a_spawn() -> io::Result<()> {
        let left = SPAWNS_LEFT.get();
        if left == 0 {
            return Err(io::Error::other("no threads left"));
        }
        SPAWNS_LEFT.set(left - 1);
        Ok(())
    }

    fn with_spawns<T>(left: u32, work: impl FnOnce() -> T) -> T {
        SPAWNS_LEFT.set(left);
        let done = work();
        SPAWNS_LEFT.set(u32::MAX);
        done
    }

    #[test]
    fn a_connection_with_no_thread_gives_its_place_back() {
        let node = Node::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _far = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (socket, _) = listener.accept().unwrap();
        with_spawns(0, || admit(&node.inner, socket, false));
        assert_eq!(node.inner.strangers.load(Ordering::Acquire), 0);
    }

    fn stage(index: u16) -> (Node, Address) {
        let node = Node::new().unwrap();
        node.listen("127.0.0.1", 0).unwrap();
        node.open(Plan { epoch: 1, index, count: 2, key: vec![7], ack_timeout_ms: 60_000, silence_timeout_ms: 60_000 });
        let listening = node.inner.listening.lock();
        let door = listening.as_ref().unwrap();
        let address = Address { address: format!("127.0.0.1:{}", door.port), fingerprint: door.fingerprint.clone() };
        drop(listening);
        (node, address)
    }

    #[test]
    fn a_next_stage_with_no_reader_or_no_watcher_is_lost() {
        for spawns in [0, 1] {
            let (near, _) = stage(0);
            let (_far, door) = stage(1);
            let links = Links { next: Some(door), entry: None, timeout_ms: 3000 };
            assert!(with_spawns(spawns, || near.connect(links)).is_err());
            let stats = near.stats();
            assert!(!stats.next_up);
            assert_eq!(stats.refusals, 1);
        }
    }

    #[test]
    fn a_next_stage_with_both_threads_is_up() {
        let (near, _) = stage(0);
        let (_far, door) = stage(1);
        let links = Links { next: Some(door), entry: None, timeout_ms: 3000 };
        assert_eq!(with_spawns(2, || near.connect(links)), Ok(false));
        assert!(near.stats().next_up);
    }

    #[test]
    fn a_closing_link_with_no_thread_costs_nothing_but_speed() {
        let (last, _) = stage(1);
        let (_entry, door) = stage(0);
        let links = Links { next: None, entry: Some(door), timeout_ms: 3000 };
        assert_eq!(with_spawns(1, || last.connect(links)), Ok(false));
        let stats = last.stats();
        assert!(!stats.closing_up);
        assert_eq!(stats.refusals, 0);
    }
}
