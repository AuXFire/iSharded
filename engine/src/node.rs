//! One stage's place in a ring.
//!
//! A pass moves forward from stage to stage, and nobody waits for it: each stage hands it on
//! and is free again. The last stage sends its answer straight to the entry over a closing
//! link. If that link cannot exist, the answer walks back along the route, which is slower
//! and still right, so the closing link is only ever an optimization. The last stage keeps
//! an answer until the entry acknowledges it, and keeps redialling a closing link it lost.
//!
//! A link can go silent with both its ends alive. So a stage keeps a pass until the next stage
//! acknowledges it, and when that link goes silent, it dials again once and sends what it
//! kept. The next stage drops what it has already seen, and the reply never knows.
//!
//! This code never decides what a failure means. It says what it saw, and the entry raises
//! the same errors the HTTP path raises, so every failure lands on recovery that is already
//! proven: a reply replays, and a lost worker is replaced.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::net::{IpAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use parking_lot::lock_api::RawMutex as _;
use parking_lot::{Condvar, Mutex, RawMutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::average;
use crate::link::{self, Frames, Identity, Link, LinkError};
use crate::wire::{self, Ack, Answer, Hello, Kind, Nack, Pass, Refusal, Role, FLAG_BF16, FLAG_TOKENS, HEADER_BYTES, MAC_BYTES};

/// How many connections may be mid-handshake at once, so strangers cannot pile up threads.
const MAX_STRANGERS: usize = 64;
/// And how many of those from one address, so one machine cannot take every place. A stage
/// is dialled by its neighbour and, at the entry, by the last stage: a few at most.
const STRANGERS_PER_ADDRESS: usize = 4;
/// Shorter than this, a key is a mistake. The coordinator makes 32 random bytes.
const MIN_KEY_BYTES: usize = 16;
/// Linux starts a connection over from a small window when it has sent nothing for one
/// retransmission timeout, which is never under 200 ms. Beats this often keep both ways of a
/// link from ever sitting that long, so a row after a pause is not held back a round trip.
const PING_EVERY: Duration = Duration::from_millis(100);
/// Padding a ring sends on each link it dials, before any pass, so the first long prompt finds
/// the window already open. A fresh connection starts at about 14 KB.
const WARM_BYTES: usize = 1 << 20;
const WARM_PIECE: usize = 64 << 10;
const WATCH_EVERY: Duration = Duration::from_millis(50);
const ACCEPT_EVERY: Duration = Duration::from_millis(20);
const DIAL_ATTEMPT: Duration = Duration::from_secs(5);
const DIAL_RETRY: Duration = Duration::from_millis(250);
/// Redials of a closing link start at `DIAL_RETRY` and double up to this.
const REDIAL_MOST: Duration = Duration::from_secs(5);
/// A closing link that lasted this long has earned a quick redial.
const REDIAL_STEADY: Duration = Duration::from_secs(30);
/// How long a stage tries to get the link to the next stage back. Replies are waiting, and
/// the replay that takes over if this fails needs no link at all.
const REDIAL_HOP: Duration = Duration::from_secs(2);
/// The most replies that share one trip through the model.
pub(crate) const MAX_MERGED: usize = 64;
/// How far back a slot remembers. Passes on a link keep their order, so what is left over
/// from a reply is close behind it, and a repeated answer is close behind the first.
const GENERATIONS_KEPT: usize = 16;
const ANSWERS_KEPT: usize = 1024;

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

/// The keys of the links a stage is part of, and of no others. Each link has its own, given
/// out by the coordinator over the authenticated control plane, so a stage can speak only
/// where the route puts it.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Keys {
    /// The link the stage before this one dials.
    #[serde(default)]
    pub back: Option<Vec<u8>>,
    /// The link this stage dials to the next.
    #[serde(default)]
    pub next: Option<Vec<u8>>,
    /// Held by the entry and the last stage only: the closing link, and the seal on answers.
    #[serde(default)]
    pub closing: Option<Vec<u8>>,
}

impl Keys {
    /// Whether these are exactly the keys a place in a route needs.
    fn fit(&self, index: u16, count: u16) -> bool {
        let ends = count > 1 && (index == 0 || index + 1 == count);
        let needed = [(&self.back, index > 0), (&self.next, index + 1 < count), (&self.closing, ends)];
        index < count && needed.iter().all(|(key, wanted)| key.as_ref().map_or(!wanted, |key| *wanted && key.len() >= MIN_KEY_BYTES))
    }
}

#[derive(Debug, Deserialize)]
pub struct Plan {
    pub epoch: u64,
    pub index: u16,
    pub count: u16,
    pub keys: Keys,
    #[serde(default = "default_timeout_ms")]
    pub ack_timeout_ms: u64,
    #[serde(default = "default_timeout_ms")]
    pub silence_timeout_ms: u64,
    /// Whether one-row passes from different slots share a trip. A caller that does not say
    /// gets them merged, as before the plan carried it.
    #[serde(default = "default_merge_passes")]
    pub merge_passes: bool,
    /// The most passes one trip may hold. A route that wants exact replies sets it to what
    /// this stage's probe found merges as rows run alone.
    #[serde(default)]
    pub merge_most: Option<u32>,
}

impl Plan {
    /// How many passes one trip takes, which is one when nothing merges.
    fn trip_most(&self) -> usize {
        match (self.merge_passes, self.merge_most) {
            (false, _) => 1,
            (true, most) => most.map_or(MAX_MERGED, |most| (most as usize).clamp(1, MAX_MERGED)),
        }
    }
}

fn default_timeout_ms() -> u64 {
    5000
}

fn default_merge_passes() -> bool {
    true
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
    /// Answers the entry threw away because the last stage had not sealed them.
    pub forged: u64,
    /// Passes and answers sent again, because their link died or nobody acknowledged them.
    pub resent: u64,
    /// Answers at the last stage still waiting for the entry's acknowledgement.
    pub answers_held: u64,
    pub compute_us: u64,
    /// Microseconds of compute since the ring opened, and the batches that took them, so a
    /// reader can tell how busy the stage was between two looks.
    pub compute_us_total: u64,
    pub batches_total: u64,
    /// The hop trace, each an average of local times, as no two stages share a clock. From a
    /// pass's compute ending to its frame being queued on the link onward.
    pub queue_us: u64,
    /// From that frame being queued to its last byte reaching the socket.
    pub write_us: u64,
    /// From a pass's frame being read whole to it being queued for compute.
    pub enqueue_us: u64,
    /// From being queued for compute to compute starting.
    pub wait_us: u64,
    pub link_rtt_us: u64,
    /// A pass's trip to the next stage and its acknowledgement's trip back.
    pub hop_us: u64,
    /// At the last stage: picking a token from the logits, inside compute.
    pub sample_us: u64,
    /// At the entry: from an answer reaching the worker to the worker's next pass for that
    /// reply arriving, which is the worker's own turnaround and the local link both ways.
    pub turnaround_us: u64,
    pub next_up: bool,
    pub closing_up: bool,
    pub epoch: Option<u64>,
    /// The next route, opened beside the current one and not yet promoted.
    pub staged_epoch: Option<u64>,
    /// Former routes still finishing what they took.
    pub retiring_epochs: Vec<u64>,
    /// "idle", "dialling", "up", or why it failed.
    pub connect: String,
    /// The same for the staged ring.
    pub staged_connect: String,
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
    forged: AtomicU64,
    resent: AtomicU64,
    compute_us: AtomicU64,
    compute_us_total: AtomicU64,
    batches_total: AtomicU64,
    queue_us: AtomicU64,
    enqueue_us: AtomicU64,
    wait_us: AtomicU64,
    link_rtt_us: AtomicU64,
    hop_us: AtomicU64,
    turnaround_us: AtomicU64,
}

struct Job {
    pass: Pass,
    arrived: Instant,
    /// Why this pass is refused, when it is. A refusal waits its turn behind its slot's older
    /// passes, so it cannot reach the entry, free the slot there and let a new reply begin
    /// while an older pass of the same slot is still to run here.
    refused: Option<String>,
}

/// What a stage knows of the replies on one slot. Generations are random, so the only way to
/// know one is old is to have seen the one that replaced it.
#[derive(Default)]
struct Line {
    generation: u64,
    newest_pass: u64,
    ended: VecDeque<u64>,
    answered: VecDeque<u64>,
}

impl Line {
    /// Whether a pass is an old reply's, or one already seen. A client numbers its passes in
    /// order, and a link keeps them in it.
    fn refuses(&self, pass: &Pass) -> bool {
        self.ended.contains(&pass.generation) || (pass.generation == self.generation && pass.id <= self.newest_pass)
    }

    fn begin(&mut self, generation: u64) {
        self.ended.push_back(std::mem::replace(&mut self.generation, generation));
        keep_newest(&mut self.ended, GENERATIONS_KEPT);
        self.answered.clear();
    }

    /// Whether an answer belongs to this slot's reply and is the first with its id.
    fn takes(&mut self, answer: &Answer) -> bool {
        if answer.generation != self.generation || self.answered.contains(&answer.id) {
            return false;
        }
        self.answered.push_back(answer.id);
        keep_newest(&mut self.answered, ANSWERS_KEPT);
        true
    }
}

fn keep_newest(list: &mut VecDeque<u64>, keep: usize) {
    if list.len() > keep {
        list.drain(..list.len() - keep);
    }
}

/// Connections that have not proven themselves yet, in all and from each address.
#[derive(Default)]
struct Strangers {
    total: usize,
    from: HashMap<IpAddr, usize>,
}

impl Strangers {
    fn enter(&mut self, address: IpAddr) -> bool {
        let mine = self.from.get(&address).copied().unwrap_or(0);
        if self.total >= MAX_STRANGERS || mine >= STRANGERS_PER_ADDRESS {
            return false;
        }
        self.from.insert(address, mine + 1);
        self.total += 1;
        true
    }

    fn leave(&mut self, address: IpAddr) {
        if let Some(mine) = self.from.get_mut(&address) {
            *mine -= 1;
            self.total -= 1;
            if *mine == 0 {
                self.from.remove(&address);
            }
        }
    }
}

struct Peer {
    link: Arc<Link>,
    stage: u16,
    closing: bool,
    heard: Mutex<Instant>,
    lost: AtomicBool,
    /// The far end has answered something on this link, so another try may be worth it.
    proven: AtomicBool,
    /// The clock heartbeats are stamped against, so a round trip needs no shared time.
    born: Instant,
}

impl Peer {
    fn new(link: Arc<Link>, stage: u16, closing: bool) -> Arc<Peer> {
        let now = Instant::now();
        Arc::new(Peer { link, stage, closing, heard: Mutex::new(now), lost: AtomicBool::new(false), proven: AtomicBool::new(false), born: now })
    }
}

/// A pass the next stage has not acknowledged, or an answer the entry has not.
struct Held {
    id: u64,
    frame: Vec<u8>,
    sent: Instant,
}

struct Ring {
    epoch: u64,
    index: u16,
    count: u16,
    keys: Keys,
    ack_timeout: Duration,
    silence_timeout: Duration,
    next: Mutex<Option<Arc<Peer>>>,
    closing: Mutex<Option<Arc<Peer>>>,
    back: Mutex<Option<Arc<Link>>>,
    entry_closing: Mutex<Option<Arc<Link>>>,
    answers: Mutex<VecDeque<Held>>,
    /// Passes on their way to the next stage. Only this order can go wrong, as the next stage
    /// takes a pass numbered below one it has seen for a repeat.
    sent: Mutex<VecDeque<Held>>,
    /// Held while a pass joins `sent` and goes on the link, so a pass and a resend cannot
    /// cross. Separate, as a send can wait and the reader and the watcher must not.
    forwarding: Mutex<()>,
    next_door: Mutex<Option<Address>>,
    next_dialling: AtomicBool,
    shut: AtomicBool,
    /// The most passes one of this ring's trips may hold, from its plan.
    merge_most: usize,
    /// At the entry: (slot, id) of passes sent on and not yet answered or refused, so a
    /// retiring ring knows when nothing of it is still out on the route.
    in_flight: Mutex<HashSet<(u32, u64)>>,
}

impl Ring {
    fn is_entry(&self) -> bool {
        self.index == 0
    }

    fn is_last(&self) -> bool {
        self.index + 1 == self.count
    }

    fn is_shut(&self) -> bool {
        self.shut.load(Ordering::Acquire)
    }

    fn shut(&self) {
        self.shut.store(true, Ordering::Release);
        for peer in [self.next.lock().take(), self.closing.lock().take()].into_iter().flatten() {
            peer.lost.store(true, Ordering::Release);
            peer.link.close();
        }
        for link in [self.back.lock().take(), self.entry_closing.lock().take()].into_iter().flatten() {
            link.close();
        }
    }

    /// Whether the oldest in a list has waited too long for its acknowledgement.
    fn is_stale(&self, list: &Mutex<VecDeque<Held>>) -> bool {
        list.lock().front().is_some_and(|held| held.sent.elapsed() > self.ack_timeout)
    }

    fn answer_acknowledged(&self, id: u64) {
        let mut answers = self.answers.lock();
        if let Some(at) = answers.iter().position(|held| held.id == id) {
            answers.remove(at);
        }
    }
}

/// The rings a node holds. `current` carries replies; `staged` is the next route, opened
/// beside it so its links are up and proven before the flip; `retiring` are former current
/// rings, refusing new passes and finishing what they already took.
///
/// While more than one epoch is held, every slot has one owner, and a pass whose epoch is not
/// its slot's owner is refused, so two replies never share a slot of one KV. That holds from
/// `stage` through `promote` until the retiring ring has drained or is closed: the old
/// route's replies keep their slots while their last passes are still out on the route.
#[derive(Default)]
struct Rings {
    current: Option<Arc<Ring>>,
    staged: Option<Arc<Ring>>,
    retiring: Vec<Arc<Ring>>,
    /// Slots given to an epoch while epochs overlap.
    owners: HashMap<u32, u64>,
    /// Who owns a slot not in `owners`: the epoch that was current when the overlap began.
    default_owner: Option<u64>,
}

impl Rings {
    /// A ring that takes new passes for this epoch.
    fn taking(&self, epoch: u64) -> Option<&Arc<Ring>> {
        [&self.current, &self.staged].into_iter().flatten().find(|ring| ring.epoch == epoch)
    }

    /// Any ring with this epoch, a retiring one included, for what it already took.
    fn any(&self, epoch: u64) -> Option<&Arc<Ring>> {
        self.taking(epoch).or_else(|| self.retiring.iter().find(|ring| ring.epoch == epoch))
    }

    fn is_empty(&self) -> bool {
        self.current.is_none() && self.staged.is_none()
    }

    /// Whether more than one epoch is held, so slots have owners.
    fn overlapping(&self) -> bool {
        self.staged.is_some() || !self.retiring.is_empty()
    }

    /// The epoch a slot belongs to while epochs overlap.
    fn owner(&self, slot: u32) -> Option<u64> {
        self.owners.get(&slot).copied().or(self.default_owner).or(self.current.as_ref().map(|ring| ring.epoch))
    }

    /// After a ring leaves: slots of an epoch no longer held go back to the current one, and
    /// with one epoch left nothing has an owner any more.
    fn settle(&mut self) {
        if !self.overlapping() {
            self.owners.clear();
            self.default_owner = None;
            return;
        }
        let held: Vec<u64> = [&self.current, &self.staged].into_iter().flatten().chain(&self.retiring).map(|ring| ring.epoch).collect();
        self.owners.retain(|_, epoch| held.contains(epoch));
        if !self.default_owner.is_some_and(|epoch| held.contains(&epoch)) {
            self.default_owner = self.current.as_ref().map(|ring| ring.epoch);
        }
    }

    /// Takes a ring of this epoch out, wherever it is held.
    fn remove(&mut self, epoch: u64) -> Option<Arc<Ring>> {
        let ring = if self.current.as_ref().is_some_and(|ring| ring.epoch == epoch) {
            self.current.take()
        } else if self.staged.as_ref().is_some_and(|ring| ring.epoch == epoch) {
            self.staged.take()
        } else {
            let at = self.retiring.iter().position(|ring| ring.epoch == epoch)?;
            Some(self.retiring.remove(at))
        };
        self.settle();
        ring
    }
}

/// No batch is running.
const IDLE: u64 = u64::MAX;

struct Inner {
    identity: Identity,
    /// Whoever holds this may touch the model. Python takes it for what it does rarely.
    gate: RawMutex,
    compute: Mutex<Option<Box<dyn Compute>>>,
    rings: RwLock<Rings>,
    /// The epoch of the batch the compute thread is running, or `IDLE`. Set inside `jobs`.
    busy: AtomicU64,
    local: Mutex<Option<Arc<Link>>>,
    local_key: [u8; 32],
    jobs: Mutex<VecDeque<Job>>,
    /// Taken only inside `jobs`, or with no other lock held.
    lines: Mutex<HashMap<u32, Line>>,
    ready: Condvar,
    /// The most passes one trip takes, as the route chose. Kept here so taking work needs no
    /// lock but `jobs`.
    merge_most: AtomicUsize,
    stopping: AtomicBool,
    strangers: Mutex<Strangers>,
    counters: Counters,
    /// At the entry: when each slot's newest answer went to the worker, and for which reply.
    answered: Mutex<HashMap<u32, (u64, Instant)>>,
    listening: Mutex<Option<Listening>>,
    accepting: Mutex<Vec<thread::JoinHandle<()>>>,
    connect: Mutex<String>,
    staged_connect: Mutex<String>,
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
            rings: RwLock::default(),
            busy: AtomicU64::new(IDLE),
            local: Mutex::new(None),
            local_key: link::random_bytes(),
            jobs: Mutex::new(VecDeque::new()),
            lines: Mutex::default(),
            ready: Condvar::new(),
            merge_most: AtomicUsize::new(MAX_MERGED),
            stopping: AtomicBool::new(false),
            strangers: Mutex::default(),
            counters: Counters::default(),
            answered: Mutex::default(),
            listening: Mutex::new(None),
            accepting: Mutex::new(Vec::new()),
            connect: Mutex::new("idle".to_string()),
            staged_connect: Mutex::new("idle".to_string()),
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

    /// Takes a place in a route. Anything from an earlier route ends here, a staged one too.
    pub fn open(&self, plan: Plan) -> Result<(), String> {
        let ring = new_ring(plan)?;
        self.inner.merge_most.store(ring.merge_most, Ordering::Release);
        let mut rings = self.inner.rings.write();
        // With the route held, so a batch of the old one that ends now is not counted here.
        self.inner.counters.compute_us_total.store(0, Ordering::Relaxed);
        self.inner.counters.batches_total.store(0, Ordering::Relaxed);
        let old = std::mem::replace(&mut *rings, Rings { current: Some(ring), ..Rings::default() });
        drop(rings);
        shut_all(old);
        Ok(())
    }

    /// Opens the next route's ring beside the current one, which goes on carrying replies.
    /// Its links are dialled with `connect_staged`, under its own epoch's keys.
    pub fn stage(&self, plan: Plan) -> Result<(), String> {
        let ring = new_ring(plan)?;
        let mut rings = self.inner.rings.write();
        let restaging = rings.staged.as_ref().is_some_and(|staged| staged.epoch == ring.epoch);
        if rings.any(ring.epoch).is_some() && !restaging {
            return Err(format!("route {} is already held here", ring.epoch));
        }
        if rings.default_owner.is_none() {
            rings.default_owner = rings.current.as_ref().map(|ring| ring.epoch);
        }
        *self.inner.staged_connect.lock() = "idle".to_string();
        let old = rings.staged.replace(ring);
        drop(rings);
        if let Some(old) = old {
            old.shut();
        }
        Ok(())
    }

    /// Makes the staged ring the current one. The old current ring takes no new passes and is
    /// shut once what it already took has finished and been acknowledged, or after its limits.
    pub fn promote(&self, epoch: u64) -> Result<(), String> {
        let mut rings = self.inner.rings.write();
        if !rings.staged.as_ref().is_some_and(|ring| ring.epoch == epoch) {
            return Err(format!("route {epoch} is not staged here"));
        }
        let new = rings.staged.take();
        let old = std::mem::replace(&mut rings.current, new);
        if let Some(current) = &rings.current {
            self.inner.merge_most.store(current.merge_most, Ordering::Release);
        }
        // The totals are the current route's, so they start again with it.
        self.inner.counters.compute_us_total.store(0, Ordering::Relaxed);
        self.inner.counters.batches_total.store(0, Ordering::Relaxed);
        // Its links were dialled as the staged ring's, so their word goes with it.
        let dialled = std::mem::replace(&mut *self.inner.staged_connect.lock(), "idle".to_string());
        *self.inner.connect.lock() = dialled;
        // Slots keep their owners: the old route's replies hold theirs until it has drained.
        if let Some(old) = &old {
            rings.retiring.push(old.clone());
        }
        rings.settle();
        drop(rings);
        if let Some(old) = old {
            let inner = self.inner.clone();
            let ring = old.clone();
            if spawn("isharded-retire", move || retire(&inner, &ring)).is_err() {
                // Nothing will wait for it, so it ends now.
                take_retiring(&self.inner, &old);
                old.shut();
            }
        }
        Ok(())
    }

    /// While two epochs overlap: these slots now belong to this epoch. A pass on one of them
    /// for the other epoch is refused, so two replies never share a slot of one KV.
    /// With one epoch held there is nothing to give, and nothing is kept.
    pub fn give_slots(&self, epoch: u64, slots: &[u32]) -> Result<(), String> {
        let mut rings = self.inner.rings.write();
        if rings.taking(epoch).is_none() {
            return Err(format!("route {epoch} is not held here"));
        }
        if rings.overlapping() {
            for slot in slots {
                rings.owners.insert(*slot, epoch);
            }
        }
        Ok(())
    }

    /// Closes one epoch's ring, staged, current or retiring, and leaves any other alone: a
    /// staged route that is given up, or a retiring one that should not wait any longer.
    pub fn close_epoch(&self, epoch: u64) -> bool {
        let ring = self.inner.rings.write().remove(epoch);
        let closed = ring.is_some();
        if let Some(ring) = ring {
            ring.shut();
        }
        closed
    }

    /// Dials the next stage and, at the last stage, the entry. The closing link may fail.
    pub fn connect(&self, links: Links) -> Result<bool, String> {
        let ring = self.inner.rings.read().current.clone().ok_or("no route has been opened")?;
        connect(&self.inner, &ring, &links)
    }

    /// The same for the staged ring. The current ring's links are left alone.
    pub fn connect_staged(&self, links: Links) -> Result<bool, String> {
        let ring = self.inner.rings.read().staged.clone().ok_or("no route has been staged")?;
        connect(&self.inner, &ring, &links)
    }

    /// The same, off this thread, so the stage goes on serving meanwhile.
    pub fn connect_in_background(&self, links: Links) {
        dial_in_background(&self.inner, links, false);
    }

    /// The same for the staged ring; `staged_connect` in the stats says how it went.
    pub fn connect_staged_in_background(&self, links: Links) {
        dial_in_background(&self.inner, links, true);
    }

    /// The flip at a stage that serves both routes: every slot goes to the staged route, which
    /// becomes the current one, and the old one retires. Returns once nothing of the old
    /// route is queued or running here, or after `wait`, saying which. From then on a pass of
    /// the old route is refused here before it touches a slot, so a reply of the new route can
    /// begin on any slot.
    pub fn flip(&self, epoch: u64, slots: &[u32], wait: Duration) -> Result<bool, String> {
        let old = self.inner.rings.read().current.as_ref().map(|ring| ring.epoch);
        self.give_slots(epoch, slots)?;
        self.promote(epoch)?;
        let Some(old) = old else {
            return Ok(true);
        };
        let deadline = Instant::now() + wait;
        while !quiet(&self.inner, old) {
            if Instant::now() >= deadline {
                return Ok(false);
            }
            thread::sleep(Duration::from_millis(1));
        }
        Ok(true)
    }

    /// Tells the route that nothing here will finish. Id zero covers every pass at once, so
    /// this needs no word from the compute thread, which may be the thing that hung.
    pub fn refuser(&self) -> impl FnOnce(String) + Send + 'static {
        let inner = self.inner.clone();
        move |message| {
            let ring = inner.rings.read().current.clone();
            if let Some(ring) = ring {
                let nack = Nack { id: 0, generation: 0, epoch: ring.epoch, refusal: Refusal::FailedStage, stage: ring.index, message };
                inner.counters.refusals.fetch_add(1, Ordering::Relaxed);
                send_back(&inner, &ring, &nack.encode());
            }
        }
    }

    pub fn close_ring(&self) {
        let old = std::mem::take(&mut *self.inner.rings.write());
        shut_all(old);
    }

    pub fn stats(&self) -> Stats {
        let counters = &self.inner.counters;
        let (ring, staged, retiring) = {
            let rings = self.inner.rings.read();
            (rings.current.clone(), rings.staged.as_ref().map(|ring| ring.epoch), rings.retiring.iter().map(|ring| ring.epoch).collect())
        };
        Stats {
            staged_epoch: staged,
            retiring_epochs: retiring,
            staged_connect: self.inner.staged_connect.lock().clone(),
            passes: counters.passes.load(Ordering::Relaxed),
            rows: counters.rows.load(Ordering::Relaxed),
            merged_passes: counters.merged_passes.load(Ordering::Relaxed),
            merged_rows: counters.merged_rows.load(Ordering::Relaxed),
            answers: counters.answers.load(Ordering::Relaxed),
            refusals: counters.refusals.load(Ordering::Relaxed),
            dropped: counters.dropped.load(Ordering::Relaxed),
            forged: counters.forged.load(Ordering::Relaxed),
            resent: counters.resent.load(Ordering::Relaxed),
            answers_held: ring.as_ref().map_or(0, |item| item.answers.lock().len() as u64),
            compute_us: counters.compute_us.load(Ordering::Relaxed),
            compute_us_total: counters.compute_us_total.load(Ordering::Relaxed),
            batches_total: counters.batches_total.load(Ordering::Relaxed),
            queue_us: counters.queue_us.load(Ordering::Relaxed),
            write_us: ring.as_ref().and_then(|item| item.next.lock().clone().or_else(|| item.closing.lock().clone())).map_or(0, |peer| peer.link.write_us()),
            enqueue_us: counters.enqueue_us.load(Ordering::Relaxed),
            wait_us: counters.wait_us.load(Ordering::Relaxed),
            link_rtt_us: counters.link_rtt_us.load(Ordering::Relaxed),
            hop_us: counters.hop_us.load(Ordering::Relaxed),
            sample_us: crate::sample::average_us(),
            turnaround_us: counters.turnaround_us.load(Ordering::Relaxed),
            next_up: ring.as_ref().is_some_and(|item| item.next.lock().is_some() || item.next_dialling.load(Ordering::Acquire)),
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

fn new_ring(plan: Plan) -> Result<Arc<Ring>, String> {
    if !plan.keys.fit(plan.index, plan.count) {
        return Err("these keys do not fit this place in the route".to_string());
    }
    let merge_most = plan.trip_most();
    Ok(Arc::new(Ring {
        epoch: plan.epoch,
        index: plan.index,
        count: plan.count,
        keys: plan.keys,
        ack_timeout: Duration::from_millis(plan.ack_timeout_ms),
        silence_timeout: Duration::from_millis(plan.silence_timeout_ms),
        next: Mutex::new(None),
        closing: Mutex::new(None),
        back: Mutex::new(None),
        entry_closing: Mutex::new(None),
        answers: Mutex::new(VecDeque::new()),
        sent: Mutex::new(VecDeque::new()),
        forwarding: Mutex::new(()),
        next_door: Mutex::new(None),
        next_dialling: AtomicBool::new(false),
        shut: AtomicBool::new(false),
        merge_most,
        in_flight: Mutex::default(),
    }))
}

fn shut_all(rings: Rings) {
    for ring in [rings.current, rings.staged].into_iter().flatten().chain(rings.retiring) {
        ring.shut();
    }
}

fn dial_in_background(inner: &Arc<Inner>, links: Links, staged: bool) {
    let state = if staged { &inner.staged_connect } else { &inner.connect };
    *state.lock() = "dialling".to_string();
    let ring = {
        let rings = inner.rings.read();
        if staged { rings.staged.clone() } else { rings.current.clone() }
    };
    let worker = inner.clone();
    let spawned = thread::Builder::new().name("isharded-dial".into()).spawn(move || {
        let what = if staged { "no route has been staged" } else { "no route has been opened" };
        let outcome = ring.ok_or_else(|| what.to_string()).and_then(|ring| connect(&worker, &ring, &links));
        let state = if staged { &worker.staged_connect } else { &worker.connect };
        *state.lock() = match outcome {
            Ok(_) => "up".to_string(),
            Err(error) => error,
        };
    });
    if spawned.is_err() {
        *state.lock() = "could not start a thread".to_string();
    }
}

/// Whether nothing of this epoch is queued or running here.
fn quiet(inner: &Inner, epoch: u64) -> bool {
    let jobs = inner.jobs.lock();
    inner.busy.load(Ordering::Acquire) != epoch && !jobs.iter().any(|job| job.pass.epoch == epoch)
}

fn take_retiring(inner: &Inner, ring: &Arc<Ring>) {
    let mut rings = inner.rings.write();
    // By pointer: a ring closed and opened again under the same epoch is not this one.
    if let Some(at) = rings.retiring.iter().position(|held| Arc::ptr_eq(held, ring)) {
        rings.retiring.remove(at);
        rings.settle();
    }
}

/// Whether a retiring ring has nothing left: no job of its epoch queued or running, nothing
/// waiting for the next stage's acknowledgement, and no answer waiting for the entry's.
fn drained(inner: &Inner, ring: &Ring) -> bool {
    let idle = quiet(inner, ring.epoch);
    // The entry waits for every answer; a later stage also waits for the stage before it to
    // let go of the hop, as only that stage knows nothing more of this epoch is coming.
    let route_quiet = if ring.is_entry() { ring.in_flight.lock().is_empty() } else { ring.back.lock().is_none() };
    idle && route_quiet && ring.sent.lock().is_empty() && ring.answers.lock().is_empty()
}

/// Waits for an old current ring to finish what it took, then shuts it.
fn retire(inner: &Inner, ring: &Arc<Ring>) {
    let deadline = Instant::now() + ring.ack_timeout + ring.silence_timeout;
    while !ring.is_shut() && !drained(inner, ring) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    take_retiring(inner, ring);
    ring.shut();
}

fn spawn(name: &str, work: impl FnOnce() + Send + 'static) -> io::Result<thread::JoinHandle<()>> {
    #[cfg(test)]
    tests::spend_a_spawn()?;
    thread::Builder::new().name(name.into()).spawn(work)
}

fn connect(inner: &Arc<Inner>, ring: &Arc<Ring>, links: &Links) -> Result<bool, String> {
    if let Some(next) = &links.next {
        let key = ring.keys.next.as_deref().ok_or("this stage has no key for a next stage")?;
        let link = dial_with_retry(next, key, ring, Role::Hop, Duration::from_millis(links.timeout_ms)).map_err(|error| error.to_string())?;
        let peer = Peer::new(link, ring.index + 1, false);
        *ring.next.lock() = Some(peer.clone());
        start_peer(inner, ring, &peer);
        if peer.lost.load(Ordering::Acquire) {
            return Err("the link to the next stage could not start".to_string());
        }
        warm(&peer.link);
        *ring.next_door.lock() = Some(next.clone());
    }
    let mut closing = false;
    // Without the closing key there is no closing link, and answers walk back.
    if let (Some(entry), Some(key)) = (&links.entry, &ring.keys.closing) {
        if let Ok(link) = dial_with_retry(entry, key, ring, Role::Closing, Duration::from_millis(links.timeout_ms)) {
            closing = start_closing(inner, ring, link.clone());
            if closing {
                warm(&link);
            }
        }
        let (inner, ring, entry) = (inner.clone(), ring.clone(), entry.clone());
        // With no thread for this, answers still walk back. Only speed is lost.
        let _ = spawn("isharded-closing", move || keep_closing(&inner, &ring, &entry));
    }
    Ok(closing)
}

/// Queues padding ahead of the first pass. The queue holds far more than this, so nothing
/// waits, and a link that fails here is found by its watcher like any other. A link dialled
/// again later carries passes it kept at once, so it is not padded.
fn warm(link: &Link) {
    let piece = wire::pad(WARM_PIECE);
    for _ in 0..WARM_BYTES / WARM_PIECE {
        if link.send(&piece).is_err() {
            return;
        }
    }
}

fn start_closing(inner: &Arc<Inner>, ring: &Arc<Ring>, link: Arc<Link>) -> bool {
    let peer = Peer::new(link, 0, true);
    *ring.closing.lock() = Some(peer.clone());
    start_peer(inner, ring, &peer);
    !peer.lost.load(Ordering::Acquire)
}

/// At the last stage, for as long as the ring lasts: brings a lost closing link back, and
/// while it is down, tries again with any answer that has heard nothing.
fn keep_closing(inner: &Arc<Inner>, ring: &Arc<Ring>, entry: &Address) {
    let Some(key) = ring.keys.closing.as_deref() else {
        return;
    };
    let mut wait = DIAL_RETRY;
    let mut next_dial = Instant::now() + wait;
    let mut began = Instant::now();
    while !ring.is_shut() {
        thread::sleep(WATCH_EVERY);
        if ring.closing.lock().is_some() {
            if began.elapsed() > REDIAL_STEADY {
                wait = DIAL_RETRY;
            }
            next_dial = Instant::now() + wait;
            continue;
        }
        if ring.is_stale(&ring.answers) {
            resend_answers(inner, ring);
        }
        if Instant::now() < next_dial {
            continue;
        }
        if let Ok(link) = link::dial(&entry.address, Some(&entry.fingerprint), key, Role::Closing, ring.epoch, DIAL_ATTEMPT) {
            began = Instant::now();
            start_closing(inner, ring, link);
            if ring.is_shut() {
                ring.shut();
            }
        }
        wait = (wait * 2).min(REDIAL_MOST);
        next_dial = Instant::now() + wait;
    }
}

fn dial_with_retry(target: &Address, key: &[u8], ring: &Ring, role: Role, patience: Duration) -> Result<Arc<Link>, LinkError> {
    // A neighbour may still be starting, so the first refusal means little.
    let deadline = Instant::now() + patience;
    loop {
        match link::dial(&target.address, Some(&target.fingerprint), key, role, ring.epoch, DIAL_ATTEMPT) {
            Ok(link) => return Ok(link),
            Err(error) if Instant::now() >= deadline => return Err(error),
            Err(_) => thread::sleep(DIAL_RETRY),
        }
    }
}

fn accept_loop(inner: &Arc<Inner>, listener: &TcpListener, encrypted: bool) {
    while !inner.stopping.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((socket, peer)) => admit(inner, socket, peer.ip(), encrypted),
            Err(_) => thread::sleep(ACCEPT_EVERY),
        }
    }
}

/// Gives a connection a thread, if there is room for another stranger from where it came.
fn admit(inner: &Arc<Inner>, socket: TcpStream, peer: IpAddr, encrypted: bool) {
    // On some systems a socket takes this from the listener, and links block.
    if socket.set_nonblocking(false).is_err() || !inner.strangers.lock().enter(peer) {
        return;
    }
    let serving = inner.clone();
    if spawn("isharded-link", move || serve(&serving, socket, peer, encrypted)).is_err() {
        inner.strangers.lock().leave(peer);
    }
}

/// The key a link someone dials here must prove. Each link has its own, so holding one opens
/// no other: only the stage before dials in as a hop, and only the last stage closes the ring.
fn key_for(inner: &Inner, role: Role, epoch: u64, encrypted: bool) -> Option<Vec<u8>> {
    // Each epoch's ring has its own keys, so a link proves which ring it belongs to.
    let ring = || inner.rings.read().any(epoch).cloned();
    match (role, encrypted) {
        (Role::Local, false) => Some(inner.local_key.to_vec()),
        (Role::Hop, true) => ring()?.keys.back.clone(),
        (Role::Closing, true) => ring().filter(|ring| ring.is_entry())?.keys.closing.clone(),
        _ => None,
    }
}

fn serve(inner: &Arc<Inner>, socket: TcpStream, peer: IpAddr, encrypted: bool) {
    let identity = encrypted.then_some(&inner.identity);
    let accepted = link::accept(socket, identity, link::HANDSHAKE, |role, epoch| key_for(inner, role, epoch, encrypted));
    inner.strangers.lock().leave(peer);
    let Ok((link, frames, hello)) = accepted else {
        return;
    };
    // Whatever happens below, the link's writer ends with this.
    match hello.role {
        Role::Local => {
            if let Some(old) = inner.local.lock().replace(link.clone()) {
                old.close();
            }
            serve_passes(inner, None, &link, frames);
        }
        Role::Hop => {
            if let Some(ring) = attach(inner, &hello, |ring| ring.back.lock().replace(link.clone())) {
                if keep_to_limits(&link, &ring) {
                    serve_passes(inner, Some(&ring), &link, frames);
                }
                forget(&ring.back, &link);
            }
        }
        Role::Closing => {
            if let Some(ring) = attach(inner, &hello, |ring| ring.entry_closing.lock().replace(link.clone())) {
                if keep_to_limits(&link, &ring) {
                    serve_answers(inner, &ring, &link, frames);
                }
                forget(&ring.entry_closing, &link);
            }
        }
    }
    link.close();
}

fn attach(inner: &Inner, hello: &Hello, put: impl Fn(&Ring) -> Option<Arc<Link>>) -> Option<Arc<Ring>> {
    let ring = inner.rings.read().any(hello.epoch).cloned()?;
    if let Some(old) = put(&ring) {
        old.close();
    }
    Some(ring)
}

/// Gives a link someone dialled the limits of a link with a watcher. The far end's heartbeats
/// mean silence past the limit is a dead link, and only a read limit can see that here.
fn keep_to_limits(link: &Link, ring: &Ring) -> bool {
    link.limit_writes(ring.silence_timeout / 2).and_then(|()| link.limit_reads(ring.silence_timeout)).is_ok()
}

/// Lets go of a link, unless a newer one has already taken its place.
fn forget(place: &Mutex<Option<Arc<Link>>>, link: &Arc<Link>) {
    let mut place = place.lock();
    if place.as_ref().is_some_and(|held| Arc::ptr_eq(held, link)) {
        *place = None;
    }
}

/// Answers a heartbeat. False means the link is finished.
fn answer_ping(link: &Link, body: &[u8]) -> bool {
    // A pong that finds no room is only a missed heartbeat.
    wire::read_beat(body).is_ok_and(|stamp| !matches!(link.beat(&wire::beat(Kind::Pong, stamp)), Err(LinkError::Closed | LinkError::Io(_))))
}

/// Reads passes off a link. Taking one off is what the acknowledgement means, so a stage
/// busy with a long pass still answers its link and is not taken for dead.
/// `ring` is the ring a hop belongs to; the worker's local link has none and is not acknowledged.
fn serve_passes(inner: &Inner, ring: Option<&Arc<Ring>>, link: &Link, mut frames: Frames) {
    let acknowledge = ring.is_some();
    while let Ok((kind, body)) = frames.next() {
        match kind {
            Kind::Pass => {
                let complete = Instant::now();
                let Ok(pass) = Pass::decode(body) else {
                    return;
                };
                if acknowledge && link.beat(&Ack { id: pass.id }.encode()).is_err() {
                    return;
                }
                if !acknowledge {
                    note_turnaround(inner, &pass, complete);
                }
                enqueue(inner, pass, ring);
                average(&inner.counters.enqueue_us, complete.elapsed().as_micros() as u64);
            }
            Kind::Ack if acknowledge => {
                let (Ok(ack), Some(ring)) = (Ack::decode(body), ring) else {
                    return;
                };
                pass_on_acknowledgement(ring, ack);
            }
            Kind::Ping if answer_ping(link, body) => {}
            Kind::Pad => {}
            _ => return,
        }
    }
}

/// At the entry: how long the worker took to send a reply's next pass after its answer.
/// Only the same reply counts, as the next reply on a slot may start minutes later.
fn note_turnaround(inner: &Inner, pass: &Pass, arrived: Instant) {
    let answered = inner.take_answered(pass.slot);
    if let Some((generation, at)) = answered {
        if generation == pass.generation {
            average(&inner.counters.turnaround_us, arrived.saturating_duration_since(at).as_micros() as u64);
        }
    }
}

impl Inner {
    fn take_answered(&self, slot: u32) -> Option<(u64, Instant)> {
        self.answered.lock().remove(&slot)
    }

    fn answered_now(&self, slot: u32, generation: u64) {
        self.answered.lock().insert(slot, (generation, Instant::now()));
    }
}

/// An answer that walked back has its acknowledgement walk forward, the way passes go.
/// An acknowledgement carries no epoch, so it goes by the ring of the link it came on.
fn pass_on_acknowledgement(ring: &Ring, ack: Ack) {
    if ring.is_last() {
        ring.answer_acknowledged(ack.id);
    } else if let Some(peer) = ring.next.lock().clone() {
        // Lost, it costs one resend, and the entry drops the repeat.
        let _ = peer.link.beat(&ack.encode());
    }
}

/// Queues a pass. A rewind says that whatever waits past it has already been thrown away,
/// so it goes now and the rewind does not sit behind it. Nobody expects an answer to those:
/// passes run in order, so once the newest pass of a reply is back, nothing of it is left.
/// A new generation says the same of all its slot's queue. The entry gave up on that reply.
/// `via` is the ring whose hop carried the pass; none for the worker's local link.
fn enqueue(inner: &Inner, pass: Pass, via: Option<&Arc<Ring>>) {
    if let Err(message) = admits(inner, &pass, via) {
        // The slot's line is left alone: this pass may not touch another route's replies.
        inner.jobs.lock().push_back(Job { pass, arrived: Instant::now(), refused: Some(message) });
        inner.ready.notify_one();
        return;
    }
    let mut jobs = inner.jobs.lock();
    let before = jobs.len();
    let mut lines = inner.lines.lock();
    let line = lines.entry(pass.slot).or_default();
    if line.refuses(&pass) {
        inner.counters.dropped.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if pass.generation != line.generation {
        line.begin(pass.generation);
        // A refusal still owes its answer, whatever reply now has the slot.
        jobs.retain(|job| job.pass.slot != pass.slot || job.refused.is_some());
    }
    line.newest_pass = pass.id;
    drop(lines);
    if let Some(keep) = pass.rollback_to {
        jobs.retain(|job| job.pass.slot != pass.slot || job.pass.epoch != pass.epoch || job.pass.position < keep || job.refused.is_some());
    }
    inner.counters.dropped.fetch_add((before - jobs.len()) as u64, Ordering::Relaxed);
    jobs.push_back(Job { pass, arrived: Instant::now(), refused: None });
    drop(jobs);
    inner.ready.notify_one();
}

/// Whether a pass may join the queue: its epoch is one this node takes passes for, and while two
/// epochs overlap, its slot belongs to that epoch. Checked before the slot's line is touched,
/// so a pass that is refused cannot clear another epoch's queue on a shared slot.
///
/// A hop carries only its own ring's passes. A retiring ring still takes passes that arrive on
/// its own hops: they were in flight when this stage was promoted, and only the entry decides
/// which epoch new passes go to.
fn admits(inner: &Inner, pass: &Pass, via: Option<&Arc<Ring>>) -> Result<(), String> {
    if let Some(via) = via {
        if via.epoch != pass.epoch {
            return Err(format!("a pass for route {} came on a link of route {}", pass.epoch, via.epoch));
        }
    }
    let rings = inner.rings.read();
    if rings.is_empty() {
        return Ok(()); // the compute thread says there is no route, as before
    }
    let retiring = via.is_some_and(|via| !via.is_shut() && rings.retiring.iter().any(|ring| Arc::ptr_eq(ring, via)));
    if !retiring && rings.taking(pass.epoch).is_none() {
        let held = rings.current.as_ref().map_or("no route".to_string(), |ring| format!("route {}", ring.epoch));
        return Err(format!("this stage is on {held}"));
    }
    if rings.overlapping() {
        let owner = rings.owner(pass.slot);
        if owner != Some(pass.epoch) {
            return Err(format!("slot {} belongs to route {}", pass.slot, owner.unwrap_or_default()));
        }
    }
    Ok(())
}

/// Refuses a pass before it is queued, along the ring it came by, or the current one.
fn refuse_early(inner: &Inner, pass: &Pass, message: String) {
    let ring = {
        let rings = inner.rings.read();
        rings.any(pass.epoch).or(rings.current.as_ref()).cloned()
    };
    match ring {
        Some(ring) => refuse(inner, &ring, pass, Refusal::Epoch, message),
        None => {
            let nack = Nack { id: pass.id, generation: pass.generation, epoch: pass.epoch, refusal: Refusal::Epoch, stage: 0, message };
            inner.counters.refusals.fetch_add(1, Ordering::Relaxed);
            to_local(inner, &nack.encode());
        }
    }
}

/// At the entry: what the last stage sends straight back.
fn serve_answers(inner: &Inner, ring: &Ring, link: &Link, mut frames: Frames) {
    while let Ok((kind, body)) = frames.next() {
        match kind {
            Kind::Result => receive_answer(inner, ring, link, frames.whole()),
            Kind::Ping if answer_ping(link, body) => {}
            Kind::Pad => {}
            _ => return,
        }
    }
}

/// At the entry: a result, by whichever way it came. Only the first of it, for the reply now
/// on its slot, goes on to the worker. Every copy is acknowledged, by the link it came on, as
/// the last stage holds it until it hears, and the first acknowledgement may have been lost.
/// An answer the last stage did not seal is dropped, and the link it came on stays up: the
/// stage that lied is somewhere behind it, and the real answer may still come the same way.
fn receive_answer(inner: &Inner, ring: &Ring, link: &Link, frame: &[u8]) {
    let Ok(answer) = Answer::decode(&frame[HEADER_BYTES..]) else {
        return;
    };
    if !ring.keys.closing.as_deref().is_some_and(|key| answer.is_sealed_by(key)) {
        inner.counters.forged.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let _ = link.beat(&Ack { id: answer.id }.encode());
    ring.in_flight.lock().remove(&(answer.slot, answer.id));
    let fresh = inner.lines.lock().get_mut(&answer.slot).is_some_and(|line| line.takes(&answer));
    let counter = if fresh { &inner.counters.answers } else { &inner.counters.dropped };
    counter.fetch_add(1, Ordering::Relaxed);
    if fresh {
        inner.answered_now(answer.slot, answer.generation);
        to_local(inner, frame);
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
    // A stuck write must fail inside the silence limit, or the writer holds the link past
    // it. Half leaves room for a piece that only just finished before the one that stuck.
    let limited = peer.link.limit_writes(ring.silence_timeout / 2);
    let reading = (inner.clone(), ring.clone(), peer.clone());
    let watching = (inner.clone(), ring.clone(), peer.clone());
    let started = limited
        .and_then(|()| {
            spawn("isharded-peer", move || {
                let (inner, ring, peer) = reading;
                read_peer(&inner, &ring, &peer);
                // A link that had already gone quiet was ended by the far end's read limit,
                // and a stage that died was still answering heartbeats.
                let quiet = peer.heard.lock().elapsed() > ring.silence_timeout / 2;
                peer_lost(&inner, &ring, &peer, "the link closed", quiet || peer.link.write_failed());
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
        peer_lost(inner, ring, peer, &error.to_string(), false);
    }
}

fn read_peer(inner: &Inner, ring: &Ring, peer: &Peer) {
    let mut frames = peer.link.reader(wire::MAX_BODY_BYTES);
    while let Ok((kind, body)) = frames.next() {
        *peer.heard.lock() = Instant::now();
        if matches!(kind, Kind::Ack | Kind::Pong) {
            peer.proven.store(true, Ordering::Release);
        }
        match kind {
            Kind::Ack => {
                let Ok(ack) = Ack::decode(body) else {
                    return;
                };
                if peer.closing {
                    ring.answer_acknowledged(ack.id);
                    continue;
                }
                let mut sent = ring.sent.lock();
                let at = sent.iter().position(|held| held.id == ack.id);
                // A round trip that carried a real row, so unlike a heartbeat it shows
                // what the row's size costs. It adds no traffic.
                if let Some(held) = at.and_then(|at| sent.remove(at)) {
                    average(&inner.counters.hop_us, held.sent.elapsed().as_micros() as u64);
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
            Kind::Result if ring.is_entry() => receive_answer(inner, ring, &peer.link, frames.whole()),
            // Coming back along the route, towards the entry.
            Kind::Nack if ring.is_entry() => {
                if let Ok(nack) = Nack::decode(body) {
                    let mut in_flight = ring.in_flight.lock();
                    match nack.id {
                        0 => in_flight.clear(),
                        id => in_flight.retain(|(_, sent)| *sent != id),
                    }
                }
                send_back(inner, ring, frames.whole());
            }
            Kind::Result | Kind::Nack => send_back(inner, ring, frames.whole()),
            Kind::Pad => {}
            _ => return,
        }
    }
}

/// Queues heartbeats and checks deadlines, and nothing here can wait on the socket.
fn watch_peer(inner: &Arc<Inner>, ring: &Arc<Ring>, peer: &Arc<Peer>) {
    let mut pinged = Instant::now();
    // A link with nothing to carry must still be heard from, well inside its limit.
    let ping_every = PING_EVERY.min(ring.silence_timeout / 3);
    while !peer.lost.load(Ordering::Acquire) && !peer.link.is_closed() {
        thread::sleep(WATCH_EVERY);
        if ring.is_stale(if peer.closing { &ring.answers } else { &ring.sent }) {
            peer_lost(inner, ring, peer, "a pass was not acknowledged", true);
            return;
        }
        if peer.heard.lock().elapsed() > ring.silence_timeout {
            peer_lost(inner, ring, peer, "the link went silent", true);
            return;
        }
        if pinged.elapsed() >= ping_every {
            pinged = Instant::now();
            let stamp = peer.born.elapsed().as_micros() as u64;
            // No room means the writer is stuck, and the deadlines here will say so. A write
            // that failed is told once, to whoever asks first, so a beat that hears it must act.
            if let Err(LinkError::Io(_)) = peer.link.beat(&wire::beat(Kind::Ping, stamp)) {
                peer_lost(inner, ring, peer, "a write did not finish in time", true);
                return;
            }
        }
    }
    // A write that ran out of time closes the link from the writer's thread. On Windows that
    // wakes no read of the socket, so the reader may never say so, and this is all that will.
    if !peer.lost.load(Ordering::Acquire) && peer.link.write_failed() && !ring.is_shut() {
        peer_lost(inner, ring, peer, "a write did not finish in time", true);
    }
}

/// `silent` says the far end may be alive. A link that closed is usually a stage that died,
/// and replies should not wait for a dial to find that out.
fn peer_lost(inner: &Arc<Inner>, ring: &Arc<Ring>, peer: &Arc<Peer>, why: &str, silent: bool) {
    if peer.lost.swap(true, Ordering::AcqRel) {
        return;
    }
    peer.link.close();
    if peer.closing {
        // Answers walk back along the route until it returns. Nothing has failed, and
        // whatever it was carrying may never have arrived.
        ring.closing.lock().take();
        resend_answers(inner, ring);
        return;
    }
    // A link that never answered anything shows the way is shut, so going round again is no use.
    let door = ring.next_door.lock().clone().filter(|_| silent && peer.proven.load(Ordering::Acquire) && !ring.is_shut());
    ring.next_dialling.store(door.is_some(), Ordering::Release);
    ring.next.lock().take();
    let Some(door) = door else {
        give_up_on_next(inner, ring, peer.stage, why);
        return;
    };
    let (dialling, place, stage, reason) = (inner.clone(), ring.clone(), peer.stage, why.to_string());
    if spawn("isharded-redial", move || redial_next(&dialling, &place, &door, stage, &reason)).is_err() {
        give_up_on_next(inner, ring, stage, why);
    }
}

fn redial_next(inner: &Arc<Inner>, ring: &Arc<Ring>, door: &Address, stage: u16, why: &str) {
    let key = ring.keys.next.as_deref().unwrap_or_default();
    match link::dial(&door.address, Some(&door.fingerprint), key, Role::Hop, ring.epoch, REDIAL_HOP) {
        Ok(link) if !ring.is_shut() => resume_next(inner, ring, link, stage),
        Ok(link) => link.close(),
        Err(_) => give_up_on_next(inner, ring, stage, why),
    }
}

/// Puts a new link in the old one's place. What the next stage has not acknowledged goes
/// first and in order, and whatever it has seen already, it drops.
fn resume_next(inner: &Arc<Inner>, ring: &Arc<Ring>, link: Arc<Link>, stage: u16) {
    let peer = Peer::new(link, stage, false);
    let _order = ring.forwarding.lock();
    *ring.next.lock() = Some(peer.clone());
    ring.next_dialling.store(false, Ordering::Release);
    start_peer(inner, ring, &peer);
    if ring.is_shut() {
        ring.shut();
    }
    for frame in restamped(&ring.sent, inner) {
        if peer.link.send(&frame).is_err() {
            break; // its reader is finding out too
        }
    }
}

fn give_up_on_next(inner: &Inner, ring: &Ring, stage: u16, why: &str) {
    ring.next_dialling.store(false, Ordering::Release);
    ring.sent.lock().clear();
    // Id zero means every pass this stage has sent on: nothing can say which ones were lost.
    let nack = Nack { id: 0, generation: 0, epoch: ring.epoch, refusal: Refusal::FailedStage, stage, message: why.to_string() };
    inner.counters.refusals.fetch_add(1, Ordering::Relaxed);
    send_back(inner, ring, &nack.encode());
}

/// Copies of what a list holds, each with a fresh chance to be acknowledged.
fn restamped(list: &Mutex<VecDeque<Held>>, inner: &Inner) -> Vec<Vec<u8>> {
    let mut list = list.lock();
    let now = Instant::now();
    list.iter_mut().for_each(|held| held.sent = now);
    inner.counters.resent.fetch_add(list.len() as u64, Ordering::Relaxed);
    list.iter().map(|held| held.frame.clone()).collect()
}

fn compute_loop(inner: &Arc<Inner>) {
    let mut last = None;
    while let Some(batch) = next_batch(inner, last) {
        last = Some(batch[0].pass.epoch);
        handle(inner, batch);
        inner.busy.store(IDLE, Ordering::Release);
    }
}

/// Waits for work. Nothing comes back once the node is stopping. Marks the batch's epoch busy
/// before letting go of the queue, so a retiring ring never looks drained while it runs.
fn next_batch(inner: &Inner, last: Option<u64>) -> Option<Vec<Job>> {
    let mut jobs = inner.jobs.lock();
    loop {
        if inner.stopping.load(Ordering::Acquire) {
            return None;
        }
        let at = pick(&jobs, last);
        if let Some(job) = jobs.remove(at) {
            let most = trip_most(inner, job.pass.epoch);
            let batch = take_batch(job, &mut jobs, most);
            inner.busy.store(batch[0].pass.epoch, Ordering::Release);
            return Some(batch);
        }
        inner.ready.wait(&mut jobs);
    }
}

/// The most passes a trip of this epoch may hold: the current ring's from the node, a staged
/// or retiring ring's from its own plan.
fn trip_most(inner: &Inner, epoch: u64) -> usize {
    let rings = inner.rings.read();
    match rings.any(epoch) {
        Some(ring) if !rings.current.as_ref().is_some_and(|current| Arc::ptr_eq(current, ring)) => ring.merge_most,
        _ => inner.merge_most.load(Ordering::Acquire),
    }
}

/// Which job goes next. Oldest first, except that when two epochs share the model, the one
/// that did not run last goes, so a flood on one ring never starves the other. A job whose
/// slot has an older job queued never jumps it, so a slot's passes keep their order.
fn pick(jobs: &VecDeque<Job>, last: Option<u64>) -> usize {
    let Some(first) = jobs.front() else {
        return 0;
    };
    if last != Some(first.pass.epoch) {
        return 0;
    }
    let mut slots = HashSet::new();
    for (at, job) in jobs.iter().enumerate() {
        if job.pass.epoch != first.pass.epoch && !slots.contains(&job.pass.slot) {
            return at;
        }
        slots.insert(job.pass.slot);
    }
    0
}

/// A job, and whatever waiting anywhere behind it can share its trip. There is no window to
/// wait out: passes gather while the stage is busy, so a lone reply goes straight through.
/// A route that asked for no merging gets its passes one at a time, and one that asked for
/// exact replies no more together than were seen to round as they do alone.
fn take_batch(first: Job, jobs: &mut VecDeque<Job>, most: usize) -> Vec<Job> {
    let mut batch = vec![first];
    if most < 2 || batch[0].refused.is_some() || !can_merge(&batch[0].pass) {
        return batch;
    }
    let epoch = batch[0].pass.epoch;
    // A slot's passes run in the order they came, so a slot is done with once one of its
    // passes has been looked at, whether that pass went or stayed.
    let mut seen = HashSet::from([batch[0].pass.slot]);
    let mut index = 0;
    while index < jobs.len() && batch.len() < most.min(MAX_MERGED) {
        let pass = &jobs[index].pass;
        if seen.insert(pass.slot) && can_merge(pass) && pass.epoch == epoch && jobs[index].refused.is_none() {
            batch.extend(jobs.remove(index));
        } else {
            index += 1;
        }
    }
    batch
}

fn refuse(inner: &Inner, ring: &Ring, pass: &Pass, refusal: Refusal, message: String) {
    let nack = Nack { id: pass.id, generation: pass.generation, epoch: pass.epoch, refusal, stage: ring.index, message };
    inner.counters.refusals.fetch_add(1, Ordering::Relaxed);
    send_back(inner, ring, &nack.encode());
}

fn handle(inner: &Arc<Inner>, mut jobs: Vec<Job>) {
    if let Some(message) = jobs[0].refused.take() {
        refuse_early(inner, &jobs[0].pass, message);
        return;
    }
    let (ring, current) = {
        let rings = inner.rings.read();
        (rings.any(jobs[0].pass.epoch).cloned(), rings.current.clone())
    };
    let Some(current) = current.or_else(|| ring.clone()) else {
        // No route: the link that carried this has nothing to do any more. Say so at the entry.
        for job in &jobs {
            let pass = &job.pass;
            let nack = Nack { id: pass.id, generation: pass.generation, epoch: pass.epoch, refusal: Refusal::Epoch, stage: 0, message: "no route here".into() };
            to_local(inner, &nack.encode());
        }
        return;
    };
    // A batch is one epoch's. One that lost its ring while it waited is refused as before.
    let Some(ring) = ring else {
        for job in &jobs {
            refuse(inner, &current, &job.pass, Refusal::Epoch, format!("this stage is on route {}", current.epoch));
        }
        return;
    };
    average(&inner.counters.wait_us, jobs[0].arrived.elapsed().as_micros() as u64);
    let passes: Vec<Pass> = jobs.into_iter().map(|job| job.pass).collect();
    inner.gate.lock();
    let started = Instant::now();
    let results = match inner.compute.lock().as_mut() {
        Some(compute) if passes.len() > 1 => compute.run_merged(&passes),
        Some(compute) => vec![compute.run(&passes[0])],
        None => passes.iter().map(|_| Err(Failure { refusal: Refusal::Retry, message: "this stage has nothing loaded".into() })).collect(),
    };
    let spent = started.elapsed().as_micros() as u64;
    let done = Instant::now();
    // Safety: taken just above, on this thread.
    unsafe { inner.gate.unlock() };
    let counters = &inner.counters;
    average(&counters.compute_us, spent);
    // The totals are the current route's, and a reopen while this ran started them again.
    if inner.rings.read().current.as_ref().is_some_and(|now| Arc::ptr_eq(now, &ring)) {
        counters.compute_us_total.fetch_add(spent, Ordering::Relaxed);
        counters.batches_total.fetch_add(1, Ordering::Relaxed);
    }
    counters.passes.fetch_add(passes.len() as u64, Ordering::Relaxed);
    counters.rows.fetch_add(passes.iter().map(|pass| pass.row_count() as u64).sum(), Ordering::Relaxed);
    if passes.len() > 1 {
        counters.merged_passes.fetch_add(1, Ordering::Relaxed);
        counters.merged_rows.fetch_add(passes.len() as u64, Ordering::Relaxed);
    }
    for (pass, result) in passes.into_iter().zip(results) {
        finish(inner, &ring, pass, result, done);
    }
}

/// `done` is when compute ended, for the hop trace.
fn finish(inner: &Arc<Inner>, ring: &Arc<Ring>, pass: Pass, result: Result<Output, Failure>, done: Instant) {
    match result {
        Err(failure) => refuse(inner, ring, &pass, failure.refusal, failure.message),
        Ok(Output::Tokens(tokens)) => {
            inner.counters.answers.fetch_add(1, Ordering::Relaxed);
            let made = Answer { id: pass.id, generation: pass.generation, epoch: pass.epoch, slot: pass.slot, tokens, seal: [0; MAC_BYTES] };
            let frame = match ring.keys.closing.as_deref() {
                Some(key) => made.sealed(key),
                None => made, // the entry is the last stage too, and nothing crosses a link
            }
            .encode();
            answer(inner, ring, &pass, frame, done);
        }
        Ok(Output::Rows { rows, width, bf16, data }) => {
            let mut flags = pass.flags & !(FLAG_TOKENS | FLAG_BF16);
            if bf16 {
                flags |= FLAG_BF16;
            }
            let onward = Pass { flags, tokens: Vec::new(), rows, width, data, ..pass };
            forward(inner, ring, &onward, done);
        }
    }
}

/// An answer lives until the entry acknowledges it.
fn answer(inner: &Arc<Inner>, ring: &Arc<Ring>, pass: &Pass, frame: Vec<u8>, done: Instant) {
    if ring.is_entry() && ring.is_last() {
        inner.answered_now(pass.slot, pass.generation);
        to_local(inner, &frame);
        return;
    }
    ring.answers.lock().push_back(Held { id: pass.id, frame: frame.clone(), sent: Instant::now() });
    carry_answer(inner, ring, &frame);
    average(&inner.counters.queue_us, done.elapsed().as_micros() as u64);
}

fn carry_answer(inner: &Arc<Inner>, ring: &Arc<Ring>, frame: &[u8]) {
    let closing = ring.closing.lock().clone();
    match closing {
        // Losing the link sends this and everything else it may have held back.
        Some(peer) if peer.link.send(frame).is_err() => peer_lost(inner, ring, &peer, "the closing link failed", true),
        Some(_) => {}
        None => send_back(inner, ring, frame),
    }
}

fn resend_answers(inner: &Arc<Inner>, ring: &Arc<Ring>) {
    for frame in restamped(&ring.answers, inner) {
        carry_answer(inner, ring, &frame);
    }
}

fn forward(inner: &Arc<Inner>, ring: &Arc<Ring>, pass: &Pass, done: Instant) {
    let order = ring.forwarding.lock();
    let peer = ring.next.lock().clone();
    if peer.is_none() && !ring.next_dialling.load(Ordering::Acquire) {
        drop(order);
        refuse(inner, ring, pass, Refusal::FailedStage, "no link to the next stage".into());
        return;
    }
    let frame = pass.encode();
    ring.sent.lock().push_back(Held { id: pass.id, frame: frame.clone(), sent: Instant::now() });
    if ring.is_entry() {
        ring.in_flight.lock().insert((pass.slot, pass.id));
    }
    // With no link, this waits with the rest for the one being dialled.
    let sent = peer.map(|peer| (peer.link.send(&frame), peer));
    drop(order);
    match sent {
        Some((Err(error), peer)) => peer_lost(inner, ring, &peer, &error.to_string(), matches!(error, LinkError::Io(_))),
        Some((Ok(()), _)) => average(&inner.counters.queue_us, done.elapsed().as_micros() as u64),
        None => {}
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

    fn job(slot: u32, epoch: u64, rows: u32, flags: u8) -> Job {
        let pass = Pass {
            id: u64::from(slot),
            generation: 1,
            epoch,
            slot,
            position: 0,
            rollback_to: None,
            flags,
            sampling: wire::Sampling::GREEDY,
            guesses: vec![],
            tokens: vec![],
            rows,
            width: 1,
            data: vec![0; rows as usize * 4],
        };
        Job { pass, arrived: Instant::now(), refused: None }
    }

    fn batch_of(jobs: Vec<Job>) -> (Vec<u32>, Vec<u32>) {
        let mut queue: VecDeque<Job> = jobs.into();
        let first = queue.pop_front().unwrap();
        let slots = |jobs: &mut dyn Iterator<Item = &Job>| jobs.map(|job| job.pass.slot).collect();
        (slots(&mut take_batch(first, &mut queue, MAX_MERGED).iter()), slots(&mut queue.iter()))
    }

    #[test]
    fn two_epochs_take_turns_and_a_slot_never_jumps_its_own_older_pass() {
        let queue: VecDeque<Job> = vec![job(0, 1, 1, 0), job(1, 1, 1, 0), job(2, 2, 1, 0), job(3, 2, 1, 0)].into();
        assert_eq!(pick(&queue, None), 0);
        assert_eq!(pick(&queue, Some(2)), 0);
        assert_eq!(pick(&queue, Some(1)), 2, "epoch 2 waited behind epoch 1, which ran last");
        // Slot 0 has an epoch 1 pass ahead of its epoch 2 pass, so the turn goes to slot 3.
        let queue: VecDeque<Job> = vec![job(0, 1, 1, 0), job(0, 2, 1, 0), job(3, 2, 1, 0)].into();
        assert_eq!(pick(&queue, Some(1)), 2);
        let queue: VecDeque<Job> = vec![job(0, 1, 1, 0), job(0, 2, 1, 0)].into();
        assert_eq!(pick(&queue, Some(1)), 0);
    }

    #[test]
    fn a_route_that_merges_nothing_takes_passes_one_at_a_time() {
        let mut queue: VecDeque<Job> = vec![job(1, 1, 1, 0), job(2, 1, 1, 0)].into();
        assert_eq!(take_batch(job(0, 1, 1, 0), &mut queue, 1).len(), 1);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn a_plan_says_whether_passes_merge_and_merges_them_when_it_does_not_say() {
        let node = Node::new().unwrap();
        let plan = |text: &str| serde_json::from_str::<Plan>(text).unwrap();
        node.open(plan(r#"{"epoch": 1, "index": 0, "count": 2, "keys": {"next": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], "closing": [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]}, "merge_passes": false}"#)).unwrap();
        assert_eq!(node.inner.merge_most.load(Ordering::Acquire), 1);
        node.open(plan(r#"{"epoch": 2, "index": 0, "count": 2, "keys": {"next": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], "closing": [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]}}"#)).unwrap();
        assert_eq!(node.inner.merge_most.load(Ordering::Acquire), MAX_MERGED);
    }

    #[test]
    fn an_exact_route_merges_no_more_than_its_plan_allows() {
        let node = Node::new().unwrap();
        let keys = r#""keys": {"next": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], "closing": [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]}"#;
        let most = |extra: &str| {
            node.open(serde_json::from_str(&format!(r#"{{"epoch": 1, "index": 0, "count": 2, {keys}{extra}}}"#)).unwrap()).unwrap();
            node.inner.merge_most.load(Ordering::Acquire)
        };
        assert_eq!(most(r#", "merge_most": 3"#), 3);
        assert_eq!(most(r#", "merge_most": 0"#), 1);
        assert_eq!(most(r#", "merge_most": 900"#), MAX_MERGED);
        assert_eq!(most(r#", "merge_most": 3, "merge_passes": false"#), 1);
        let queue = (1..6).map(|slot| job(slot, 1, 1, 0)).collect::<VecDeque<_>>();
        let mut left = queue;
        assert_eq!(take_batch(job(0, 1, 1, 0), &mut left, 3).len(), 3);
        assert_eq!(left.len(), 3);
    }

    #[test]
    fn a_merge_reaches_past_what_cannot_go_with_it() {
        let verify = job(1, 1, 3, wire::FLAG_VERIFY);
        let queue = vec![job(4, 1, 1, 0), verify, job(9, 1, 1, 0), job(4, 1, 1, 0), job(2, 2, 1, 0), job(0, 1, 1, 0)];
        assert_eq!(batch_of(queue), (vec![4, 9, 0], vec![1, 4, 2]));
    }

    #[test]
    fn a_pass_never_overtakes_one_of_its_own_slot() {
        let verify = job(1, 1, 3, wire::FLAG_VERIFY);
        assert_eq!(batch_of(vec![job(0, 1, 1, 0), verify, job(1, 1, 1, 0)]), (vec![0], vec![1, 1]));
        let other_route = job(1, 2, 1, 0);
        assert_eq!(batch_of(vec![job(0, 1, 1, 0), other_route, job(1, 1, 1, 0)]), (vec![0], vec![1, 1]));
    }

    #[test]
    fn a_pass_that_cannot_merge_runs_alone() {
        let verify = job(1, 1, 3, wire::FLAG_VERIFY);
        assert_eq!(batch_of(vec![verify, job(2, 1, 1, 0)]), (vec![1], vec![2]));
    }

    #[test]
    fn a_merge_stops_at_the_most_one_trip_holds() {
        let queue = (0..MAX_MERGED as u32 + 2).map(|slot| job(slot, 1, 1, 0)).collect();
        let (taken, left) = batch_of(queue);
        assert_eq!(taken.len(), MAX_MERGED);
        assert_eq!(left, vec![MAX_MERGED as u32, MAX_MERGED as u32 + 1]);
    }

    #[test]
    fn a_connection_with_no_thread_gives_its_place_back() {
        let node = Node::new().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _far = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (socket, peer) = listener.accept().unwrap();
        with_spawns(0, || admit(&node.inner, socket, peer.ip(), false));
        assert_eq!(node.inner.strangers.lock().total, 0);
        assert!(node.inner.strangers.lock().from.is_empty());
    }

    #[test]
    fn one_address_cannot_take_every_place_for_strangers() {
        let mut strangers = Strangers::default();
        let one: IpAddr = "10.0.0.1".parse().unwrap();
        assert!((0..STRANGERS_PER_ADDRESS).all(|_| strangers.enter(one)));
        assert!(!strangers.enter(one));
        let others = (2..).map(|last| IpAddr::from([10, 0, 0, last]));
        let admitted = others.take(MAX_STRANGERS).filter(|other| strangers.enter(*other)).count();
        assert_eq!(admitted, MAX_STRANGERS - STRANGERS_PER_ADDRESS);
        assert!(!strangers.enter("10.0.1.1".parse().unwrap()), "past the total");
        strangers.leave(one);
        assert!(strangers.enter(one), "a place given back can be taken again");
    }

    #[test]
    fn a_place_in_a_route_needs_exactly_the_keys_of_its_own_links() {
        let key = || Some(vec![1; MIN_KEY_BYTES]);
        let keys = |back, next, closing| Keys { back, next, closing };
        assert!(keys(None, None, None).fit(0, 1));
        assert!(keys(None, key(), key()).fit(0, 3));
        assert!(keys(key(), key(), None).fit(1, 3));
        assert!(keys(key(), None, key()).fit(2, 3));
        // A middle stage never holds the closing key, and nobody holds a key it has no use for.
        assert!(!keys(key(), key(), key()).fit(1, 3));
        assert!(!keys(key(), key(), key()).fit(0, 3));
        assert!(!keys(None, key(), None).fit(0, 3));
        assert!(!keys(None, Some(vec![1; MIN_KEY_BYTES - 1]), key()).fit(0, 3));
        assert!(!keys(None, None, None).fit(1, 1));
    }

    #[test]
    fn a_plan_whose_keys_do_not_fit_is_refused_and_the_old_route_stays() {
        let (node, _) = stage(0);
        let wrong = Plan { epoch: 2, index: 1, count: 3, keys: keys_for(0, 3), ack_timeout_ms: 1, silence_timeout_ms: 1, merge_passes: true, merge_most: None };
        assert!(node.open(wrong).is_err());
        assert_eq!(node.stats().epoch, Some(1));
    }

    /// The keys a place needs, the same at every node of a test route.
    fn keys_for(index: u16, count: u16) -> Keys {
        let hop = |edge: u16| Some(vec![edge as u8 + 1; 32]);
        let ends = count > 1 && (index == 0 || index + 1 == count);
        Keys { back: (index > 0).then(|| hop(index - 1)).flatten(), next: (index + 1 < count).then(|| hop(index)).flatten(), closing: ends.then(|| vec![9; 32]) }
    }

    fn stage(index: u16) -> (Node, Address) {
        let node = Node::new().unwrap();
        node.listen("127.0.0.1", 0).unwrap();
        node.open(Plan { epoch: 1, index, count: 2, keys: keys_for(index, 2), ack_timeout_ms: 60_000, silence_timeout_ms: 60_000, merge_passes: true, merge_most: None }).unwrap();
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

    #[test]
    fn a_closing_link_that_could_not_open_is_tried_again() {
        let (last, _) = stage(1);
        let entry = Node::new().unwrap();
        entry.listen("127.0.0.1", 0).unwrap();
        let door = {
            let listening = entry.inner.listening.lock();
            let door = listening.as_ref().unwrap();
            Address { address: format!("127.0.0.1:{}", door.port), fingerprint: door.fingerprint.clone() }
        };
        // No route is open at the entry yet, so it turns the dial away.
        assert_eq!(last.connect(Links { next: None, entry: Some(door), timeout_ms: 100 }), Ok(false));
        assert!(!last.stats().closing_up);
        entry.open(Plan { epoch: 1, index: 0, count: 2, keys: keys_for(0, 2), ack_timeout_ms: 60_000, silence_timeout_ms: 60_000, merge_passes: true, merge_most: None }).unwrap();
        let began = Instant::now();
        while !last.stats().closing_up {
            assert!(began.elapsed() < Duration::from_secs(5), "it never came up");
            thread::sleep(WATCH_EVERY);
        }
    }
}
