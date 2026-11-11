//! Real nodes over real TLS links on loopback, with a stand-in for the model.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, Weak};
use std::thread;
use std::time::{Duration, Instant};

use isharded_engine::link::{self, Frames, Identity, Link};
use isharded_engine::watchdog::{Limits, Watchdog};
use isharded_engine::node::{Address, Compute, Failure, Keys, Links, Node, Output, Plan};
use isharded_engine::wire::{self, Ack, Answer, Kind, Nack, Pass, Refusal, Role, Sampling, FLAG_TOKENS, MAC_BYTES};

const CLOSING: &[u8] = &[9; 32];
const WIDTH: u32 = 4;
/// How long one test may run. Every wait inside a test has its own limit, so this is only
/// reached by a hang, which used to hold the whole run under a busy machine until CI gave up.
const TEST_SECONDS: u64 = 120;

/// The nodes each running test made, by the name of the thread it runs on, so a deadline
/// can say what they were doing.
static WATCHED: Mutex<Vec<(String, Weak<Node>)>> = Mutex::new(Vec::new());

fn watch(node: &Arc<Node>) {
    let name = thread::current().name().unwrap_or_default().to_string();
    let mut watched = WATCHED.lock().unwrap();
    watched.retain(|(_, node)| node.strong_count() > 0);
    watched.push((name, Arc::downgrade(node)));
}

/// Runs a test on a thread of its own and fails it, instead of hanging, if it runs past
/// `TEST_SECONDS` (or ISHARDED_TEST_SECONDS). On the way out it prints every node the test
/// made, with its counters and link state, which is where a hung ring shows what it was
/// waiting for. Rust cannot print another thread's stack, so this is the state instead.
fn within_deadline(name: &'static str, body: impl FnOnce() + Send + 'static) {
    let seconds = std::env::var("ISHARDED_TEST_SECONDS").ok().and_then(|text| text.parse().ok()).unwrap_or(TEST_SECONDS);
    within(name, seconds, body);
}

fn within(name: &'static str, seconds: u64, body: impl FnOnce() + Send + 'static) {
    let (done, finished) = mpsc::channel();
    let runner = thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            body();
            let _ = done.send(());
        })
        .unwrap();
    match finished.recv_timeout(Duration::from_secs(seconds)) {
        Ok(()) => runner.join().unwrap(),
        // The body panicked: that is the test's own failure, with its own message.
        Err(RecvTimeoutError::Disconnected) => {
            if let Err(panic) = runner.join() {
                std::panic::resume_unwind(panic);
            }
        }
        Err(RecvTimeoutError::Timeout) => {
            let nodes: Vec<String> = WATCHED
                .lock()
                .unwrap()
                .iter()
                .filter(|(owner, _)| owner == name)
                .filter_map(|(_, node)| node.upgrade())
                .enumerate()
                .map(|(index, node)| format!("  node {index}: {}", serde_json::to_string(&node.stats()).unwrap_or_default()))
                .collect();
            eprintln!("{name} ran past {seconds} s. Its nodes, in the order it made them:\n{}", nodes.join("\n"));
            // Its thread is left behind: the process ends with the run, and nothing waits on it.
            panic!("{name} ran past {seconds} s and was stopped; the nodes it made are printed above");
        }
    }
}

/// The key of the link from stage `edge` to the next.
fn hop_key(edge: u16) -> Vec<u8> {
    vec![edge as u8 + 1; 32]
}

/// The keys a coordinator gives a place in a route: those of its own links and no others.
fn keys_for(index: u16, count: u16) -> Keys {
    let ends = count > 1 && (index == 0 || index + 1 == count);
    Keys {
        back: (index > 0).then(|| hop_key(index - 1)),
        next: (index + 1 < count).then(|| hop_key(index)),
        closing: ends.then(|| CLOSING.to_vec()),
    }
}

/// An answer as the last stage makes it.
fn sealed(id: u64, tokens: Vec<u32>) -> Answer {
    Answer { id, generation: 1, epoch: 7, slot: 0, tokens, seal: [0; MAC_BYTES] }.sealed(CLOSING)
}

/// Adds its stage number to every value, so an answer shows each stage ran, in order.
struct Adder {
    stage: u16,
    last: bool,
    delay: Duration,
    seen: Arc<AtomicU64>,
}

impl Compute for Adder {
    fn run(&mut self, pass: &Pass) -> Result<Output, Failure> {
        self.seen.fetch_add(1, Ordering::SeqCst);
        if !self.delay.is_zero() {
            thread::sleep(self.delay);
        }
        if pass.position == 666 {
            return Err(Failure { refusal: Refusal::StageError, message: format!("stage {} refused it", self.stage) });
        }
        let mut values: Vec<f32> = if pass.carries_tokens() {
            pass.tokens.iter().flat_map(|token| vec![*token as f32; WIDTH as usize]).collect()
        } else {
            pass.data.chunks_exact(4).map(|item| f32::from_le_bytes([item[0], item[1], item[2], item[3]])).collect()
        };
        values.iter_mut().for_each(|value| *value = *value * 10.0 + f32::from(self.stage));
        if self.last {
            return Ok(Output::Tokens(vec![values[0] as u32, pass.position]));
        }
        Ok(Output::Rows {
            rows: pass.row_count() as u32,
            width: WIDTH,
            bf16: false,
            data: values.iter().flat_map(|value| value.to_le_bytes()).collect(),
        })
    }
}

struct Listening {
    port: u16,
    local_port: u16,
    local_key: Vec<u8>,
    fingerprint: String,
}

fn listening(node: &Node) -> Listening {
    let text = node.listen("127.0.0.1", 0).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&text).unwrap();
    Listening {
        port: value["port"].as_u64().unwrap() as u16,
        local_port: value["local_port"].as_u64().unwrap() as u16,
        local_key: value["local_key"].as_array().unwrap().iter().map(|item| item.as_u64().unwrap() as u8).collect(),
        fingerprint: value["fingerprint"].as_str().unwrap().to_string(),
    }
}

/// One connection through a relay: its sockets, and a hold on each way.
struct Carried {
    sockets: [TcpStream; 2],
    forth: Arc<AtomicBool>,
    back: Arc<AtomicBool>,
}

#[derive(Default)]
struct RelayState {
    carried: Mutex<Vec<Carried>>,
    refusing: AtomicBool,
    swallowing: AtomicBool,
    attempts: AtomicU64,
    gone: AtomicBool,
}

/// Stands in the middle of one link, so a test can do to it what a network does. A connection
/// made after a hold or a cut goes through, unless `refuse` says it does not.
struct Relay {
    port: u16,
    state: Arc<RelayState>,
}

impl Relay {
    fn to(port: u16) -> Relay {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let relay = Relay { port: listener.local_addr().unwrap().port(), state: Arc::default() };
        let state = relay.state.clone();
        thread::spawn(move || {
            while !state.gone.load(Ordering::SeqCst) {
                let Ok((near, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(2));
                    continue;
                };
                state.attempts.fetch_add(1, Ordering::SeqCst);
                if state.refusing.load(Ordering::SeqCst) {
                    continue;
                }
                near.set_nonblocking(false).unwrap();
                let far = TcpStream::connect(("127.0.0.1", port)).unwrap();
                let swallow = state.swallowing.load(Ordering::SeqCst);
                let (forth, back) = (Arc::new(AtomicBool::new(swallow)), Arc::new(AtomicBool::new(swallow)));
                let sockets = [near.try_clone().unwrap(), far.try_clone().unwrap()];
                state.carried.lock().unwrap().push(Carried { sockets, forth: forth.clone(), back: back.clone() });
                let (near_copy, far_copy) = (near.try_clone().unwrap(), far.try_clone().unwrap());
                thread::spawn(move || carry(near_copy, far_copy, &forth));
                thread::spawn(move || carry(far, near, &back));
            }
        });
        relay
    }

    /// Nothing is read or passed on any more on what is open, as when the far machine freezes.
    fn hold(&self) {
        for item in self.state.carried.lock().unwrap().iter() {
            item.forth.store(true, Ordering::SeqCst);
            item.back.store(true, Ordering::SeqCst);
        }
    }

    /// The same, for what comes back from the far end only.
    fn hold_back(&self) {
        for item in self.state.carried.lock().unwrap().iter() {
            item.back.store(true, Ordering::SeqCst);
        }
    }

    /// The same, for what goes towards the far end only.
    fn hold_forth(&self) {
        for item in self.state.carried.lock().unwrap().iter() {
            item.forth.store(true, Ordering::SeqCst);
        }
    }

    /// New connections are closed as soon as they arrive.
    fn refuse(&self, refusing: bool) {
        self.state.refusing.store(refusing, Ordering::SeqCst);
    }

    /// New connections are taken and held from their first byte.
    fn swallow(&self) {
        self.state.swallowing.store(true, Ordering::SeqCst);
    }

    fn attempts(&self) -> u64 {
        self.state.attempts.load(Ordering::SeqCst)
    }

    fn cut(&self) {
        for socket in self.state.carried.lock().unwrap().iter().flat_map(|item| &item.sockets) {
            let _ = socket.shutdown(Shutdown::Both);
        }
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.state.gone.store(true, Ordering::SeqCst);
        for item in self.state.carried.lock().unwrap().iter() {
            item.forth.store(false, Ordering::SeqCst);
            item.back.store(false, Ordering::SeqCst);
        }
        self.cut();
    }
}

fn carry(mut from: TcpStream, mut to: TcpStream, held: &AtomicBool) {
    let mut buffer = vec![0; 1 << 16];
    while let Ok(count) = from.read(&mut buffer) {
        while held.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(5));
        }
        if count == 0 || to.write_all(&buffer[..count]).is_err() {
            break;
        }
    }
    let _ = to.shutdown(Shutdown::Both);
}

struct Ring {
    nodes: Vec<Arc<Node>>,
    seen: Vec<Arc<AtomicU64>>,
    client: Arc<Link>,
    frames: Frames,
    relay: Option<Relay>,
    doors: Vec<Listening>,
}

fn ring(count: u16, closing: bool, delay: Duration, ack_timeout_ms: u64) -> Ring {
    build_ring(count, closing, delay, (ack_timeout_ms, ack_timeout_ms), None)
}

/// A ring whose dial to one stage goes through a relay. Only the closing link dials stage
/// zero, so naming stage zero puts the relay on that link.
fn ring_through(door: usize, count: u16, closing: bool, delay: Duration, ack_timeout_ms: u64) -> (Ring, Relay) {
    ring_with_limits(door, count, closing, delay, (ack_timeout_ms, ack_timeout_ms))
}

/// The same, with the acknowledgement and silence limits apart.
fn ring_with_limits(door: usize, count: u16, closing: bool, delay: Duration, limits: (u64, u64)) -> (Ring, Relay) {
    let mut ring = build_ring(count, closing, delay, limits, Some(door));
    let relay = ring.relay.take().unwrap();
    (ring, relay)
}

fn build_ring(count: u16, closing: bool, delay: Duration, (ack_timeout_ms, silence_timeout_ms): (u64, u64), relayed: Option<usize>) -> Ring {
    let mut nodes = vec![];
    let mut seen = vec![];
    let mut doors = vec![];
    for stage in 0..count {
        let node = Arc::new(Node::new().unwrap());
        watch(&node);
        let counter = Arc::new(AtomicU64::new(0));
        node.set_compute(Box::new(Adder { stage: stage + 1, last: stage + 1 == count, delay, seen: counter.clone() }));
        doors.push(listening(&node));
        nodes.push(node);
        seen.push(counter);
    }
    for (index, node) in nodes.iter().enumerate() {
        node.open(Plan { epoch: 7, index: index as u16, count, keys: keys_for(index as u16, count), ack_timeout_ms, silence_timeout_ms, merge_passes: true, merge_most: None }).unwrap();
    }
    let relay = relayed.map(|door| Relay::to(doors[door].port));
    let address = |at: usize| {
        let port = relay.as_ref().filter(|_| relayed == Some(at)).map_or(doors[at].port, |item| item.port);
        Address { address: format!("127.0.0.1:{port}"), fingerprint: doors[at].fingerprint.clone() }
    };
    for (index, node) in nodes.iter().enumerate() {
        let next = (index + 1 < doors.len()).then(|| address(index + 1));
        let entry = (closing && index + 1 == count as usize && count > 1).then(|| address(0));
        node.connect(Links { next, entry, timeout_ms: 3000 }).unwrap();
    }
    let client = link::dial(&format!("127.0.0.1:{}", doors[0].local_port), None, &doors[0].local_key, Role::Local, 0, Duration::from_secs(3)).unwrap();
    let mut frames = client.reader(1 << 20);
    // The entry takes this link as its worker's only after the handshake, on its own thread.
    // A test that fails a stage at once would lose the refusal to that moment, so wait for it:
    // the answer to a beat comes from that same thread.
    client.beat(&wire::beat(Kind::Ping, 1)).unwrap();
    assert_eq!(frames.next().unwrap().0, Kind::Pong);
    // A frame that never comes fails the test instead of holding it for ever.
    client.limit_reads(Duration::from_secs(30)).unwrap();
    Ring { nodes, seen, client, frames, relay, doors }
}

fn pass(id: u64, token: u32, position: u32) -> Pass {
    Pass {
        id,
        generation: 1,
        epoch: 7,
        slot: 0,
        position,
        rollback_to: None,
        flags: FLAG_TOKENS,
        sampling: Sampling::GREEDY,
        guesses: vec![],
        tokens: vec![token],
        rows: 0,
        width: 0,
        data: vec![],
    }
}

/// A pass that already carries rows, as big as a test needs.
fn rows(id: u64, count: u32) -> Pass {
    let data = vec![0; count as usize * WIDTH as usize * 4];
    Pass { flags: 0, tokens: vec![], rows: count, width: WIDTH, data, ..pass(id, 0, 9) }
}

fn wait_until(what: &str, done: impl Fn() -> bool) {
    let began = Instant::now();
    while !done() {
        assert!(began.elapsed() < Duration::from_secs(5), "{what} never happened");
        thread::sleep(Duration::from_millis(5));
    }
}

/// Reads refusals until the one that says a link was lost, and checks which stage it names.
fn wait_for_loss(ring: &mut Ring, stage: u16) {
    loop {
        let (kind, body) = ring.frames.next().unwrap();
        assert_eq!(kind, Kind::Nack);
        let nack = Nack::decode(&body).unwrap();
        assert_eq!(nack.refusal, Refusal::FailedStage);
        if nack.id == 0 {
            assert_eq!(nack.stage, stage);
            return;
        }
    }
}

fn answer(ring: &mut Ring) -> Answer {
    let (kind, body) = ring.frames.next().unwrap();
    assert_eq!(kind, Kind::Result);
    Answer::decode(&body).unwrap()
}

#[test]
fn a_pass_runs_every_stage_in_order_and_the_answer_reaches_the_entry() {
    within_deadline("a_pass_runs_every_stage_in_order_and_the_answer_reaches_the_entry", || {
        let mut ring = ring(3, true, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        let got = answer(&mut ring);
        // ((5 * 10 + 1) * 10 + 2) * 10 + 3
        assert_eq!(got, sealed(1, vec![5123, 9]));
        assert!(ring.nodes[2].stats().closing_up);
    });
}

#[test]
fn stages_count_their_compute_and_the_entry_times_the_workers_turnaround() {
    within_deadline("stages_count_their_compute_and_the_entry_times_the_workers_turnaround", || {
        let mut ring = ring(3, true, Duration::from_millis(2), 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        // The worker thinks for a while before its next pass.
        thread::sleep(Duration::from_millis(40));
        ring.client.send(&pass(2, 5, 10).encode()).unwrap();
        answer(&mut ring);
        for node in &ring.nodes {
            let stats = node.stats();
            assert_eq!(stats.batches_total, 2);
            assert!(stats.compute_us_total >= 4000, "{}", stats.compute_us_total);
        }
        let turnaround = ring.nodes[0].stats().turnaround_us;
        assert!((40_000..2_000_000).contains(&turnaround), "{turnaround}");
        // The next reply on the slot starts long after, and that wait is no turnaround.
        thread::sleep(Duration::from_millis(300));
        ring.client.send(&Pass { generation: 2, ..pass(3, 5, 0) }.encode()).unwrap();
        answer(&mut ring);
        assert_eq!(ring.nodes[0].stats().turnaround_us, turnaround);
        // A new route starts the totals again, so a reader can tell busy time on this one.
        ring.nodes[1].open(Plan { epoch: 8, index: 1, count: 3, keys: keys_for(1, 3), ack_timeout_ms: 2000, silence_timeout_ms: 2000, merge_passes: true, merge_most: None }).unwrap();
        let stats = ring.nodes[1].stats();
        assert_eq!((stats.batches_total, stats.compute_us_total), (0, 0));
    });
}

#[test]
fn a_batch_of_the_old_route_that_ends_after_a_reopen_is_not_counted_on_the_new_one() {
    within_deadline("a_batch_of_the_old_route_that_ends_after_a_reopen_is_not_counted_on_the_new_one", || {
        let ring = ring(1, false, Duration::from_millis(300), 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        wait_until("the pass to start", || ring.seen[0].load(Ordering::SeqCst) == 1);
        // A repair reopens the stage at the same epoch while the old route's pass still runs.
        ring.nodes[0].open(Plan { epoch: 7, index: 0, count: 1, keys: keys_for(0, 1), ack_timeout_ms: 2000, silence_timeout_ms: 2000, merge_passes: true, merge_most: None }).unwrap();
        wait_until("the pass to end", || ring.nodes[0].stats().passes == 1);
        let stats = ring.nodes[0].stats();
        assert_eq!((stats.batches_total, stats.compute_us_total), (0, 0));
    });
}

#[test]
fn without_a_closing_link_the_answer_walks_back() {
    within_deadline("without_a_closing_link_the_answer_walks_back", || {
        let mut ring = ring(3, false, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        assert_eq!(answer(&mut ring).tokens, vec![5123, 9]);
        assert!(!ring.nodes[2].stats().closing_up);
    });
}

#[test]
fn one_stage_answers_by_itself() {
    within_deadline("one_stage_answers_by_itself", || {
        let mut ring = ring(1, false, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        assert_eq!(answer(&mut ring).tokens, vec![51, 9]);
    });
}

#[test]
fn many_passes_in_flight_all_arrive_in_order() {
    within_deadline("many_passes_in_flight_all_arrive_in_order", || {
        let mut ring = ring(4, true, Duration::ZERO, 2000);
        for id in 1..=500 {
            ring.client.send(&pass(id, 1, id as u32).encode()).unwrap();
        }
        for id in 1..=500 {
            assert_eq!(answer(&mut ring).id, id);
        }
    });
}

#[test]
fn a_stage_that_refuses_is_named_and_later_passes_still_run() {
    within_deadline("a_stage_that_refuses_is_named_and_later_passes_still_run", || {
        let mut ring = ring(3, true, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 666).encode()).unwrap();
        let (kind, body) = ring.frames.next().unwrap();
        assert_eq!(kind, Kind::Nack);
        let nack = Nack::decode(&body).unwrap();
        assert_eq!((nack.id, nack.refusal, nack.stage), (1, Refusal::StageError, 0));
        ring.client.send(&pass(2, 5, 9).encode()).unwrap();
        assert_eq!(answer(&mut ring).id, 2);
    });
}

#[test]
fn a_pass_from_another_route_is_refused_and_nothing_runs() {
    within_deadline("a_pass_from_another_route_is_refused_and_nothing_runs", || {
        let mut ring = ring(2, true, Duration::ZERO, 2000);
        let mut stale = pass(1, 5, 9);
        stale.epoch = 6;
        ring.client.send(&stale.encode()).unwrap();
        let (kind, body) = ring.frames.next().unwrap();
        assert_eq!(kind, Kind::Nack);
        assert_eq!(Nack::decode(&body).unwrap().refusal, Refusal::Epoch);
        assert_eq!(ring.seen[0].load(Ordering::SeqCst), 0);
    });
}

#[test]
fn losing_a_stage_is_reported_to_the_entry_naming_it() {
    within_deadline("losing_a_stage_is_reported_to_the_entry_naming_it", || {
        let mut ring = ring(3, true, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        // A process dying closes its sockets, and this does the same.
        ring.nodes[2].stop();
        let began = Instant::now();
        let (kind, body) = ring.frames.next().unwrap();
        assert_eq!(kind, Kind::Nack);
        let nack = Nack::decode(&body).unwrap();
        assert_eq!((nack.id, nack.refusal, nack.stage), (0, Refusal::FailedStage, 2));
        assert!(began.elapsed() < Duration::from_secs(1));
    });
}

#[test]
fn a_stage_that_hangs_says_so_before_it_ends() {
    within_deadline("a_stage_that_hangs_says_so_before_it_ends", || {
        let mut ring = ring(3, true, Duration::ZERO, 60_000);
        let ended = Arc::new(AtomicBool::new(false));
        let end = ended.clone();
        let watchdog = Watchdog::new(ring.nodes[1].refuser(), move || end.store(true, Ordering::Release));
        watchdog.set_limits(Limits { pass_ms: 200, row_ms: 0 });
        let _hung = watchdog.time(1);
        let (kind, body) = ring.frames.next().unwrap();
        assert_eq!(kind, Kind::Nack);
        let nack = Nack::decode(&body).unwrap();
        assert_eq!((nack.id, nack.refusal, nack.stage), (0, Refusal::FailedStage, 1));
        assert!(nack.message.contains("hung"), "{}", nack.message);
        wait_until("the process to end", || ended.load(Ordering::Acquire));
        assert_eq!(ring.nodes[1].stats().refusals, 1);
    });
}

/// A stage that hangs, under a watchdog whose end is the real thing: the node stops, its
/// sockets close, as when the process ends. Returns the refusals that reached the entry.
fn hang_and_end(ring: &mut Ring, stage: usize) -> Vec<Nack> {
    let hung = std::mem::replace(&mut ring.nodes[stage], Arc::new(Node::new().unwrap()));
    let refuse = hung.refuser();
    let watchdog = Watchdog::new(refuse, move || hung.stop());
    watchdog.set_limits(Limits { pass_ms: 200, row_ms: 0 });
    let _hung = watchdog.time(1);
    let mut heard = vec![];
    // Read until the link ends, or until a second has passed with nothing more.
    ring.client.limit_reads(Duration::from_secs(3)).unwrap();
    while let Ok((kind, body)) = ring.frames.next() {
        assert_eq!(kind, Kind::Nack);
        let nack = Nack::decode(&body).unwrap();
        let hung_said_so = nack.message.contains("hung");
        heard.push(nack);
        if hung_said_so {
            break;
        }
    }
    heard
}

#[test]
fn a_middle_stage_that_hangs_is_heard_saying_so_though_its_process_ends_at_once() {
    within_deadline("a_middle_stage_that_hangs_is_heard_saying_so_though_its_process_ends_at_once", || {
        for _ in 0..5 {
            let mut ring = ring(3, true, Duration::ZERO, 60_000);
            let heard = hang_and_end(&mut ring, 1);
            let said = heard.iter().find(|nack| nack.message.contains("hung"));
            assert!(said.is_some(), "only these reached the entry: {:?}", heard.iter().map(|nack| &nack.message).collect::<Vec<_>>());
            assert_eq!((said.unwrap().id, said.unwrap().refusal, said.unwrap().stage), (0, Refusal::FailedStage, 1));
        }
    });
}

#[test]
fn a_single_stage_that_hangs_tells_its_own_worker_before_its_process_ends() {
    within_deadline("a_single_stage_that_hangs_tells_its_own_worker_before_its_process_ends", || {
        for _ in 0..5 {
            let mut ring = ring(1, false, Duration::ZERO, 60_000);
            let heard = hang_and_end(&mut ring, 0);
            assert!(heard.iter().any(|nack| nack.message.contains("hung") && nack.stage == 0), "{:?}", heard.iter().map(|nack| &nack.message).collect::<Vec<_>>());
        }
    });
}

#[test]
fn a_last_stage_that_hangs_is_heard_over_the_closing_link_when_the_way_back_is_stuck() {
    within_deadline("a_last_stage_that_hangs_is_heard_over_the_closing_link_when_the_way_back_is_stuck", || {
        for _ in 0..3 {
            // The hop into the last stage goes through a relay that passes nothing back, so the
            // walk back along the route cannot carry the refusal. Only the closing link can.
            let (mut ring, relay) = ring_through(2, 3, true, Duration::ZERO, 60_000);
            ring.client.send(&pass(1, 5, 9).encode()).unwrap();
            answer(&mut ring);
            relay.hold_back();
            let heard = hang_and_end(&mut ring, 2);
            assert!(heard.iter().any(|nack| nack.message.contains("hung") && nack.stage == 2), "{:?}", heard.iter().map(|nack| &nack.message).collect::<Vec<_>>());
        }
    });
}

#[test]
#[should_panic(expected = "a_test_that_hangs ran past 1 s and was stopped")]
fn a_test_that_hangs_is_stopped_at_its_deadline_and_shows_its_nodes() {
    within("a_test_that_hangs", 1, || {
        let _ring = ring(2, true, Duration::ZERO, 2000);
        thread::sleep(Duration::from_secs(600));
    });
}

#[test]
fn a_test_that_fails_inside_its_deadline_fails_with_its_own_message() {
    let failed = std::panic::catch_unwind(|| within("fails_by_itself", 60, || panic!("its own words")));
    let message = failed.unwrap_err();
    assert_eq!(message.downcast_ref::<&str>(), Some(&"its own words"));
}

#[test]
fn a_stage_with_no_route_has_nobody_to_tell() {
    within_deadline("a_stage_with_no_route_has_nobody_to_tell", || {
        Node::new().unwrap().refuser()("nothing was in flight".into());
    });
}

#[test]
fn a_stage_that_goes_silent_is_noticed_without_any_traffic() {
    within_deadline("a_stage_that_goes_silent_is_noticed_without_any_traffic", || {
        // The middle stage's links stop answering when its whole process freezes. Stopping its
        // heartbeat answers is what a frozen process looks like from outside.
        let mut ring = ring(2, false, Duration::ZERO, 300);
        ring.nodes[1].close_ring();
        let began = Instant::now();
        let (kind, body) = ring.frames.next().unwrap();
        assert_eq!(kind, Kind::Nack);
        assert_eq!(Nack::decode(&body).unwrap().stage, 1);
        assert!(began.elapsed() < Duration::from_secs(2));
    });
}

#[test]
fn a_peer_that_stops_reading_is_lost_within_the_silence_limit() {
    within_deadline("a_peer_that_stops_reading_is_lost_within_the_silence_limit", || {
        let silence = Duration::from_millis(1000);
        let (mut ring, relay) = ring_through(1, 2, false, Duration::ZERO, silence.as_millis() as u64);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        relay.refuse(true);
        relay.hold();
        let began = Instant::now();
        // More than the sockets on the way can hold, so a write stops partway through a frame.
        for id in 2..514 {
            ring.client.send(&rows(id, 2048).encode()).unwrap();
        }
        wait_for_loss(&mut ring, 1);
        // The margin is for a busy machine: one turn of the watcher and some scheduling.
        let took = began.elapsed();
        assert!(took < silence + Duration::from_millis(500), "took {took:?}");
        assert!(!ring.nodes[0].stats().next_up);
    });
}

#[test]
fn a_write_stuck_on_a_long_frame_does_not_hold_up_the_acknowledgement_check() {
    within_deadline("a_write_stuck_on_a_long_frame_does_not_hold_up_the_acknowledgement_check", || {
        // Only the acknowledgement limit can say the link is lost in time. It is longer than the
        // gap between heartbeats, so a watcher that waits on the socket gets stuck on a heartbeat
        // first, and sees nothing until the write limit of half the silence limit.
        let ack = Duration::from_millis(1500);
        let (mut ring, relay) = ring_with_limits(1, 2, false, Duration::ZERO, (ack.as_millis() as u64, 8000));
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        relay.refuse(true);
        relay.hold();
        let began = Instant::now();
        for id in 2..514 {
            ring.client.send(&rows(id, 2048).encode()).unwrap();
        }
        wait_for_loss(&mut ring, 1);
        let took = began.elapsed();
        assert!(took < ack + Duration::from_millis(800), "took {took:?}");
    });
}

#[test]
fn a_frame_too_big_for_one_tls_buffer_crosses_a_link() {
    within_deadline("a_frame_too_big_for_one_tls_buffer_crosses_a_link", || {
        let mut ring = ring(3, true, Duration::ZERO, 2000);
        ring.client.send(&rows(1, 1 << 16).encode()).unwrap();
        assert_eq!(answer(&mut ring).tokens, vec![123, 9]);
    });
}

#[test]
fn losing_the_closing_link_costs_nothing_but_speed() {
    within_deadline("losing_the_closing_link_costs_nothing_but_speed", || {
        let (mut ring, relay) = ring_through(0, 3, true, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        assert!(ring.nodes[2].stats().closing_up);
        relay.cut();
        wait_until("the closing link's loss", || !ring.nodes[2].stats().closing_up);
        ring.client.send(&pass(2, 5, 9).encode()).unwrap();
        let got = answer(&mut ring);
        assert_eq!(got, sealed(2, vec![5123, 9]));
        assert_eq!(ring.nodes[2].stats().refusals, 0);
    });
}

/// Keeps a test that loses an answer from waiting for ever.
fn patient_client(ring: &Ring) {
    ring.client.limit_reads(Duration::from_secs(4)).unwrap();
}

/// So a hold that follows catches only what the test sends next.
fn wait_for_acknowledgements(ring: &Ring) {
    wait_until("every acknowledgement", || ring.nodes.last().unwrap().stats().answers_held == 0);
}

#[test]
fn an_answer_held_by_a_closing_link_that_goes_silent_arrives_by_walking_back() {
    within_deadline("an_answer_held_by_a_closing_link_that_goes_silent_arrives_by_walking_back", || {
        let ack = Duration::from_millis(600);
        let (mut ring, relay) = ring_through(0, 3, true, Duration::ZERO, ack.as_millis() as u64);
        patient_client(&ring);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        wait_for_acknowledgements(&ring);
        relay.refuse(true);
        relay.hold();
        let began = Instant::now();
        ring.client.send(&pass(2, 5, 10).encode()).unwrap();
        let got = answer(&mut ring);
        assert_eq!(got.tokens, vec![5123, 10]);
        assert!(began.elapsed() < ack + Duration::from_millis(400), "took {:?}", began.elapsed());
        let last = ring.nodes[2].stats();
        assert_eq!((last.resent, last.refusals), (1, 0));
        wait_until("the answer's acknowledgement", || ring.nodes[2].stats().answers_held == 0);
    });
}

#[test]
fn an_answer_on_a_closing_link_that_is_cut_arrives_by_walking_back() {
    within_deadline("an_answer_on_a_closing_link_that_is_cut_arrives_by_walking_back", || {
        let (mut ring, relay) = ring_through(0, 3, true, Duration::from_millis(100), 2000);
        patient_client(&ring);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        relay.refuse(true);
        relay.hold();
        ring.client.send(&pass(2, 5, 10).encode()).unwrap();
        wait_until("the answer leaving", || ring.nodes[2].stats().answers_held == 1);
        relay.cut();
        assert_eq!(answer(&mut ring).tokens, vec![5123, 10]);
    });
}

#[test]
fn every_answer_is_acknowledged_by_the_way_it_came() {
    within_deadline("every_answer_is_acknowledged_by_the_way_it_came", || {
        for closing in [true, false] {
            let mut ring = ring(3, closing, Duration::ZERO, 2000);
            for id in 1..=20 {
                ring.client.send(&pass(id, 5, id as u32).encode()).unwrap();
            }
            for _ in 0..20 {
                answer(&mut ring);
            }
            wait_for_acknowledgements(&ring);
            assert_eq!(ring.nodes[2].stats().resent, 0);
        }
    });
}

#[test]
fn an_answer_whose_acknowledgement_is_lost_reaches_the_worker_once() {
    within_deadline("an_answer_whose_acknowledgement_is_lost_reaches_the_worker_once", || {
        let (mut ring, relay) = ring_through(0, 3, true, Duration::ZERO, 500);
        patient_client(&ring);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        wait_for_acknowledgements(&ring);
        relay.refuse(true);
        relay.hold_back();
        ring.client.send(&pass(2, 5, 10).encode()).unwrap();
        assert_eq!(answer(&mut ring).id, 2);
        wait_until("the repeat", || ring.nodes[0].stats().dropped == 1);
        wait_until("its acknowledgement", || ring.nodes[2].stats().answers_held == 0);
        ring.client.send(&pass(3, 5, 11).encode()).unwrap();
        assert_eq!(answer(&mut ring).id, 3);
    });
}

#[test]
fn a_lost_closing_link_comes_back_at_both_ends() {
    within_deadline("a_lost_closing_link_comes_back_at_both_ends", || {
        let (mut ring, relay) = ring_through(0, 3, true, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        wait_for_acknowledgements(&ring); // or the first answer is still held, and goes again
        relay.cut();
        wait_until("the closing link's loss", || relay.attempts() == 2);
        wait_until("the closing link's return", || ring.nodes[2].stats().closing_up && ring.nodes[0].stats().closing_up);
        ring.client.send(&pass(2, 5, 10).encode()).unwrap();
        assert_eq!(answer(&mut ring).tokens, vec![5123, 10]);
        assert_eq!(ring.nodes[2].stats().resent, 0);
    });
}

#[test]
fn the_entry_forgets_a_closing_link_that_died_or_went_silent() {
    within_deadline("the_entry_forgets_a_closing_link_that_died_or_went_silent", || {
        for silent in [false, true] {
            let (ring, relay) = ring_through(0, 2, true, Duration::ZERO, 400);
            assert!(ring.nodes[0].stats().closing_up);
            relay.refuse(true);
            if silent {
                relay.hold();
            } else {
                relay.cut();
            }
            wait_until("the entry noticing", || !ring.nodes[0].stats().closing_up);
            wait_until("the last stage noticing", || !ring.nodes[1].stats().closing_up);
        }
    });
}

#[test]
fn redialling_backs_off_and_stops_with_the_ring() {
    within_deadline("redialling_backs_off_and_stops_with_the_ring", || {
        let (ring, relay) = ring_through(0, 2, true, Duration::ZERO, 2000);
        relay.refuse(true);
        relay.cut();
        wait_until("the first redial", || relay.attempts() >= 2);
        thread::sleep(Duration::from_millis(1800));
        // At 0.25 s apart that would be eight more. Doubling gives three.
        assert!(relay.attempts() <= 5, "{} attempts", relay.attempts());
        relay.refuse(false);
        let began = Instant::now();
        wait_until("the closing link's return", || ring.nodes[1].stats().closing_up);
        assert!(began.elapsed() < Duration::from_secs(3));
        relay.refuse(true);
        relay.cut();
        ring.nodes[1].close_ring();
        let attempts = relay.attempts();
        thread::sleep(Duration::from_millis(700));
        assert_eq!(relay.attempts(), attempts);
    });
}

/// Waits for a first heartbeat, which is what shows a link was ever worth redialling.
fn let_links_prove_themselves(ring: &Ring, stages: usize) {
    wait_until("the heartbeats", || ring.nodes[..stages].iter().all(|node| node.stats().link_rtt_us > 0));
}

#[test]
fn a_silent_hop_is_redialled_and_what_it_held_goes_again_in_order() {
    within_deadline("a_silent_hop_is_redialled_and_what_it_held_goes_again_in_order", || {
        for door in [1, 2] {
            let ack = Duration::from_millis(600);
            let (mut ring, relay) = ring_through(door, 3, true, Duration::ZERO, ack.as_millis() as u64);
            patient_client(&ring);
            ring.client.send(&pass(1, 5, 9).encode()).unwrap();
            answer(&mut ring);
            let_links_prove_themselves(&ring, 2);
            relay.hold();
            let began = Instant::now();
            for id in 2..=20 {
                ring.client.send(&pass(id, 5, id as u32).encode()).unwrap();
            }
            for id in 2..=20 {
                let got = answer(&mut ring);
                assert_eq!((got.id, got.tokens), (id, vec![5123, id as u32]));
            }
            assert!(began.elapsed() < ack + Duration::from_millis(500), "took {:?}", began.elapsed());
            assert_eq!(relay.attempts(), 2);
            assert!(ring.nodes[door - 1].stats().next_up);
            assert_eq!(ring.nodes[door - 1].stats().refusals, 0);
        }
    });
}

#[test]
fn a_hop_silent_one_way_is_redialled_even_when_the_far_end_gives_up_on_it_first() {
    within_deadline("a_hop_silent_one_way_is_redialled_even_when_the_far_end_gives_up_on_it_first", || {
        // The far end's read limit ends the link, and that close still gets through. It is not a
        // stage dying, so the reply should not pay for a replay. Which side's limit fires first
        // depends on the system, so the close is made here, just inside the near end's own limit.
        let (mut ring, relay) = ring_through(1, 3, true, Duration::ZERO, 600);
        patient_client(&ring);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        let_links_prove_themselves(&ring, 2);
        relay.hold_forth();
        for id in 2..=5 {
            ring.client.send(&pass(id, 5, id as u32).encode()).unwrap();
        }
        thread::sleep(Duration::from_millis(450));
        relay.cut();
        for id in 2..=5 {
            assert_eq!(answer(&mut ring).id, id);
        }
        assert_eq!(relay.attempts(), 2);
        assert_eq!(ring.nodes[0].stats().refusals, 0);
    });
}

#[test]
fn a_pass_that_arrived_but_whose_acknowledgement_was_lost_runs_once() {
    within_deadline("a_pass_that_arrived_but_whose_acknowledgement_was_lost_runs_once", || {
        let (mut ring, relay) = ring_through(1, 3, true, Duration::ZERO, 500);
        patient_client(&ring);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        let_links_prove_themselves(&ring, 2);
        relay.hold_back();
        ring.client.send(&pass(2, 5, 10).encode()).unwrap();
        assert_eq!(answer(&mut ring).id, 2);
        wait_until("the repeat reaching the stage", || ring.nodes[1].stats().dropped == 1);
        ring.client.send(&pass(3, 5, 11).encode()).unwrap();
        assert_eq!(answer(&mut ring).id, 3);
        assert_eq!(ring.seen[1].load(Ordering::SeqCst), 3);
        assert_eq!(ring.seen[2].load(Ordering::SeqCst), 3);
    });
}

#[test]
fn a_hop_that_cannot_be_redialled_is_refused_soon_after_it_is_lost() {
    within_deadline("a_hop_that_cannot_be_redialled_is_refused_soon_after_it_is_lost", || {
        let ack = Duration::from_millis(500);
        let (mut ring, relay) = ring_through(1, 2, false, Duration::ZERO, ack.as_millis() as u64);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        let_links_prove_themselves(&ring, 1);
        relay.swallow();
        relay.hold();
        let began = Instant::now();
        ring.client.send(&pass(2, 5, 10).encode()).unwrap();
        wait_for_loss(&mut ring, 1);
        assert!(began.elapsed() < ack + Duration::from_millis(2500), "took {:?}", began.elapsed());
        assert_eq!(relay.attempts(), 2);
        assert!(!ring.nodes[0].stats().next_up);
    });
}

#[test]
fn a_redialled_hop_that_says_nothing_is_not_redialled_again() {
    within_deadline("a_redialled_hop_that_says_nothing_is_not_redialled_again", || {
        let ack = 400;
        let (mut ring, relay) = ring_through(1, 2, false, Duration::ZERO, ack);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        let_links_prove_themselves(&ring, 1);
        relay.hold();
        wait_until("the redial", || relay.attempts() == 2);
        // The new link's first beat goes out a tenth of a second after it opens, and the answer
        // to it would prove the link and earn a third dial. Hold it well before that. Holding it
        // before its handshake ends is a dial that failed, which must not dial again either.
        thread::sleep(Duration::from_millis(40));
        relay.hold();
        wait_for_loss(&mut ring, 1);
        assert_eq!(relay.attempts(), 2);
    });
}

#[test]
fn a_busy_stage_is_not_taken_for_dead() {
    within_deadline("a_busy_stage_is_not_taken_for_dead", || {
        // Each pass takes longer than the timeout, and the acknowledgement must not wait for it.
        let mut ring = ring(2, true, Duration::from_millis(400), 300);
        for id in 1..=3 {
            ring.client.send(&pass(id, 1, 1).encode()).unwrap();
        }
        for id in 1..=3 {
            assert_eq!(answer(&mut ring).id, id);
        }
    });
}

#[test]
fn passes_that_wait_share_a_trip_and_each_still_gets_its_own_answer() {
    within_deadline("passes_that_wait_share_a_trip_and_each_still_gets_its_own_answer", || {
        let mut ring = ring(2, true, Duration::from_millis(30), 2000);
        let total: u64 = 12;
        for id in 1..=total {
            let mut item = pass(id, id as u32, 3);
            item.slot = id as u32;
            ring.client.send(&item.encode()).unwrap();
        }
        let mut answers: Vec<_> = (0..total).map(|_| answer(&mut ring)).collect();
        answers.sort_by_key(|item| item.id);
        for (index, item) in answers.iter().enumerate() {
            let id = index as u32 + 1;
            assert_eq!(item.tokens[0], (id * 10 + 1) * 10 + 2, "reply {id} got another reply's result");
        }
    });
}

#[test]
fn a_merge_reaches_past_a_slots_second_pass_and_refuses_only_its_own_row() {
    within_deadline("a_merge_reaches_past_a_slots_second_pass_and_refuses_only_its_own_row", || {
        let mut ring = ring(2, true, Duration::from_millis(30), 2000);
        // The first holds the stage while the rest queue up, slot 1's two passes side by side.
        let queued = [(1, 9, 1), (2, 1, 7), (3, 1, 8), (4, 2, 9), (5, 3, 666), (6, 4, 10)];
        for (id, slot, position) in queued {
            ring.client.send(&Pass { slot, ..pass(id, 3, position) }.encode()).unwrap();
        }
        let (mut answered, mut refused) = (vec![], vec![]);
        for _ in queued {
            let (kind, body) = ring.frames.next().unwrap();
            match kind {
                Kind::Result => answered.push(Answer::decode(&body).unwrap().id),
                Kind::Nack => refused.push(Nack::decode(&body).unwrap().id),
                other => panic!("{other:?}"),
            }
        }
        answered.sort_unstable();
        assert_eq!(answered, vec![1, 2, 3, 4, 6]);
        assert_eq!(refused, vec![5]);
        assert!(ring.nodes[0].stats().merged_rows >= 4, "{:?}", ring.nodes[0].stats());
    });
}

#[test]
fn passes_on_one_sequence_never_share_a_trip() {
    within_deadline("passes_on_one_sequence_never_share_a_trip", || {
        let mut ring = ring(2, true, Duration::from_millis(5), 2000);
        for id in 1..=20 {
            ring.client.send(&pass(id, 1, id as u32).encode()).unwrap();
        }
        for id in 1..=20 {
            assert_eq!(answer(&mut ring).tokens[1], id as u32);
        }
        assert_eq!(ring.nodes[0].stats().merged_passes, 0);
    });
}

#[test]
fn replies_that_arrive_a_moment_apart_are_gathered_into_one_trip() {
    within_deadline("replies_that_arrive_a_moment_apart_are_gathered_into_one_trip", || {
        let mut ring = ring(2, true, Duration::from_millis(20), 4000);
        let slots: u64 = 4;
        for round in 0..6u64 {
            // Each reply sends its next pass once the last round's answers are all in, one a
            // moment after another, as replies do that were answered together.
            for slot in 1..=slots {
                let mut item = pass(round * slots + slot, slot as u32, round as u32);
                item.slot = slot as u32;
                ring.client.send(&item.encode()).unwrap();
                std::thread::sleep(Duration::from_millis(2));
            }
            let mut answers: Vec<_> = (0..slots).map(|_| answer(&mut ring)).collect();
            answers.sort_by_key(|item| item.id);
            for (index, item) in answers.iter().enumerate() {
                let slot = index as u32 + 1;
                assert_eq!(item.id, round * slots + u64::from(slot));
                assert_eq!(item.tokens[0], (slot * 10 + 1) * 10 + 2, "reply {slot} got another reply's result");
            }
        }
        // The first pass of a round finds the stage free, and waits for the ones it knows
        // are coming instead of going alone.
        let stats = ring.nodes[0].stats();
        assert!(stats.gathers >= 1 && stats.gather_us > 0, "{stats:?}");
        assert!(stats.gather_us <= 20_000, "{stats:?}");
    });
}

#[test]
fn a_rewind_clears_what_is_queued_past_it() {
    within_deadline("a_rewind_clears_what_is_queued_past_it", || {
        // Long passes, so waiting behind eight of them stands well clear of a slow machine's noise.
        let mut ring = ring(2, true, Duration::from_millis(120), 4000);
        for id in 1..=8 {
            ring.client.send(&pass(id, 1, 100 + id as u32).encode()).unwrap();
        }
        // The answer to the first pass said the rest were built on a wrong guess.
        let rewind = Pass { rollback_to: Some(102), ..pass(9, 2, 102) };
        ring.client.send(&rewind.encode()).unwrap();
        let began = Instant::now();
        let mut seen = vec![];
        while seen.last() != Some(&9) {
            seen.push(answer(&mut ring).id);
        }
        assert!(seen.len() < 5, "{seen:?}");
        let took = began.elapsed();
        assert!(took < Duration::from_millis(840), "the rewind waited behind passes nobody wants: {took:?}");
        assert!(ring.nodes[0].stats().dropped >= 5);
    });
}

#[test]
fn a_rewind_leaves_other_replies_and_earlier_places_alone() {
    within_deadline("a_rewind_leaves_other_replies_and_earlier_places_alone", || {
        let mut ring = ring(2, true, Duration::from_millis(20), 2000);
        let other = Pass { slot: 1, ..pass(1, 1, 500) };
        ring.client.send(&pass(2, 1, 10).encode()).unwrap();
        ring.client.send(&other.encode()).unwrap();
        ring.client.send(&pass(3, 1, 11).encode()).unwrap();
        ring.client.send(&Pass { rollback_to: Some(12), ..pass(4, 2, 12) }.encode()).unwrap();
        let mut seen: Vec<u64> = (0..4).map(|_| answer(&mut ring).id).collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![1, 2, 3, 4]);
        assert_eq!(ring.nodes[0].stats().dropped, 0);
    });
}

#[test]
fn a_new_reply_clears_what_an_older_one_left_queued_on_its_slot() {
    within_deadline("a_new_reply_clears_what_an_older_one_left_queued_on_its_slot", || {
        let mut ring = ring(2, true, Duration::from_millis(30), 2000);
        for id in 1..=8 {
            ring.client.send(&pass(id, 1, id as u32).encode()).unwrap();
        }
        let other = Pass { slot: 1, ..pass(9, 1, 3) };
        ring.client.send(&other.encode()).unwrap();
        ring.client.send(&Pass { generation: 2, ..pass(10, 1, 0) }.encode()).unwrap();
        let mut seen = vec![];
        while !seen.contains(&10) || !seen.contains(&9) {
            seen.push(answer(&mut ring).id);
        }
        assert!(seen.len() < 5, "{seen:?}");
        assert!(ring.nodes[0].stats().dropped >= 5);
    });
}

#[test]
fn a_pass_from_a_generation_that_has_ended_never_runs() {
    within_deadline("a_pass_from_a_generation_that_has_ended_never_runs", || {
        let mut ring = ring(2, true, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        ring.client.send(&Pass { generation: 2, ..pass(2, 5, 0) }.encode()).unwrap();
        answer(&mut ring);
        let ran = ring.seen[0].load(Ordering::SeqCst);
        // It reaches the stage after the reply that replaced it, as a pass orphaned by a lost link can.
        ring.client.send(&pass(3, 5, 10).encode()).unwrap();
        ring.client.send(&Pass { generation: 2, ..pass(4, 5, 1) }.encode()).unwrap();
        assert_eq!(answer(&mut ring).id, 4);
        assert_eq!(ring.seen[0].load(Ordering::SeqCst), ran + 1);
        assert_eq!(ring.nodes[0].stats().dropped, 1);
    });
}

#[test]
fn a_pass_that_arrives_twice_runs_once() {
    within_deadline("a_pass_that_arrives_twice_runs_once", || {
        let mut ring = ring(2, true, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        ring.client.send(&pass(2, 5, 10).encode()).unwrap();
        assert_eq!(answer(&mut ring).id, 1);
        assert_eq!(answer(&mut ring).id, 2);
        assert_eq!(ring.seen[1].load(Ordering::SeqCst), 2);
    });
}

#[test]
fn the_entry_drops_a_late_result_from_an_older_generation() {
    within_deadline("the_entry_drops_a_late_result_from_an_older_generation", || {
        let mut ring = ring(2, true, Duration::from_millis(100), 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        ring.client.send(&Pass { generation: 2, ..pass(2, 5, 0) }.encode()).unwrap();
        assert_eq!(answer(&mut ring).id, 2);
        assert!(ring.nodes[0].stats().dropped >= 1);
    });
}

/// Says what the last stage says, from outside it, as a stage that lost track could.
fn closing_voice(ring: &Ring) -> Arc<Link> {
    let door = &ring.doors[0];
    link::dial(&format!("127.0.0.1:{}", door.port), Some(&door.fingerprint), CLOSING, Role::Closing, 7, Duration::from_secs(3)).unwrap()
}

#[test]
fn the_entry_passes_on_an_answer_it_has_already_seen_only_once() {
    within_deadline("the_entry_passes_on_an_answer_it_has_already_seen_only_once", || {
        let mut ring = ring(2, false, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        let first = answer(&mut ring);
        let voice = closing_voice(&ring);
        voice.send(&first.encode()).unwrap();
        voice.send(&Answer { id: 2, ..first.clone() }.sealed(CLOSING).encode()).unwrap();
        voice.send(&Answer { id: 2, ..first.clone() }.sealed(CLOSING).encode()).unwrap();
        voice.send(&Answer { id: 3, ..first }.sealed(CLOSING).encode()).unwrap();
        assert_eq!(answer(&mut ring).id, 2);
        assert_eq!(answer(&mut ring).id, 3);
        wait_until("both repeats dropped", || ring.nodes[0].stats().dropped == 2);
        voice.close();
    });
}

#[test]
fn an_answer_from_no_reply_this_entry_knows_is_dropped() {
    within_deadline("an_answer_from_no_reply_this_entry_knows_is_dropped", || {
        let mut ring = ring(2, false, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        let first = answer(&mut ring);
        let voice = closing_voice(&ring);
        voice.send(&Answer { id: 2, generation: 99, ..first.clone() }.sealed(CLOSING).encode()).unwrap();
        voice.send(&Answer { id: 3, slot: 4, ..first.clone() }.sealed(CLOSING).encode()).unwrap();
        voice.send(&Answer { id: 4, ..first }.sealed(CLOSING).encode()).unwrap();
        assert_eq!(answer(&mut ring).id, 4);
        assert_eq!(ring.nodes[0].stats().dropped, 2);
        voice.close();
    });
}

#[test]
fn a_stage_can_speak_only_on_the_links_it_is_part_of() {
    within_deadline("a_stage_can_speak_only_on_the_links_it_is_part_of", || {
        let mut ring = ring(3, true, Duration::ZERO, 2000);
        {
            let dial = |at: usize, key: &[u8], role: Role| {
                let door = &ring.doors[at];
                link::dial(&format!("127.0.0.1:{}", door.port), Some(&door.fingerprint), key, role, 7, Duration::from_secs(3))
            };
            // Stage one holds the keys of the hops on either side of it, and nothing else.
            for key in [hop_key(0), hop_key(1)] {
                assert!(dial(0, &key, Role::Closing).is_err(), "it closed the ring as if it were the last stage");
            }
            assert!(dial(2, &hop_key(0), Role::Hop).is_err(), "it dialled past its neighbour");
            assert!(dial(1, &hop_key(1), Role::Hop).is_err(), "a key went the wrong way along its link");
            assert!(dial(1, CLOSING, Role::Closing).is_err(), "a stage in the middle took a closing link");
            assert!(dial(0, &hop_key(0), Role::Hop).is_err(), "the entry took a hop");
        }
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        assert_eq!(answer(&mut ring), sealed(1, vec![5123, 9]));
        let (entry, middle, last) = (ring.nodes[0].stats(), ring.nodes[1].stats(), ring.nodes[2].stats());
        assert!(entry.next_up && entry.closing_up && middle.next_up && last.closing_up);
        assert_eq!(ring.nodes[2].stats().resent, 0);
    });
}

#[test]
fn an_answer_a_middle_stage_makes_up_is_dropped_and_its_link_stays_up() {
    within_deadline("an_answer_a_middle_stage_makes_up_is_dropped_and_its_link_stays_up", || {
        // The test stands in for stage one of three, holding only what stage one would hold.
        let entry = Node::new().unwrap();
        entry.set_compute(Box::new(Adder { stage: 1, last: false, delay: Duration::ZERO, seen: Arc::default() }));
        let door = listening(&entry);
        entry.open(Plan { epoch: 7, index: 0, count: 3, keys: keys_for(0, 3), ack_timeout_ms: 5000, silence_timeout_ms: 5000, merge_passes: true, merge_most: None }).unwrap();
        let identity = Identity::generate().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = Address { address: listener.local_addr().unwrap().to_string(), fingerprint: identity.fingerprint.clone() };
        let middle = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            let (link, frames, _) = link::accept(socket, Some(&identity), link::HANDSHAKE, |role, _| (role == Role::Hop).then(|| hop_key(0))).unwrap();
            (link, frames)
        });
        entry.connect(Links { next: Some(address), entry: None, timeout_ms: 3000 }).unwrap();
        let (link, mut frames) = middle.join().unwrap();
        let client = link::dial(&format!("127.0.0.1:{}", door.local_port), None, &door.local_key, Role::Local, 0, Duration::from_secs(3)).unwrap();
        let mut answers = client.reader(1 << 20);
        client.send(&pass(1, 5, 9).encode()).unwrap();
        let id = loop {
            let (kind, body) = frames.next().unwrap();
            if kind == Kind::Pass {
                break Pass::decode(body).unwrap().id;
            }
        };
        link.beat(&Ack { id }.encode()).unwrap();
        let made_up = Answer { id, generation: 1, epoch: 7, slot: 0, tokens: vec![42, 9], seal: [0; MAC_BYTES] };
        link.send(&made_up.encode()).unwrap();
        for key in [hop_key(0), hop_key(1)] {
            link.send(&made_up.clone().sealed(&key).encode()).unwrap();
        }
        // What the last stage really sealed, walking back through this stage, still lands.
        link.send(&sealed(id, vec![5123, 9]).encode()).unwrap();
        let (kind, body) = answers.next().unwrap();
        assert_eq!(kind, Kind::Result);
        assert_eq!(Answer::decode(body).unwrap().tokens, vec![5123, 9]);
        let stats = entry.stats();
        assert_eq!((stats.forged, stats.answers), (3, 1));
        assert!(stats.next_up && !link.is_closed());
        link.close();
        client.close();
    });
}

#[test]
fn stopping_frees_a_nodes_ports_at_once() {
    within_deadline("stopping_frees_a_nodes_ports_at_once", || {
        let mut ring = ring(2, true, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        for (node, door) in ring.nodes.iter().zip(&ring.doors) {
            node.stop();
            for port in [door.port, door.local_port] {
                TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|error| panic!("port {port} is still held: {error}"));
            }
        }
    });
}

#[test]
fn hop_time_is_a_round_trip_that_leaves_out_the_next_stages_work() {
    within_deadline("hop_time_is_a_round_trip_that_leaves_out_the_next_stages_work", || {
        let delay = Duration::from_millis(300);
        let mut ring = ring(2, true, delay, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        wait_until("a hop time", || ring.nodes[0].stats().hop_us > 0);
        let stats = ring.nodes[0].stats();
        assert!(stats.hop_us < delay.as_micros() as u64, "{}", stats.hop_us);
        assert_eq!(serde_json::to_value(&stats).unwrap()["hop_us"], stats.hop_us);
        assert_eq!(ring.nodes[1].stats().hop_us, 0);
    });
}

#[test]
fn nobody_reaches_the_local_door_without_its_key() {
    within_deadline("nobody_reaches_the_local_door_without_its_key", || {
        let node = Node::new().unwrap();
        let door = listening(&node);
        let got = link::dial(&format!("127.0.0.1:{}", door.local_port), None, b"wrong", Role::Local, 0, Duration::from_secs(3));
        assert!(got.is_err());
    });
}

#[test]
fn the_local_role_is_refused_on_the_public_door() {
    within_deadline("the_local_role_is_refused_on_the_public_door", || {
        let node = Node::new().unwrap();
        let door = listening(&node);
        let got = link::dial(&format!("127.0.0.1:{}", door.port), Some(&door.fingerprint), &door.local_key, Role::Local, 0, Duration::from_secs(3));
        assert!(got.is_err());
    });
}

#[test]
fn a_route_cannot_be_joined_with_the_wrong_key_or_epoch() {
    within_deadline("a_route_cannot_be_joined_with_the_wrong_key_or_epoch", || {
        let node = Node::new().unwrap();
        let door = listening(&node);
        node.open(Plan { epoch: 7, index: 1, count: 2, keys: keys_for(1, 2), ack_timeout_ms: 1000, silence_timeout_ms: 1000, merge_passes: true, merge_most: None }).unwrap();
        let address = format!("127.0.0.1:{}", door.port);
        let key = hop_key(0);
        assert!(link::dial(&address, Some(&door.fingerprint), b"wrong", Role::Hop, 7, Duration::from_secs(3)).is_err());
        assert!(link::dial(&address, Some(&door.fingerprint), CLOSING, Role::Hop, 7, Duration::from_secs(3)).is_err());
        assert!(link::dial(&address, Some(&door.fingerprint), &key, Role::Hop, 8, Duration::from_secs(3)).is_err());
        assert!(link::dial(&address, Some(&door.fingerprint), &key, Role::Hop, 7, Duration::from_secs(3)).is_ok());
    });
}

/// What a next stage that only listens heard from a ring's first stage.
struct Heard {
    padding: usize,
    padding_after_the_pass: usize,
    pings_in_a_second_and_a_half: usize,
}

/// Stands in for a next stage: takes the link a ring dials, reads it without answering, and
/// counts what came. It stops a second and a half after the first pass.
fn listen_as_next_stage() -> (Address, thread::JoinHandle<Heard>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let identity = link::Identity::generate().unwrap();
    let address = Address { address: listener.local_addr().unwrap().to_string(), fingerprint: identity.fingerprint.clone() };
    let reading = thread::spawn(move || {
        let (socket, _) = listener.accept().unwrap();
        let key_for = |role, epoch| (role == Role::Hop && epoch == 7).then(|| hop_key(0));
        let (_link, mut frames, _) = link::accept(socket, Some(&identity), Duration::from_secs(10), key_for).unwrap();
        let mut heard = Heard { padding: 0, padding_after_the_pass: 0, pings_in_a_second_and_a_half: 0 };
        let mut passed: Option<Instant> = None;
        while passed.is_none_or(|at| at.elapsed() < Duration::from_millis(1500)) {
            let (kind, body) = frames.next().unwrap();
            match (kind, passed) {
                (Kind::Pad, None) => heard.padding += body.len(),
                (Kind::Pad, Some(_)) => heard.padding_after_the_pass += body.len(),
                (Kind::Pass, None) => passed = Some(Instant::now()),
                (Kind::Ping, Some(_)) => heard.pings_in_a_second_and_a_half += 1,
                _ => {}
            }
        }
        heard
    });
    (address, reading)
}

#[test]
fn a_ring_pads_the_link_it_dials_before_any_pass_and_beats_ten_times_a_second() {
    within_deadline("a_ring_pads_the_link_it_dials_before_any_pass_and_beats_ten_times_a_second", || {
        let node = Node::new().unwrap();
        let door = listening(&node);
        node.set_compute(Box::new(Adder { stage: 1, last: false, delay: Duration::ZERO, seen: Arc::default() }));
        node.open(Plan { epoch: 7, index: 0, count: 2, keys: keys_for(0, 2), ack_timeout_ms: 5000, silence_timeout_ms: 5000, merge_passes: true, merge_most: None }).unwrap();
        let (next, reading) = listen_as_next_stage();
        node.connect(Links { next: Some(next), entry: None, timeout_ms: 3000 }).unwrap();
        let client = link::dial(&format!("127.0.0.1:{}", door.local_port), None, &door.local_key, Role::Local, 0, Duration::from_secs(3)).unwrap();
        client.send(&pass(1, 5, 9).encode()).unwrap();
        let heard = reading.join().unwrap();
        assert_eq!(heard.padding, 1 << 20);
        assert_eq!(heard.padding_after_the_pass, 0);
        // Once a second before, so even a slow machine stays far from that.
        assert!((7..=16).contains(&heard.pings_in_a_second_and_a_half), "{} pings", heard.pings_in_a_second_and_a_half);
        node.stop();
    });
}

#[test]
fn padding_is_dropped_by_whoever_gets_it_and_what_follows_still_arrives() {
    within_deadline("padding_is_dropped_by_whoever_gets_it_and_what_follows_still_arrives", || {
        let mut ring = ring(3, true, Duration::ZERO, 2000);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        let first = answer(&mut ring);
        // Every link of this ring was padded as it opened, and none of them was lost for it.
        assert!(ring.nodes.iter().all(|node| node.stats().resent == 0 && node.stats().refusals == 0));
        assert!(ring.nodes[2].stats().closing_up);
        let voice = closing_voice(&ring);
        voice.send(&wire::pad(1000)).unwrap();
        voice.send(&Answer { id: 2, ..first }.sealed(CLOSING).encode()).unwrap();
        assert_eq!(answer(&mut ring).id, 2);
        voice.close();
    });
}

// ---- Two epochs on one node (spike P3) ----

const NEXT: u64 = 8;
const NEXT_CLOSING: &[u8] = &[200; 32];

/// The next route's keys: other keys for the same links, as a coordinator gives each epoch.
fn next_keys(index: u16, count: u16) -> Keys {
    let ends = count > 1 && (index == 0 || index + 1 == count);
    let hop = |edge: u16| vec![edge as u8 + 101; 32];
    Keys { back: (index > 0).then(|| hop(index - 1)), next: (index + 1 < count).then(|| hop(index)), closing: ends.then(|| NEXT_CLOSING.to_vec()) }
}

fn next_plan(index: u16, count: u16, epoch: u64, limit_ms: u64) -> Plan {
    Plan { epoch, index, count, keys: next_keys(index, count), ack_timeout_ms: limit_ms, silence_timeout_ms: limit_ms, merge_passes: true, merge_most: None }
}

/// Opens the next route beside the current one on the same nodes, dials its links, and gives
/// it `slots`. With `relayed`, the staged hop into that stage goes through a relay.
fn stage_next(ring: &Ring, epoch: u64, limit_ms: u64, slots: &[u32], relayed: Option<usize>) -> Option<Relay> {
    let count = ring.nodes.len() as u16;
    for (index, node) in ring.nodes.iter().enumerate() {
        node.stage(next_plan(index as u16, count, epoch, limit_ms)).unwrap();
    }
    let relay = relayed.map(|door| Relay::to(ring.doors[door].port));
    let address = |at: usize| {
        let port = relay.as_ref().filter(|_| relayed == Some(at)).map_or(ring.doors[at].port, |item| item.port);
        Address { address: format!("127.0.0.1:{port}"), fingerprint: ring.doors[at].fingerprint.clone() }
    };
    for (index, node) in ring.nodes.iter().enumerate().rev() {
        let next = (index + 1 < ring.nodes.len()).then(|| address(index + 1));
        let entry = (index + 1 == ring.nodes.len() && count > 1).then(|| address(0));
        node.connect_staged(Links { next, entry, timeout_ms: 3000 }).unwrap();
        node.give_slots(epoch, slots).unwrap();
    }
    relay
}

fn on(epoch: u64, slot: u32, id: u64, token: u32) -> Pass {
    Pass { epoch, slot, ..pass(id, token, id as u32) }
}

/// What two stages make of a token.
fn through_two(token: u32) -> u32 {
    (token * 10 + 1) * 10 + 2
}

fn refusal(ring: &mut Ring) -> Nack {
    let (kind, body) = ring.frames.next().unwrap();
    assert_eq!(kind, Kind::Nack, "expected a refusal");
    Nack::decode(&body).unwrap()
}

#[test]
fn two_epochs_share_one_node_each_on_its_own_ring_and_keys() {
    within_deadline("two_epochs_share_one_node_each_on_its_own_ring_and_keys", || {
        let mut ring = ring(2, true, Duration::ZERO, 2000);
        patient_client(&ring);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        assert_eq!(answer(&mut ring).tokens, vec![through_two(5), 9]);
        stage_next(&ring, NEXT, 2000, &[1], None);
        let stats = ring.nodes[0].stats();
        assert_eq!((stats.epoch, stats.staged_epoch), (Some(7), Some(NEXT)));
        // Ring 7 goes on carrying replies, ring 8 carries its own, interleaved one by one.
        for id in 2..=61 {
            let (epoch, slot) = if id % 2 == 0 { (7, 0) } else { (NEXT, 1) };
            ring.client.send(&on(epoch, slot, id, id as u32 % 9 + 1).encode()).unwrap();
        }
        let mut last = [1, 0];
        for _ in 2..=61 {
            let got = answer(&mut ring);
            let (epoch, key) = if got.slot == 0 { (7, CLOSING) } else { (NEXT, NEXT_CLOSING) };
            assert_eq!(got.epoch, epoch, "{got:?}");
            assert!(got.is_sealed_by(key), "answer {} not sealed by its own route's key", got.id);
            assert_eq!(got.tokens, vec![through_two(got.id as u32 % 9 + 1), got.id as u32]);
            assert!(got.id > last[got.slot as usize], "slot {} out of order", got.slot);
            last[got.slot as usize] = got.id;
        }
        assert!(ring.nodes.iter().all(|node| node.stats().refusals == 0));
        assert!(ring.nodes[1].stats().closing_up);
    });
}

#[test]
fn after_promotion_the_old_epoch_is_refused_and_what_it_had_in_flight_still_lands() {
    within_deadline("after_promotion_the_old_epoch_is_refused_and_what_it_had_in_flight_still_lands", || {
        let mut ring = ring(2, true, Duration::from_millis(40), 2000);
        patient_client(&ring);
        stage_next(&ring, NEXT, 2000, &[1], None);
        ring.client.send(&on(NEXT, 1, 1, 3).encode()).unwrap();
        assert_eq!(answer(&mut ring).epoch, NEXT);
        // Five passes of the old route in flight when every stage flips.
        for id in 2..=6 {
            ring.client.send(&on(7, 0, id, 4).encode()).unwrap();
        }
        // The entry is the gate: a pass it has not taken yet is refused after its flip. So flip
        // once the first has reached the second stage and the rest wait at the entry or between.
        wait_until("the old route's first pass reaching the second stage", || ring.seen[1].load(Ordering::SeqCst) >= 2); // the first was route 8's
        for node in &ring.nodes {
            node.promote(NEXT).unwrap();
        }
        assert!(ring.nodes[0].promote(NEXT).is_err(), "nothing is staged any more");
        let stats = ring.nodes[0].stats();
        assert_eq!((stats.epoch, stats.staged_epoch), (Some(NEXT), None));
        for id in 2..=6 {
            let (kind, body) = ring.frames.next().unwrap();
            assert_eq!(kind, Kind::Result, "pass {id}: {:?}", Nack::decode(body).ok());
            let got = Answer::decode(body).unwrap();
            assert_eq!((got.id, got.epoch, got.tokens[0]), (id, 7, through_two(4)));
        }
        ring.client.send(&on(7, 0, 7, 4).encode()).unwrap();
        let nack = refusal(&mut ring);
        assert_eq!((nack.id, nack.refusal, nack.epoch), (7, Refusal::Epoch, 7));
        // Once the old route has drained everywhere, every slot belongs to the new route.
        wait_until("the old route draining", || ring.nodes.iter().all(|node| node.stats().retiring_epochs.is_empty()));
        for (id, slot) in [(8, 0), (9, 1)] {
            ring.client.send(&on(NEXT, slot, id, 2).encode()).unwrap();
            let got = answer(&mut ring);
            assert_eq!((got.id, got.epoch, got.tokens[0]), (id, NEXT, through_two(2)));
        }
        // Drained, the old ring is shut: its hop key no longer opens a link.
        let door = &ring.doors[1];
        let dial = |key: &[u8], epoch: u64| link::dial(&format!("127.0.0.1:{}", door.port), Some(&door.fingerprint), key, Role::Hop, epoch, Duration::from_secs(1));
        wait_until("the old ring shutting", || dial(&hop_key(0), 7).is_err());
        assert!(dial(&next_keys(1, 2).back.unwrap(), NEXT).is_ok());
    });
}

#[test]
fn a_silent_link_on_the_staged_ring_leaves_the_current_rings_replies_alone() {
    within_deadline("a_silent_link_on_the_staged_ring_leaves_the_current_rings_replies_alone", || {
        let mut ring = ring(2, true, Duration::ZERO, 2000);
        patient_client(&ring);
        let relay = stage_next(&ring, NEXT, 400, &[1], Some(1)).unwrap();
        ring.client.send(&on(NEXT, 1, 1, 3).encode()).unwrap();
        assert_eq!(answer(&mut ring).epoch, NEXT);
        thread::sleep(Duration::from_millis(300)); // heartbeats prove the staged hop
        relay.refuse(true);
        relay.hold();
        let began = Instant::now();
        let (mut worst, mut staged_loss) = (Duration::ZERO, None);
        let mut id = 1;
        while began.elapsed() < Duration::from_millis(2500) {
            id += 1;
            let sent = Instant::now();
            ring.client.send(&on(7, 0, id, 5).encode()).unwrap();
            loop {
                let (kind, body) = ring.frames.next().unwrap();
                if kind == Kind::Nack {
                    let nack = Nack::decode(body).unwrap();
                    assert_eq!((nack.epoch, nack.refusal, nack.stage), (NEXT, Refusal::FailedStage, 1), "{nack:?}");
                    staged_loss.get_or_insert(began.elapsed());
                    continue;
                }
                let got = Answer::decode(body).unwrap();
                assert_eq!((got.id, got.epoch, got.tokens[0]), (id, 7, through_two(5)));
                break;
            }
            worst = worst.max(sent.elapsed());
            thread::sleep(Duration::from_millis(20));
        }
        let staged_loss = staged_loss.expect("the staged ring never said its hop was lost");
        println!("P3SILENT {{\"passes\": {}, \"worst_ms\": {:.2}, \"staged_loss_ms\": {}}}", id - 1, worst.as_secs_f64() * 1000.0, staged_loss.as_millis());
        assert!(worst < Duration::from_millis(250), "the current ring was held up: {worst:?}");
        assert!(ring.nodes[0].stats().next_up);
        assert_eq!(ring.nodes[0].stats().staged_epoch, Some(NEXT));
    });
}

#[test]
fn a_pass_on_a_slot_the_other_epoch_owns_is_refused_and_touches_nothing() {
    within_deadline("a_pass_on_a_slot_the_other_epoch_owns_is_refused_and_touches_nothing", || {
        let mut ring = ring(2, true, Duration::from_millis(30), 2000);
        patient_client(&ring); // a regression fails here instead of hanging
        stage_next(&ring, NEXT, 2000, &[1], None);
        // Slot 0 is busy on route 7, with passes queued behind the one running.
        for id in 1..=4 {
            ring.client.send(&on(7, 0, id, 1).encode()).unwrap();
        }
        // Route 8 tries slot 0 as a new reply, which on one KV would wipe route 7's.
        ring.client.send(&Pass { generation: 2, ..on(NEXT, 0, 5, 1) }.encode()).unwrap();
        ring.client.send(&on(7, 1, 6, 1).encode()).unwrap();
        let (mut answered, mut refused) = (vec![], vec![]);
        for _ in 0..6 {
            let (kind, body) = ring.frames.next().unwrap();
            match kind {
                Kind::Result => answered.push(Answer::decode(body).unwrap().id),
                Kind::Nack => refused.push(Nack::decode(body).unwrap()),
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(answered, vec![1, 2, 3, 4]);
        let refused: Vec<_> = refused.iter().map(|nack| (nack.id, nack.refusal, nack.message.clone())).collect();
        assert_eq!(refused, vec![(5, Refusal::Epoch, "slot 0 belongs to route 7".to_string()), (6, Refusal::Epoch, "slot 1 belongs to route 8".to_string())]);
        assert_eq!(ring.nodes[0].stats().dropped, 0, "a refused pass cleared another route's queue");
        assert_eq!(ring.seen[0].load(Ordering::SeqCst), 4);
        // The slot handed over later is the new route's.
        for node in &ring.nodes {
            node.give_slots(NEXT, &[0]).unwrap();
        }
        ring.client.send(&Pass { generation: 2, ..on(NEXT, 0, 7, 1) }.encode()).unwrap();
        assert_eq!(answer(&mut ring).epoch, NEXT);
    });
}

#[test]
fn a_flood_on_one_epoch_does_not_starve_the_other() {
    within_deadline("a_flood_on_one_epoch_does_not_starve_the_other", || {
        let mut ring = ring(2, true, Duration::from_millis(15), 4000);
        patient_client(&ring);
        stage_next(&ring, NEXT, 4000, &[1], None);
        // One slot, so nothing merges and each pass costs a full trip.
        for id in 1..=40 {
            ring.client.send(&on(7, 0, id, 1).encode()).unwrap();
        }
        ring.client.send(&on(NEXT, 1, 41, 1).encode()).unwrap();
        let order: Vec<u64> = (0..41).map(|_| answer(&mut ring).id).collect();
        let at = order.iter().position(|id| *id == 41).unwrap();
        println!("P3FAIR {{\"queued_ahead\": 40, \"answered_at\": {at}}}");
        assert!(at <= 5, "route 8 waited behind route 7's flood: answered {at}th");
    });
}

fn percentile(sorted: &[f64], share: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * share).round() as usize]
}

#[test]
fn time_a_plain_reopen_and_a_staged_promotion() {
    within_deadline("time_a_plain_reopen_and_a_staged_promotion", || {
        let repetitions = 20;
        let mut ring = ring(2, true, Duration::ZERO, 2000);
        patient_client(&ring);
        ring.client.send(&pass(1, 5, 9).encode()).unwrap();
        answer(&mut ring);
        let mut id = 1;
        let count = ring.nodes.len() as u16;
        let doors: Vec<Address> = ring.doors.iter().map(|door| Address { address: format!("127.0.0.1:{}", door.port), fingerprint: door.fingerprint.clone() }).collect();
        // The fallback: every stage opens the new route while idle, dials it, and the first pass runs.
        let mut reopen = vec![];
        for rep in 0..repetitions {
            let epoch = 100 + rep as u64;
            let began = Instant::now();
            for (index, node) in ring.nodes.iter().enumerate() {
                node.open(next_plan(index as u16, count, epoch, 2000)).unwrap();
            }
            for (index, node) in ring.nodes.iter().enumerate().rev() {
                let next = (index + 1 < ring.nodes.len()).then(|| doors[index + 1].clone());
                let entry = (index + 1 == ring.nodes.len()).then(|| doors[0].clone());
                node.connect(Links { next, entry, timeout_ms: 3000 }).unwrap();
            }
            id += 1;
            ring.client.send(&on(epoch, 0, id, 5).encode()).unwrap();
            let got = answer(&mut ring);
            assert_eq!((got.id, got.epoch), (id, epoch));
            reopen.push(began.elapsed().as_secs_f64() * 1000.0);
        }
        // Dual epoch: the next route is staged and dialled beforehand; only the flip is timed.
        let mut promote = vec![];
        for rep in 0..repetitions {
            let epoch = 200 + rep as u64;
            stage_next(&ring, epoch, 2000, &[], None);
            thread::sleep(Duration::from_millis(50));
            let began = Instant::now();
            // As a worker flips: every slot to the new route, which then takes over.
            for node in &ring.nodes {
                assert!(node.flip(epoch, &[0], Duration::from_secs(1)).unwrap());
            }
            id += 1;
            ring.client.send(&on(epoch, 0, id, 5).encode()).unwrap();
            let got = answer(&mut ring);
            assert_eq!((got.id, got.epoch), (id, epoch));
            promote.push(began.elapsed().as_secs_f64() * 1000.0);
        }
        for (what, times) in [("reopen", &mut reopen), ("promote", &mut promote)] {
            let raw = times.iter().map(|ms| format!("{ms:.3}")).collect::<Vec<_>>().join(", ");
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "P3TIME {{\"what\": \"{what}\", \"n\": {}, \"p50_ms\": {:.3}, \"max_ms\": {:.3}, \"min_ms\": {:.3}, \"raw_ms\": [{raw}]}}",
                times.len(),
                percentile(times, 0.5),
                times[times.len() - 1],
                times[0]
            );
        }
    });
}

/// What two stages make of a frame that is either an answer or a refusal.
fn landed(ring: &mut Ring) -> Result<Answer, Nack> {
    let (kind, body) = ring.frames.next().unwrap();
    match kind {
        Kind::Result => Ok(Answer::decode(&body).unwrap()),
        Kind::Nack => Err(Nack::decode(&body).unwrap()),
        other => panic!("{other:?}"),
    }
}

#[test]
fn after_promotion_a_slot_stays_the_old_routes_until_it_has_drained() {
    within_deadline("after_promotion_a_slot_stays_the_old_routes_until_it_has_drained", || {
        let mut ring = ring(2, true, Duration::from_millis(60), 2000);
        patient_client(&ring); // a regression fails instead of hanging
        stage_next(&ring, NEXT, 2000, &[1], None);
        // Route 7 has three passes of one reply on slot 0, spread along the route.
        for id in 1..=3 {
            ring.client.send(&on(7, 0, id, 1).encode()).unwrap();
        }
        wait_until("route 7's first pass reaching the second stage", || ring.seen[1].load(Ordering::SeqCst) >= 1);
        for node in &ring.nodes {
            node.promote(NEXT).unwrap();
        }
        // A new reply of route 8 on slot 0 now would wipe route 7's passes still on that slot.
        ring.client.send(&Pass { generation: 2, ..on(NEXT, 0, 4, 1) }.encode()).unwrap();
        let (mut answered, mut refused) = (vec![], vec![]);
        for _ in 0..4 {
            match landed(&mut ring) {
                Ok(got) => answered.push((got.id, got.epoch)),
                Err(nack) => refused.push((nack.id, nack.refusal, nack.message)),
            }
        }
        assert_eq!(answered, vec![(1, 7), (2, 7), (3, 7)], "route 7 lost passes to route 8's new reply");
        assert_eq!(refused, vec![(4, Refusal::Epoch, "slot 0 belongs to route 7".to_string())]);
        assert_eq!(ring.nodes[0].stats().dropped, 0);
        // Drained, the slot is the new route's, as every other slot is.
        wait_until("the old route draining", || ring.nodes.iter().all(|node| node.stats().retiring_epochs.is_empty()));
        ring.client.send(&Pass { generation: 2, ..on(NEXT, 0, 5, 1) }.encode()).unwrap();
        let got = answer(&mut ring);
        assert_eq!((got.id, got.epoch, got.tokens[0]), (5, NEXT, through_two(1)));
    });
}

#[test]
fn a_refusal_waits_behind_the_older_passes_of_its_slot() {
    within_deadline("a_refusal_waits_behind_the_older_passes_of_its_slot", || {
        let mut ring = ring(2, true, Duration::ZERO, 2000);
        patient_client(&ring);
        // A slow last stage, so passes queue there.
        let slow = Adder { stage: 2, last: true, delay: Duration::from_millis(80), seen: ring.seen[1].clone() };
        ring.nodes[1].set_compute(Box::new(slow));
        stage_next(&ring, NEXT, 2000, &[], None);
        for id in 1..=3 {
            ring.client.send(&on(7, 0, id, 1).encode()).unwrap();
        }
        wait_until("the first pass running at the last stage", || ring.seen[1].load(Ordering::SeqCst) >= 1);
        // The slot goes to route 8 there while passes 2 and 3 still wait; a later one is refused.
        ring.nodes[1].give_slots(NEXT, &[0]).unwrap();
        ring.client.send(&on(7, 0, 4, 1).encode()).unwrap();
        // Answers come by the closing link and refusals walk back, so they may cross on the way.
        // What matters is when the refusal left: sent at once, it would free the slot at the entry
        // while 2 and 3 were still to run on it there, and a new reply would share the slot.
        let mut answered = vec![];
        for _ in 0..4 {
            match landed(&mut ring) {
                Ok(got) => answered.push(got.id),
                Err(nack) => {
                    assert_eq!((nack.id, nack.refusal), (4, Refusal::Epoch));
                    assert_eq!(ring.nodes[1].stats().passes, 3, "refused before the slot's older passes had run");
                }
            }
        }
        assert_eq!(answered, vec![1, 2, 3]);
    });
}

#[test]
fn a_flip_hands_every_slot_over_once_the_old_routes_queue_here_has_run() {
    within_deadline("a_flip_hands_every_slot_over_once_the_old_routes_queue_here_has_run", || {
        let mut ring = ring(2, true, Duration::from_millis(40), 2000);
        patient_client(&ring);
        stage_next(&ring, NEXT, 2000, &[], None);
        assert!(ring.nodes[0].flip(NEXT + 1, &[0], Duration::ZERO).is_err(), "only a staged route can take over");
        for id in 1..=3 {
            ring.client.send(&on(7, 0, id, 1).encode()).unwrap();
        }
        wait_until("route 7's first pass running at the entry", || ring.seen[0].load(Ordering::SeqCst) >= 1);
        // The entry flips first and returns once its own queue of route 7 has run, then the next.
        assert!(ring.nodes[0].flip(NEXT, &[0, 1], Duration::from_secs(2)).unwrap());
        // Counted, not timed: when the first pass was seen running is up to the scheduler.
        assert_eq!(ring.seen[0].load(Ordering::SeqCst), 3, "the flip came back with route 7 still queued");
        // A pass still on its way here when this stage flips would be refused, as its slot is
        // route 8's by then. The orchestrator flips every stage at once and replays what that
        // cuts; here the last stage flips once all three have reached it.
        wait_until("route 7's last pass running at the last stage", || ring.seen[1].load(Ordering::SeqCst) >= 3);
        assert!(ring.nodes[1].flip(NEXT, &[0, 1], Duration::from_secs(2)).unwrap());
        for id in 1..=3 {
            let got = answer(&mut ring);
            assert_eq!((got.id, got.epoch, got.tokens[0]), (id, 7, through_two(1)));
        }
        // From now on route 7 is refused, and route 8 has every slot at once.
        ring.client.send(&on(7, 0, 4, 1).encode()).unwrap();
        assert_eq!(refusal(&mut ring).refusal, Refusal::Epoch);
        for (id, slot) in [(5, 0), (6, 1)] {
            ring.client.send(&Pass { generation: 2, ..on(NEXT, slot, id, 2) }.encode()).unwrap();
            let got = answer(&mut ring);
            assert_eq!((got.id, got.epoch, got.tokens[0]), (id, NEXT, through_two(2)));
        }
        let stats = ring.nodes[0].stats();
        assert_eq!((stats.epoch, stats.staged_epoch), (Some(NEXT), None));
    });
}

#[test]
fn a_stage_that_hangs_after_a_flip_tells_the_old_route_too() {
    within_deadline("a_stage_that_hangs_after_a_flip_tells_the_old_route_too", || {
        let mut ring = ring(2, true, Duration::ZERO, 60_000);
        patient_client(&ring);
        // A slow last stage, so route 7's passes are still queued there after the flip.
        let slow = Adder { stage: 2, last: true, delay: Duration::from_millis(400), seen: ring.seen[1].clone() };
        ring.nodes[1].set_compute(Box::new(slow));
        stage_next(&ring, NEXT, 60_000, &[], None);
        for id in 1..=3 {
            ring.client.send(&on(7, 0, id, 1).encode()).unwrap();
        }
        wait_until("route 7's first pass running at the last stage", || ring.seen[1].load(Ordering::SeqCst) >= 1);
        for node in &ring.nodes {
            node.promote(NEXT).unwrap();
        }
        assert_eq!(ring.nodes[1].stats().retiring_epochs, vec![7], "route 7 retired at once, so this proves nothing");
        // Whichever route's pass hung, the refusal must reach that route's entry.
        let hung = std::mem::replace(&mut ring.nodes[1], Arc::new(Node::new().unwrap()));
        let watchdog = Watchdog::new(hung.refuser(), move || hung.stop());
        watchdog.set_limits(Limits { pass_ms: 200, row_ms: 0 });
        let _hung = watchdog.time(1);
        ring.client.limit_reads(Duration::from_secs(3)).unwrap();
        let mut told = vec![];
        while let Ok((kind, body)) = ring.frames.next() {
            if kind == Kind::Nack {
                let nack = Nack::decode(&body).unwrap();
                if nack.message.contains("hung") {
                    told.push((nack.epoch, nack.refusal, nack.stage));
                }
            }
            if told.iter().any(|&(epoch, _, _)| epoch == 7) {
                break;
            }
        }
        assert!(told.contains(&(7, Refusal::FailedStage, 1)), "the old route's entry heard only {told:?}");
    });
}

#[test]
fn closing_one_epoch_leaves_the_other_serving() {
    within_deadline("closing_one_epoch_leaves_the_other_serving", || {
        let mut ring = ring(2, true, Duration::ZERO, 2000);
        patient_client(&ring);
        stage_next(&ring, NEXT, 2000, &[1], None);
        // A staged route given up: route 7 goes on, and slot 1 is route 7's again.
        for node in &ring.nodes {
            assert!(node.close_epoch(NEXT));
            assert!(!node.close_epoch(NEXT), "closed twice");
        }
        let stats = ring.nodes[0].stats();
        assert_eq!((stats.epoch, stats.staged_epoch), (Some(7), None));
        ring.client.send(&on(7, 1, 1, 3).encode()).unwrap();
        let got = answer(&mut ring);
        assert_eq!((got.id, got.epoch, got.tokens[0]), (1, 7, through_two(3)));
        ring.client.send(&on(NEXT, 1, 2, 3).encode()).unwrap();
        assert_eq!(refusal(&mut ring).message, "this stage is on route 7");
        // The staged ring's keys open nothing any more.
        let door = &ring.doors[1];
        let dial = |key: &[u8], epoch: u64| link::dial(&format!("127.0.0.1:{}", door.port), Some(&door.fingerprint), key, Role::Hop, epoch, Duration::from_secs(1));
        assert!(dial(&next_keys(1, 2).back.unwrap(), NEXT).is_err());
        assert!(dial(&hop_key(0), 7).is_ok());
    });
}

#[test]
fn a_pass_dropped_further_along_does_not_hold_the_old_ring_open() {
    within_deadline("a_pass_dropped_further_along_does_not_hold_the_old_ring_open", || {
        // Limits long enough that the retiring ring's own deadline cannot be what ends it.
        let mut ring = ring(2, true, Duration::ZERO, 3000);
        patient_client(&ring);
        let slow = Adder { stage: 2, last: true, delay: Duration::from_millis(80), seen: ring.seen[1].clone() };
        ring.nodes[1].set_compute(Box::new(slow));
        for id in 1..=3 {
            ring.client.send(&pass(id, 1, id as u32).encode()).unwrap();
        }
        wait_until("the first pass running at the last stage", || ring.seen[1].load(Ordering::SeqCst) >= 1);
        // A new reply on the slot: the last stage drops passes 2 and 3, which are never answered.
        ring.client.send(&Pass { generation: 2, ..pass(4, 1, 0) }.encode()).unwrap();
        while answer(&mut ring).id != 4 {}
        assert!(ring.nodes[1].stats().dropped >= 1, "nothing was dropped, so this proves nothing");
        stage_next(&ring, NEXT, 3000, &[], None);
        for node in &ring.nodes {
            node.promote(NEXT).unwrap();
        }
        let began = Instant::now();
        wait_until("the old route draining", || ring.nodes.iter().all(|node| node.stats().retiring_epochs.is_empty()));
        assert!(began.elapsed() < Duration::from_millis(1500), "the old ring waited out its limits: {:?}", began.elapsed());
    });
}
