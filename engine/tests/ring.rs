//! Real nodes over real TLS links on loopback, with a stand-in for the model.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use isharded_engine::link::{self, Frames, Identity, Link};
use isharded_engine::watchdog::{Limits, Watchdog};
use isharded_engine::node::{Address, Compute, Failure, Keys, Links, Node, Output, Plan};
use isharded_engine::wire::{Ack, Answer, Kind, Nack, Pass, Refusal, Role, Sampling, FLAG_TOKENS, MAC_BYTES};

const CLOSING: &[u8] = &[9; 32];
const WIDTH: u32 = 4;

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
    nodes: Vec<Node>,
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
        let node = Node::new().unwrap();
        let counter = Arc::new(AtomicU64::new(0));
        node.set_compute(Box::new(Adder { stage: stage + 1, last: stage + 1 == count, delay, seen: counter.clone() }));
        doors.push(listening(&node));
        nodes.push(node);
        seen.push(counter);
    }
    for (index, node) in nodes.iter().enumerate() {
        node.open(Plan { epoch: 7, index: index as u16, count, keys: keys_for(index as u16, count), ack_timeout_ms, silence_timeout_ms }).unwrap();
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
    let frames = client.reader(1 << 20);
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
    let mut ring = ring(3, true, Duration::ZERO, 2000);
    ring.client.send(&pass(1, 5, 9).encode()).unwrap();
    let got = answer(&mut ring);
    // ((5 * 10 + 1) * 10 + 2) * 10 + 3
    assert_eq!(got, sealed(1, vec![5123, 9]));
    assert!(ring.nodes[2].stats().closing_up);
}

#[test]
fn without_a_closing_link_the_answer_walks_back() {
    let mut ring = ring(3, false, Duration::ZERO, 2000);
    ring.client.send(&pass(1, 5, 9).encode()).unwrap();
    assert_eq!(answer(&mut ring).tokens, vec![5123, 9]);
    assert!(!ring.nodes[2].stats().closing_up);
}

#[test]
fn one_stage_answers_by_itself() {
    let mut ring = ring(1, false, Duration::ZERO, 2000);
    ring.client.send(&pass(1, 5, 9).encode()).unwrap();
    assert_eq!(answer(&mut ring).tokens, vec![51, 9]);
}

#[test]
fn many_passes_in_flight_all_arrive_in_order() {
    let mut ring = ring(4, true, Duration::ZERO, 2000);
    for id in 1..=500 {
        ring.client.send(&pass(id, 1, id as u32).encode()).unwrap();
    }
    for id in 1..=500 {
        assert_eq!(answer(&mut ring).id, id);
    }
}

#[test]
fn a_stage_that_refuses_is_named_and_later_passes_still_run() {
    let mut ring = ring(3, true, Duration::ZERO, 2000);
    ring.client.send(&pass(1, 5, 666).encode()).unwrap();
    let (kind, body) = ring.frames.next().unwrap();
    assert_eq!(kind, Kind::Nack);
    let nack = Nack::decode(&body).unwrap();
    assert_eq!((nack.id, nack.refusal, nack.stage), (1, Refusal::StageError, 0));
    ring.client.send(&pass(2, 5, 9).encode()).unwrap();
    assert_eq!(answer(&mut ring).id, 2);
}

#[test]
fn a_pass_from_another_route_is_refused_and_nothing_runs() {
    let mut ring = ring(2, true, Duration::ZERO, 2000);
    let mut stale = pass(1, 5, 9);
    stale.epoch = 6;
    ring.client.send(&stale.encode()).unwrap();
    let (kind, body) = ring.frames.next().unwrap();
    assert_eq!(kind, Kind::Nack);
    assert_eq!(Nack::decode(&body).unwrap().refusal, Refusal::Epoch);
    assert_eq!(ring.seen[0].load(Ordering::SeqCst), 0);
}

#[test]
fn losing_a_stage_is_reported_to_the_entry_naming_it() {
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
}

#[test]
fn a_stage_that_hangs_says_so_before_it_ends() {
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
}

#[test]
fn a_stage_with_no_route_has_nobody_to_tell() {
    Node::new().unwrap().refuser()("nothing was in flight".into());
}

#[test]
fn a_stage_that_goes_silent_is_noticed_without_any_traffic() {
    // The middle stage's links stop answering when its whole process freezes. Stopping its
    // heartbeat answers is what a frozen process looks like from outside.
    let mut ring = ring(2, false, Duration::ZERO, 300);
    ring.nodes[1].close_ring();
    let began = Instant::now();
    let (kind, body) = ring.frames.next().unwrap();
    assert_eq!(kind, Kind::Nack);
    assert_eq!(Nack::decode(&body).unwrap().stage, 1);
    assert!(began.elapsed() < Duration::from_secs(2));
}

#[test]
fn a_peer_that_stops_reading_is_lost_within_the_silence_limit() {
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
}

#[test]
fn a_write_stuck_on_a_long_frame_does_not_hold_up_the_acknowledgement_check() {
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
}

#[test]
fn a_frame_too_big_for_one_tls_buffer_crosses_a_link() {
    let mut ring = ring(3, true, Duration::ZERO, 2000);
    ring.client.send(&rows(1, 1 << 16).encode()).unwrap();
    assert_eq!(answer(&mut ring).tokens, vec![123, 9]);
}

#[test]
fn losing_the_closing_link_costs_nothing_but_speed() {
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
}

#[test]
fn an_answer_on_a_closing_link_that_is_cut_arrives_by_walking_back() {
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
}

#[test]
fn every_answer_is_acknowledged_by_the_way_it_came() {
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
}

#[test]
fn an_answer_whose_acknowledgement_is_lost_reaches_the_worker_once() {
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
}

#[test]
fn a_lost_closing_link_comes_back_at_both_ends() {
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
}

#[test]
fn the_entry_forgets_a_closing_link_that_died_or_went_silent() {
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
}

#[test]
fn redialling_backs_off_and_stops_with_the_ring() {
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
}

/// Waits for a first heartbeat, which is what shows a link was ever worth redialling.
fn let_links_prove_themselves(ring: &Ring, stages: usize) {
    wait_until("the heartbeats", || ring.nodes[..stages].iter().all(|node| node.stats().link_rtt_us > 0));
}

#[test]
fn a_silent_hop_is_redialled_and_what_it_held_goes_again_in_order() {
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
}

#[test]
fn a_hop_silent_one_way_is_redialled_even_when_the_far_end_gives_up_on_it_first() {
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
}

#[test]
fn a_pass_that_arrived_but_whose_acknowledgement_was_lost_runs_once() {
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
}

#[test]
fn a_hop_that_cannot_be_redialled_is_refused_soon_after_it_is_lost() {
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
}

#[test]
fn a_redialled_hop_that_says_nothing_is_not_redialled_again() {
    let ack = 400;
    let (mut ring, relay) = ring_through(1, 2, false, Duration::ZERO, ack);
    ring.client.send(&pass(1, 5, 9).encode()).unwrap();
    answer(&mut ring);
    let_links_prove_themselves(&ring, 1);
    relay.hold();
    wait_until("the redial", || relay.attempts() == 2);
    thread::sleep(Duration::from_millis(100));
    relay.hold();
    wait_for_loss(&mut ring, 1);
    assert_eq!(relay.attempts(), 2);
}

#[test]
fn a_busy_stage_is_not_taken_for_dead() {
    // Each pass takes longer than the timeout, and the acknowledgement must not wait for it.
    let mut ring = ring(2, true, Duration::from_millis(400), 300);
    for id in 1..=3 {
        ring.client.send(&pass(id, 1, 1).encode()).unwrap();
    }
    for id in 1..=3 {
        assert_eq!(answer(&mut ring).id, id);
    }
}

#[test]
fn passes_that_wait_share_a_trip_and_each_still_gets_its_own_answer() {
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
}

#[test]
fn a_merge_reaches_past_a_slots_second_pass_and_refuses_only_its_own_row() {
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
}

#[test]
fn passes_on_one_sequence_never_share_a_trip() {
    let mut ring = ring(2, true, Duration::from_millis(5), 2000);
    for id in 1..=20 {
        ring.client.send(&pass(id, 1, id as u32).encode()).unwrap();
    }
    for id in 1..=20 {
        assert_eq!(answer(&mut ring).tokens[1], id as u32);
    }
    assert_eq!(ring.nodes[0].stats().merged_passes, 0);
}

#[test]
fn a_rewind_clears_what_is_queued_past_it() {
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
}

#[test]
fn a_rewind_leaves_other_replies_and_earlier_places_alone() {
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
}

#[test]
fn a_new_reply_clears_what_an_older_one_left_queued_on_its_slot() {
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
}

#[test]
fn a_pass_from_a_generation_that_has_ended_never_runs() {
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
}

#[test]
fn a_pass_that_arrives_twice_runs_once() {
    let mut ring = ring(2, true, Duration::ZERO, 2000);
    ring.client.send(&pass(1, 5, 9).encode()).unwrap();
    ring.client.send(&pass(1, 5, 9).encode()).unwrap();
    ring.client.send(&pass(2, 5, 10).encode()).unwrap();
    assert_eq!(answer(&mut ring).id, 1);
    assert_eq!(answer(&mut ring).id, 2);
    assert_eq!(ring.seen[1].load(Ordering::SeqCst), 2);
}

#[test]
fn the_entry_drops_a_late_result_from_an_older_generation() {
    let mut ring = ring(2, true, Duration::from_millis(100), 2000);
    ring.client.send(&pass(1, 5, 9).encode()).unwrap();
    ring.client.send(&Pass { generation: 2, ..pass(2, 5, 0) }.encode()).unwrap();
    assert_eq!(answer(&mut ring).id, 2);
    assert!(ring.nodes[0].stats().dropped >= 1);
}

/// Says what the last stage says, from outside it, as a stage that lost track could.
fn closing_voice(ring: &Ring) -> Arc<Link> {
    let door = &ring.doors[0];
    link::dial(&format!("127.0.0.1:{}", door.port), Some(&door.fingerprint), CLOSING, Role::Closing, 7, Duration::from_secs(3)).unwrap()
}

#[test]
fn the_entry_passes_on_an_answer_it_has_already_seen_only_once() {
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
}

#[test]
fn an_answer_from_no_reply_this_entry_knows_is_dropped() {
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
}

#[test]
fn a_stage_can_speak_only_on_the_links_it_is_part_of() {
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
}

#[test]
fn an_answer_a_middle_stage_makes_up_is_dropped_and_its_link_stays_up() {
    // The test stands in for stage one of three, holding only what stage one would hold.
    let entry = Node::new().unwrap();
    entry.set_compute(Box::new(Adder { stage: 1, last: false, delay: Duration::ZERO, seen: Arc::default() }));
    let door = listening(&entry);
    entry.open(Plan { epoch: 7, index: 0, count: 3, keys: keys_for(0, 3), ack_timeout_ms: 5000, silence_timeout_ms: 5000 }).unwrap();
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
}

#[test]
fn stopping_frees_a_nodes_ports_at_once() {
    let mut ring = ring(2, true, Duration::ZERO, 2000);
    ring.client.send(&pass(1, 5, 9).encode()).unwrap();
    answer(&mut ring);
    for (node, door) in ring.nodes.iter().zip(&ring.doors) {
        node.stop();
        for port in [door.port, door.local_port] {
            TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|error| panic!("port {port} is still held: {error}"));
        }
    }
}

#[test]
fn hop_time_is_a_round_trip_that_leaves_out_the_next_stages_work() {
    let delay = Duration::from_millis(300);
    let mut ring = ring(2, true, delay, 2000);
    ring.client.send(&pass(1, 5, 9).encode()).unwrap();
    answer(&mut ring);
    wait_until("a hop time", || ring.nodes[0].stats().hop_us > 0);
    let stats = ring.nodes[0].stats();
    assert!(stats.hop_us < delay.as_micros() as u64, "{}", stats.hop_us);
    assert_eq!(serde_json::to_value(&stats).unwrap()["hop_us"], stats.hop_us);
    assert_eq!(ring.nodes[1].stats().hop_us, 0);
}

#[test]
fn nobody_reaches_the_local_door_without_its_key() {
    let node = Node::new().unwrap();
    let door = listening(&node);
    let got = link::dial(&format!("127.0.0.1:{}", door.local_port), None, b"wrong", Role::Local, 0, Duration::from_secs(3));
    assert!(got.is_err());
}

#[test]
fn the_local_role_is_refused_on_the_public_door() {
    let node = Node::new().unwrap();
    let door = listening(&node);
    let got = link::dial(&format!("127.0.0.1:{}", door.port), Some(&door.fingerprint), &door.local_key, Role::Local, 0, Duration::from_secs(3));
    assert!(got.is_err());
}

#[test]
fn a_route_cannot_be_joined_with_the_wrong_key_or_epoch() {
    let node = Node::new().unwrap();
    let door = listening(&node);
    node.open(Plan { epoch: 7, index: 1, count: 2, keys: keys_for(1, 2), ack_timeout_ms: 1000, silence_timeout_ms: 1000 }).unwrap();
    let address = format!("127.0.0.1:{}", door.port);
    let key = hop_key(0);
    assert!(link::dial(&address, Some(&door.fingerprint), b"wrong", Role::Hop, 7, Duration::from_secs(3)).is_err());
    assert!(link::dial(&address, Some(&door.fingerprint), CLOSING, Role::Hop, 7, Duration::from_secs(3)).is_err());
    assert!(link::dial(&address, Some(&door.fingerprint), &key, Role::Hop, 8, Duration::from_secs(3)).is_err());
    assert!(link::dial(&address, Some(&door.fingerprint), &key, Role::Hop, 7, Duration::from_secs(3)).is_ok());
}
