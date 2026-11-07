//! Real nodes over real TLS links on loopback, with a stand-in for the model.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use isharded_engine::link::{self, Frames, Link};
use isharded_engine::node::{Address, Compute, Failure, Links, Node, Output, Plan};
use isharded_engine::wire::{Answer, Kind, Nack, Pass, Refusal, Role, Sampling, FLAG_TOKENS};

const KEY: &[u8] = b"a route key for testing";
const WIDTH: u32 = 4;

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

/// Stands in the middle of one link, so a test can do to it what a network does.
struct Relay {
    port: u16,
    held: Arc<AtomicBool>,
    sockets: Arc<Mutex<Vec<TcpStream>>>,
}

impl Relay {
    fn to(port: u16) -> Relay {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let relay = Relay { port: listener.local_addr().unwrap().port(), held: Arc::default(), sockets: Arc::default() };
        let (held, sockets) = (relay.held.clone(), relay.sockets.clone());
        thread::spawn(move || {
            let (near, _) = listener.accept().unwrap();
            let far = TcpStream::connect(("127.0.0.1", port)).unwrap();
            sockets.lock().unwrap().extend([near.try_clone().unwrap(), far.try_clone().unwrap()]);
            for (from, to) in [(near.try_clone().unwrap(), far.try_clone().unwrap()), (far, near)] {
                let held = held.clone();
                thread::spawn(move || carry(from, to, &held));
            }
        });
        relay
    }

    /// Nothing is read or passed on any more, as when the far machine freezes.
    fn hold(&self) {
        self.held.store(true, Ordering::SeqCst);
    }

    fn cut(&self) {
        for socket in self.sockets.lock().unwrap().iter() {
            let _ = socket.shutdown(Shutdown::Both);
        }
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.held.store(false, Ordering::SeqCst);
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
        node.open(Plan { epoch: 7, index: index as u16, count, key: KEY.to_vec(), ack_timeout_ms, silence_timeout_ms });
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
    assert_eq!(got, Answer { id: 1, generation: 1, epoch: 7, slot: 0, tokens: vec![5123, 9] });
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
    assert_eq!(got, Answer { id: 2, generation: 1, epoch: 7, slot: 0, tokens: vec![5123, 9] });
    assert_eq!(ring.nodes[2].stats().refusals, 0);
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
    let mut ring = ring(2, true, Duration::from_millis(30), 2000);
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
    assert!(began.elapsed() < Duration::from_millis(250), "the rewind waited behind passes nobody wants");
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
    link::dial(&format!("127.0.0.1:{}", door.port), Some(&door.fingerprint), KEY, Role::Closing, 7, Duration::from_secs(3)).unwrap()
}

#[test]
fn the_entry_passes_on_an_answer_it_has_already_seen_only_once() {
    let mut ring = ring(2, false, Duration::ZERO, 2000);
    ring.client.send(&pass(1, 5, 9).encode()).unwrap();
    let first = answer(&mut ring);
    let voice = closing_voice(&ring);
    voice.send(&first.encode()).unwrap();
    voice.send(&Answer { id: 2, ..first.clone() }.encode()).unwrap();
    voice.send(&Answer { id: 2, ..first.clone() }.encode()).unwrap();
    voice.send(&Answer { id: 3, ..first }.encode()).unwrap();
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
    voice.send(&Answer { id: 2, generation: 99, ..first.clone() }.encode()).unwrap();
    voice.send(&Answer { id: 3, slot: 4, ..first.clone() }.encode()).unwrap();
    voice.send(&Answer { id: 4, ..first }.encode()).unwrap();
    assert_eq!(answer(&mut ring).id, 4);
    assert_eq!(ring.nodes[0].stats().dropped, 2);
    voice.close();
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
    node.open(Plan { epoch: 7, index: 1, count: 2, key: KEY.to_vec(), ack_timeout_ms: 1000, silence_timeout_ms: 1000 });
    let address = format!("127.0.0.1:{}", door.port);
    assert!(link::dial(&address, Some(&door.fingerprint), b"wrong", Role::Hop, 7, Duration::from_secs(3)).is_err());
    assert!(link::dial(&address, Some(&door.fingerprint), KEY, Role::Hop, 8, Duration::from_secs(3)).is_err());
    assert!(link::dial(&address, Some(&door.fingerprint), KEY, Role::Hop, 7, Duration::from_secs(3)).is_ok());
}
