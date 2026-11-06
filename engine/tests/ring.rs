//! Real nodes over real TLS links on loopback, with a stand-in for the model.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
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

struct Ring {
    nodes: Vec<Node>,
    seen: Vec<Arc<AtomicU64>>,
    client: Arc<Link>,
    frames: Frames,
}

fn ring(count: u16, closing: bool, delay: Duration, ack_timeout_ms: u64) -> Ring {
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
        node.open(Plan { epoch: 7, index: index as u16, count, key: KEY.to_vec(), ack_timeout_ms, silence_timeout_ms: ack_timeout_ms });
    }
    let address = |door: &Listening| Address { address: format!("127.0.0.1:{}", door.port), fingerprint: door.fingerprint.clone() };
    for (index, node) in nodes.iter().enumerate() {
        let next = doors.get(index + 1).map(address);
        let entry = (closing && index + 1 == count as usize && count > 1).then(|| address(&doors[0]));
        node.connect(Links { next, entry, timeout_ms: 3000 }).unwrap();
    }
    let client = link::dial(&format!("127.0.0.1:{}", doors[0].local_port), None, &doors[0].local_key, Role::Local, 0, Duration::from_secs(3)).unwrap();
    let frames = client.reader(1 << 20);
    Ring { nodes, seen, client, frames }
}

fn pass(id: u64, token: u32, position: u32) -> Pass {
    Pass {
        id,
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
    assert_eq!(got, Answer { id: 1, epoch: 7, tokens: vec![5123, 9] });
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
fn losing_the_closing_link_costs_nothing_but_speed() {
    let mut ring = ring(3, true, Duration::ZERO, 2000);
    ring.client.send(&pass(1, 5, 9).encode()).unwrap();
    answer(&mut ring);
    // Dropping the entry's end of the closing link makes the last stage fall back.
    ring.nodes[0].stats();
    ring.client.send(&pass(2, 5, 9).encode()).unwrap();
    assert_eq!(answer(&mut ring).id, 2);
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
