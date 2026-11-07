//! Frames between stages. Everything here reads data from the network, so nothing may panic.
//!
//! A frame is an 8 byte header and a body. Lengths are checked against what is left
//! before anything is allocated, so a peer cannot make a stage reserve memory it did
//! not send.

use std::fmt;

use ring::hmac;

pub const MAGIC: u16 = 0x5369;
/// Two added a generation to passes, answers and refusals. Three added the seal on answers,
/// and padding, which an older node would take for a broken link. Nodes that differ refuse
/// each other.
pub const VERSION: u8 = 3;
pub const HEADER_BYTES: usize = 8;
/// The most a body may hold. Enough for a 512 row pass at a width of 32,768 in FP32.
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
/// The most tokens a pass may name, whether as input or as guesses.
pub const MAX_TOKENS: usize = 4096;
pub const MAX_MESSAGE_BYTES: usize = 1024;

pub const FLAG_VERIFY: u8 = 1;
pub const FLAG_TOKENS: u8 = 2;
pub const FLAG_BF16: u8 = 4;
/// Keep state after this pass, so a later rewind can reach it.
pub const FLAG_KEEP: u8 = 8;
const KNOWN_FLAGS: u8 = FLAG_VERIFY | FLAG_TOKENS | FLAG_BF16 | FLAG_KEEP;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Hello = 1,
    HelloOk = 2,
    Auth = 3,
    Pass = 4,
    Ack = 5,
    Result = 6,
    Nack = 7,
    Ping = 8,
    Pong = 9,
    /// Bytes that only open a link's window. The receiver drops them unread.
    Pad = 10,
}

impl Kind {
    fn from_byte(byte: u8) -> Option<Kind> {
        Some(match byte {
            1 => Kind::Hello,
            2 => Kind::HelloOk,
            3 => Kind::Auth,
            4 => Kind::Pass,
            5 => Kind::Ack,
            6 => Kind::Result,
            7 => Kind::Nack,
            8 => Kind::Ping,
            9 => Kind::Pong,
            10 => Kind::Pad,
            _ => return None,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum WireError {
    BadMagic,
    BadVersion(u8),
    BadKind(u8),
    TooLarge(usize),
    Short,
    Invalid(&'static str),
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WireError::BadMagic => write!(f, "not an iSharded frame"),
            WireError::BadVersion(version) => write!(f, "frame version {version} is not supported"),
            WireError::BadKind(kind) => write!(f, "unknown frame kind {kind}"),
            WireError::TooLarge(size) => write!(f, "a frame of {size} bytes is too large"),
            WireError::Short => write!(f, "a frame ended early"),
            WireError::Invalid(what) => write!(f, "invalid frame: {what}"),
        }
    }
}

impl std::error::Error for WireError {}

/// Reads a header, and says what kind of body follows and how long it is.
pub fn parse_header(header: &[u8; HEADER_BYTES]) -> Result<(Kind, usize), WireError> {
    if u16::from_le_bytes([header[0], header[1]]) != MAGIC {
        return Err(WireError::BadMagic);
    }
    if header[2] != VERSION {
        return Err(WireError::BadVersion(header[2]));
    }
    let kind = Kind::from_byte(header[3]).ok_or(WireError::BadKind(header[3]))?;
    let length = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    if length > MAX_BODY_BYTES {
        return Err(WireError::TooLarge(length));
    }
    Ok((kind, length))
}

fn start_frame(kind: Kind, capacity: usize) -> Vec<u8> {
    let mut frame = Vec::with_capacity(HEADER_BYTES + capacity);
    frame.extend_from_slice(&MAGIC.to_le_bytes());
    frame.push(VERSION);
    frame.push(kind as u8);
    frame.extend_from_slice(&[0; 4]);
    frame
}

fn finish_frame(mut frame: Vec<u8>) -> Vec<u8> {
    let length = (frame.len() - HEADER_BYTES) as u32;
    frame[4..8].copy_from_slice(&length.to_le_bytes());
    frame
}

struct Cursor<'a> {
    data: &'a [u8],
}

impl<'a> Cursor<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], WireError> {
        if count > self.data.len() {
            return Err(WireError::Short);
        }
        let (head, tail) = self.data.split_at(count);
        self.data = tail;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], WireError> {
        let mut value = [0; N];
        value.copy_from_slice(self.take(N)?);
        Ok(value)
    }

    fn byte(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }

    fn word(&mut self) -> Result<u16, WireError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn int(&mut self) -> Result<u32, WireError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn long(&mut self) -> Result<u64, WireError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn float(&mut self) -> Result<f32, WireError> {
        Ok(f32::from_bits(self.int()?))
    }

    fn tokens(&mut self) -> Result<Vec<u32>, WireError> {
        let count = self.int()? as usize;
        if count > MAX_TOKENS {
            return Err(WireError::Invalid("too many tokens"));
        }
        let bytes = self.take(count * 4)?;
        Ok(bytes.chunks_exact(4).map(|item| u32::from_le_bytes([item[0], item[1], item[2], item[3]])).collect())
    }

    fn finished(&self) -> Result<(), WireError> {
        if self.data.is_empty() {
            Ok(())
        } else {
            Err(WireError::Invalid("bytes left after the end"))
        }
    }
}

fn put_tokens(frame: &mut Vec<u8>, tokens: &[u32]) {
    frame.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for token in tokens {
        frame.extend_from_slice(&token.to_le_bytes());
    }
}

/// How the last stage turns logits into tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub seed: Option<u64>,
}

impl Sampling {
    pub const GREEDY: Sampling = Sampling { temperature: 0.0, top_p: 1.0, top_k: 0, seed: None };
}

/// One pass along the route: rows of hidden state, or at the entry, token ids.
#[derive(Debug, Clone, PartialEq)]
pub struct Pass {
    pub id: u64,
    /// Random, and the same for every pass of one reply on a slot, so nothing left over from
    /// an earlier reply on that slot can be taken for this one.
    pub generation: u64,
    pub epoch: u64,
    pub slot: u32,
    pub position: u32,
    /// Forget positions from here on before running, because guesses were wrong.
    pub rollback_to: Option<u32>,
    pub flags: u8,
    pub sampling: Sampling,
    /// What the rows after the first are guessing, for the last stage to check.
    pub guesses: Vec<u32>,
    /// Input ids, when the flag says this pass carries tokens instead of rows.
    pub tokens: Vec<u32>,
    pub rows: u32,
    pub width: u32,
    pub data: Vec<u8>,
}

impl Pass {
    pub fn verifying(&self) -> bool {
        self.flags & FLAG_VERIFY != 0
    }

    pub fn carries_tokens(&self) -> bool {
        self.flags & FLAG_TOKENS != 0
    }

    pub fn is_bf16(&self) -> bool {
        self.flags & FLAG_BF16 != 0
    }

    pub fn keeps_state(&self) -> bool {
        self.flags & FLAG_KEEP != 0
    }

    /// How many rows run through a stage, however they arrived.
    pub fn row_count(&self) -> usize {
        if self.carries_tokens() {
            self.tokens.len()
        } else {
            self.rows as usize
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut frame = start_frame(Kind::Pass, 72 + self.data.len() + 4 * (self.guesses.len() + self.tokens.len()));
        frame.extend_from_slice(&self.id.to_le_bytes());
        frame.extend_from_slice(&self.generation.to_le_bytes());
        frame.extend_from_slice(&self.epoch.to_le_bytes());
        frame.extend_from_slice(&self.slot.to_le_bytes());
        frame.extend_from_slice(&self.position.to_le_bytes());
        frame.extend_from_slice(&self.rollback_to.unwrap_or(u32::MAX).to_le_bytes());
        frame.push(self.flags);
        frame.extend_from_slice(&self.sampling.temperature.to_bits().to_le_bytes());
        frame.extend_from_slice(&self.sampling.top_p.to_bits().to_le_bytes());
        frame.extend_from_slice(&self.sampling.top_k.to_le_bytes());
        frame.push(u8::from(self.sampling.seed.is_some()));
        frame.extend_from_slice(&self.sampling.seed.unwrap_or(0).to_le_bytes());
        put_tokens(&mut frame, &self.guesses);
        put_tokens(&mut frame, &self.tokens);
        frame.extend_from_slice(&self.rows.to_le_bytes());
        frame.extend_from_slice(&self.width.to_le_bytes());
        frame.extend_from_slice(&self.data);
        finish_frame(frame)
    }

    pub fn decode(body: &[u8]) -> Result<Pass, WireError> {
        let mut cursor = Cursor { data: body };
        let id = cursor.long()?;
        let generation = cursor.long()?;
        let epoch = cursor.long()?;
        let slot = cursor.int()?;
        let position = cursor.int()?;
        let rollback = cursor.int()?;
        let flags = cursor.byte()?;
        if flags & !KNOWN_FLAGS != 0 {
            return Err(WireError::Invalid("unknown flags"));
        }
        let temperature = cursor.float()?;
        let top_p = cursor.float()?;
        let top_k = cursor.int()?;
        let seeded = cursor.byte()?;
        let seed = cursor.long()?;
        if !temperature.is_finite() || temperature < 0.0 || !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(WireError::Invalid("sampling is out of range"));
        }
        let guesses = cursor.tokens()?;
        let tokens = cursor.tokens()?;
        let rows = cursor.int()?;
        let width = cursor.int()?;
        let element = if flags & FLAG_BF16 != 0 { 2 } else { 4 };
        let expected = (rows as usize).checked_mul(width as usize).and_then(|cells| cells.checked_mul(element));
        if expected != Some(cursor.data.len()) {
            return Err(WireError::Invalid("rows do not match their size"));
        }
        if flags & FLAG_TOKENS != 0 && (tokens.is_empty() || rows != 0) {
            return Err(WireError::Invalid("a token pass needs tokens and no rows"));
        }
        if flags & FLAG_TOKENS == 0 && (!tokens.is_empty() || rows == 0) {
            return Err(WireError::Invalid("a row pass needs rows and no tokens"));
        }
        let sampling = Sampling { temperature, top_p, top_k, seed: (seeded != 0).then_some(seed) };
        let rollback_to = (rollback != u32::MAX).then_some(rollback);
        Ok(Pass { id, generation, epoch, slot, position, rollback_to, flags, sampling, guesses, tokens, rows, width, data: cursor.data.to_vec() })
    }
}

/// A stage saying it has taken a pass off the link and is working on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ack {
    pub id: u64,
}

impl Ack {
    pub fn encode(&self) -> Vec<u8> {
        beat(Kind::Ack, self.id)
    }

    pub fn decode(body: &[u8]) -> Result<Ack, WireError> {
        Ok(Ack { id: read_beat(body)? })
    }
}

/// What a pass produced: one token, or the tokens a verify pass accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub id: u64,
    pub generation: u64,
    pub epoch: u64,
    pub slot: u32,
    pub tokens: Vec<u32>,
    /// Made by the last stage under a key only it and the entry hold, so a stage in between
    /// cannot make an answer up. Zero where the entry is the last stage too.
    pub seal: [u8; MAC_BYTES],
}

const SEAL_LABEL: &[u8] = b"isharded answer";

impl Answer {
    fn put_fields(&self, frame: &mut Vec<u8>) {
        frame.extend_from_slice(&self.id.to_le_bytes());
        frame.extend_from_slice(&self.generation.to_le_bytes());
        frame.extend_from_slice(&self.epoch.to_le_bytes());
        frame.extend_from_slice(&self.slot.to_le_bytes());
        put_tokens(frame, &self.tokens);
    }

    fn sealed_fields(&self) -> Vec<u8> {
        let mut message = SEAL_LABEL.to_vec();
        self.put_fields(&mut message);
        message
    }

    pub fn sealed(mut self, key: &[u8]) -> Answer {
        let tag = hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, key), &self.sealed_fields());
        self.seal.copy_from_slice(tag.as_ref());
        self
    }

    pub fn is_sealed_by(&self, key: &[u8]) -> bool {
        hmac::verify(&hmac::Key::new(hmac::HMAC_SHA256, key), &self.sealed_fields(), &self.seal).is_ok()
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut frame = start_frame(Kind::Result, 32 + 4 * self.tokens.len() + MAC_BYTES);
        self.put_fields(&mut frame);
        frame.extend_from_slice(&self.seal);
        finish_frame(frame)
    }

    pub fn decode(body: &[u8]) -> Result<Answer, WireError> {
        let mut cursor = Cursor { data: body };
        let id = cursor.long()?;
        let generation = cursor.long()?;
        let epoch = cursor.long()?;
        let slot = cursor.int()?;
        let tokens = cursor.tokens()?;
        let seal = cursor.array()?;
        cursor.finished()?;
        Ok(Answer { id, generation, epoch, slot, tokens, seal })
    }
}

/// Why a pass did not finish. The entry turns these into the errors the HTTP path raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Refusal {
    /// The stage restarted or lost state; replaying the reply is enough.
    Retry = 1,
    /// The route moved on. Nothing failed.
    Epoch = 2,
    /// A stage stopped answering. The index says which.
    FailedStage = 3,
    /// A stage is up and could not process this pass.
    StageError = 4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nack {
    pub id: u64,
    /// Zero with an id of zero: a link failed, and every pass on it is gone.
    pub generation: u64,
    pub epoch: u64,
    pub refusal: Refusal,
    pub stage: u16,
    pub message: String,
}

impl Nack {
    pub fn encode(&self) -> Vec<u8> {
        let text = self.message.as_bytes();
        let cut = floor_char_boundary(&self.message, MAX_MESSAGE_BYTES.min(text.len()));
        let mut frame = start_frame(Kind::Nack, 32 + cut);
        frame.extend_from_slice(&self.id.to_le_bytes());
        frame.extend_from_slice(&self.generation.to_le_bytes());
        frame.extend_from_slice(&self.epoch.to_le_bytes());
        frame.push(self.refusal as u8);
        frame.extend_from_slice(&self.stage.to_le_bytes());
        frame.extend_from_slice(&(cut as u32).to_le_bytes());
        frame.extend_from_slice(&text[..cut]);
        finish_frame(frame)
    }

    pub fn decode(body: &[u8]) -> Result<Nack, WireError> {
        let mut cursor = Cursor { data: body };
        let id = cursor.long()?;
        let generation = cursor.long()?;
        let epoch = cursor.long()?;
        let refusal = match cursor.byte()? {
            1 => Refusal::Retry,
            2 => Refusal::Epoch,
            3 => Refusal::FailedStage,
            4 => Refusal::StageError,
            _ => return Err(WireError::Invalid("unknown refusal")),
        };
        let stage = cursor.word()?;
        let length = cursor.int()? as usize;
        if length > MAX_MESSAGE_BYTES {
            return Err(WireError::Invalid("message too long"));
        }
        let message = String::from_utf8_lossy(cursor.take(length)?).into_owned();
        cursor.finished()?;
        Ok(Nack { id, generation, epoch, refusal, stage, message })
    }
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// A link's heartbeat, and its answer. The stamp comes back unchanged, to time the link.
/// An acknowledgement has the same shape, with the pass's id for a stamp.
pub fn beat(kind: Kind, stamp: u64) -> Vec<u8> {
    let mut frame = start_frame(kind, 8);
    frame.extend_from_slice(&stamp.to_le_bytes());
    finish_frame(frame)
}

pub fn read_beat(body: &[u8]) -> Result<u64, WireError> {
    let mut cursor = Cursor { data: body };
    let stamp = cursor.long()?;
    cursor.finished()?;
    Ok(stamp)
}

/// A frame of `size` zero bytes that carries nothing.
pub fn pad(size: usize) -> Vec<u8> {
    let mut frame = start_frame(Kind::Pad, size);
    frame.resize(HEADER_BYTES + size, 0);
    finish_frame(frame)
}

/// What a link is for, said once when it opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Role {
    /// The previous stage, sending passes forward.
    Hop = 1,
    /// The last stage, sending answers straight to the entry.
    Closing = 2,
    /// The worker on this machine, starting passes at the entry.
    Local = 3,
}

pub const NONCE_BYTES: usize = 32;
pub const MAC_BYTES: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hello {
    pub role: Role,
    pub epoch: u64,
    pub nonce: [u8; NONCE_BYTES],
}

impl Hello {
    pub fn encode(&self) -> Vec<u8> {
        let mut frame = start_frame(Kind::Hello, 9 + NONCE_BYTES);
        frame.push(self.role as u8);
        frame.extend_from_slice(&self.epoch.to_le_bytes());
        frame.extend_from_slice(&self.nonce);
        finish_frame(frame)
    }

    pub fn decode(body: &[u8]) -> Result<Hello, WireError> {
        let mut cursor = Cursor { data: body };
        let role = match cursor.byte()? {
            1 => Role::Hop,
            2 => Role::Closing,
            3 => Role::Local,
            _ => return Err(WireError::Invalid("unknown role")),
        };
        let epoch = cursor.long()?;
        let nonce = cursor.array()?;
        cursor.finished()?;
        Ok(Hello { role, epoch, nonce })
    }
}

/// The accepting side's answer: its nonce, and proof it holds the route's key.
pub fn hello_ok(nonce: &[u8; NONCE_BYTES], mac: &[u8]) -> Vec<u8> {
    let mut frame = start_frame(Kind::HelloOk, NONCE_BYTES + MAC_BYTES);
    frame.extend_from_slice(nonce);
    frame.extend_from_slice(mac);
    finish_frame(frame)
}

pub fn read_hello_ok(body: &[u8]) -> Result<([u8; NONCE_BYTES], [u8; MAC_BYTES]), WireError> {
    if body.len() != NONCE_BYTES + MAC_BYTES {
        return Err(WireError::Invalid("wrong size"));
    }
    let mut cursor = Cursor { data: body };
    Ok((cursor.array()?, cursor.array()?))
}

/// The dialling side's proof that it holds the route's key.
pub fn auth(mac: &[u8]) -> Vec<u8> {
    let mut frame = start_frame(Kind::Auth, MAC_BYTES);
    frame.extend_from_slice(mac);
    finish_frame(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass() -> Pass {
        Pass {
            id: 7,
            generation: 11,
            epoch: 3,
            slot: 1,
            position: 42,
            rollback_to: Some(40),
            flags: FLAG_VERIFY,
            sampling: Sampling { temperature: 0.7, top_p: 0.9, top_k: 40, seed: Some(5) },
            guesses: vec![1, 2],
            tokens: vec![],
            rows: 3,
            width: 2,
            data: vec![9; 24],
        }
    }

    fn body(frame: &[u8]) -> &[u8] {
        &frame[HEADER_BYTES..]
    }

    #[test]
    fn a_pass_survives_the_wire() {
        let original = pass();
        let frame = original.encode();
        let mut header = [0; HEADER_BYTES];
        header.copy_from_slice(&frame[..HEADER_BYTES]);
        assert_eq!(parse_header(&header), Ok((Kind::Pass, frame.len() - HEADER_BYTES)));
        assert_eq!(Pass::decode(body(&frame)).unwrap(), original);
    }

    #[test]
    fn a_token_pass_survives_the_wire() {
        let original = Pass { flags: FLAG_TOKENS, tokens: vec![5, 6], rows: 0, width: 0, data: vec![], rollback_to: None, ..pass() };
        assert_eq!(Pass::decode(body(&original.encode())).unwrap(), original);
    }

    #[test]
    fn the_keep_flag_survives_the_wire_and_the_next_flag_is_still_refused() {
        let original = Pass { flags: FLAG_VERIFY | FLAG_KEEP, ..pass() };
        let decoded = Pass::decode(body(&original.encode())).unwrap();
        assert!(decoded.keeps_state() && !pass().keeps_state());
        assert_eq!(decoded, original);
        let unknown = Pass { flags: 16, ..pass() }.encode();
        assert_eq!(Pass::decode(body(&unknown)), Err(WireError::Invalid("unknown flags")));
    }

    #[test]
    fn every_truncation_of_a_pass_is_refused_without_panicking() {
        let frame = pass().encode();
        for end in 0..body(&frame).len() {
            assert!(Pass::decode(&body(&frame)[..end]).is_err(), "accepted {end} bytes");
        }
    }

    #[test]
    fn a_pass_that_lies_about_its_rows_is_refused() {
        let mut bad = pass();
        bad.rows = 1_000_000;
        assert_eq!(Pass::decode(body(&bad.encode())), Err(WireError::Invalid("rows do not match their size")));
        bad.rows = u32::MAX;
        bad.width = u32::MAX;
        assert!(Pass::decode(body(&bad.encode())).is_err());
    }

    #[test]
    fn a_huge_token_count_is_refused_before_it_is_read() {
        let mut frame = pass().encode();
        // The guess count sits after the fixed fields.
        let at = HEADER_BYTES + 8 + 8 + 8 + 4 + 4 + 4 + 1 + 4 + 4 + 4 + 1 + 8;
        frame[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(Pass::decode(body(&frame)), Err(WireError::Invalid("too many tokens")));
    }

    #[test]
    fn sampling_out_of_range_is_refused() {
        for (temperature, top_p) in [(-1.0, 1.0), (f32::NAN, 1.0), (1.0, 0.0), (1.0, 2.0), (f32::INFINITY, 1.0)] {
            let bad = Pass { sampling: Sampling { temperature, top_p, top_k: 0, seed: None }, ..pass() };
            assert!(Pass::decode(body(&bad.encode())).is_err());
        }
    }

    #[test]
    fn headers_are_checked() {
        let good = Ack { id: 1 }.encode();
        let mut header = [0; HEADER_BYTES];
        header.copy_from_slice(&good[..HEADER_BYTES]);
        let mut bad = header;
        bad[0] = 0;
        assert_eq!(parse_header(&bad), Err(WireError::BadMagic));
        bad = header;
        bad[2] = 99;
        assert_eq!(parse_header(&bad), Err(WireError::BadVersion(99)));
        bad = header;
        bad[3] = 99;
        assert_eq!(parse_header(&bad), Err(WireError::BadKind(99)));
        bad = header;
        bad[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(parse_header(&bad), Err(WireError::TooLarge(_))));
    }

    #[test]
    fn answers_refusals_and_beats_survive_the_wire() {
        let answer = Answer { id: 4, generation: 9, epoch: 2, slot: 5, tokens: vec![1, 2, 3], seal: [0; MAC_BYTES] }.sealed(b"key");
        assert_eq!(Answer::decode(body(&answer.encode())).unwrap(), answer);
        let nack = Nack { id: 4, generation: 9, epoch: 2, refusal: Refusal::FailedStage, stage: 3, message: "gone".into() };
        assert_eq!(Nack::decode(body(&nack.encode())).unwrap(), nack);
        assert_eq!(read_beat(body(&beat(Kind::Ping, 11))).unwrap(), 11);
        assert_eq!(Ack::decode(body(&Ack { id: 8 }.encode())).unwrap().id, 8);
        let hello = Hello { role: Role::Closing, epoch: 9, nonce: [3; NONCE_BYTES] };
        assert_eq!(Hello::decode(body(&hello.encode())).unwrap(), hello);
    }

    #[test]
    fn a_seal_covers_every_field_and_only_its_key_makes_it() {
        let answer = Answer { id: 4, generation: 9, epoch: 2, slot: 5, tokens: vec![1, 2, 3], seal: [0; MAC_BYTES] };
        assert!(!answer.is_sealed_by(b"key"));
        let sealed = answer.clone().sealed(b"key");
        assert!(sealed.is_sealed_by(b"key") && !sealed.is_sealed_by(b"another key"));
        let changed = [
            Answer { id: 5, ..sealed.clone() },
            Answer { generation: 8, ..sealed.clone() },
            Answer { epoch: 3, ..sealed.clone() },
            Answer { slot: 6, ..sealed.clone() },
            Answer { tokens: vec![1, 2, 4], ..sealed.clone() },
        ];
        assert!(changed.iter().all(|item| !item.is_sealed_by(b"key")));
        let decoded = Answer::decode(body(&sealed.encode())).unwrap();
        assert!(decoded.is_sealed_by(b"key"));
        let frame = sealed.encode();
        assert!(Answer::decode(&body(&frame)[..body(&frame).len() - 1]).is_err());
    }

    #[test]
    fn padding_is_a_frame_of_its_own_kind_and_size() {
        let frame = pad(1000);
        let mut header = [0; HEADER_BYTES];
        header.copy_from_slice(&frame[..HEADER_BYTES]);
        assert_eq!(parse_header(&header), Ok((Kind::Pad, 1000)));
        assert_eq!(frame.len(), HEADER_BYTES + 1000);
        assert_eq!(parse_header(&pad(0)[..HEADER_BYTES].try_into().unwrap()), Ok((Kind::Pad, 0)));
    }

    #[test]
    fn a_long_message_is_cut_at_a_character_boundary() {
        let nack = Nack { id: 1, generation: 1, epoch: 1, refusal: Refusal::Retry, stage: 0, message: "é".repeat(MAX_MESSAGE_BYTES) };
        let decoded = Nack::decode(body(&nack.encode())).unwrap();
        assert!(decoded.message.len() <= MAX_MESSAGE_BYTES);
        assert!(decoded.message.chars().all(|item| item == 'é'));
    }
}
