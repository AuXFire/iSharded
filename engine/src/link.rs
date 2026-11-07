//! Links between stages: one long-lived connection each way along the route.
//!
//! Each node makes a fresh key and certificate when it starts, and the coordinator tells
//! the neighbours its fingerprint over the control plane, which is already authenticated.
//! So a link needs no certificate authority, works in every pool mode, and is encrypted
//! even on a network where the control plane is not. Once TLS is up, both ends prove they
//! hold that link's key, so a stranger who reaches the port gets nothing, and a stage of the
//! route gets no further than the links the coordinator gave it keys for.

use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use ring::digest;
use ring::hmac;
use ring::rand::{SecureRandom, SystemRandom};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{ring as provider, verify_tls12_signature, verify_tls13_signature, WebPkiSupportedAlgorithms};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{ClientConfig, ClientConnection, Connection, DigitallySignedStruct, ServerConfig, ServerConnection, SignatureScheme};

use crate::wire::{self, Hello, Kind, Role, HEADER_BYTES, MAC_BYTES, NONCE_BYTES};

/// The most a whole handshake may take. A limit on each read is not enough: a peer that
/// sends a byte now and then would hold its place for ever.
pub const HANDSHAKE: Duration = Duration::from_secs(10);
/// A frame's write must finish in this long, or the link is treated as dead.
const WRITE_SECONDS: u64 = 15;
/// How much may wait for a link's writer. A frame longer than the whole limit still goes,
/// alone. Past these a sender waits, up to the write limit.
const QUEUE_FRAMES: usize = 1024;
const QUEUE_BYTES: usize = 64 << 20;
/// Heartbeats and acknowledgements are tiny. This many waiting means the writer is stuck.
const QUEUE_BEATS: usize = 4096;
/// The most a peer may send before proving it holds the key.
const STRANGER_BYTES: usize = 256;
const SERVER_LABEL: &[u8] = b"isharded link accept";
const CLIENT_LABEL: &[u8] = b"isharded link dial";

#[derive(Debug)]
pub enum LinkError {
    Closed,
    Io(io::Error),
    Tls(rustls::Error),
    Wire(wire::WireError),
    Refused(&'static str),
}

impl std::fmt::Display for LinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkError::Closed => write!(f, "the link closed"),
            LinkError::Io(error) => write!(f, "{error}"),
            LinkError::Tls(error) => write!(f, "TLS: {error}"),
            LinkError::Wire(error) => write!(f, "{error}"),
            LinkError::Refused(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for LinkError {}

impl From<io::Error> for LinkError {
    fn from(error: io::Error) -> LinkError {
        LinkError::Io(error)
    }
}

impl From<rustls::Error> for LinkError {
    fn from(error: rustls::Error) -> LinkError {
        LinkError::Tls(error)
    }
}

impl From<wire::WireError> for LinkError {
    fn from(error: wire::WireError) -> LinkError {
        LinkError::Wire(error)
    }
}

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0; N];
    SystemRandom::new().fill(&mut bytes).expect("the system's random source works");
    bytes
}

fn fingerprint_of(certificate: &CertificateDer<'_>) -> String {
    let hash = digest::digest(&digest::SHA256, certificate.as_ref());
    hash.as_ref().iter().map(|byte| format!("{byte:02x}")).collect()
}

/// This process's key and certificate.
pub struct Identity {
    pub fingerprint: String,
    server: Arc<ServerConfig>,
}

impl Identity {
    pub fn generate() -> Result<Identity, String> {
        let made = rcgen::generate_simple_self_signed(vec!["isharded".to_string()]).map_err(|error| error.to_string())?;
        let certificate: CertificateDer<'static> = made.cert.der().clone();
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(made.key_pair.serialize_der()));
        let fingerprint = fingerprint_of(&certificate);
        let server = ServerConfig::builder_with_provider(Arc::new(provider::default_provider()))
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|error| error.to_string())?
            .with_no_client_auth()
            .with_single_cert(vec![certificate], key)
            .map_err(|error| error.to_string())?;
        Ok(Identity { fingerprint, server: Arc::new(server) })
    }
}

/// Accepts exactly the certificate the coordinator named.
#[derive(Debug)]
struct Pinned {
    fingerprint: String,
    algorithms: WebPkiSupportedAlgorithms,
}

impl ServerCertVerifier for Pinned {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        if fingerprint_of(end_entity) == self.fingerprint {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General("this is not the certificate the coordinator named".into()))
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        certificate: &CertificateDer<'_>,
        signature: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(message, certificate, signature, &self.algorithms)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        certificate: &CertificateDer<'_>,
        signature: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(message, certificate, signature, &self.algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algorithms.supported_schemes()
    }
}

fn client_config(fingerprint: &str) -> Result<Arc<ClientConfig>, LinkError> {
    let crypto = provider::default_provider();
    let algorithms = crypto.signature_verification_algorithms;
    let config = ClientConfig::builder_with_provider(Arc::new(crypto))
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(Pinned { fingerprint: fingerprint.to_string(), algorithms }))
        .with_no_client_auth();
    Ok(Arc::new(config))
}

/// What waits for a link's writer.
#[derive(Default)]
struct Outbox {
    /// These go ahead of frames, between one frame and the next, so a heartbeat or an
    /// acknowledgement never waits behind a queue of long frames.
    beats: VecDeque<Vec<u8>>,
    frames: VecDeque<Vec<u8>>,
    bytes: usize,
    /// When each real frame was queued, oldest first. Padding and beats have none.
    queued: VecDeque<Instant>,
    /// When the frame `take` last gave out was queued, if it was a real one.
    taken: Option<Instant>,
}

impl Outbox {
    fn has_room(&self, length: usize) -> bool {
        self.frames.is_empty() || (self.frames.len() < QUEUE_FRAMES && self.bytes + length <= QUEUE_BYTES)
    }

    fn take(&mut self) -> Option<Vec<u8>> {
        let beat = self.beats.pop_front();
        if beat.is_some() {
            self.taken = None;
            return beat;
        }
        // Padding only warms a link nothing uses yet. A real frame queued behind it warms the
        // link itself, and on a slow link padding would hold it back past its acknowledgement.
        if self.frames.front().is_some_and(|frame| wire::is_pad(frame)) && self.frames.iter().any(|frame| !wire::is_pad(frame)) {
            let mut dropped = 0;
            self.frames.retain(|frame| {
                let pad = wire::is_pad(frame);
                dropped += if pad { frame.len() } else { 0 };
                !pad
            });
            self.bytes -= dropped;
        }
        let frame = self.frames.pop_front()?;
        self.bytes -= frame.len();
        self.taken = if wire::is_pad(&frame) { None } else { self.queued.pop_front() };
        Some(frame)
    }
}

/// One open, authenticated connection. Any thread may send; one thread reads; and one thread
/// writes, so no sender ever waits on the socket. It runs until `close`, which everyone who
/// holds a link must call in the end.
pub struct Link {
    socket: TcpStream,
    /// Never held across a socket write, so a stalled peer cannot stop the reader.
    tls: Option<Mutex<Connection>>,
    outbox: Mutex<Outbox>,
    work: Condvar,
    room: Condvar,
    closed: AtomicBool,
    /// Why the writer stopped, for every sender after it to hear. It stays, so a beat that
    /// hears it first cannot hide it from the watcher that would report the link lost.
    failure: Mutex<Option<io::ErrorKind>>,
    /// The average from a real frame being queued to its last byte reaching the socket.
    write_us: AtomicU64,
}

impl Link {
    fn new(socket: TcpStream, tls: Option<Connection>) -> Arc<Link> {
        Arc::new(Link {
            socket,
            tls: tls.map(Mutex::new),
            outbox: Mutex::default(),
            work: Condvar::new(),
            room: Condvar::new(),
            closed: AtomicBool::new(false),
            failure: Mutex::new(None),
            write_us: AtomicU64::new(0),
        })
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Stops the link and wakes whatever is reading from it.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        let _ = self.socket.shutdown(Shutdown::Both);
        // Taken so that nobody can be between looking and waiting.
        let _outbox = self.outbox.lock();
        self.work.notify_all();
        self.room.notify_all();
    }

    /// Whether a write stalled or failed, as opposed to the far end closing the link.
    pub fn write_us(&self) -> u64 {
        self.write_us.load(Ordering::Relaxed)
    }

    pub fn write_failed(&self) -> bool {
        self.failure.lock().is_some()
    }

    /// Closes the link, and keeps the reason if this is what closed it.
    fn fail(&self, kind: io::ErrorKind) {
        let mut failure = self.failure.lock();
        if !self.is_closed() {
            *failure = Some(kind);
        }
        drop(failure);
        self.close();
    }

    fn closed_error(&self) -> LinkError {
        let failure = *self.failure.lock();
        failure.map_or(LinkError::Closed, |kind| LinkError::Io(kind.into()))
    }

    /// Shortens how long a stalled write may hold on, for a link with a tighter limit.
    pub fn limit_writes(&self, limit: Duration) -> io::Result<()> {
        self.socket.set_write_timeout(Some(limit.min(Duration::from_secs(WRITE_SECONDS))))
    }

    /// Ends the link if nothing at all arrives for this long, for the end that has no watcher.
    pub fn limit_reads(&self, limit: Duration) -> io::Result<()> {
        self.socket.set_read_timeout(Some(limit))
    }

    /// Queues a frame for the writer, in order. A full queue makes this wait, and one that
    /// stays full past the write limit means the link is lost.
    pub fn send(&self, frame: &[u8]) -> Result<(), LinkError> {
        let mut outbox = self.outbox.lock();
        let mut give_up = None;
        while !outbox.has_room(frame.len()) && !self.is_closed() {
            let limit = self.socket.write_timeout()?.unwrap_or(Duration::from_secs(WRITE_SECONDS));
            let deadline = *give_up.get_or_insert_with(|| Instant::now() + limit);
            if self.room.wait_until(&mut outbox, deadline).timed_out() && !outbox.has_room(frame.len()) {
                drop(outbox);
                self.fail(io::ErrorKind::TimedOut);
                return Err(self.closed_error());
            }
        }
        if self.is_closed() {
            drop(outbox);
            return Err(self.closed_error());
        }
        outbox.bytes += frame.len();
        if !wire::is_pad(frame) {
            outbox.queued.push_back(Instant::now());
        }
        outbox.frames.push_back(frame.to_vec());
        self.work.notify_one();
        Ok(())
    }

    /// Queues a heartbeat or an acknowledgement, and never waits, so a reader or a watcher
    /// is always free. A refusal means the writer is stuck or gone.
    pub fn beat(&self, frame: &[u8]) -> Result<(), LinkError> {
        let mut outbox = self.outbox.lock();
        if self.is_closed() {
            drop(outbox);
            return Err(self.closed_error());
        }
        if outbox.beats.len() >= QUEUE_BEATS {
            return Err(LinkError::Refused("too many beats are waiting"));
        }
        outbox.beats.push_back(frame.to_vec());
        self.work.notify_one();
        Ok(())
    }

    fn start_writer(self: &Arc<Self>) -> io::Result<()> {
        let link = self.clone();
        thread::Builder::new().name("isharded-write".into()).spawn(move || link.write_out()).map(|_| ())
    }

    fn write_out(&self) {
        let mut sealed = Vec::new();
        while let Some((frame, queued)) = self.next_frame() {
            match self.write_frame(&frame, &mut sealed) {
                // Part of the frame may have gone, and nothing after it could be read.
                Err(error) => self.fail(error.kind()),
                Ok(()) => queued.into_iter().for_each(|at| crate::average(&self.write_us, at.elapsed().as_micros() as u64)),
            }
        }
    }

    fn next_frame(&self) -> Option<(Vec<u8>, Option<Instant>)> {
        let mut outbox = self.outbox.lock();
        while !self.is_closed() {
            if let Some(frame) = outbox.take() {
                self.room.notify_all();
                return Some((frame, outbox.taken.take()));
            }
            self.work.wait(&mut outbox);
        }
        None
    }

    /// Writes a frame now, before the writer starts. Only a handshake does this.
    fn greet(&self, frame: &[u8]) -> Result<(), LinkError> {
        Ok(self.write_frame(frame, &mut Vec::new())?)
    }

    fn write_frame(&self, frame: &[u8], sealed: &mut Vec<u8>) -> io::Result<()> {
        match &self.tls {
            None => self.write_piece(frame),
            Some(connection) => self.write_sealed(connection, frame, sealed),
        }
    }

    /// Encrypts under the lock and writes outside it. TLS buffers a limited amount, so a
    /// long frame goes in pieces.
    fn write_sealed(&self, connection: &Mutex<Connection>, frame: &[u8], sealed: &mut Vec<u8>) -> io::Result<()> {
        let mut left = frame;
        loop {
            sealed.clear();
            let mut connection = connection.lock();
            let taken = connection.writer().write(left)?;
            left = &left[taken..];
            while connection.wants_write() {
                connection.write_tls(sealed)?;
            }
            drop(connection);
            self.write_piece(sealed)?;
            if left.is_empty() {
                return Ok(());
            }
        }
    }

    /// Writes to the socket inside its write limit. The system applies that limit to one
    /// call, and on Linux a call that moves a few bytes returns and starts it over, so a
    /// stalled peer could otherwise hold a write for several limits.
    fn write_piece(&self, mut bytes: &[u8]) -> io::Result<()> {
        let began = Instant::now();
        while !bytes.is_empty() {
            match (&self.socket).write(bytes) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(count) => bytes = &bytes[count..],
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
            if !bytes.is_empty() && self.socket.write_timeout()?.is_some_and(|limit| began.elapsed() >= limit) {
                return Err(io::ErrorKind::TimedOut.into());
            }
        }
        Ok(())
    }

    pub fn reader(self: &Arc<Self>, limit: usize) -> Frames {
        Frames { link: self.clone(), plain: Vec::with_capacity(1 << 16), last: 0, start: 0, raw: vec![0; 1 << 16].into_boxed_slice(), limit, until: None }
    }
}

/// How long is left before a deadline, or a refusal once it has passed.
fn time_left(until: Instant) -> io::Result<Duration> {
    let left = until.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(io::Error::new(io::ErrorKind::TimedOut, "the handshake took too long"));
    }
    Ok(left)
}

/// A socket whose reads all share one deadline, for the TLS handshake.
struct Hurried<'a> {
    socket: &'a TcpStream,
    until: Instant,
}

impl Read for Hurried<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.socket.set_read_timeout(Some(time_left(self.until)?))?;
        (&mut &*self.socket).read(buffer)
    }
}

impl Write for Hurried<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        time_left(self.until)?;
        (&mut &*self.socket).write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Moves what TLS has decrypted to the end of `plain`.
fn take_plaintext(connection: &mut Connection, plain: &mut Vec<u8>, ready: usize) -> io::Result<()> {
    let end = plain.len();
    plain.resize(end + ready, 0);
    connection.reader().read_exact(&mut plain[end..])
}

/// Reads whole frames from a link.
pub struct Frames {
    link: Arc<Link>,
    plain: Vec<u8>,
    /// Where the frame `next` gave out begins. It ends at `start`.
    last: usize,
    /// Where what has not been read yet begins.
    start: usize,
    raw: Box<[u8]>,
    limit: usize,
    /// While the link is being proven, when that must be done by.
    until: Option<Instant>,
}

impl Frames {
    pub fn set_limit(&mut self, limit: usize) {
        self.limit = limit;
    }

    /// The frame `next` gave out, header and all, for passing on as it came.
    pub fn whole(&self) -> &[u8] {
        &self.plain[self.last..self.start]
    }

    /// Finds the next frame's body in `plain`. The body handed out before this one lived
    /// in there, so this is the first moment the buffer may move.
    fn take_frame(&mut self) -> Result<Option<(Kind, Range<usize>)>, LinkError> {
        if self.start == self.plain.len() {
            self.plain.clear();
            self.start = 0;
        } else if self.start > 1 << 20 {
            self.plain.drain(..self.start);
            self.start = 0;
        }
        self.last = self.start;
        let have = self.plain.len() - self.start;
        if have < HEADER_BYTES {
            return Ok(None);
        }
        let mut header = [0; HEADER_BYTES];
        header.copy_from_slice(&self.plain[self.start..self.start + HEADER_BYTES]);
        let (kind, length) = wire::parse_header(&header)?;
        if length > self.limit {
            return Err(LinkError::Refused("a frame is larger than this link allows"));
        }
        if have < HEADER_BYTES + length {
            return Ok(None);
        }
        let begin = self.start + HEADER_BYTES;
        self.start = begin + length;
        Ok(Some((kind, begin..self.start)))
    }

    /// Moves what TLS already holds into this reader. Data can arrive with the handshake's
    /// last message, and then no socket read would ever announce it.
    fn pull_held(&mut self) -> Result<bool, LinkError> {
        let Some(connection) = &self.link.tls else {
            return Ok(false);
        };
        let mut connection = connection.lock();
        let ready = connection.process_new_packets()?.plaintext_bytes_to_read();
        if ready > 0 {
            take_plaintext(&mut connection, &mut self.plain, ready)?;
        }
        Ok(ready > 0)
    }

    /// Blocks until a whole frame has arrived. The body stays in this reader's buffer, and
    /// is good until the next call.
    pub fn next(&mut self) -> Result<(Kind, &[u8]), LinkError> {
        loop {
            if let Some((kind, body)) = self.take_frame()? {
                return Ok((kind, &self.plain[body]));
            }
            if self.pull_held()? {
                continue;
            }
            if let Some(until) = self.until {
                self.link.socket.set_read_timeout(Some(time_left(until)?))?;
            }
            let count = (&self.link.socket).read(&mut self.raw)?;
            if count == 0 {
                return Err(LinkError::Closed);
            }
            match &self.link.tls {
                None => self.plain.extend_from_slice(&self.raw[..count]),
                Some(connection) => {
                    let mut connection = connection.lock();
                    let mut left = &self.raw[..count];
                    while !left.is_empty() {
                        connection.read_tls(&mut left)?;
                        let state = connection.process_new_packets()?;
                        let ready = state.plaintext_bytes_to_read();
                        if ready > 0 {
                            take_plaintext(&mut connection, &mut self.plain, ready)?;
                        }
                        if state.peer_has_closed() {
                            return Err(LinkError::Closed);
                        }
                    }
                    // Whatever TLS wants to say by itself leaves with the next send, and
                    // heartbeats keep those coming. Written here, it could land inside a
                    // frame going out, and it could stall this reader.
                }
            }
        }
    }
}

fn proof(key: &[u8], label: &[u8], hello: &Hello, nonce: &[u8; NONCE_BYTES], fingerprint: &str) -> [u8; MAC_BYTES] {
    let key = hmac::Key::new(hmac::HMAC_SHA256, key);
    let mut context = hmac::Context::with_key(&key);
    context.update(label);
    context.update(&[hello.role as u8]);
    context.update(&hello.epoch.to_le_bytes());
    context.update(&hello.nonce);
    context.update(nonce);
    // Tying the proof to the certificate stops a relay passing it on over another channel.
    context.update(fingerprint.as_bytes());
    let mut mac = [0; MAC_BYTES];
    mac.copy_from_slice(context.sign().as_ref());
    mac
}

fn same(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len() && left.iter().zip(right).fold(0u8, |sum, (a, b)| sum | (a ^ b)) == 0
}

fn prepare(socket: &TcpStream) -> Result<(), LinkError> {
    socket.set_nodelay(true)?;
    socket.set_write_timeout(Some(Duration::from_secs(WRITE_SECONDS)))?;
    Ok(())
}

fn shake_hands(socket: &TcpStream, until: Instant, connection: impl Into<Connection>) -> Result<Connection, LinkError> {
    let mut connection: Connection = connection.into();
    let mut hurried = Hurried { socket, until };
    while connection.is_handshaking() {
        connection.complete_io(&mut hurried)?;
    }
    Ok(connection)
}

/// Opens a link to another node, proving this end holds `key`.
pub fn dial(address: &str, fingerprint: Option<&str>, key: &[u8], role: Role, epoch: u64, timeout: Duration) -> Result<Arc<Link>, LinkError> {
    let until = Instant::now() + timeout.min(HANDSHAKE);
    let target = address.to_socket_addrs()?.next().ok_or(LinkError::Refused("the address resolved to nothing"))?;
    let socket = TcpStream::connect_timeout(&target, timeout)?;
    prepare(&socket)?;
    let tls = match fingerprint {
        None => None,
        Some(fingerprint) => {
            let name = ServerName::try_from("isharded").map_err(|_| LinkError::Refused("bad server name"))?;
            Some(shake_hands(&socket, until, ClientConnection::new(client_config(fingerprint)?, name)?)?)
        }
    };
    let link = Link::new(socket, tls);
    let hello = Hello { role, epoch, nonce: random_bytes() };
    link.greet(&hello.encode())?;
    let mut frames = link.reader(STRANGER_BYTES);
    frames.until = Some(until);
    let (kind, body) = frames.next()?;
    if kind != Kind::HelloOk {
        return Err(LinkError::Refused("the other node did not greet this one"));
    }
    let (nonce, mac) = wire::read_hello_ok(body)?;
    let pin = fingerprint.unwrap_or("");
    if !same(&mac, &proof(key, SERVER_LABEL, &hello, &nonce, pin)) {
        return Err(LinkError::Refused("the other node does not hold this link's key"));
    }
    link.greet(&wire::auth(&proof(key, CLIENT_LABEL, &hello, &nonce, pin)))?;
    link.socket.set_read_timeout(None)?;
    link.start_writer()?;
    Ok(link)
}

/// Takes a connection someone made to this node, if it proves itself within `patience`.
/// `key_for` says which key a role and epoch need.
pub fn accept(
    socket: TcpStream,
    identity: Option<&Identity>,
    patience: Duration,
    key_for: impl Fn(Role, u64) -> Option<Vec<u8>>,
) -> Result<(Arc<Link>, Frames, Hello), LinkError> {
    let until = Instant::now() + patience;
    prepare(&socket)?;
    let tls = match identity {
        None => None,
        Some(identity) => Some(shake_hands(&socket, until, ServerConnection::new(identity.server.clone())?)?),
    };
    let link = Link::new(socket, tls);
    let mut frames = link.reader(STRANGER_BYTES);
    frames.until = Some(until);
    let (kind, body) = frames.next()?;
    if kind != Kind::Hello {
        return Err(LinkError::Refused("a link must open with a greeting"));
    }
    let hello = Hello::decode(body)?;
    let key = key_for(hello.role, hello.epoch).ok_or(LinkError::Refused("no route here for that greeting"))?;
    let nonce: [u8; NONCE_BYTES] = random_bytes();
    let pin = identity.map_or("", |item| item.fingerprint.as_str());
    link.greet(&wire::hello_ok(&nonce, &proof(&key, SERVER_LABEL, &hello, &nonce, pin)))?;
    let (kind, body) = frames.next()?;
    if kind != Kind::Auth || !same(body, &proof(&key, CLIENT_LABEL, &hello, &nonce, pin)) {
        return Err(LinkError::Refused("the other node does not hold this link's key"));
    }
    link.socket.set_read_timeout(None)?;
    link.start_writer()?;
    frames.set_limit(wire::MAX_BODY_BYTES);
    frames.until = None;
    Ok((link, frames, hello))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::Ack;
    use std::net::TcpListener;
    use std::sync::mpsc;

    const KEY: &[u8] = b"route key";

    fn serve(identity: Arc<Identity>, key: &'static [u8]) -> (String, thread::JoinHandle<Result<u64, LinkError>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let handle = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            let (link, mut frames, hello) = accept(socket, Some(&identity), HANDSHAKE, |_, _| Some(key.to_vec()))?;
            let (kind, body) = frames.next()?;
            assert_eq!(kind, Kind::Ack);
            link.send(&Ack { id: 2 }.encode())?;
            Ok(hello.epoch + Ack::decode(body)?.id)
        });
        (address, handle)
    }

    #[test]
    fn two_nodes_with_the_key_talk_both_ways() {
        let identity = Arc::new(Identity::generate().unwrap());
        let pin = identity.fingerprint.clone();
        let (address, server) = serve(identity, KEY);
        let link = dial(&address, Some(&pin), KEY, Role::Hop, 40, Duration::from_secs(5)).unwrap();
        link.send(&Ack { id: 2 }.encode()).unwrap();
        let mut frames = link.reader(1024);
        let (kind, body) = frames.next().unwrap();
        assert_eq!((kind, Ack::decode(body).unwrap().id), (Kind::Ack, 2));
        assert_eq!(frames.whole(), Ack { id: 2 }.encode());
        assert_eq!(server.join().unwrap().unwrap(), 42);
    }

    #[test]
    fn the_wrong_certificate_is_refused() {
        let identity = Arc::new(Identity::generate().unwrap());
        let (address, _server) = serve(identity, KEY);
        let wrong = "0".repeat(64);
        assert!(matches!(dial(&address, Some(&wrong), KEY, Role::Hop, 1, Duration::from_secs(5)), Err(LinkError::Tls(_) | LinkError::Io(_))));
    }

    #[test]
    fn a_dialler_without_the_key_is_refused_by_both_ends() {
        let identity = Arc::new(Identity::generate().unwrap());
        let pin = identity.fingerprint.clone();
        let (address, server) = serve(identity, KEY);
        let got = dial(&address, Some(&pin), b"another key", Role::Hop, 1, Duration::from_secs(5));
        assert!(matches!(got, Err(LinkError::Refused(_))));
        assert!(server.join().unwrap().is_err());
    }

    #[test]
    fn a_node_on_another_wire_version_is_refused_at_its_greeting() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            accept(socket, None, HANDSHAKE, |_, _| Some(KEY.to_vec())).map(|_| ())
        });
        let mut frame = Hello { role: Role::Hop, epoch: 1, nonce: [0; NONCE_BYTES] }.encode();
        frame[2] = wire::VERSION - 1;
        TcpStream::connect(address).unwrap().write_all(&frame).unwrap();
        assert!(matches!(handle.join().unwrap(), Err(LinkError::Wire(wire::WireError::BadVersion(_)))));
    }

    #[test]
    fn a_stranger_cannot_send_a_large_frame_before_proving_anything() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            accept(socket, None, HANDSHAKE, |_, _| Some(KEY.to_vec())).map(|_| ())
        });
        let mut stranger = TcpStream::connect(address).unwrap();
        let frame = crate::wire::Pass {
            id: 1,
            generation: 1,
            epoch: 1,
            slot: 0,
            position: 0,
            rollback_to: None,
            flags: 0,
            sampling: crate::wire::Sampling::GREEDY,
            guesses: vec![],
            tokens: vec![],
            rows: 1,
            width: 1024,
            data: vec![0; 4096],
        };
        stranger.write_all(&frame.encode()).unwrap();
        assert!(matches!(handle.join().unwrap(), Err(LinkError::Refused(_))));
    }

    /// A peer that sends one byte at a time, never quite slow enough for a limit on each read.
    fn trickle(encrypted: bool, opening: Vec<u8>) {
        let patience = Duration::from_millis(600);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let identity = Identity::generate().unwrap();
        let handle = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            let began = Instant::now();
            let accepted = accept(socket, encrypted.then_some(&identity), patience, |_, _| Some(KEY.to_vec())).map(|_| ());
            (accepted, began.elapsed())
        });
        let mut stranger = TcpStream::connect(address).unwrap();
        for byte in opening.iter().cycle().take(40) {
            if stranger.write_all(&[*byte]).is_err() || handle.is_finished() {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        let (accepted, took) = handle.join().unwrap();
        assert!(accepted.is_err());
        assert!(took < patience + Duration::from_millis(400), "took {took:?}");
    }

    #[test]
    fn a_stranger_who_trickles_bytes_is_refused_when_the_whole_handshake_runs_out() {
        // A TLS record header that promises a long record, and then the slow rest of it.
        trickle(true, vec![22, 3, 1, 0x40, 0]);
        // A greeting's header, where the far end waits for a body that comes a byte at a time.
        trickle(false, Hello { role: Role::Hop, epoch: 1, nonce: [0; NONCE_BYTES] }.encode());
    }

    #[test]
    fn a_stalled_write_leaves_the_reader_free_and_then_ends_the_link() {
        let limit = Duration::from_secs(1);
        let (link, finish, far) = stalled_link(limit);
        let began = Instant::now();
        let writer = {
            let link = link.clone();
            thread::spawn(move || {
                let frame = vec![0; 1 << 20];
                loop {
                    if let Err(error) = link.send(&frame) {
                        return error;
                    }
                }
            })
        };
        thread::sleep(Duration::from_millis(300));
        // The lock is what the reader needs. Hearing the far end through a stalled socket
        // is not something every loopback allows, so this looks at the lock itself.
        assert!(link.tls.as_ref().unwrap().try_lock_for(Duration::from_millis(200)).is_some());
        assert!(!writer.is_finished(), "the write was meant to be stuck during that");
        assert!(matches!(writer.join().unwrap(), LinkError::Io(_)));
        assert!(began.elapsed() < limit * 3);
        assert!(link.is_closed());
        // Every sender after the failure hears why, not only the first.
        assert!(matches!(link.send(b"more"), Err(LinkError::Io(_))));
        drop(finish);
        far.join().unwrap();
    }

    #[test]
    fn beats_go_ahead_of_frames_and_frames_keep_their_order() {
        let mut outbox = Outbox::default();
        for frame in [vec![1; 10], vec![2; 10]] {
            outbox.bytes += frame.len();
            outbox.frames.push_back(frame);
        }
        outbox.beats.push_back(vec![3]);
        assert_eq!(outbox.take(), Some(vec![3]));
        assert_eq!(outbox.take(), Some(vec![1; 10]));
        assert_eq!(outbox.take(), Some(vec![2; 10]));
        assert_eq!((outbox.take(), outbox.bytes), (None, 0));
    }

    #[test]
    fn padding_steps_aside_for_a_real_frame_and_goes_out_on_an_idle_link() {
        let mut outbox = Outbox::default();
        let queue = |outbox: &mut Outbox, frame: Vec<u8>| {
            outbox.bytes += frame.len();
            outbox.frames.push_back(frame);
        };
        queue(&mut outbox, wire::pad(100));
        queue(&mut outbox, wire::pad(100));
        assert_eq!(outbox.take(), Some(wire::pad(100)));
        let ack = Ack { id: 7 }.encode();
        queue(&mut outbox, ack.clone());
        queue(&mut outbox, wire::pad(100));
        assert_eq!(outbox.take(), Some(ack));
        assert_eq!((outbox.take(), outbox.bytes), (None, 0));
        assert!(!wire::is_pad(&[10; 10]));
    }

    #[test]
    fn a_write_failure_is_heard_by_every_sender_after_it() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let socket = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let link = Link::new(socket, None);
        link.fail(io::ErrorKind::TimedOut);
        assert!(matches!(link.beat(&[1]), Err(LinkError::Io(_))));
        assert!(matches!(link.send(&[1]), Err(LinkError::Io(_))));
        assert!(link.write_failed());
    }

    #[test]
    fn a_queue_is_bounded_by_count_and_by_bytes_and_a_long_frame_can_still_go() {
        let mut outbox = Outbox::default();
        assert!(outbox.has_room(QUEUE_BYTES * 2));
        outbox.bytes += QUEUE_BYTES;
        outbox.frames.push_back(vec![]);
        assert!(!outbox.has_room(1));
        outbox.bytes = 0;
        outbox.frames.resize(QUEUE_FRAMES, vec![]);
        assert!(!outbox.has_room(1));
    }

    /// A link to a far end that took it and will never read, and a way to end that.
    fn stalled_link(limit: Duration) -> (Arc<Link>, mpsc::Sender<()>, thread::JoinHandle<()>) {
        let identity = Arc::new(Identity::generate().unwrap());
        let pin = identity.fingerprint.clone();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let (finish, finished) = mpsc::channel::<()>();
        let far = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            let _held = accept(socket, Some(&identity), HANDSHAKE, |_, _| Some(KEY.to_vec())).unwrap();
            let _ = finished.recv();
        });
        let link = dial(&address, Some(&pin), KEY, Role::Hop, 1, Duration::from_secs(5)).unwrap();
        link.limit_writes(limit).unwrap();
        (link, finish, far)
    }

    #[test]
    fn a_queue_that_stays_full_loses_the_link_and_a_beat_never_waits() {
        let limit = Duration::from_secs(1);
        let (link, finish, far) = stalled_link(limit);
        let began = Instant::now();
        let frame = vec![0; 8 << 20];
        let error = loop {
            if let Err(error) = link.send(&frame) {
                break error;
            }
            let queued = Instant::now();
            assert!(link.beat(&Ack { id: 1 }.encode()).is_ok() || link.is_closed());
            assert!(queued.elapsed() < Duration::from_millis(100));
        };
        assert!(matches!(error, LinkError::Io(_)));
        assert!(began.elapsed() < limit * 3);
        assert!(link.is_closed());
        assert!(matches!(link.beat(b"more"), Err(LinkError::Io(_))));
        drop(finish);
        far.join().unwrap();
    }

    #[test]
    fn a_read_limit_ends_a_silent_link() {
        let (link, finish, far) = stalled_link(Duration::from_secs(5));
        link.limit_reads(Duration::from_millis(100)).unwrap();
        let began = Instant::now();
        assert!(link.reader(1024).next().is_err());
        assert!(began.elapsed() < Duration::from_secs(2));
        link.close();
        drop(finish);
        far.join().unwrap();
    }

    #[test]
    fn frames_arrive_whole_in_order_from_several_senders() {
        let identity = Arc::new(Identity::generate().unwrap());
        let pin = identity.fingerprint.clone();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let server = thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            let (_link, mut frames, _) = accept(socket, Some(&identity), HANDSHAKE, |_, _| Some(KEY.to_vec())).unwrap();
            let mut seen = vec![];
            for _ in 0..4000 {
                let (_, body) = frames.next().unwrap();
                seen.push(Ack::decode(body).unwrap().id);
            }
            seen
        });
        let link = dial(&address, Some(&pin), KEY, Role::Hop, 1, Duration::from_secs(5)).unwrap();
        let senders: Vec<_> = (0..4u64)
            .map(|sender| {
                let link = link.clone();
                thread::spawn(move || {
                    for index in 0..1000 {
                        link.send(&Ack { id: sender * 1000 + index }.encode()).unwrap();
                    }
                })
            })
            .collect();
        senders.into_iter().for_each(|item| item.join().unwrap());
        let mut seen = server.join().unwrap();
        for sender in 0..4 {
            let mine: Vec<_> = seen.iter().filter(|id| **id / 1000 == sender).collect();
            assert!(mine.windows(2).all(|pair| pair[0] < pair[1]));
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..4000).collect::<Vec<_>>());
    }
}
