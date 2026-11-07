//! Notices a pass that will never end.
//!
//! A stage that hangs inside llama.cpp keeps its links up and answers its heartbeats, so
//! nothing on the route can tell it from a stage with a long pass. Only this process can see
//! that a pass has run past anything real. It says so, and then the process ends, because the
//! thread that hung cannot be got back. The worker sees a dead stage, which it knows how to
//! replace, and every reply replays.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;

const TICK: Duration = Duration::from_millis(100);
/// The most one tick may count for, however late it wakes.
const LONGEST_TICKS: u32 = 5;
/// How long a refusal gets to leave. It writes to a link, and a stuck link cannot be let
/// keep a hung process alive.
const REFUSAL_PATIENCE: Duration = Duration::from_secs(1);

/// How long a pass may take, as `pass_seconds(rows, 1)` says. These travel in a route's plan.
/// With no limit there is no watchdog, which is how a plan from an older worker reads.
#[derive(Debug, Default, Clone, Copy, Deserialize)]
pub struct Limits {
    #[serde(default)]
    pub pass_ms: u64,
    #[serde(default)]
    pub row_ms: u64,
}

impl Limits {
    fn for_rows(self, rows: u64) -> Option<Duration> {
        (self.pass_ms > 0).then(|| Duration::from_millis(self.pass_ms + rows * self.row_ms))
    }
}

#[derive(Default)]
struct Shared {
    pass_ms: AtomicU64,
    row_ms: AtomicU64,
    /// Goes up with each pass, so a new pass with as many rows is not taken for the old one.
    passes: AtomicU64,
    /// The running pass's rows. Zero when there is none.
    rows: AtomicU64,
    stopping: AtomicBool,
}

impl Shared {
    fn limit(&self, rows: u64) -> Option<Duration> {
        Limits { pass_ms: self.pass_ms.load(Ordering::Relaxed), row_ms: self.row_ms.load(Ordering::Relaxed) }.for_rows(rows)
    }
}

pub struct Watchdog {
    shared: Arc<Shared>,
}

/// A pass being timed. Dropping it says the pass is over.
pub struct Timed<'a> {
    shared: &'a Shared,
}

impl Drop for Timed<'_> {
    fn drop(&mut self) {
        self.shared.rows.store(0, Ordering::Release);
    }
}

impl Watchdog {
    /// `refuse` tells the route what was in flight, and `end` then finishes the process.
    pub fn new(refuse: impl FnOnce(String) + Send + 'static, end: impl FnOnce() + Send + 'static) -> Watchdog {
        let shared = Arc::<Shared>::default();
        let watching = shared.clone();
        let started = thread::Builder::new().name("isharded-watchdog".into()).spawn(move || {
            if let Some(why) = watch(&watching) {
                eprintln!("isharded engine: {why}");
                give_up(why, refuse);
                end();
            }
        });
        // Without this thread a hang is left to the worker's own deadlines, so it is no failure.
        drop(started);
        Watchdog { shared }
    }

    pub fn set_limits(&self, limits: Limits) {
        self.shared.pass_ms.store(limits.pass_ms, Ordering::Relaxed);
        self.shared.row_ms.store(limits.row_ms, Ordering::Relaxed);
    }

    /// Only for whoever holds the gate, so the clock leaves out any wait for it.
    pub fn time(&self, rows: u64) -> Timed<'_> {
        self.shared.passes.fetch_add(1, Ordering::AcqRel);
        self.shared.rows.store(rows.max(1), Ordering::Release);
        Timed { shared: &self.shared }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.shared.stopping.store(true, Ordering::Release);
    }
}

/// Adds up ticks, each by its real length but never more than a few ticks' worth. A process
/// frozen and later woken, as a closed laptop is, has run for none of that time, so its pass
/// has not overrun. A busy machine that wakes this thread late still counts the time that
/// went by: macOS runners slept 100 ms ticks for more than twice that.
fn watch(shared: &Shared) -> Option<String> {
    let mut seen = 0;
    let mut held = Duration::ZERO;
    while !shared.stopping.load(Ordering::Acquire) {
        let began = Instant::now();
        thread::sleep(TICK);
        let slept = began.elapsed().min(TICK * LONGEST_TICKS);
        let rows = shared.rows.load(Ordering::Acquire);
        let pass = shared.passes.load(Ordering::Acquire);
        if rows == 0 || pass != seen {
            seen = pass;
            held = Duration::ZERO;
            continue;
        }
        held += slept;
        if shared.limit(rows).is_some_and(|limit| held > limit) {
            return Some(format!("a pass of {rows} rows has run for {} s, so this stage has hung", held.as_secs()));
        }
    }
    None
}

fn give_up(why: String, refuse: impl FnOnce(String) + Send + 'static) {
    let (done, finished) = mpsc::channel();
    let refusing = thread::Builder::new().name("isharded-refuse".into()).spawn(move || {
        refuse(why);
        let _ = done.send(());
    });
    if refusing.is_ok() {
        let _ = finished.recv_timeout(REFUSAL_PATIENCE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Instant;

    const LIMITS: Limits = Limits { pass_ms: 300, row_ms: 100 };

    /// What a watchdog did, in order.
    #[derive(Clone, Default)]
    struct Seen(Arc<Mutex<Vec<String>>>);

    impl Seen {
        fn watchdog(&self, limits: Limits) -> Watchdog {
            let (refused, ended) = (self.clone(), self.clone());
            let watchdog = Watchdog::new(move |why| refused.0.lock().unwrap().push(why), move || ended.0.lock().unwrap().push("end".into()));
            watchdog.set_limits(limits);
            watchdog
        }

        fn events(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }

        fn wait_for_end(&self, patience: Duration) -> Duration {
            let began = Instant::now();
            while !self.events().iter().any(|item| item == "end") {
                assert!(began.elapsed() < patience, "the watchdog never acted");
                thread::sleep(Duration::from_millis(10));
            }
            began.elapsed()
        }
    }

    #[test]
    fn a_pass_past_its_limit_is_refused_and_then_the_process_ends() {
        let seen = Seen::default();
        let watchdog = seen.watchdog(LIMITS);
        let _timed = watchdog.time(1);
        seen.wait_for_end(Duration::from_secs(5));
        let events = seen.events();
        assert_eq!(events.len(), 2);
        assert!(events[0].contains("1 rows"), "{events:?}");
    }

    #[test]
    fn the_limit_scales_with_the_rows_of_the_pass() {
        let seen = Seen::default();
        let watchdog = seen.watchdog(LIMITS);
        let _timed = watchdog.time(20);
        thread::sleep(Duration::from_millis(1000));
        assert!(seen.events().is_empty(), "20 rows have 2.3 s");
        assert!(seen.wait_for_end(Duration::from_secs(5)) > Duration::from_millis(1000));
    }

    #[test]
    fn passes_that_finish_never_add_up_to_an_overrun() {
        let seen = Seen::default();
        let watchdog = seen.watchdog(LIMITS);
        for _ in 0..12 {
            let _timed = watchdog.time(1);
            thread::sleep(Duration::from_millis(90));
        }
        assert!(seen.events().is_empty());
    }

    #[test]
    fn time_between_passes_does_not_count_against_the_next() {
        let seen = Seen::default();
        let watchdog = seen.watchdog(LIMITS);
        drop(watchdog.time(1));
        thread::sleep(Duration::from_millis(700));
        let _timed = watchdog.time(1);
        thread::sleep(Duration::from_millis(200));
        assert!(seen.events().is_empty());
    }

    #[test]
    fn no_limit_means_no_watchdog() {
        let seen = Seen::default();
        let watchdog = seen.watchdog(Limits::default());
        let _timed = watchdog.time(1);
        thread::sleep(Duration::from_millis(400));
        assert!(seen.events().is_empty());
    }

    #[test]
    fn a_refusal_that_sticks_does_not_keep_the_process_alive() {
        let ended = Seen::default();
        let end = ended.clone();
        let (_hold, stuck) = mpsc::channel::<()>();
        let watchdog = Watchdog::new(
            move |_| {
                let _ = stuck.recv();
            },
            move || end.0.lock().unwrap().push("end".into()),
        );
        watchdog.set_limits(LIMITS);
        let _timed = watchdog.time(1);
        ended.wait_for_end(Duration::from_secs(5));
    }

    #[test]
    fn a_plan_with_no_limits_still_reads() {
        let limits: Limits = serde_json::from_str(r#"{"epoch": 1, "index": 0}"#).unwrap();
        assert_eq!(limits.pass_ms, 0);
        let limits: Limits = serde_json::from_str(r#"{"epoch": 1, "pass_ms": 30000, "row_ms": 1000}"#).unwrap();
        assert_eq!(limits.for_rows(4), Some(Duration::from_millis(34_000)));
    }
}
