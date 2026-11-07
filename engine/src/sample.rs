//! Turning logits into tokens at the last stage.
//!
//! This follows `filtered_distribution` and `speculative_choice` in llamacpp.py. Speculation
//! changes speed and never the distribution: a guess `g` is kept with probability `q(g)`, and
//! a refusal draws from `q` without `g`, so any token `x` comes out with probability `q(x)`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::wire::Sampling;

/// Candidates kept when only top-p is set, since sorting a whole vocabulary costs milliseconds.
const NUCLEUS_CANDIDATES: usize = 4096;
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// How long picking tokens takes, for the hop trace. One engine runs in a process.
static SAMPLE_US: AtomicU64 = AtomicU64::new(0);

pub fn average_us() -> u64 {
    SAMPLE_US.load(Ordering::Relaxed)
}

/// Adds its own lifetime to the average when it ends, whichever way the sampler returns.
struct Timed(Instant);

impl Timed {
    fn start() -> Timed {
        Timed(Instant::now())
    }
}

impl Drop for Timed {
    fn drop(&mut self) {
        crate::average(&SAMPLE_US, self.0.elapsed().as_micros() as u64);
    }
}

/// A small generator. Draws follow from the seed and the position, so a replay repeats them.
pub struct Draws {
    /// Empty until the first draw with no seed, so greedy tokens never ask the system.
    state: Option<u64>,
}

impl Draws {
    pub fn new(seed: Option<u64>, position: u32) -> Draws {
        Draws { state: seed.map(|seed| mix(mix(seed) ^ u64::from(position).wrapping_mul(GOLDEN))) }
    }

    fn next(&mut self) -> u64 {
        let state = self.state.get_or_insert_with(fresh);
        *state = state.wrapping_add(GOLDEN);
        mix(*state)
    }

    /// A number in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn mix(value: u64) -> u64 {
    let mut item = value;
    item = (item ^ (item >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    item = (item ^ (item >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    item ^ (item >> 31)
}

fn fresh() -> u64 {
    use ring::rand::{SecureRandom, SystemRandom};
    let mut bytes = [0; 8];
    // The system's generator failing is no reason to lose a reply: any start will do.
    let _ = SystemRandom::new().fill(&mut bytes);
    u64::from_le_bytes(bytes)
}

/// Candidates past this many are picked by sorting keys rather than by keeping a short list.
const SHORT_LIST: usize = 64;
/// Lanes for finding the highest logit, enough for the compiler to use vector registers.
const LANES: usize = 16;

/// The highest logit. The first of a tie, as `numpy.argmax` picks. NaN never wins, and a
/// row with nothing above minus infinity gives 0.
///
/// The highest value is found first, in lanes, and then its first place. Each lane keeps a
/// value only for one strictly above it, as a plain loop does, so the two agree on every row.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut lanes = [f32::NEG_INFINITY; LANES];
    let mut chunks = logits.chunks_exact(LANES);
    for chunk in &mut chunks {
        for (lane, value) in lanes.iter_mut().zip(chunk) {
            *lane = if *value > *lane { *value } else { *lane };
        }
    }
    let highest = lanes.iter().chain(chunks.remainder()).fold(f32::NEG_INFINITY, |high, value| if *value > high { *value } else { high });
    if highest == f32::NEG_INFINITY {
        return 0;
    }
    // Equal also matches -0.0 against 0.0, and a plain loop keeps whichever of them came first.
    logits.iter().position(|value| *value == highest).unwrap_or(0) as u32
}

/// A logit as a number whose order is `f32::total_cmp`'s. Adding zero makes -0.0 and 0.0
/// the tie they are in Python.
fn order_key(logit: f32) -> u32 {
    let bits = (logit + 0.0).to_bits() as i32;
    ((bits ^ (((bits >> 31) as u32) >> 1) as i32) as u32) ^ 0x8000_0000
}

/// The `limit` best ids, highest logit first and the lower id on a tie. That order is total,
/// so any way of finding them gives the same list.
fn best_ids(logits: &[f32], limit: usize) -> Vec<u32> {
    if limit <= SHORT_LIST && limit < logits.len() {
        return short_list(logits, limit);
    }
    // Highest key first and then the lower id, as one number to sort.
    let pack = |id: usize, key: u32| u64::from(!key) << 32 | id as u64;
    let floor = likely_floor(logits, limit);
    let mut packed: Vec<u64> = logits.iter().enumerate().filter_map(|(id, logit)| Some(order_key(*logit)).filter(|key| *key >= floor).map(|key| pack(id, key))).collect();
    if packed.len() < limit.min(logits.len()) {
        // The guess was too high. Everything is a candidate, as it was before the guess.
        packed = logits.iter().enumerate().map(|(id, logit)| pack(id, order_key(*logit))).collect();
    }
    if limit < packed.len() {
        packed.select_nth_unstable(limit - 1);
        packed.truncate(limit);
    }
    packed.sort_unstable();
    packed.into_iter().map(|item| item as u32).collect()
}

/// A key that about twice `limit` logits reach, guessed from every eighth one. Any key that
/// at least `limit` logits reach is safe to drop the rest below, as the best ones all reach it.
fn likely_floor(logits: &[f32], limit: usize) -> u32 {
    const STRIDE: usize = 8;
    let mut sample: Vec<u32> = logits.iter().step_by(STRIDE).map(|logit| order_key(*logit)).collect();
    let wanted = 2 * limit / STRIDE;
    if wanted == 0 || wanted >= sample.len() {
        return 0;
    }
    *sample.select_nth_unstable_by(wanted, |a, b| b.cmp(a)).1
}

/// A few best ids in one pass. Most logits lose to the shortest of the list at once, and ids
/// come in order, so a tie with it is always a loss.
fn short_list(logits: &[f32], limit: usize) -> Vec<u32> {
    let mut list: Vec<(u32, u32)> = Vec::with_capacity(limit + 1);
    let mut floor = 0;
    // Blocks whose best cannot join are passed over whole, which the compiler can vectorize.
    for (block, chunk) in logits.chunks(LANES).enumerate() {
        if list.len() == limit && chunk.iter().map(|logit| order_key(*logit)).max().unwrap_or(0) <= floor {
            continue;
        }
        for (offset, logit) in chunk.iter().enumerate() {
            let key = order_key(*logit);
            if list.len() == limit && key <= floor {
                continue;
            }
            let at = list.partition_point(|(held, _)| *held >= key);
            list.insert(at, (key, (block * LANES + offset) as u32));
            list.truncate(limit);
            floor = list.last().map_or(0, |(held, _)| *held);
        }
    }
    list.into_iter().map(|(_, id)| id).collect()
}

fn normalize(chances: &mut [f64]) {
    let total: f64 = chances.iter().sum();
    chances.iter_mut().for_each(|chance| *chance /= total);
}

/// Token ids and probabilities after top-k, temperature, and top-p, in that order.
///
/// A draw walks the candidates in order, so this order is part of a seeded reply. Highest
/// logit first and the lower id on a tie, as `filtered_distribution` keeps them, so a reply
/// that leaves the engine carries on as it would have.
pub fn filtered(logits: &[f32], sampling: &Sampling) -> (Vec<u32>, Vec<f64>) {
    let limit = if sampling.top_k > 0 {
        sampling.top_k as usize
    } else if sampling.top_p < 1.0 {
        NUCLEUS_CANDIDATES
    } else {
        0
    };
    let mut ids = if limit > 0 { best_ids(logits, limit) } else { (0..logits.len() as u32).collect() };
    let temperature = f64::from(sampling.temperature);
    let highest = ids.iter().map(|id| f64::from(logits[*id as usize])).fold(f64::NEG_INFINITY, f64::max);
    let mut chances: Vec<f64> = ids.iter().map(|id| ((f64::from(logits[*id as usize]) - highest) / temperature).exp()).collect();
    normalize(&mut chances);
    if sampling.top_p < 1.0 {
        let top_p = f64::from(sampling.top_p);
        let mut before = 0.0;
        let kept = chances.iter().take_while(|chance| {
            let keep = before <= top_p;
            before += **chance;
            keep
        });
        let count = kept.count();
        ids.truncate(count);
        chances.truncate(count);
        normalize(&mut chances);
    }
    (ids, chances)
}

fn draw(ids: &[u32], chances: &[f64], draws: &mut Draws) -> u32 {
    let mut point = draws.unit();
    for (id, chance) in ids.iter().zip(chances) {
        point -= chance;
        if point < 0.0 {
            return *id;
        }
    }
    // Rounding can leave a sliver past the end.
    *ids.last().expect("a distribution is never empty")
}

/// One token from one row of logits.
pub fn choose(logits: &[f32], sampling: &Sampling, draws: &mut Draws) -> u32 {
    let _timed = Timed::start();
    if sampling.temperature == 0.0 {
        return argmax(logits);
    }
    if sampling.top_k == 0 && sampling.top_p >= 1.0 {
        return draw_from_all(logits, f64::from(sampling.temperature), draws);
    }
    let (ids, chances) = filtered(logits, sampling);
    draw(&ids, &chances, draws)
}

/// `filtered` and `draw` with no filter, by the same arithmetic in the same order. Every id
/// is a candidate, so none are listed, and a chance is only divided out once the walk reaches it.
fn draw_from_all(logits: &[f32], temperature: f64, draws: &mut Draws) -> u32 {
    let highest = logits.iter().map(|logit| f64::from(*logit)).fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|logit| ((f64::from(*logit) - highest) / temperature).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut point = draws.unit();
    for (id, weight) in weights.iter().enumerate() {
        point -= weight / total;
        if point < 0.0 {
            return id as u32;
        }
    }
    // Rounding can leave a sliver past the end.
    logits.len().saturating_sub(1) as u32
}

/// What a speculative pass produced. `row(i)` gives logits for row `i`.
///
/// Row 0 follows the last real token, and row `i` follows guess `i - 1`. The result is
/// the guesses that stood, then the model's own next token.
pub fn speculative<'a>(mut row: impl FnMut(usize) -> &'a [f32], guesses: &[u32], sampling: &Sampling, draws: &mut Draws) -> Vec<u32> {
    let _timed = Timed::start();
    let mut accepted = Vec::with_capacity(guesses.len() + 1);
    for index in 0..=guesses.len() {
        let logits = row(index);
        let guess = guesses.get(index).copied();
        if sampling.temperature == 0.0 {
            let choice = argmax(logits);
            accepted.push(choice);
            if Some(choice) != guess {
                return accepted;
            }
            continue;
        }
        let (mut ids, mut chances) = filtered(logits, sampling);
        if let Some(guess) = guess {
            let found = ids.iter().position(|id| *id == guess);
            let chance = found.map_or(0.0, |at| chances[at]);
            if draws.unit() < chance {
                accepted.push(guess);
                continue;
            }
            if let Some(at) = found {
                ids.remove(at);
                chances.remove(at);
            }
            if ids.is_empty() {
                accepted.push(argmax(logits));
                return accepted;
            }
            normalize(&mut chances);
        }
        accepted.push(draw(&ids, &chances, draws));
        return accepted;
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;

    const WARM: Sampling = Sampling { temperature: 1.0, top_p: 1.0, top_k: 0, seed: Some(1) };

    #[test]
    fn the_first_of_a_tie_wins() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn greedy_needs_no_draws_and_ignores_the_seed() {
        let mut draws = Draws::new(Some(9), 0);
        assert_eq!(choose(&[0.0, 5.0, 1.0], &Sampling::GREEDY, &mut draws), 1);
    }

    #[test]
    fn top_k_keeps_the_highest() {
        let (mut ids, _) = filtered(&[1.0, 9.0, 3.0, 8.0], &Sampling { top_k: 2, ..WARM });
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn candidates_come_highest_first_and_ties_by_id() {
        let logits = [2.0, 5.0, 2.0, 7.0, 2.0, 2.0, -0.0, 0.0];
        let (ids, _) = filtered(&logits, &Sampling { top_k: 4, ..WARM });
        assert_eq!(ids, vec![3, 1, 0, 2]);
        let (ids, _) = filtered(&logits, &Sampling { top_k: 8, ..WARM });
        assert_eq!(ids, vec![3, 1, 0, 2, 4, 5, 6, 7]);
    }

    /// The same logits and draws as `test_seeded_top_k_follows_the_engine` in tests/test_gguf.py.
    #[test]
    fn a_seeded_top_k_reply_draws_what_the_python_path_draws() {
        let mut state = 12345u64;
        let logits: Vec<f32> = (0..1000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 40) % 64) as f32 / 8.0
            })
            .collect();
        let got: Vec<u32> = (0..12).map(|position| choose(&logits, &Sampling { top_k: 40, temperature: 0.75, ..WARM }, &mut Draws::new(Some(7), position))).collect();
        assert_eq!(got, vec![575, 613, 168, 912, 575, 815, 668, 290, 899, 216, 815, 575]);
    }

    #[test]
    fn top_p_keeps_the_token_that_crosses_the_line() {
        let logits = [(0.5f32).ln(), (0.3f32).ln(), (0.15f32).ln(), (0.05f32).ln()];
        let (ids, chances) = filtered(&logits, &Sampling { top_p: 0.6, ..WARM });
        assert_eq!(ids, vec![0, 1]);
        assert!((chances.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_seed_repeats_and_a_position_differs() {
        let first: Vec<u64> = (0..4).map(|_| Draws::new(Some(5), 9).next()).collect();
        assert!(first.windows(2).all(|pair| pair[0] == pair[1]));
        assert_ne!(Draws::new(Some(5), 9).next(), Draws::new(Some(5), 10).next());
        // Seed 5 at position 9 must not share a draw with seed 6 at position 8.
        assert_ne!(Draws::new(Some(5), 9).next(), Draws::new(Some(6), 8).next());
    }

    #[test]
    fn greedy_verification_stops_at_the_first_wrong_guess() {
        let rows: [&[f32]; 3] = [&[0.0, 9.0], &[9.0, 0.0], &[0.0, 9.0]];
        let mut draws = Draws::new(None, 0);
        let got = speculative(|row| rows[row], &[1, 1], &Sampling::GREEDY, &mut draws);
        assert_eq!(got, vec![1, 0]);
    }

    #[test]
    fn all_guesses_right_gives_one_more_token() {
        let rows: [&[f32]; 3] = [&[0.0, 9.0], &[9.0, 0.0], &[0.0, 9.0]];
        let mut draws = Draws::new(None, 0);
        let got = speculative(|row| rows[row], &[1, 0], &Sampling::GREEDY, &mut draws);
        assert_eq!(got, vec![1, 0, 1]);
    }

    #[test]
    fn speculation_keeps_the_models_distribution() {
        // q = [0.7, 0.2, 0.1], and the guess is always token 1. Rejection must still give q.
        let logits = [(0.7f32).ln(), (0.2f32).ln(), (0.1f32).ln()];
        let rows: [&[f32]; 2] = [&logits, &logits];
        let mut counts = [0u32; 3];
        let total = 200_000u32;
        for index in 0..total {
            let mut draws = Draws::new(Some(u64::from(index)), 0);
            let got = speculative(|row| rows[row], &[1], &WARM, &mut draws);
            counts[got[0] as usize] += 1;
        }
        for (count, wanted) in counts.iter().zip([0.7, 0.2, 0.1]) {
            let seen = f64::from(*count) / f64::from(total);
            assert!((seen - wanted).abs() < 0.005, "{seen} against {wanted}");
        }
    }

    /// The sampler as it was before it was made faster, which the fast one must match.
    fn plain_argmax(logits: &[f32]) -> u32 {
        let mut best = 0;
        let mut highest = f32::NEG_INFINITY;
        for (index, value) in logits.iter().enumerate() {
            if *value > highest {
                highest = *value;
                best = index;
            }
        }
        best as u32
    }

    fn plain_best_ids(logits: &[f32], limit: usize) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
        let logit = |id: &u32| logits[*id as usize] + 0.0;
        let order = |a: &u32, b: &u32| logit(b).total_cmp(&logit(a)).then(a.cmp(b));
        if limit < logits.len() {
            ids.select_nth_unstable_by(limit - 1, order);
            ids.truncate(limit);
        }
        ids.sort_unstable_by(order);
        ids
    }

    /// Logits full of what breaks samplers: ties, both zeros, both infinities, and NaN
    /// with either sign and several payloads.
    fn nasty_row(state: &mut u64, length: usize) -> Vec<f32> {
        let mut next = || {
            *state = mix(state.wrapping_add(GOLDEN));
            *state
        };
        let specials = [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -f32::NAN, f32::from_bits(0x7F80_0001), f32::from_bits(0xFFC0_1234), 1.5, 1.5, -3.25];
        (0..length)
            .map(|_| {
                let roll = next();
                match roll % 10 {
                    0 => specials[(roll >> 8) as usize % specials.len()],
                    // A few values only, so ties are everywhere.
                    1..=4 => ((roll >> 8) % 7) as f32 - 3.0,
                    _ => ((roll >> 11) as f64 / (1u64 << 53) as f64 * 40.0 - 20.0) as f32,
                }
            })
            .collect()
    }

    #[test]
    fn the_fast_sampler_picks_what_the_plain_one_picks_bit_for_bit() {
        let mut state = 2026u64;
        let mut draws_made = 0u64;
        let mut row_index = 0u64;
        while draws_made < 1_000_000 {
            row_index += 1;
            let length = 1 + (mix(row_index) % 300) as usize;
            let logits = nasty_row(&mut state, length);
            assert_eq!(argmax(&logits), plain_argmax(&logits), "argmax on row {row_index}");
            let limit = 1 + (mix(row_index ^ 7) % (length as u64 + 80)) as usize;
            assert_eq!(best_ids(&logits, limit), plain_best_ids(&logits, limit.min(length)), "row {row_index}, limit {limit}");
            let sampling = Sampling {
                temperature: [0.0, 0.3, 0.7, 1.0, 1.7][(row_index % 5) as usize],
                top_k: [0, 1, 2, 20, 64, 65, 500][(row_index % 7) as usize],
                top_p: [1.0, 0.8, 0.95, 0.0][(row_index % 4) as usize],
                seed: Some(row_index),
            };
            for position in 0..50 {
                let fast = choose(&logits, &sampling, &mut Draws::new(sampling.seed, position));
                let plain = plain_choose(&logits, &sampling, &mut Draws::new(sampling.seed, position));
                assert_eq!(fast, plain, "row {row_index}, position {position}, {sampling:?}");
                draws_made += 1;
            }
        }
    }

    /// `choose` as it was, built from the plain parts.
    fn plain_choose(logits: &[f32], sampling: &Sampling, draws: &mut Draws) -> u32 {
        if sampling.temperature == 0.0 {
            return plain_argmax(logits);
        }
        let limit = if sampling.top_k > 0 {
            sampling.top_k as usize
        } else if sampling.top_p < 1.0 {
            NUCLEUS_CANDIDATES
        } else {
            0
        };
        let fast = filtered(logits, sampling);
        let ids = if limit > 0 { plain_best_ids(logits, limit.max(1)) } else { (0..logits.len() as u32).collect() };
        // The rest of `filtered` is unchanged, so the same ids must give the same chances.
        let (ids, chances) = rest_of_filtered(logits, sampling, ids);
        assert_eq!(fast.0, ids);
        assert_eq!(fast.1.iter().map(|chance| chance.to_bits()).collect::<Vec<_>>(), chances.iter().map(|chance| chance.to_bits()).collect::<Vec<_>>());
        draw(&ids, &chances, draws)
    }

    fn rest_of_filtered(logits: &[f32], sampling: &Sampling, mut ids: Vec<u32>) -> (Vec<u32>, Vec<f64>) {
        let temperature = f64::from(sampling.temperature);
        let highest = ids.iter().map(|id| f64::from(logits[*id as usize])).fold(f64::NEG_INFINITY, f64::max);
        let mut chances: Vec<f64> = ids.iter().map(|id| ((f64::from(logits[*id as usize]) - highest) / temperature).exp()).collect();
        normalize(&mut chances);
        if sampling.top_p < 1.0 {
            let top_p = f64::from(sampling.top_p);
            let mut before = 0.0;
            let count = chances
                .iter()
                .take_while(|chance| {
                    let keep = before <= top_p;
                    before += **chance;
                    keep
                })
                .count();
            ids.truncate(count);
            chances.truncate(count);
            normalize(&mut chances);
        }
        (ids, chances)
    }

    #[test]
    fn the_fast_sampler_matches_on_whole_vocabulary_rows() {
        let mut state = 7u64;
        for round in 0..6u64 {
            let logits = nasty_row(&mut state, 151_936);
            assert_eq!(argmax(&logits), plain_argmax(&logits));
            for limit in [1, 20, 64, 65, 4096, 151_936] {
                assert_eq!(best_ids(&logits, limit), plain_best_ids(&logits, limit), "round {round}, limit {limit}");
            }
        }
        // Rising logits make every one of them join the short list.
        let rising: Vec<f32> = (0..151_936).map(|index| index as f32 / 1000.0).collect();
        assert_eq!(best_ids(&rising, 20), plain_best_ids(&rising, 20));
        assert_eq!(argmax(&rising), 151_935);
        assert_eq!(argmax(&[f32::NAN, f32::NEG_INFINITY]), 0);
        assert_eq!(argmax(&[f32::NAN, -0.0, 0.0, 0.0]), 1);
    }

    #[test]
    fn picking_a_token_is_timed_for_the_trace() {
        let logits: Vec<f32> = (0..200_000).map(|index| (index % 977) as f32).collect();
        choose(&logits, &WARM, &mut Draws::new(Some(3), 0));
        assert!(average_us() > 0);
    }

    /// Times each kind of draw on a row the size of Qwen3's vocabulary. Run with
    /// `cargo test --release -- --ignored --nocapture times_sampling`.
    #[test]
    #[ignore]
    fn times_sampling() {
        let mut state = 99u64;
        let logits: Vec<f32> = (0..151_936)
            .map(|_| {
                state = mix(state.wrapping_add(GOLDEN));
                ((state >> 11) as f64 / (1u64 << 53) as f64 * 30.0 - 15.0) as f32
            })
            .collect();
        let cases = [
            ("greedy", Sampling::GREEDY),
            ("top_k 20, top_p 0.8, T 0.7", Sampling { temperature: 0.7, top_p: 0.8, top_k: 20, seed: Some(1) }),
            ("top_p 0.8 alone, T 0.7", Sampling { temperature: 0.7, top_p: 0.8, top_k: 0, seed: Some(1) }),
            ("temperature 0.7 alone", Sampling { temperature: 0.7, top_p: 1.0, top_k: 0, seed: Some(1) }),
        ];
        // The plain sampler, without the checks `plain_choose` makes, so both are timed alike.
        let plain = |sampling: &Sampling, draws: &mut Draws| {
            if sampling.temperature == 0.0 {
                return plain_argmax(&logits);
            }
            let limit = if sampling.top_k > 0 { sampling.top_k as usize } else { NUCLEUS_CANDIDATES };
            let ids = if sampling.top_k > 0 || sampling.top_p < 1.0 { plain_best_ids(&logits, limit) } else { (0..logits.len() as u32).collect() };
            let (ids, chances) = rest_of_filtered(&logits, sampling, ids);
            draw(&ids, &chances, draws)
        };
        let time = |work: &mut dyn FnMut() -> u32| {
            let began = Instant::now();
            std::hint::black_box(work());
            began.elapsed().as_secs_f64() * 1e6
        };
        for (name, sampling) in cases {
            // Taken in turns, so a busy moment on the machine lands on both.
            let (mut fast, mut slow): (Vec<f64>, Vec<f64>) = (0..200)
                .map(|position| {
                    let fast = time(&mut || choose(std::hint::black_box(&logits), &sampling, &mut Draws::new(Some(5), position)));
                    let slow = time(&mut || plain(&sampling, &mut Draws::new(Some(5), position)));
                    (fast, slow)
                })
                .unzip();
            fast.sort_by(f64::total_cmp);
            slow.sort_by(f64::total_cmp);
            println!("{name}: now median {:.0} us (p10 {:.0}, p90 {:.0}), before {:.0} us (p10 {:.0}, p90 {:.0})", fast[100], fast[20], fast[180], slow[100], slow[20], slow[180]);
        }
    }

    #[test]
    fn sampling_follows_the_distribution() {
        let logits = [(0.25f32).ln(), (0.75f32).ln()];
        let mut ones = 0u32;
        for index in 0..100_000 {
            ones += choose(&logits, &WARM, &mut Draws::new(Some(index), 0));
        }
        assert!((f64::from(ones) / 100_000.0 - 0.75).abs() < 0.006);
    }
}
