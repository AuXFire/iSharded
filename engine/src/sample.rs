//! Turning logits into tokens at the last stage.
//!
//! This follows `filtered_distribution` and `speculative_choice` in llamacpp.py. Speculation
//! changes speed and never the distribution: a guess `g` is kept with probability `q(g)`, and
//! a refusal draws from `q` without `g`, so any token `x` comes out with probability `q(x)`.

use crate::wire::Sampling;

/// Candidates kept when only top-p is set, since sorting a whole vocabulary costs milliseconds.
const NUCLEUS_CANDIDATES: usize = 4096;

/// A small generator. Draws follow from the seed and the position, so a replay repeats them.
pub struct Draws {
    state: u64,
}

impl Draws {
    pub fn new(seed: Option<u64>, position: u32) -> Draws {
        let start = match seed {
            Some(seed) => mix(mix(seed) ^ u64::from(position).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            None => fresh(),
        };
        Draws { state: start }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.state)
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

/// The highest logit. The first of a tie, as `numpy.argmax` picks.
pub fn argmax(logits: &[f32]) -> u32 {
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

/// Token ids and probabilities after top-k, temperature, and top-p, in that order.
pub fn filtered(logits: &[f32], sampling: &Sampling) -> (Vec<u32>, Vec<f64>) {
    let limit = if sampling.top_k > 0 {
        sampling.top_k as usize
    } else if sampling.top_p < 1.0 {
        NUCLEUS_CANDIDATES
    } else {
        0
    };
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    if limit > 0 && limit < logits.len() {
        let order = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
        ids.select_nth_unstable_by(limit - 1, order);
        ids.truncate(limit);
    }
    let temperature = f64::from(sampling.temperature);
    let highest = ids.iter().map(|id| f64::from(logits[*id as usize])).fold(f64::NEG_INFINITY, f64::max);
    let mut chances: Vec<f64> = ids.iter().map(|id| ((f64::from(logits[*id as usize]) - highest) / temperature).exp()).collect();
    let total: f64 = chances.iter().sum();
    chances.iter_mut().for_each(|chance| *chance /= total);
    if sampling.top_p < 1.0 {
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_unstable_by(|a, b| chances[*b].total_cmp(&chances[*a]));
        let top_p = f64::from(sampling.top_p);
        let mut before = 0.0;
        let mut kept_ids = Vec::new();
        let mut kept = Vec::new();
        for index in order {
            if before > top_p {
                break;
            }
            before += chances[index];
            kept_ids.push(ids[index]);
            kept.push(chances[index]);
        }
        let sum: f64 = kept.iter().sum();
        kept.iter_mut().for_each(|chance| *chance /= sum);
        return (kept_ids, kept);
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
    if sampling.temperature == 0.0 {
        return argmax(logits);
    }
    let (ids, chances) = filtered(logits, sampling);
    draw(&ids, &chances, draws)
}

/// What a speculative pass produced. `row(i)` gives logits for row `i`.
///
/// Row 0 follows the last real token, and row `i` follows guess `i - 1`. The result is
/// the guesses that stood, then the model's own next token.
pub fn speculative<'a>(mut row: impl FnMut(usize) -> &'a [f32], guesses: &[u32], sampling: &Sampling, draws: &mut Draws) -> Vec<u32> {
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
            let total: f64 = chances.iter().sum();
            chances.iter_mut().for_each(|chance| *chance /= total);
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
