//! The threads of the 3D analysis: one pool of worker threads per thread count, and the two operations that
//! share work out over a pool, a sum and a write, built so that the result does not depend on how many
//! threads there are.
//!
//! The pools come from the `rayon` crate. A pool is started the first time a thread count is asked for and
//! then kept for the life of the process, so every object that asks for 4 threads shares the same 4 threads.
//! The thread count is always an argument: nothing here reads the environment or counts the processor's
//! cores, and a count above the number of cores is allowed (the threads then take turns). One thread needs
//! no pool: the calling thread does the work.
//!
//! Why the result does not depend on the thread count:
//! - **Writes.** The output is cut into pieces (the three entries of one node, the value of one element).
//!   Each piece is computed from the inputs alone, with the same arithmetic whichever thread runs it, and no
//!   two threads write the same piece.
//! - **Sums.** Floating-point addition rounds, so a sum depends on the order of its terms. Every sum here
//!   runs over blocks of `BLOCK_LENGTH` entries: each block is summed in index order, and the block sums are
//!   added in block order. Which thread sums which block, and when it finishes, cannot change the result.

use std::sync::{Arc, Mutex};

use rayon::ThreadPool;
use rayon::prelude::*;

use crate::error::{EngineError, ErrorCode};

/// The length, in entries, of the blocks every sum runs over.
///
/// It is a fixed number and must never be derived from the thread count: the blocks decide the order in
/// which the terms of a sum are added, and so its rounding; blocks that changed with the number of threads
/// would make the result change with it. The same length is also the least work one task takes and the
/// size below which work stays on the calling thread; those two uses change only the speed.
pub const BLOCK_LENGTH: usize = 1024;

/// The pools started so far, with their thread counts.
static POOLS: Mutex<Vec<(usize, Arc<ThreadPool>)>> = Mutex::new(Vec::new());

/// A handle on the threads of one thread count.
#[derive(Clone, Debug)]
pub struct Threads {
    count: usize,
    /// The shared pool; `None` for one thread, where the calling thread does all the work.
    pool: Option<Arc<ThreadPool>>,
}

impl Threads {
    /// The threads for `count` threads, starting the pool if no object has asked for this count before.
    /// A count of 0 is refused.
    pub fn new(count: usize) -> Result<Threads, EngineError> {
        if count == 0 {
            return Err(
                EngineError::invalid_input("the number of threads must be at least 1")
                    .with_integer("threads", count),
            );
        }
        if count == 1 {
            return Ok(Threads { count, pool: None });
        }
        // A poisoned lock only means another thread panicked while holding it; the list is still usable.
        let mut pools = POOLS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, pool)) = pools.iter().find(|(c, _)| *c == count) {
            return Ok(Threads {
                count,
                pool: Some(Arc::clone(pool)),
            });
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(count)
            .thread_name(|index| format!("michell-{index}"))
            .build()
            .map_err(|error| {
                EngineError::new(
                    ErrorCode::InternalError,
                    format!("could not start {count} threads: {error}"),
                )
                .with_integer("threads", count)
            })?;
        let pool = Arc::new(pool);
        pools.push((count, Arc::clone(&pool)));
        Ok(Threads {
            count,
            pool: Some(pool),
        })
    }

    /// The number of threads.
    pub fn count(&self) -> usize {
        self.count
    }

    /// a . b, summed block by block: each block of `BLOCK_LENGTH` entries in index order, then the block
    /// sums in block order. The two vectors must have the same length.
    pub fn inner(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(
            a.len(),
            b.len(),
            "inner product of vectors of different length"
        );
        let block_sums: Vec<f64> = match self.pool_for(a.len()) {
            None => a
                .chunks(BLOCK_LENGTH)
                .zip(b.chunks(BLOCK_LENGTH))
                .map(|(a, b)| block_inner(a, b))
                .collect(),
            // `collect` keeps the blocks in their order, whichever thread summed them.
            Some(pool) => pool.install(|| {
                a.par_chunks(BLOCK_LENGTH)
                    .zip(b.par_chunks(BLOCK_LENGTH))
                    .map(|(a, b)| block_inner(a, b))
                    .collect()
            }),
        };
        let mut sum = 0.0;
        for block_sum in block_sums {
            sum += block_sum;
        }
        sum
    }

    /// Cuts `out` into pieces of `width` entries (3 for the displacements of one node, 1 for the value of
    /// one element) and calls `work(piece, values)` for each, where `piece` counts the pieces from 0 and
    /// `values` is that piece of `out`. Each piece is written by exactly one thread.
    pub fn for_each_piece<F>(&self, out: &mut [f64], width: usize, work: F)
    where
        F: Fn(usize, &mut [f64]) + Send + Sync,
    {
        match self.pool_for(out.len()) {
            None => {
                for (piece, values) in out.chunks_mut(width).enumerate() {
                    work(piece, values);
                }
            }
            Some(pool) => {
                let pieces_per_task = (BLOCK_LENGTH / width).max(1);
                pool.install(|| {
                    out.par_chunks_mut(width)
                        .enumerate()
                        .with_min_len(pieces_per_task)
                        .for_each(|(piece, values)| work(piece, values));
                });
            }
        }
    }

    /// Calls `work(i, &mut out[i])` for every entry of `out`: a vector update such as x += alpha p.
    pub fn update<F>(&self, out: &mut [f64], work: F)
    where
        F: Fn(usize, &mut f64) + Send + Sync,
    {
        self.for_each_piece(out, 1, |i, value| work(i, &mut value[0]));
    }

    /// The pool to share out work on `length` entries, or `None` when the calling thread does it alone: with
    /// one thread, or when the work fits in one block and sharing it out would cost more than it saves.
    fn pool_for(&self, length: usize) -> Option<&ThreadPool> {
        if length <= BLOCK_LENGTH {
            return None;
        }
        self.pool.as_deref()
    }
}

/// a . b for one block, in index order.
fn block_inner(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (a_i, b_i) in a.iter().zip(b) {
        sum += a_i * b_i;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values whose sum depends on the order of addition: magnitudes from 1e-8 to 1e8, both signs.
    fn order_sensitive(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let x = i as f64;
                (1.3 * x).sin() * 10f64.powi((i % 17) as i32 - 8)
            })
            .collect()
    }

    /// The block rule written out by hand: blocks of `block` entries, each in index order, then in order.
    fn blocked_sum(a: &[f64], b: &[f64], block: usize) -> f64 {
        let mut total = 0.0;
        let mut start = 0;
        while start < a.len() {
            let end = (start + block).min(a.len());
            let mut block_sum = 0.0;
            for i in start..end {
                block_sum += a[i] * b[i];
            }
            total += block_sum;
            start = end;
        }
        total
    }

    #[test]
    fn zero_threads_are_refused() {
        let error = Threads::new(0).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidInput);
    }

    #[test]
    fn one_pool_per_thread_count_and_more_threads_than_cores_work() {
        let first = Threads::new(3).unwrap();
        let second = Threads::new(3).unwrap();
        assert!(Arc::ptr_eq(
            first.pool.as_ref().unwrap(),
            second.pool.as_ref().unwrap()
        ));
        assert!(Threads::new(1).unwrap().pool.is_none());
        let many = Threads::new(64).unwrap();
        assert_eq!(many.count(), 64);
        let a = vec![1.0; 10 * BLOCK_LENGTH];
        assert_eq!(many.inner(&a, &a), (10 * BLOCK_LENGTH) as f64);
    }

    #[test]
    fn the_sum_follows_the_block_rule_on_every_thread_count() {
        // 13 full blocks and a short one, so that every thread count below gets several blocks.
        let n = 13 * BLOCK_LENGTH + 357;
        let a = order_sensitive(n);
        let b: Vec<f64> = order_sensitive(n + 5)[5..].to_vec();
        let expected = blocked_sum(&a, &b, BLOCK_LENGTH);
        // The data are sensitive to the order: one running sum, or other blocks, round differently.
        assert_ne!(expected.to_bits(), blocked_sum(&a, &b, n).to_bits());
        assert_ne!(
            expected.to_bits(),
            blocked_sum(&a, &b, BLOCK_LENGTH / 2).to_bits()
        );
        for count in [1, 2, 5, 12] {
            let threads = Threads::new(count).unwrap();
            assert_eq!(
                threads.inner(&a, &b).to_bits(),
                expected.to_bits(),
                "{count}"
            );
        }
        // A vector that fits in one block is one running sum.
        let short = &a[..BLOCK_LENGTH];
        assert_eq!(
            Threads::new(5).unwrap().inner(short, short).to_bits(),
            blocked_sum(short, short, BLOCK_LENGTH).to_bits()
        );
        assert_eq!(Threads::new(2).unwrap().inner(&[], &[]), 0.0);
    }

    #[test]
    fn every_piece_is_written_once_with_its_own_index() {
        for count in [1, 2, 5, 12] {
            let threads = Threads::new(count).unwrap();
            let mut out = vec![0.0; 3 * 5000];
            threads.for_each_piece(&mut out, 3, |piece, values| {
                for (c, value) in values.iter_mut().enumerate() {
                    *value += (3 * piece + c) as f64;
                }
            });
            assert!(out.iter().enumerate().all(|(i, v)| *v == i as f64));
            let mut x = vec![1.0; 3000];
            threads.update(&mut x, |i, x_i| *x_i += i as f64);
            assert!(x.iter().enumerate().all(|(i, v)| *v == i as f64 + 1.0));
        }
    }
}
