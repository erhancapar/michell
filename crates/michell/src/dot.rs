//! The dot product of two vectors, for the optimisation loop (the compliance f . u, the mass, the adjoint
//! shortcut of `michell.adjoint`).
//!
//! Why the engine computes it: NumPy hands `a @ b` to its BLAS library, and the OpenBLAS build that NumPy
//! ships runs a product of more than 10000 entries on every processor core. Its worker threads then keep
//! waiting for work, at full load, between the calls of the loop. Measured on the MBB beam of
//! `examples/mbb_beam.py` (150 x 50 elements, 15402 degrees of freedom, 362 passes, 16 logical cores):
//! 32.1 s of wall time and 504 s of processor time with NumPy's products, 22.9 s and 24.3 s with this
//! function. The engine itself runs on one thread; the 1.3 s of processor time left above the wall time
//! is spent once, by the BLAS threads starting up when NumPy is imported.
//!
//! The sum runs in index order, 0 to n - 1, with one accumulator, so the same input gives bit-for-bit the
//! same result on every call (`docs/design/engine-boundary.md`, "Same input, same output").

use crate::error::EngineError;

/// a . b = sum over i of a_i b_i, summed in index order. The two vectors must have the same length.
pub fn dot(a: &[f64], b: &[f64]) -> Result<f64, EngineError> {
    if a.len() != b.len() {
        return Err(EngineError::invalid_input(format!(
            "the two vectors of a dot product must have the same length; got {} and {}",
            a.len(),
            b.len()
        ))
        .with_integer("length_a", a.len())
        .with_integer("length_b", b.len()));
    }
    let mut sum = 0.0;
    for (a_i, b_i) in a.iter().zip(b) {
        sum += a_i * b_i;
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorCode;

    #[test]
    fn a_small_product_by_hand() {
        // 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, -5.0, 6.0]).unwrap(), 12.0);
    }

    #[test]
    fn empty_vectors_give_zero() {
        assert_eq!(dot(&[], &[]).unwrap(), 0.0);
    }

    #[test]
    fn the_sum_runs_in_index_order() {
        // In index order, 1e16 + 1 rounds back to 1e16, so the sum is 0. Any other order that adds the 1
        // last would give 1: the result shows the order.
        assert_eq!(dot(&[1e16, 1.0, -1e16], &[1.0, 1.0, 1.0]).unwrap(), 0.0);
        assert_eq!(dot(&[1e16, -1e16, 1.0], &[1.0, 1.0, 1.0]).unwrap(), 1.0);
    }

    #[test]
    fn different_lengths_are_refused() {
        let error = dot(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidInput);
        assert_eq!(error.details.len(), 2);
    }
}
