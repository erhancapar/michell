//! The preconditioned conjugate-gradient method for a symmetric positive definite system A x = b.
//!
//! The matrix and the preconditioner are not stored here: the caller passes two functions, one that
//! computes the product `A p` and one that applies the preconditioner `M^-1 r`. The 3D analysis passes
//! its matrix-free product and the inverse of the diagonal of K (Jacobi); any symmetric positive definite
//! preconditioner fits the same place, such as one multigrid V-cycle.
//!
//! The method as in Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed. (2003), algorithm 9.1
//! (preconditioned conjugate gradient), and Shewchuk, "An Introduction to the Conjugate Gradient Method
//! Without the Agonizing Pain" (1994), section B3. It stops when the relative residual |b - A x| / |b|
//! is at most the tolerance, using the residual the method updates step by step.
//!
//! Every sum (the dot products) runs in index order with one accumulator, so the same input gives
//! bit-for-bit the same result.

use crate::error::{EngineError, ErrorCode};

/// How a solve ended: the number of steps taken and the relative residual |b - A x| / |b| reached.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveReport {
    pub steps: usize,
    pub relative_residual: f64,
}

/// Refuses a tolerance that is not a number between 0 and 1 (both excluded).
pub fn check_tolerance(tolerance: f64) -> Result<(), EngineError> {
    if !(tolerance.is_finite() && tolerance > 0.0 && tolerance < 1.0) {
        return Err(EngineError::invalid_input(
            "the solver tolerance must lie between 0 and 1 (both excluded)",
        )
        .with_number("tolerance", tolerance));
    }
    Ok(())
}

/// Solves A x = b. `x` holds the starting vector on entry and the solution on return.
///
/// `multiply(p, q)` must write `q = A p`; `precondition(r, z)` must write `z = M^-1 r` for a symmetric
/// positive definite M. Both overwrite every entry of their output. More than `max_steps` steps without
/// reaching `tolerance` give the error `solver_not_converged`, with the step count and the residual.
pub fn solve(
    mut multiply: impl FnMut(&[f64], &mut [f64]),
    mut precondition: impl FnMut(&[f64], &mut [f64]),
    b: &[f64],
    x: &mut [f64],
    tolerance: f64,
    max_steps: usize,
) -> Result<SolveReport, EngineError> {
    check_tolerance(tolerance)?;
    let n = b.len();
    if x.len() != n {
        return Err(EngineError::invalid_input(
            "the starting vector must have the length of the right-hand side",
        )
        .with_integer("expected", n)
        .with_integer("got", x.len()));
    }
    let b_norm = inner(b, b).sqrt();
    if b_norm == 0.0 {
        // A positive definite A has only x = 0 for b = 0.
        x.fill(0.0);
        return Ok(SolveReport {
            steps: 0,
            relative_residual: 0.0,
        });
    }

    // r = b - A x
    let mut q = vec![0.0; n];
    multiply(x, &mut q);
    let mut r: Vec<f64> = b.iter().zip(&q).map(|(b_i, q_i)| b_i - q_i).collect();
    let mut relative_residual = inner(&r, &r).sqrt() / b_norm;
    if relative_residual <= tolerance {
        return Ok(SolveReport {
            steps: 0,
            relative_residual,
        });
    }

    let mut z = vec![0.0; n];
    precondition(&r, &mut z);
    let mut p = z.clone();
    let mut rz = inner(&r, &z);

    for step in 1..=max_steps {
        multiply(&p, &mut q);
        let curvature = inner(&p, &q);
        if !(curvature > 0.0 && rz > 0.0) {
            // A x = b has no positive definite A, or M is not positive definite: the method cannot go on.
            return Err(not_converged(
                "the conjugate-gradient method broke down: a search direction has no positive curvature",
                step,
                relative_residual,
                tolerance,
            )
            .with_number("curvature", curvature));
        }
        // Step length alpha = (r . z) / (p . A p); x += alpha p; r -= alpha A p.
        let alpha = rz / curvature;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        relative_residual = inner(&r, &r).sqrt() / b_norm;
        if !relative_residual.is_finite() {
            return Err(not_converged(
                "the conjugate-gradient method produced a residual that is not a finite number",
                step,
                relative_residual,
                tolerance,
            ));
        }
        if relative_residual <= tolerance {
            return Ok(SolveReport {
                steps: step,
                relative_residual,
            });
        }
        // New search direction p = z + beta p, beta = (r_new . z_new) / (r . z).
        precondition(&r, &mut z);
        let rz_new = inner(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    Err(not_converged(
        &format!(
            "the solver did not reach the relative residual {tolerance:e} within {max_steps} steps"
        ),
        max_steps,
        relative_residual,
        tolerance,
    ))
}

/// The `solver_not_converged` error with the numbers of where the method stopped.
fn not_converged(message: &str, steps: usize, residual: f64, tolerance: f64) -> EngineError {
    EngineError::new(ErrorCode::SolverNotConverged, message)
        .with_integer("steps", steps)
        .with_number("relative_residual", residual)
        .with_number("tolerance", tolerance)
}

/// a . b, summed in index order.
fn inner(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (a_i, b_i) in a.iter().zip(b) {
        sum += a_i * b_i;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tridiagonal matrix with 4 on the diagonal and -1 beside it (positive definite).
    fn tridiagonal(p: &[f64], q: &mut [f64]) {
        let n = p.len();
        for i in 0..n {
            let left = if i > 0 { p[i - 1] } else { 0.0 };
            let right = if i + 1 < n { p[i + 1] } else { 0.0 };
            q[i] = 4.0 * p[i] - left - right;
        }
    }

    fn identity(r: &[f64], z: &mut [f64]) {
        z.copy_from_slice(r);
    }

    #[test]
    fn solves_a_small_positive_definite_system() {
        let n = 30;
        let exact: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let mut b = vec![0.0; n];
        tridiagonal(&exact, &mut b);
        let mut x = vec![0.0; n];
        let report = solve(tridiagonal, identity, &b, &mut x, 1e-12, 100).unwrap();
        assert!(report.relative_residual <= 1e-12);
        assert!(report.steps <= n);
        for (x_i, e_i) in x.iter().zip(&exact) {
            assert!((x_i - e_i).abs() < 1e-11);
        }
    }

    #[test]
    fn a_diagonal_preconditioner_solves_a_diagonal_system_in_one_step() {
        let diagonal: Vec<f64> = (1..=10).map(|i| i as f64 * i as f64).collect();
        let multiply = |p: &[f64], q: &mut [f64]| {
            for i in 0..p.len() {
                q[i] = diagonal[i] * p[i];
            }
        };
        let precondition = |r: &[f64], z: &mut [f64]| {
            for i in 0..r.len() {
                z[i] = r[i] / diagonal[i];
            }
        };
        let b = vec![1.0; 10];
        let mut x = vec![0.0; 10];
        let report = solve(multiply, precondition, &b, &mut x, 1e-14, 5).unwrap();
        assert_eq!(report.steps, 1);
        assert!((x[9] - 0.01).abs() < 1e-15);
    }

    #[test]
    fn an_exact_start_takes_no_step_and_a_zero_right_hand_side_gives_zero() {
        let exact = vec![1.0, -2.0, 3.0];
        let mut b = vec![0.0; 3];
        tridiagonal(&exact, &mut b);
        let mut x = exact.clone();
        let report = solve(tridiagonal, identity, &b, &mut x, 1e-10, 10).unwrap();
        assert_eq!(report.steps, 0);
        let mut x = vec![5.0; 3];
        let report = solve(tridiagonal, identity, &[0.0; 3], &mut x, 1e-10, 10).unwrap();
        assert_eq!((report.steps, x), (0, vec![0.0; 3]));
    }

    #[test]
    fn too_few_steps_give_solver_not_converged() {
        let b = vec![1.0; 40];
        let mut x = vec![0.0; 40];
        let error = solve(tridiagonal, identity, &b, &mut x, 1e-12, 3).unwrap_err();
        assert_eq!(error.code, ErrorCode::SolverNotConverged);
        assert_eq!(
            error.details[0],
            ("steps".to_string(), crate::DetailValue::Integer(3))
        );
    }

    #[test]
    fn a_matrix_that_is_not_positive_definite_stops_with_an_error() {
        let negative = |p: &[f64], q: &mut [f64]| {
            for i in 0..p.len() {
                q[i] = -p[i];
            }
        };
        let mut x = vec![0.0; 4];
        let error = solve(negative, identity, &[1.0; 4], &mut x, 1e-10, 10).unwrap_err();
        assert_eq!(error.code, ErrorCode::SolverNotConverged);
    }

    #[test]
    fn wrong_settings_are_refused() {
        let mut x = vec![0.0; 3];
        for tolerance in [0.0, 1.0, f64::NAN] {
            let error = solve(tridiagonal, identity, &[1.0; 3], &mut x, tolerance, 10).unwrap_err();
            assert_eq!(error.code, ErrorCode::InvalidInput);
        }
        let mut short = vec![0.0; 2];
        assert!(solve(tridiagonal, identity, &[1.0; 3], &mut short, 1e-6, 10).is_err());
    }
}
