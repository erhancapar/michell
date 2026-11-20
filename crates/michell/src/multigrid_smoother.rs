//! The smoother of the multigrid: damped point Jacobi, x <- x + omega D^-1 (r - A x), with the damping
//! omega tied to a guaranteed bound.
//!
//! Damped Jacobi reduces the error in the energy norm of a positive (semi-)definite A whenever
//! 0 < omega < 2 / lambda_max(D^-1 A); above 2 / lambda_max it makes some error components grow, the
//! V-cycle is then no longer a positive definite preconditioner, and conjugate gradients can break down.
//! A damping tuned on a uniform design can cross that line on an irregular one, so omega is set from an
//! upper bound L of lambda_max(D^-1 A) on that level for that density: omega = c * 2 / L with c < 1.
//!
//! The bound. With A = sum of the element matrices A_e (each positive semi-definite, extended by zeros)
//! and D_e the diagonal of A_e, the diagonal of A is D = sum of D_e. If x_e^T A_e x_e <= L_e x_e^T D_e x_e
//! for every element and every x, then
//!
//!   x^T A x = sum_e x_e^T A_e x_e <= sum_e L_e x_e^T D_e x_e <= (max_e L_e) x^T D x,
//!
//! so lambda_max(D^-1 A) <= max_e L_e. L_e is an upper bound of the largest eigenvalue of
//! D_e^-1/2 A_e D_e^-1/2 taken on the degrees of freedom of the element whose diagonal is positive (a row
//! of a positive semi-definite matrix with a zero diagonal is zero, so the others add nothing).
//! - On the finest level every element matrix is E_e K_unit with the rows and columns of fixed degrees of
//!   freedom removed: a principal submatrix of the scaled K_unit, whose largest eigenvalue is at most that
//!   of the scaled K_unit (Cauchy's interlacing theorem). One bound, computed once, serves every element.
//! - On a coarse level each element matrix is different, and its bound is computed on every build.
//!
//! How one L_e is found (`element_bound`): a few steps of the power method give an estimate mu from
//! below; L = 1.01 mu is then proved to be an upper bound by factoring L I - S (S the scaled matrix) by
//! Cholesky: the factorisation succeeds with positive pivots only if L I - S is positive definite, that
//! is, if L lies above every eigenvalue of S. If it fails, L is raised by 2 % and tried again. Rounding in
//! a Cholesky factorisation of order 24 changes the matrix by far less than 1e-12 L, so L (1 + 1e-12)
//! is returned. If no L is proved within 100 tries, the trace of S (its order, every diagonal entry being
//! 1) bounds lambda_max of a positive semi-definite S and is returned instead.
//!
//! The same number of sweeps runs before and after the coarse correction, so the V-cycle is a symmetric
//! preconditioner. The settings c (`DAMPING`) and the number of sweeps (`SWEEPS`) were chosen by the
//! measured study below.
//!
//! Steps of conjugate gradients to a relative residual of 1e-8 / products with the fine stiffness matrix
//! per solve (a smoothing sweep costs one product), on the four cases of verification row 16
//! (`smoother_study` builds them again; E_min = 1e-9 E, 3 multigrid levels each):
//!
//! | c    | sweeps | uniform 50x20x20 | lattice 50x20x20 | irregular 50x20x20 | L-shaped 35x25x27 |
//! |------|--------|------------------|------------------|--------------------|-------------------|
//! | 0.6  | 1      | 24 / 73          | 58 / 175         | 112 / 337          | 27 / 82           |
//! | 0.6  | 2      | 18 / 91          | 43 / 216         | 88 / 441           | 20 / 101          |
//! | 0.6  | 3      | 15 / 106         | 38 / 267         | 76 / 533           | 16 / 113          |
//! | 0.75 | 1      | 22 / 67          | 53 / 160         | 104 / 313          | 24 / 73           |
//! | 0.75 | 2      | 16 / 81          | 40 / 201         | 80 / 401           | 18 / 91           |
//! | 0.75 | 3      | 13 / 92          | 35 / 246         | 71 / 498           | 15 / 106          |
//! | 0.9  | 1      | 20 / 61          | 49 / 148         | 97 / 292           | 23 / 70           |
//! | 0.9  | 2      | 14 / 71          | 37 / 186         | 76 / 381           | 16 / 81           |
//! | 0.9  | 3      | 12 / 85          | 33 / 232         | 67 / 470           | 14 / 99           |
//! | 0.95 | 1      | 20 / 61          | 48 / 145         | 95 / 286           | 22 / 67           |
//! | 0.95 | 2      | 14 / 71          | 37 / 186         | 75 / 376           | 16 / 81           |
//! | 0.95 | 3      | 11 / 78          | 32 / 225         | 66 / 463           | 13 / 92           |
//!
//! More sweeps take fewer steps but more products; c = 0.95 with one sweep takes the fewest products in
//! all four cases, and so is the setting used. c closer to 1 was not tried: an error component along an
//! eigenvector of D^-1 A with eigenvalue lambda is multiplied by 1 - 2 c lambda / L per sweep, which tends
//! to -1 for lambda near L as c tends to 1, so the stiffest components would barely be damped.
//!
//! Sources: Amir, Aage and Lazarov, "On multigrid-CG for efficient topology optimization", Struct.
//! Multidisc. Optim. 49 (2014) (damped Jacobi smoothing in a multigrid-preconditioned conjugate-gradient
//! solver for topology optimisation); Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed. (2003),
//! section 4.2 (convergence of damped Jacobi); Horn and Johnson, "Matrix Analysis", 2nd ed. (2013),
//! theorem 4.3.28 (interlacing for principal submatrices).

use crate::threads::Threads;

/// c in omega = c * 2 / L: the share of the largest damping that is still safe.
pub const DAMPING: f64 = 0.95;

/// The number of Jacobi sweeps before, and again after, the coarse correction.
pub const SWEEPS: usize = 1;

/// The settings of the smoother.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmootherSettings {
    /// c in omega = c * 2 / L, between 0 and 1 (both excluded).
    pub damping: f64,
    /// Sweeps before and after the coarse correction, at least 1.
    pub sweeps: usize,
}

impl Default for SmootherSettings {
    fn default() -> SmootherSettings {
        SmootherSettings {
            damping: DAMPING,
            sweeps: SWEEPS,
        }
    }
}

/// Steps of the power method before the estimate is tested.
const POWER_STEPS: usize = 30;

/// A guaranteed upper bound of the largest eigenvalue of D^-1/2 A D^-1/2 for one 24 by 24 positive
/// semi-definite element matrix `matrix` (row by row), on the degrees of freedom whose diagonal is positive.
/// 0 when no diagonal is positive.
pub fn element_bound(matrix: &[f64]) -> f64 {
    let keep: Vec<usize> = (0..24).filter(|&i| matrix[24 * i + i] > 0.0).collect();
    let m = keep.len();
    if m == 0 {
        return 0.0;
    }
    // S = D^-1/2 A D^-1/2 on the kept degrees of freedom: 1 on its diagonal.
    let scale: Vec<f64> = keep
        .iter()
        .map(|&i| 1.0 / matrix[24 * i + i].sqrt())
        .collect();
    let mut s = vec![0.0; m * m];
    for (r, &i) in keep.iter().enumerate() {
        for (c, &j) in keep.iter().enumerate() {
            s[m * r + c] = scale[r] * matrix[24 * i + j] * scale[c];
        }
    }

    // The power method from a fixed start that is not orthogonal to any eigenvector in general.
    let mut v: Vec<f64> = (0..m).map(|i| 1.0 + i as f64 / m as f64).collect();
    let mut w = vec![0.0; m];
    let mut estimate = 0.0;
    for _ in 0..POWER_STEPS {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        for (r, w_r) in w.iter_mut().enumerate() {
            *w_r = (0..m).map(|c| s[m * r + c] * v[c]).sum();
        }
        // The Rayleigh quotient v^T S v of the unit vector v: at most lambda_max.
        estimate = v.iter().zip(&w).map(|(a, b)| a * b).sum();
        std::mem::swap(&mut v, &mut w);
    }

    let mut bound = 1.01 * estimate.max(1.0 / m as f64);
    for _ in 0..100 {
        if lies_above_every_eigenvalue(&s, m, bound) {
            return bound * (1.0 + 1e-12);
        }
        bound *= 1.02;
    }
    m as f64
}

/// Whether bound * I - S is positive definite: its Cholesky factorisation meets only positive pivots.
fn lies_above_every_eigenvalue(s: &[f64], m: usize, bound: f64) -> bool {
    let mut l = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..=i {
            let identity = if i == j { bound } else { 0.0 };
            let mut sum = identity - s[m * i + j];
            for k in 0..j {
                sum -= l[m * i + k] * l[m * j + k];
            }
            if i == j {
                if sum.is_nan() || sum <= 0.0 {
                    return false;
                }
                l[m * i + i] = sum.sqrt();
            } else {
                l[m * i + j] = sum / l[m * j + j];
            }
        }
    }
    true
}

/// The largest `element_bound` over the element matrices stored one after the other in `matrices`
/// (24 * 24 entries each), at least 1 (the bound of an identity row). Each element's bound is computed by
/// one thread; the largest of them does not depend on the order.
pub fn largest_bound(matrices: &[f64], threads: &Threads) -> f64 {
    let count = matrices.len() / 576;
    let mut bounds = vec![0.0; count];
    threads.update(&mut bounds, |element, bound| {
        *bound = element_bound(&matrices[576 * element..576 * (element + 1)]);
    });
    bounds.iter().fold(1.0, |largest, b| largest.max(*b))
}

/// Damped Jacobi on one level: omega / D on the free degrees of freedom, 0 on the fixed ones.
pub struct Jacobi {
    scaled_inverse: Vec<f64>,
}

impl Jacobi {
    /// The smoother for the level with this `diagonal` (positive on every free degree of freedom) and
    /// bound `bound` of lambda_max(D^-1 A): omega = damping * 2 / bound.
    pub fn new(
        diagonal: &[f64],
        fixed: &[bool],
        bound: f64,
        damping: f64,
        threads: &Threads,
    ) -> Jacobi {
        let omega = damping * 2.0 / bound;
        let mut scaled_inverse = vec![0.0; diagonal.len()];
        threads.update(&mut scaled_inverse, |dof, value| {
            if !fixed[dof] {
                *value = omega / diagonal[dof];
            }
        });
        Jacobi { scaled_inverse }
    }

    /// The first sweep from x = 0: x = omega D^-1 r.
    pub fn first_sweep(&self, threads: &Threads, r: &[f64], x: &mut [f64]) {
        threads.update(x, |dof, x_dof| *x_dof = self.scaled_inverse[dof] * r[dof]);
    }

    /// One sweep: x += omega D^-1 (r - A x), with `ax` = A x computed by the caller.
    pub fn sweep(&self, threads: &Threads, r: &[f64], ax: &[f64], x: &mut [f64]) {
        threads.update(x, |dof, x_dof| {
            *x_dof += self.scaled_inverse[dof] * (r[dof] - ax[dof]);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex8;

    /// The largest eigenvalue of a symmetric matrix by many power steps, for comparison.
    fn largest_eigenvalue(s: &[f64], m: usize) -> f64 {
        let mut v = vec![1.0; m];
        let mut value = 0.0;
        for _ in 0..5000 {
            let w: Vec<f64> = (0..m)
                .map(|r| (0..m).map(|c| s[m * r + c] * v[c]).sum())
                .collect();
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            value = v.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>()
                / v.iter().map(|x| x * x).sum::<f64>();
            v = w.iter().map(|x| x / norm).collect();
        }
        value
    }

    fn flat(k: &[[f64; 24]; 24]) -> Vec<f64> {
        k.iter().flatten().copied().collect()
    }

    #[test]
    fn the_bound_of_the_unit_element_is_close_above_its_largest_eigenvalue() {
        let k = flat(&hex8::stiffness(1.0, 0.3, 1.0));
        let bound = element_bound(&k);
        let scaled: Vec<f64> = (0..576)
            .map(|at| k[at] / (k[25 * (at / 24)] * k[25 * (at % 24)]).sqrt())
            .collect();
        let exact = largest_eigenvalue(&scaled, 24);
        assert!(bound >= exact && bound < 1.05 * exact, "{bound} {exact}");
        // Rows and columns taken out (held degrees of freedom) can only lower it.
        let mut reduced = k.clone();
        for dof in [0, 1, 2, 7, 13] {
            for other in 0..24 {
                reduced[24 * dof + other] = 0.0;
                reduced[24 * other + dof] = 0.0;
            }
        }
        assert!(element_bound(&reduced) <= 1.05 * exact);
        assert_eq!(element_bound(&[0.0; 576]), 0.0);
    }

    #[test]
    fn the_bound_holds_for_the_assembled_matrix() {
        // Two unit elements side by side (x direction) with very different moduli: the largest
        // eigenvalue of D^-1 A of the 36-dof assembled matrix stays below the element bound.
        let k = hex8::stiffness(1.0, 0.3, 1.0);
        let bound = element_bound(&flat(&k));
        let grid = crate::grid3d::Grid3d::full(2, 1, 1, 1.0).unwrap();
        let n = grid.n_dofs();
        let mut a = vec![0.0; n * n];
        for (element, modulus) in [(0, 1.0), (1, 1e-6)] {
            let dofs = grid.element_dofs(element);
            for r in 0..24 {
                for c in 0..24 {
                    a[n * dofs[r] + dofs[c]] += modulus * k[r][c];
                }
            }
        }
        let scaled: Vec<f64> = (0..n * n)
            .map(|at| a[at] / (a[(n + 1) * (at / n)] * a[(n + 1) * (at % n)]).sqrt())
            .collect();
        assert!(largest_eigenvalue(&scaled, n) <= bound);
    }

    #[test]
    fn jacobi_leaves_fixed_entries_at_zero() {
        let threads = Threads::new(1).unwrap();
        let smoother = Jacobi::new(&[2.0, 4.0, 1.0], &[false, true, false], 2.0, 0.5, &threads);
        let mut x = vec![9.0; 3];
        smoother.first_sweep(&threads, &[1.0, 1.0, 1.0], &mut x);
        assert_eq!(x, vec![0.25, 0.0, 0.5]);
        smoother.sweep(&threads, &[1.0, 1.0, 1.0], &[0.0, 5.0, 2.0], &mut x);
        assert_eq!(x, vec![0.5, 0.0, 0.0]);
    }
}
