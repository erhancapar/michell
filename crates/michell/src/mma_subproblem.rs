//! The convex subproblem of the method of moving asymptotes, solved by a primal-dual interior-point
//! method.
//!
//! The subproblem is the one of Svanberg, "MMA and GCMMA: two methods for nonlinear optimization"
//! (2007), in the problem form of that note with the artificial variables y_i and z:
//!
//!   minimise    g_0(x) + a0 z + sum_i (c_i y_i + d_i y_i^2 / 2)
//!   subject to  g_i(x) - a_i z - y_i <= 0                    i = 1..m
//!               alpha_j <= x_j <= beta_j,  y_i >= 0,  z >= 0
//!
//! with the separable convex functions (built in `mma.rs`)
//!
//!   g_i(x) = sum_j ( p_ij / (u_j - x_j) + q_ij / (x_j - l_j) ) + r_i,       p_ij, q_ij >= 0.
//!
//! The method follows the note: the optimality (KKT) conditions with every complementarity product
//! relaxed to a small epsilon, solved by Newton's method, with epsilon lowered step by step towards zero.
//! The Newton system below is derived here in full, so each line can be checked by hand.
//!
//! Unknowns: x (n), y (m), z, the multipliers lambda (m) of the limits, xi (n) and eta (n) of the bounds
//! alpha and beta, mu (m) of y >= 0, zeta of z >= 0, and the slacks s (m) of the limits. With
//! psi(x, lambda) = g_0(x) + sum_i lambda_i g_i(x), the relaxed conditions are
//!
//!   r_x     = d psi / d x  - xi + eta                     = 0
//!   r_y     = c + d y - lambda - mu                       = 0
//!   r_z     = a0 - zeta - a . lambda                      = 0
//!   r_lam   = g(x) - a z - y + s                          = 0
//!   r_xi    = xi  (x - alpha) - epsilon                   = 0
//!   r_eta   = eta (beta - x)  - epsilon                   = 0
//!   r_mu    = mu y            - epsilon                   = 0
//!   r_zeta  = zeta z          - epsilon                   = 0
//!   r_s     = lambda s        - epsilon                   = 0
//!
//! with every variable except x strictly positive and alpha < x < beta.
//!
//! One Newton step. The steps of xi, eta, mu, zeta and s follow from the last five lines once the others
//! are known:
//!
//!   d_xi   = (-r_xi  - xi d_x)   / (x - alpha)
//!   d_eta  = (-r_eta + eta d_x)  / (beta - x)
//!   d_mu   = (-r_mu  - mu d_y)   / y
//!   d_zeta = (-r_zeta - zeta d_z) / z
//!   d_s    = (-r_s   - s d_lam)  / lambda
//!
//! Put into the first four lines, with G_ij = d g_i / d x_j and psi'' the (diagonal) second derivative of
//! psi in x:
//!
//!   D_x   = psi'' + xi / (x - alpha) + eta / (beta - x)      (diagonal)
//!   b_x   = -r_x - r_xi / (x - alpha) + r_eta / (beta - x)
//!   D_y   = d + mu / y,       b_y = -r_y - r_mu / y
//!   b_z   = -r_z - r_zeta / z
//!   b_lam = -r_lam + r_s / lambda
//!
//!   d_x = (b_x - G^T d_lam) / D_x,      d_y = (b_y + d_lam) / D_y
//!
//! and what is left is a system of m + 1 equations in d_lam and d_z:
//!
//!   [ G D_x^-1 G^T + diag(1/D_y + s/lambda)    a      ] [ d_lam ]   [ G D_x^-1 b_x - b_y / D_y - b_lam ]
//!   [ -a^T                                     zeta/z ] [ d_z   ] = [ b_z                              ]
//!
//! It is small (m is the number of limits), so it is solved directly. Building it costs one pass over the
//! n variables per limit pair, so one Newton step costs O(n m^2), never O(n^2).
//!
//! The step is shortened so that every positive variable keeps at least 1 % of its value
//! (fraction to the boundary), and then halved until the norm of all residuals falls.
//!
//! The units of the limits. The Newton direction does not depend on the units in which each limit is
//! written, but the two tests of the method do: the step is halved until the Euclidean norm of all the
//! residuals falls (note, section 5.4), and epsilon is lowered once their largest falls below 0.9 epsilon
//! (section 5.5). Both add r_lam_i, in the units of g_i, to r_x, in the units of the derivative of g_0.
//! A limit whose approximation is far steeper than the goal's (a gradient a thousand times larger on some
//! variables) is also sharply curved there, so a step that moves those variables changes its r_lam_i by
//! far more than the linear model predicts. That one residual then decides the halving: the accepted
//! step falls to a few hundredths of the Newton step or less, the solver advances by that fraction per
//! step, and it runs out of steps. The note leaves the units of the problem to the user and asks for a
//! well scaled one (section 2), so `solve` writes each limit in units in which it is no steeper than the
//! goal before it starts. With omega_i > 0 the substitution
//!
//!   g_i -> omega_i g_i,  a_i -> omega_i a_i,  c_i -> c_i / omega_i,  d_i -> d_i / omega_i^2,
//!   y_i -> omega_i y_i,  s_i -> omega_i s_i,  lambda_i -> lambda_i / omega_i,  mu_i -> mu_i / omega_i
//!
//! leaves every term of the subproblem's objective, the products lambda_i s_i and mu_i y_i, a . lambda and
//! the conditions of the note's problem form (a_i c_i > a0 where a_i > 0) unchanged, and multiplies r_lam_i
//! by omega_i and r_y_i by 1 / omega_i, leaving every other residual as it is. It is the same subproblem,
//! with the same solution for every epsilon, and the same method applied to it, with the same Newton
//! direction from any given point; only the starting point and the weight of those two residuals in the
//! two tests change. The size of a function is the largest over j of the sum of the magnitudes of its two
//! terms' derivatives, p_ij / (u_j - x_j)^2 + q_ij / (x_j - l_j)^2, at the starting point (positive,
//! since every p_ij and q_ij is), and omega_i = min(1, size of g_0 / size of g_i): a limit no steeper
//! than the goal is left as it is. The solution is returned in the caller's units.
//!
//! Threads. Every loop over the n variables runs on the threads of `threads.rs`: the entries of a vector
//! over the variables are each computed by one thread with the same arithmetic on any thread count, and
//! every sum over the variables (the values g_i(x), the norm of the residuals, the sums that build the
//! (m + 1) system) and every largest or smallest value (the largest residual, the longest step, the size
//! of a function) is taken within each block of `BLOCK_LENGTH` variables in index order and then over the
//! blocks in block order, also on one thread. The result is therefore the same on every thread count. The
//! (m + 1) system itself is solved on the calling thread.

use crate::error::{EngineError, ErrorCode};
use crate::threads::Threads;

/// The relaxation epsilon starts here, ...
pub const EPSILON_START: f64 = 1.0;
/// ... is multiplied by this factor each time the residual has fallen below 0.9 epsilon, ...
pub const EPSILON_FACTOR: f64 = 0.1;
/// ... and the solver stops once the residual is below 0.9 epsilon for this epsilon.
pub const EPSILON_END: f64 = 1e-9;
/// Newton steps allowed for one value of epsilon before the solver gives up.
pub const NEWTON_STEPS_PER_EPSILON: usize = 200;
/// A step may take a positive variable down to this fraction of its distance to zero, not further.
const FRACTION_TO_BOUNDARY: f64 = 0.99;
/// Halvings of the step allowed while looking for a lower residual.
const STEP_HALVINGS: usize = 50;

/// The data of one subproblem. Arrays over variables have length n; `p` and `q` hold the limits' terms
/// row by row (limit i, variable j at index i n + j).
pub struct Subproblem<'a> {
    pub lower: &'a [f64],
    pub upper: &'a [f64],
    pub alpha: &'a [f64],
    pub beta: &'a [f64],
    pub p0: &'a [f64],
    pub q0: &'a [f64],
    pub p: &'a [f64],
    pub q: &'a [f64],
    /// The constant r_i of each limit's approximation.
    pub r: &'a [f64],
    pub a0: f64,
    pub a: &'a [f64],
    pub c: &'a [f64],
    pub d: &'a [f64],
}

/// The solution of a subproblem: the primal variables and the multipliers of the limits.
#[derive(Clone, Debug)]
pub struct Solution {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: f64,
    pub lambda: Vec<f64>,
}

/// All unknowns of the relaxed conditions (see the file header).
#[derive(Clone, Debug)]
struct Point {
    x: Vec<f64>,
    y: Vec<f64>,
    z: f64,
    lambda: Vec<f64>,
    xi: Vec<f64>,
    eta: Vec<f64>,
    mu: Vec<f64>,
    zeta: f64,
    s: Vec<f64>,
}

/// The residuals of the relaxed conditions, in the same order as the unknowns.
struct Residual {
    x: Vec<f64>,
    y: Vec<f64>,
    z: f64,
    lambda: Vec<f64>,
    xi: Vec<f64>,
    eta: Vec<f64>,
    mu: Vec<f64>,
    zeta: f64,
    s: Vec<f64>,
}

impl Residual {
    /// The residuals of the unknowns that are not indexed by the variables (y, z, lambda, mu, zeta, s).
    fn per_limit(&self) -> impl Iterator<Item = f64> + '_ {
        self.y
            .iter()
            .chain(std::iter::once(&self.z))
            .chain(&self.lambda)
            .chain(&self.mu)
            .chain(std::iter::once(&self.zeta))
            .chain(&self.s)
            .copied()
    }

    /// The Euclidean norm of all residuals: the terms of x, xi and eta summed by the block rule of
    /// `threads.rs`, then the others in order.
    fn norm(&self, threads: &Threads) -> f64 {
        let mut sum = threads.sum(self.x.len(), |j| {
            self.x[j] * self.x[j] + self.xi[j] * self.xi[j] + self.eta[j] * self.eta[j]
        });
        for v in self.per_limit() {
            sum += v * v;
        }
        sum.sqrt()
    }

    /// The largest magnitude of all residuals.
    fn largest(&self, threads: &Threads) -> f64 {
        let mut largest = 0.0_f64;
        for block in threads.blocks(self.x.len(), |block| {
            block.fold(0.0_f64, |m, j| {
                m.max(self.x[j].abs())
                    .max(self.xi[j].abs())
                    .max(self.eta[j].abs())
            })
        }) {
            largest = largest.max(block);
        }
        self.per_limit().fold(largest, |m, v| m.max(v.abs()))
    }
}

impl Subproblem<'_> {
    fn n(&self) -> usize {
        self.p0.len()
    }

    fn m(&self) -> usize {
        self.r.len()
    }

    /// g_i(x) for every limit: the sums over the variables in one pass, by the block rule.
    fn limit_values(&self, x: &[f64], threads: &Threads) -> Vec<f64> {
        let n = self.n();
        let m = self.m();
        let blocks = threads.blocks(n, |block| {
            let mut sums = vec![0.0; m];
            for j in block {
                for (i, sum) in sums.iter_mut().enumerate() {
                    *sum += self.p[i * n + j] / (self.upper[j] - x[j])
                        + self.q[i * n + j] / (x[j] - self.lower[j]);
                }
            }
            sums
        });
        (0..m)
            .map(|i| {
                let mut sum = 0.0;
                for block in &blocks {
                    sum += block[i];
                }
                self.r[i] + sum
            })
            .collect()
    }

    /// G_ij = d g_i / d x_j, row by row.
    fn limit_gradients(&self, x: &[f64], threads: &Threads) -> Vec<f64> {
        let n = self.n();
        let mut gradients = Vec::with_capacity(self.m() * n);
        for i in 0..self.m() {
            let (p, q) = (&self.p[i * n..(i + 1) * n], &self.q[i * n..(i + 1) * n]);
            gradients.extend(threads.vector(n, |j| {
                let ux = self.upper[j] - x[j];
                let xl = x[j] - self.lower[j];
                p[j] / (ux * ux) - q[j] / (xl * xl)
            }));
        }
        gradients
    }

    /// The combined terms p0_j + sum_i lambda_i p_ij and q0_j + sum_i lambda_i q_ij of psi.
    fn combined_terms(&self, lambda: &[f64], threads: &Threads) -> (Vec<f64>, Vec<f64>) {
        let n = self.n();
        let combined = |own: &[f64], terms: &[f64]| {
            threads.vector(n, |j| {
                let mut value = own[j];
                for (i, l) in lambda.iter().enumerate() {
                    value += l * terms[i * n + j];
                }
                value
            })
        };
        (combined(self.p0, self.p), combined(self.q0, self.q))
    }

    fn residual(&self, point: &Point, epsilon: f64, threads: &Threads) -> Residual {
        let n = self.n();
        let (p, q) = self.combined_terms(&point.lambda, threads);
        let x = &point.x;
        let r_x = threads.vector(n, |j| {
            let ux = self.upper[j] - x[j];
            let xl = x[j] - self.lower[j];
            p[j] / (ux * ux) - q[j] / (xl * xl) - point.xi[j] + point.eta[j]
        });
        let r_y = (0..self.m())
            .map(|i| self.c[i] + self.d[i] * point.y[i] - point.lambda[i] - point.mu[i])
            .collect();
        let a_dot_lambda: f64 = self.a.iter().zip(&point.lambda).map(|(a, l)| a * l).sum();
        let r_z = self.a0 - point.zeta - a_dot_lambda;
        let g = self.limit_values(x, threads);
        let r_lambda = (0..self.m())
            .map(|i| g[i] - self.a[i] * point.z - point.y[i] + point.s[i])
            .collect();
        Residual {
            x: r_x,
            y: r_y,
            z: r_z,
            lambda: r_lambda,
            xi: threads.vector(n, |j| point.xi[j] * (x[j] - self.alpha[j]) - epsilon),
            eta: threads.vector(n, |j| point.eta[j] * (self.beta[j] - x[j]) - epsilon),
            mu: (0..self.m())
                .map(|i| point.mu[i] * point.y[i] - epsilon)
                .collect(),
            zeta: point.zeta * point.z - epsilon,
            s: (0..self.m())
                .map(|i| point.lambda[i] * point.s[i] - epsilon)
                .collect(),
        }
    }

    /// The starting point: x in the middle of [alpha, beta], every other unknown 1 or a little more.
    fn start(&self, threads: &Threads) -> Point {
        let n = self.n();
        let m = self.m();
        let x = threads.vector(n, |j| 0.5 * (self.alpha[j] + self.beta[j]));
        Point {
            xi: threads.vector(n, |j| (1.0 / (x[j] - self.alpha[j])).max(1.0)),
            eta: threads.vector(n, |j| (1.0 / (self.beta[j] - x[j])).max(1.0)),
            x,
            y: vec![1.0; m],
            z: 1.0,
            lambda: vec![1.0; m],
            mu: (0..m).map(|i| (0.5 * self.c[i]).max(1.0)).collect(),
            zeta: 1.0,
            s: vec![1.0; m],
        }
    }

    /// The Newton step at `point` (see the file header for the derivation).
    fn newton_step(
        &self,
        point: &Point,
        res: &Residual,
        threads: &Threads,
    ) -> Result<Point, EngineError> {
        let n = self.n();
        let m = self.m();
        let x = &point.x;
        let (p, q) = self.combined_terms(&point.lambda, threads);
        let gradients = self.limit_gradients(x, threads);

        // D_x and b_x, per variable.
        let d_x_diag = threads.vector(n, |j| {
            let ux = self.upper[j] - x[j];
            let xl = x[j] - self.lower[j];
            let psi_second = 2.0 * p[j] / (ux * ux * ux) + 2.0 * q[j] / (xl * xl * xl);
            psi_second + point.xi[j] / (x[j] - self.alpha[j]) + point.eta[j] / (self.beta[j] - x[j])
        });
        let b_x = threads.vector(n, |j| {
            -res.x[j] - res.xi[j] / (x[j] - self.alpha[j]) + res.eta[j] / (self.beta[j] - x[j])
        });
        // D_y, b_y, b_lam per limit, and b_z.
        let d_y_diag: Vec<f64> = (0..m)
            .map(|i| self.d[i] + point.mu[i] / point.y[i])
            .collect();
        let b_y: Vec<f64> = (0..m).map(|i| -res.y[i] - res.mu[i] / point.y[i]).collect();
        let b_lambda: Vec<f64> = (0..m)
            .map(|i| -res.lambda[i] + res.s[i] / point.lambda[i])
            .collect();
        let b_z = -res.z - res.zeta / point.z;

        // The sums over the variables in G D_x^-1 G^T (entry (i, k) for k >= i: the matrix is symmetric)
        // and in G D_x^-1 b_x, all in one pass by the block rule.
        let pairs: Vec<(usize, usize)> = (0..m).flat_map(|i| (i..m).map(move |k| (i, k))).collect();
        let blocks = threads.blocks(n, |block| {
            let mut sums = vec![0.0; pairs.len() + m];
            for j in block {
                for (sum, &(i, k)) in sums.iter_mut().zip(&pairs) {
                    *sum += gradients[i * n + j] * gradients[k * n + j] / d_x_diag[j];
                }
                for i in 0..m {
                    sums[pairs.len() + i] += gradients[i * n + j] * b_x[j] / d_x_diag[j];
                }
            }
            sums
        });
        let mut sums = vec![0.0; pairs.len() + m];
        for block in &blocks {
            for (sum, value) in sums.iter_mut().zip(block) {
                *sum += value;
            }
        }

        // The (m + 1) system in d_lam and d_z, row by row.
        let size = m + 1;
        let mut matrix = vec![0.0; size * size];
        let mut rhs = vec![0.0; size];
        for (&(i, k), sum) in pairs.iter().zip(&sums) {
            matrix[i * size + k] = *sum;
            matrix[k * size + i] = *sum;
        }
        for i in 0..m {
            matrix[i * size + i] += 1.0 / d_y_diag[i] + point.s[i] / point.lambda[i];
            matrix[i * size + m] = self.a[i];
            matrix[m * size + i] = -self.a[i];
            rhs[i] = sums[pairs.len() + i] - b_y[i] / d_y_diag[i] - b_lambda[i];
        }
        matrix[m * size + m] = point.zeta / point.z;
        rhs[m] = b_z;
        let solution = solve_dense(&mut matrix, &mut rhs, size)?;
        let d_lambda = &solution[..m];
        let d_z = solution[m];

        // Back substitution, in the order of the file header.
        let d_x = threads.vector(n, |j| {
            let g_t_d_lambda: f64 = (0..m).map(|i| gradients[i * n + j] * d_lambda[i]).sum();
            (b_x[j] - g_t_d_lambda) / d_x_diag[j]
        });
        let d_y: Vec<f64> = (0..m)
            .map(|i| (b_y[i] + d_lambda[i]) / d_y_diag[i])
            .collect();
        Ok(Point {
            xi: threads.vector(n, |j| {
                (-res.xi[j] - point.xi[j] * d_x[j]) / (x[j] - self.alpha[j])
            }),
            eta: threads.vector(n, |j| {
                (-res.eta[j] + point.eta[j] * d_x[j]) / (self.beta[j] - x[j])
            }),
            mu: (0..m)
                .map(|i| (-res.mu[i] - point.mu[i] * d_y[i]) / point.y[i])
                .collect(),
            zeta: (-res.zeta - point.zeta * d_z) / point.z,
            s: (0..m)
                .map(|i| (-res.s[i] - point.s[i] * d_lambda[i]) / point.lambda[i])
                .collect(),
            x: d_x,
            y: d_y,
            z: d_z,
            lambda: d_lambda.to_vec(),
        })
    }

    /// The longest step t <= 1 along `step` that keeps every positive variable above 1 % of its value
    /// and x strictly between alpha and beta.
    fn longest_step(&self, point: &Point, step: &Point, threads: &Threads) -> f64 {
        let limit = |t: f64, value: f64, change: f64| {
            if change < 0.0 {
                t.min(-FRACTION_TO_BOUNDARY * value / change)
            } else {
                t
            }
        };
        let mut t: f64 = 1.0;
        for block in threads.blocks(self.n(), |block| {
            block.fold(1.0_f64, |t, j| {
                let t = limit(t, point.x[j] - self.alpha[j], step.x[j]);
                let t = limit(t, self.beta[j] - point.x[j], -step.x[j]);
                let t = limit(t, point.xi[j], step.xi[j]);
                limit(t, point.eta[j], step.eta[j])
            })
        }) {
            t = t.min(block);
        }
        for i in 0..self.m() {
            t = limit(t, point.y[i], step.y[i]);
            t = limit(t, point.lambda[i], step.lambda[i]);
            t = limit(t, point.mu[i], step.mu[i]);
            t = limit(t, point.s[i], step.s[i]);
        }
        t = limit(t, point.z, step.z);
        limit(t, point.zeta, step.zeta)
    }
}

/// point + t step.
fn moved(point: &Point, step: &Point, t: f64, threads: &Threads) -> Point {
    let add = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a + t * b).collect();
    let add_on_threads = |a: &[f64], b: &[f64]| threads.vector(a.len(), |j| a[j] + t * b[j]);
    Point {
        x: add_on_threads(&point.x, &step.x),
        y: add(&point.y, &step.y),
        z: point.z + t * step.z,
        lambda: add(&point.lambda, &step.lambda),
        xi: add_on_threads(&point.xi, &step.xi),
        eta: add_on_threads(&point.eta, &step.eta),
        mu: add(&point.mu, &step.mu),
        zeta: point.zeta + t * step.zeta,
        s: add(&point.s, &step.s),
    }
}

/// The factor omega_i for each limit (see "The units of the limits" in the file header).
fn limit_units(sub: &Subproblem<'_>, threads: &Threads) -> Vec<f64> {
    let n = sub.n();
    let start = sub.start(threads);
    let size = |p: &[f64], q: &[f64]| {
        threads
            .blocks(n, |block| {
                block.fold(0.0_f64, |largest, j| {
                    let ux = sub.upper[j] - start.x[j];
                    let xl = start.x[j] - sub.lower[j];
                    largest.max(p[j] / (ux * ux) + q[j] / (xl * xl))
                })
            })
            .into_iter()
            .fold(0.0_f64, f64::max)
    };
    let goal = size(sub.p0, sub.q0);
    (0..sub.m())
        .map(|i| {
            let limit = size(&sub.p[i * n..(i + 1) * n], &sub.q[i * n..(i + 1) * n]);
            if limit > goal { goal / limit } else { 1.0 }
        })
        .collect()
}

/// Solves the subproblem on one thread. Returns `solver_not_converged` if one value of epsilon needs more
/// than `NEWTON_STEPS_PER_EPSILON` Newton steps.
pub fn solve(sub: &Subproblem<'_>) -> Result<Solution, EngineError> {
    solve_on(sub, &Threads::new(1)?)
}

/// Solves the subproblem on `threads`, with the same result on every thread count (see "Threads" in the
/// file header). Returns `solver_not_converged` if one value of epsilon needs more than
/// `NEWTON_STEPS_PER_EPSILON` Newton steps.
pub fn solve_on(sub: &Subproblem<'_>, threads: &Threads) -> Result<Solution, EngineError> {
    let omega = limit_units(sub, threads);
    if omega.iter().all(|&w| w == 1.0) {
        return solve_in_given_units(sub, threads);
    }
    let n = sub.n();
    let scaled = |values: &[f64]| {
        let mut scaled = Vec::with_capacity(values.len());
        for (row, w) in values.chunks(n).zip(&omega) {
            scaled.extend(threads.vector(n, |j| row[j] * w));
        }
        scaled
    };
    let p = scaled(sub.p);
    let q = scaled(sub.q);
    let r: Vec<f64> = sub.r.iter().zip(&omega).map(|(v, w)| v * w).collect();
    let a: Vec<f64> = sub.a.iter().zip(&omega).map(|(v, w)| v * w).collect();
    let c: Vec<f64> = sub.c.iter().zip(&omega).map(|(v, w)| v / w).collect();
    let d: Vec<f64> = sub.d.iter().zip(&omega).map(|(v, w)| v / (w * w)).collect();
    let solution = solve_in_given_units(
        &Subproblem {
            p: &p,
            q: &q,
            r: &r,
            a: &a,
            c: &c,
            d: &d,
            ..*sub
        },
        threads,
    )?;
    Ok(Solution {
        y: solution.y.iter().zip(&omega).map(|(v, w)| v / w).collect(),
        lambda: solution
            .lambda
            .iter()
            .zip(&omega)
            .map(|(v, w)| v * w)
            .collect(),
        ..solution
    })
}

/// The primal-dual method of the file header on the subproblem as given.
fn solve_in_given_units(sub: &Subproblem<'_>, threads: &Threads) -> Result<Solution, EngineError> {
    let mut point = sub.start(threads);
    let mut epsilon = EPSILON_START;
    loop {
        let mut steps = 0;
        let mut res = sub.residual(&point, epsilon, threads);
        while res.largest(threads) > 0.9 * epsilon {
            if steps == NEWTON_STEPS_PER_EPSILON {
                return Err(EngineError::new(
                    ErrorCode::SolverNotConverged,
                    "the MMA subproblem solver did not reach its tolerance",
                )
                .with_number("epsilon", epsilon)
                .with_number("residual", res.largest(threads))
                .with_integer("newton_steps", steps));
            }
            let step = sub.newton_step(&point, &res, threads)?;
            let mut t = sub.longest_step(&point, &step, threads);
            let norm_before = res.norm(threads);
            let mut candidate = moved(&point, &step, t, threads);
            let mut candidate_res = sub.residual(&candidate, epsilon, threads);
            for _ in 0..STEP_HALVINGS {
                if candidate_res.norm(threads) < norm_before {
                    break;
                }
                t *= 0.5;
                candidate = moved(&point, &step, t, threads);
                candidate_res = sub.residual(&candidate, epsilon, threads);
            }
            point = candidate;
            res = candidate_res;
            steps += 1;
        }
        if epsilon <= EPSILON_END {
            break;
        }
        epsilon *= EPSILON_FACTOR;
    }
    Ok(Solution {
        x: point.x,
        y: point.y,
        z: point.z,
        lambda: point.lambda,
    })
}

/// Solves the dense system `matrix` (size by size, row by row) times v = `rhs` by Gaussian elimination
/// with partial pivoting. Both inputs are overwritten.
fn solve_dense(matrix: &mut [f64], rhs: &mut [f64], size: usize) -> Result<Vec<f64>, EngineError> {
    for column in 0..size {
        // The row with the largest entry in this column becomes the pivot row.
        let mut pivot_row = column;
        for row in column + 1..size {
            if matrix[row * size + column].abs() > matrix[pivot_row * size + column].abs() {
                pivot_row = row;
            }
        }
        let pivot = matrix[pivot_row * size + column];
        if pivot == 0.0 || !pivot.is_finite() {
            return Err(EngineError::new(
                ErrorCode::InternalError,
                "the Newton system of the MMA subproblem is singular",
            )
            .with_integer("column", column)
            .with_number("pivot", pivot));
        }
        if pivot_row != column {
            for k in 0..size {
                matrix.swap(column * size + k, pivot_row * size + k);
            }
            rhs.swap(column, pivot_row);
        }
        for row in column + 1..size {
            let factor = matrix[row * size + column] / pivot;
            for k in column..size {
                matrix[row * size + k] -= factor * matrix[column * size + k];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = vec![0.0; size];
    for row in (0..size).rev() {
        let mut sum = rhs[row];
        for k in row + 1..size {
            sum -= matrix[row * size + k] * solution[k];
        }
        solution[row] = sum / matrix[row * size + row];
    }
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dense_solve_matches_a_system_worked_by_hand() {
        // 2 v0 + v1 = 5, v0 + 3 v1 = 10  ->  v0 = 1, v1 = 3; the zero in the corner forces a row swap.
        let mut matrix = vec![0.0, 1.0, 2.0, 1.0];
        let mut rhs = vec![3.0, 5.0];
        let v = solve_dense(&mut matrix, &mut rhs, 2).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-15 && (v[1] - 3.0).abs() < 1e-15);
    }

    /// One variable, one limit, solved by hand: minimise 1/(2 - x) + 1/(x + 1) with the limit
    /// 1/(2 - x) - 1.5 <= 0, that is x <= 4/3. The goal's own minimum, x = 1/2, meets the limit.
    #[test]
    fn an_inactive_limit_leaves_the_unconstrained_minimum() {
        let sub = Subproblem {
            lower: &[-1.0],
            upper: &[2.0],
            alpha: &[-0.5],
            beta: &[1.5],
            p0: &[1.0],
            q0: &[1.0],
            p: &[1.0],
            q: &[0.0],
            r: &[-1.5],
            a0: 1.0,
            a: &[0.0],
            c: &[1000.0],
            d: &[1.0],
        };
        let solution = solve(&sub).unwrap();
        // d/dx (1/(2-x) + 1/(x+1)) = 0 at 2 - x = x + 1, x = 1/2.
        assert!((solution.x[0] - 0.5).abs() < 1e-8);
        assert!(solution.lambda[0] < 1e-6);
        assert!(solution.y[0] < 1e-8);
    }

    #[test]
    fn an_active_limit_is_met_with_the_multiplier_of_the_kkt_conditions() {
        // Goal: q0 / (x - l) = 1 / (x + 1), falling in x. Limit: p / (u - x) + r = 1/(2 - x) - 2 <= 0,
        // active at x = 3/2. Stationarity: -1/(x+1)^2 + lambda / (2-x)^2 = 0, lambda = (1/2)^2/(5/2)^2.
        let sub = Subproblem {
            lower: &[-1.0],
            upper: &[2.0],
            alpha: &[-0.5],
            beta: &[1.9],
            p0: &[0.0],
            q0: &[1.0],
            p: &[1.0],
            q: &[0.0],
            r: &[-2.0],
            a0: 1.0,
            a: &[0.0],
            c: &[1000.0],
            d: &[1.0],
        };
        let solution = solve(&sub).unwrap();
        assert!((solution.x[0] - 1.5).abs() < 1e-8);
        assert!((solution.lambda[0] - 0.04).abs() < 1e-8);
    }

    /// The limit of the test above in units 10^4 times smaller: it is then far steeper than the goal and is
    /// solved in other units (see the file header), but the design and the multiplier, in the caller's units,
    /// are those of the KKT conditions.
    #[test]
    fn a_steep_limit_gives_the_design_and_multiplier_in_the_callers_units() {
        let sub = Subproblem {
            lower: &[-1.0],
            upper: &[2.0],
            alpha: &[-0.5],
            beta: &[1.9],
            p0: &[0.0],
            q0: &[1.0],
            p: &[1e4],
            q: &[0.0],
            r: &[-2e4],
            a0: 1.0,
            a: &[0.0],
            c: &[1000.0],
            d: &[1.0],
        };
        assert!(limit_units(&sub, &Threads::new(1).unwrap())[0] < 1e-3);
        let solution = solve(&sub).unwrap();
        assert!((solution.x[0] - 1.5).abs() < 1e-8);
        assert!((solution.lambda[0] - 0.04e-4).abs() < 1e-12);
        assert!(solution.y[0] < 1e-8);
    }

    #[test]
    fn a_bound_holds_when_the_goal_pushes_against_it() {
        let sub = Subproblem {
            lower: &[-1.0],
            upper: &[2.0],
            alpha: &[0.0],
            beta: &[1.0],
            p0: &[0.0],
            q0: &[1.0],
            p: &[],
            q: &[],
            r: &[],
            a0: 1.0,
            a: &[],
            c: &[],
            d: &[],
        };
        let solution = solve(&sub).unwrap();
        assert!((solution.x[0] - 1.0).abs() < 1e-8);
    }
}
