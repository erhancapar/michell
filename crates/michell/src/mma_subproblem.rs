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

use crate::error::{EngineError, ErrorCode};

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
    fn all(&self) -> impl Iterator<Item = f64> + '_ {
        self.x
            .iter()
            .chain(&self.y)
            .chain(std::iter::once(&self.z))
            .chain(&self.lambda)
            .chain(&self.xi)
            .chain(&self.eta)
            .chain(&self.mu)
            .chain(std::iter::once(&self.zeta))
            .chain(&self.s)
            .copied()
    }

    fn norm(&self) -> f64 {
        self.all().map(|v| v * v).sum::<f64>().sqrt()
    }

    fn largest(&self) -> f64 {
        self.all().fold(0.0, |m, v| m.max(v.abs()))
    }
}

impl Subproblem<'_> {
    fn n(&self) -> usize {
        self.p0.len()
    }

    fn m(&self) -> usize {
        self.r.len()
    }

    /// g_i(x) for every limit.
    fn limit_values(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n();
        (0..self.m())
            .map(|i| {
                let mut value = self.r[i];
                for j in 0..n {
                    value += self.p[i * n + j] / (self.upper[j] - x[j])
                        + self.q[i * n + j] / (x[j] - self.lower[j]);
                }
                value
            })
            .collect()
    }

    /// G_ij = d g_i / d x_j, row by row.
    fn limit_gradients(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n();
        let mut gradients = vec![0.0; self.m() * n];
        for i in 0..self.m() {
            for j in 0..n {
                let ux = self.upper[j] - x[j];
                let xl = x[j] - self.lower[j];
                gradients[i * n + j] =
                    self.p[i * n + j] / (ux * ux) - self.q[i * n + j] / (xl * xl);
            }
        }
        gradients
    }

    /// The combined terms p0_j + sum_i lambda_i p_ij and q0_j + sum_i lambda_i q_ij of psi.
    fn combined_terms(&self, lambda: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n();
        let mut p = self.p0.to_vec();
        let mut q = self.q0.to_vec();
        for (i, l) in lambda.iter().enumerate() {
            for j in 0..n {
                p[j] += l * self.p[i * n + j];
                q[j] += l * self.q[i * n + j];
            }
        }
        (p, q)
    }

    fn residual(&self, point: &Point, epsilon: f64) -> Residual {
        let n = self.n();
        let (p, q) = self.combined_terms(&point.lambda);
        let x = &point.x;
        let r_x = (0..n)
            .map(|j| {
                let ux = self.upper[j] - x[j];
                let xl = x[j] - self.lower[j];
                p[j] / (ux * ux) - q[j] / (xl * xl) - point.xi[j] + point.eta[j]
            })
            .collect();
        let r_y = (0..self.m())
            .map(|i| self.c[i] + self.d[i] * point.y[i] - point.lambda[i] - point.mu[i])
            .collect();
        let a_dot_lambda: f64 = self.a.iter().zip(&point.lambda).map(|(a, l)| a * l).sum();
        let r_z = self.a0 - point.zeta - a_dot_lambda;
        let g = self.limit_values(x);
        let r_lambda = (0..self.m())
            .map(|i| g[i] - self.a[i] * point.z - point.y[i] + point.s[i])
            .collect();
        Residual {
            x: r_x,
            y: r_y,
            z: r_z,
            lambda: r_lambda,
            xi: (0..n)
                .map(|j| point.xi[j] * (x[j] - self.alpha[j]) - epsilon)
                .collect(),
            eta: (0..n)
                .map(|j| point.eta[j] * (self.beta[j] - x[j]) - epsilon)
                .collect(),
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
    fn start(&self) -> Point {
        let n = self.n();
        let m = self.m();
        let x: Vec<f64> = (0..n)
            .map(|j| 0.5 * (self.alpha[j] + self.beta[j]))
            .collect();
        Point {
            xi: (0..n)
                .map(|j| (1.0 / (x[j] - self.alpha[j])).max(1.0))
                .collect(),
            eta: (0..n)
                .map(|j| (1.0 / (self.beta[j] - x[j])).max(1.0))
                .collect(),
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
    fn newton_step(&self, point: &Point, res: &Residual) -> Result<Point, EngineError> {
        let n = self.n();
        let m = self.m();
        let x = &point.x;
        let (p, q) = self.combined_terms(&point.lambda);
        let gradients = self.limit_gradients(x);

        // D_x and b_x, per variable.
        let mut d_x_diag = vec![0.0; n];
        let mut b_x = vec![0.0; n];
        for j in 0..n {
            let ux = self.upper[j] - x[j];
            let xl = x[j] - self.lower[j];
            let to_alpha = x[j] - self.alpha[j];
            let to_beta = self.beta[j] - x[j];
            let psi_second = 2.0 * p[j] / (ux * ux * ux) + 2.0 * q[j] / (xl * xl * xl);
            d_x_diag[j] = psi_second + point.xi[j] / to_alpha + point.eta[j] / to_beta;
            b_x[j] = -res.x[j] - res.xi[j] / to_alpha + res.eta[j] / to_beta;
        }
        // D_y, b_y, b_lam per limit, and b_z.
        let d_y_diag: Vec<f64> = (0..m)
            .map(|i| self.d[i] + point.mu[i] / point.y[i])
            .collect();
        let b_y: Vec<f64> = (0..m).map(|i| -res.y[i] - res.mu[i] / point.y[i]).collect();
        let b_lambda: Vec<f64> = (0..m)
            .map(|i| -res.lambda[i] + res.s[i] / point.lambda[i])
            .collect();
        let b_z = -res.z - res.zeta / point.z;

        // The (m + 1) system in d_lam and d_z, row by row.
        let size = m + 1;
        let mut matrix = vec![0.0; size * size];
        let mut rhs = vec![0.0; size];
        for i in 0..m {
            for k in 0..m {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += gradients[i * n + j] * gradients[k * n + j] / d_x_diag[j];
                }
                matrix[i * size + k] = sum;
            }
            matrix[i * size + i] += 1.0 / d_y_diag[i] + point.s[i] / point.lambda[i];
            matrix[i * size + m] = self.a[i];
            matrix[m * size + i] = -self.a[i];
            let mut g_b_x = 0.0;
            for j in 0..n {
                g_b_x += gradients[i * n + j] * b_x[j] / d_x_diag[j];
            }
            rhs[i] = g_b_x - b_y[i] / d_y_diag[i] - b_lambda[i];
        }
        matrix[m * size + m] = point.zeta / point.z;
        rhs[m] = b_z;
        let solution = solve_dense(&mut matrix, &mut rhs, size)?;
        let d_lambda = &solution[..m];
        let d_z = solution[m];

        // Back substitution, in the order of the file header.
        let d_x: Vec<f64> = (0..n)
            .map(|j| {
                let g_t_d_lambda: f64 = (0..m).map(|i| gradients[i * n + j] * d_lambda[i]).sum();
                (b_x[j] - g_t_d_lambda) / d_x_diag[j]
            })
            .collect();
        let d_y: Vec<f64> = (0..m)
            .map(|i| (b_y[i] + d_lambda[i]) / d_y_diag[i])
            .collect();
        Ok(Point {
            xi: (0..n)
                .map(|j| (-res.xi[j] - point.xi[j] * d_x[j]) / (x[j] - self.alpha[j]))
                .collect(),
            eta: (0..n)
                .map(|j| (-res.eta[j] + point.eta[j] * d_x[j]) / (self.beta[j] - x[j]))
                .collect(),
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
    fn longest_step(&self, point: &Point, step: &Point) -> f64 {
        let mut t: f64 = 1.0;
        let mut limit = |value: f64, change: f64| {
            if change < 0.0 {
                t = t.min(-FRACTION_TO_BOUNDARY * value / change);
            }
        };
        for j in 0..self.n() {
            limit(point.x[j] - self.alpha[j], step.x[j]);
            limit(self.beta[j] - point.x[j], -step.x[j]);
            limit(point.xi[j], step.xi[j]);
            limit(point.eta[j], step.eta[j]);
        }
        for i in 0..self.m() {
            limit(point.y[i], step.y[i]);
            limit(point.lambda[i], step.lambda[i]);
            limit(point.mu[i], step.mu[i]);
            limit(point.s[i], step.s[i]);
        }
        limit(point.z, step.z);
        limit(point.zeta, step.zeta);
        t
    }
}

/// point + t step.
fn moved(point: &Point, step: &Point, t: f64) -> Point {
    let add = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a + t * b).collect();
    Point {
        x: add(&point.x, &step.x),
        y: add(&point.y, &step.y),
        z: point.z + t * step.z,
        lambda: add(&point.lambda, &step.lambda),
        xi: add(&point.xi, &step.xi),
        eta: add(&point.eta, &step.eta),
        mu: add(&point.mu, &step.mu),
        zeta: point.zeta + t * step.zeta,
        s: add(&point.s, &step.s),
    }
}

/// Solves the subproblem. Returns `solver_not_converged` if one value of epsilon needs more than
/// `NEWTON_STEPS_PER_EPSILON` Newton steps.
pub fn solve(sub: &Subproblem<'_>) -> Result<Solution, EngineError> {
    let mut point = sub.start();
    let mut epsilon = EPSILON_START;
    loop {
        let mut steps = 0;
        let mut res = sub.residual(&point, epsilon);
        while res.largest() > 0.9 * epsilon {
            if steps == NEWTON_STEPS_PER_EPSILON {
                return Err(EngineError::new(
                    ErrorCode::SolverNotConverged,
                    "the MMA subproblem solver did not reach its tolerance",
                )
                .with_number("epsilon", epsilon)
                .with_number("residual", res.largest())
                .with_integer("newton_steps", steps));
            }
            let step = sub.newton_step(&point, &res)?;
            let mut t = sub.longest_step(&point, &step);
            let norm_before = res.norm();
            let mut candidate = moved(&point, &step, t);
            let mut candidate_res = sub.residual(&candidate, epsilon);
            for _ in 0..STEP_HALVINGS {
                if candidate_res.norm() < norm_before {
                    break;
                }
                t *= 0.5;
                candidate = moved(&point, &step, t);
                candidate_res = sub.residual(&candidate, epsilon);
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
