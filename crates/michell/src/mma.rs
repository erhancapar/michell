//! The method of moving asymptotes (MMA): one goal and any number of limits, each a smooth function of the
//! design variables, with a lower and an upper bound on every variable.
//!
//! Sources: K. Svanberg, "The method of moving asymptotes: a new method for structural optimization",
//! Int. J. Numer. Meth. Engng 24 (1987) 359-373 (the method: convex approximations with asymptotes that
//! move from one iteration to the next), and K. Svanberg, "MMA and GCMMA: two methods for nonlinear
//! optimization" (2007), whose form of the method this file follows. The parts, in the order of `update`:
//!
//! Part 1. The problem form (2007 note, eq. 1.1): the limits are written f_i(x) <= 0 and the problem is
//!
//!   minimise f_0(x) + a0 z + sum_i (c_i y_i + d_i y_i^2 / 2)
//!   subject to f_i(x) - a_i z - y_i <= 0,  x_min <= x <= x_max,  y >= 0,  z >= 0.
//!
//! With a0 = 1, a_i = 0, d_i = 1 and c_i large (the note's choice for the standard problem, eq. 1.2; its
//! section 2 names 1000 as reasonably large), y_i and z are zero at the solution of a problem whose
//! limits can be met, and this is the original problem; y_i > 0 shows a limit that cannot be met.
//!
//! Part 2. The asymptotes l_j < x_j < u_j (2007 note, eqs 3.11 to 3.14; the rule goes back to the 1987
//! paper). In the first two iterations l_j = x_j - 0.5 (x_max_j - x_min_j) and
//! u_j = x_j + 0.5 (x_max_j - x_min_j).
//! Later, with x' and x'' the designs of the two previous iterations and l', u' the previous asymptotes,
//!
//!   l_j = x_j - gamma_j (x'_j - l'_j),    u_j = x_j + gamma_j (u'_j - x'_j),
//!
//! gamma_j = 0.7 if (x_j - x'_j)(x'_j - x''_j) < 0 (the variable oscillates: the asymptotes close in),
//! 1.2 if > 0 (it moves steadily: they open out), 1 if = 0; then l_j is kept between
//! x_j - 10 (x_max_j - x_min_j) and x_j - 0.01 (x_max_j - x_min_j), and u_j likewise above x_j.
//!
//! Part 3. The move limits (2007 note, eqs 3.6 and 3.7):
//!
//!   alpha_j = max( x_min_j,  l_j + 0.1 (x_j - l_j),  x_j - 0.5 (x_max_j - x_min_j) )
//!   beta_j  = min( x_max_j,  u_j - 0.1 (u_j - x_j),  x_j + 0.5 (x_max_j - x_min_j) )
//!
//! Part 4. The approximation of each function f_i at the current design x (1987 paper; in the form of
//! the 2007 note, eqs 3.2 to 3.5, whose small additions make it strictly convex):
//!
//!   f_i(x) ~ g_i(x) = sum_j ( p_ij / (u_j - x_j) + q_ij / (x_j - l_j) ) + r_i
//!   p_ij = (u_j - x_j)^2 ( 1.001 (df_i/dx_j)+ + 0.001 (df_i/dx_j)- + 1e-5 / (x_max_j - x_min_j) )
//!   q_ij = (x_j - l_j)^2 ( 0.001 (df_i/dx_j)+ + 1.001 (df_i/dx_j)- + 1e-5 / (x_max_j - x_min_j) )
//!   r_i  = f_i(x) - sum_j ( p_ij / (u_j - x_j) + q_ij / (x_j - l_j) )
//!
//! where (v)+ = max(0, v) and (v)- = max(0, -v). The approximation has the value and the gradient of
//! f_i at x (the tests check this), and it is convex.
//!
//! Part 5. The subproblem (the approximations in place of f_i, x between alpha and beta) is solved by
//! the primal-dual interior-point method of the 2007 note: see `mma_subproblem.rs` (which also says in
//! what units it writes each limit). Its x is the next design.
//!
//! The formulas and constants above were checked against the 2007 note. They are written from the
//! method's definition in the two papers; no other program's code was used.

use crate::error::EngineError;
use crate::mma_subproblem::{self, Subproblem};

/// Asymptotes of the first two iterations: this fraction of x_max - x_min on each side of x.
pub const ASYMPTOTE_START: f64 = 0.5;
/// gamma when a variable oscillates (the asymptotes move closer).
pub const ASYMPTOTE_CLOSER: f64 = 0.7;
/// gamma when a variable keeps moving the same way (the asymptotes move away).
pub const ASYMPTOTE_WIDER: f64 = 1.2;
/// The asymptotes stay at least this fraction of x_max - x_min away from x ...
pub const ASYMPTOTE_NEAREST: f64 = 0.01;
/// ... and at most this fraction.
pub const ASYMPTOTE_FARTHEST: f64 = 10.0;
/// alpha and beta lie at least this fraction of the way from the asymptote to x.
pub const ASYMPTOTE_MARGIN: f64 = 0.1;
/// The largest move of a variable in one iteration, as a fraction of x_max - x_min.
pub const MOVE_FRACTION: f64 = 0.5;
/// The weights of the positive and the negative part of a derivative in p_ij and q_ij.
const MAIN_WEIGHT: f64 = 1.001;
const SMALL_WEIGHT: f64 = 0.001;
/// The term that makes every approximation strictly convex, divided by x_max - x_min.
const CONVEXITY_TERM: f64 = 1e-5;
/// a0, a_i, c_i and d_i of the problem form (part 1 of the header).
pub const A0: f64 = 1.0;
pub const A: f64 = 0.0;
pub const C: f64 = 1000.0;
pub const D: f64 = 1.0;

/// The MMA update. It remembers the two previous designs and the previous asymptotes, so one `Mma`
/// belongs to one run.
#[derive(Clone, Debug)]
pub struct Mma {
    x_min: Vec<f64>,
    x_max: Vec<f64>,
    n_limits: usize,
    /// How many updates have been made.
    iteration: usize,
    /// The designs of the previous update (x') and of the one before (x'').
    x_previous: Vec<f64>,
    x_before_previous: Vec<f64>,
    /// The asymptotes of the previous update.
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Mma {
    /// An MMA update for variables between `x_min` and `x_max` (with `x_min < x_max` for each) and
    /// `n_limits` limits.
    pub fn new(x_min: Vec<f64>, x_max: Vec<f64>, n_limits: usize) -> Result<Mma, EngineError> {
        if x_min.len() != x_max.len() {
            return Err(
                EngineError::invalid_input("x_min and x_max need the same length")
                    .with_integer("x_min", x_min.len())
                    .with_integer("x_max", x_max.len()),
            );
        }
        if x_min.is_empty() {
            return Err(EngineError::invalid_input(
                "MMA needs at least one variable",
            ));
        }
        for j in 0..x_min.len() {
            if !(x_min[j].is_finite() && x_max[j].is_finite() && x_min[j] < x_max[j]) {
                return Err(EngineError::invalid_input(
                    "every variable needs finite bounds with x_min < x_max",
                )
                .with_integer("variable", j)
                .with_number("x_min", x_min[j])
                .with_number("x_max", x_max[j]));
            }
        }
        let n = x_min.len();
        Ok(Mma {
            x_min,
            x_max,
            n_limits,
            iteration: 0,
            x_previous: vec![0.0; n],
            x_before_previous: vec![0.0; n],
            lower: vec![0.0; n],
            upper: vec![0.0; n],
        })
    }

    pub fn n_variables(&self) -> usize {
        self.x_min.len()
    }

    pub fn n_limits(&self) -> usize {
        self.n_limits
    }

    /// How many updates have been made.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// The asymptotes l and u of the last update (zeros before the first).
    pub fn asymptotes(&self) -> (&[f64], &[f64]) {
        (&self.lower, &self.upper)
    }

    /// The next design.
    ///
    /// - `x`: the current design, within the bounds;
    /// - `goal`, `d_goal`: f_0(x) and its gradient;
    /// - `limits`: f_i(x) for each limit, with the limit written f_i(x) <= 0;
    /// - `d_limits`: the gradients of the limits, row by row (limit i, variable j at index i n + j).
    pub fn update(
        &mut self,
        x: &[f64],
        goal: f64,
        d_goal: &[f64],
        limits: &[f64],
        d_limits: &[f64],
    ) -> Result<Vec<f64>, EngineError> {
        self.check_inputs(x, goal, d_goal, limits, d_limits)?;
        let n = self.n_variables();
        let m = self.n_limits;

        let (lower, upper) = self.new_asymptotes(x);

        // Part 3: the move limits.
        let mut alpha = vec![0.0; n];
        let mut beta = vec![0.0; n];
        for j in 0..n {
            let range = self.x_max[j] - self.x_min[j];
            alpha[j] = self.x_min[j]
                .max(lower[j] + ASYMPTOTE_MARGIN * (x[j] - lower[j]))
                .max(x[j] - MOVE_FRACTION * range);
            beta[j] = self.x_max[j]
                .min(upper[j] - ASYMPTOTE_MARGIN * (upper[j] - x[j]))
                .min(x[j] + MOVE_FRACTION * range);
        }

        // Part 4: the approximations of the goal and of each limit.
        let (p0, q0, _) = self.approximation(x, goal, d_goal, &lower, &upper);
        let mut p = Vec::with_capacity(m * n);
        let mut q = Vec::with_capacity(m * n);
        let mut r = Vec::with_capacity(m);
        for i in 0..m {
            let (p_i, q_i, r_i) =
                self.approximation(x, limits[i], &d_limits[i * n..(i + 1) * n], &lower, &upper);
            p.extend(p_i);
            q.extend(q_i);
            r.push(r_i);
        }

        // Part 5: the subproblem.
        let a = vec![A; m];
        let c = vec![C; m];
        let d = vec![D; m];
        let solution = mma_subproblem::solve(&Subproblem {
            lower: &lower,
            upper: &upper,
            alpha: &alpha,
            beta: &beta,
            p0: &p0,
            q0: &q0,
            p: &p,
            q: &q,
            r: &r,
            a0: A0,
            a: &a,
            c: &c,
            d: &d,
        })?;

        // Only a successful update changes what the object remembers.
        self.x_before_previous = std::mem::replace(&mut self.x_previous, x.to_vec());
        self.lower = lower;
        self.upper = upper;
        self.iteration += 1;
        Ok(solution.x)
    }

    /// part 2 of the header: the asymptotes for the design x.
    fn new_asymptotes(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_variables();
        let mut lower = vec![0.0; n];
        let mut upper = vec![0.0; n];
        for j in 0..n {
            let range = self.x_max[j] - self.x_min[j];
            // `iteration` counts the updates already made: 0 and 1 mean this is the first or second.
            if self.iteration < 2 {
                lower[j] = x[j] - ASYMPTOTE_START * range;
                upper[j] = x[j] + ASYMPTOTE_START * range;
            } else {
                let trend =
                    (x[j] - self.x_previous[j]) * (self.x_previous[j] - self.x_before_previous[j]);
                let gamma = if trend < 0.0 {
                    ASYMPTOTE_CLOSER
                } else if trend > 0.0 {
                    ASYMPTOTE_WIDER
                } else {
                    1.0
                };
                lower[j] = x[j] - gamma * (self.x_previous[j] - self.lower[j]);
                upper[j] = x[j] + gamma * (self.upper[j] - self.x_previous[j]);
                lower[j] = lower[j]
                    .max(x[j] - ASYMPTOTE_FARTHEST * range)
                    .min(x[j] - ASYMPTOTE_NEAREST * range);
                upper[j] = upper[j]
                    .min(x[j] + ASYMPTOTE_FARTHEST * range)
                    .max(x[j] + ASYMPTOTE_NEAREST * range);
            }
        }
        (lower, upper)
    }

    /// part 4 of the header: p_j, q_j and r of one function with value `value` and gradient `gradient`
    /// at x.
    fn approximation(
        &self,
        x: &[f64],
        value: f64,
        gradient: &[f64],
        lower: &[f64],
        upper: &[f64],
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let n = self.n_variables();
        let mut p = vec![0.0; n];
        let mut q = vec![0.0; n];
        let mut r = value;
        for j in 0..n {
            let ux = upper[j] - x[j];
            let xl = x[j] - lower[j];
            let positive = gradient[j].max(0.0);
            let negative = (-gradient[j]).max(0.0);
            let convexity = CONVEXITY_TERM / (self.x_max[j] - self.x_min[j]);
            p[j] = ux * ux * (MAIN_WEIGHT * positive + SMALL_WEIGHT * negative + convexity);
            q[j] = xl * xl * (SMALL_WEIGHT * positive + MAIN_WEIGHT * negative + convexity);
            r -= p[j] / ux + q[j] / xl;
        }
        (p, q, r)
    }

    fn check_inputs(
        &self,
        x: &[f64],
        goal: f64,
        d_goal: &[f64],
        limits: &[f64],
        d_limits: &[f64],
    ) -> Result<(), EngineError> {
        let n = self.n_variables();
        let m = self.n_limits;
        for (values, name, expected) in [
            (x, "x", n),
            (d_goal, "d_goal", n),
            (limits, "limits", m),
            (d_limits, "d_limits", m * n),
        ] {
            if values.len() != expected {
                return Err(
                    EngineError::invalid_input(format!("{name} has the wrong length"))
                        .with_integer("expected", expected)
                        .with_integer("got", values.len()),
                );
            }
            if let Some(index) = values.iter().position(|v| !v.is_finite()) {
                return Err(EngineError::invalid_input(format!(
                    "{name} holds a value that is not a finite number"
                ))
                .with_integer("index", index)
                .with_number("value", values[index]));
            }
        }
        if !goal.is_finite() {
            return Err(
                EngineError::invalid_input("the goal is not a finite number")
                    .with_number("goal", goal),
            );
        }
        if let Some(j) = (0..n).find(|&j| !(x[j] >= self.x_min[j] && x[j] <= self.x_max[j])) {
            return Err(
                EngineError::invalid_input("every variable must lie within its bounds")
                    .with_integer("variable", j)
                    .with_number("x", x[j])
                    .with_number("x_min", self.x_min[j])
                    .with_number("x_max", self.x_max[j]),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The approximation of part 4 must have the value and the gradient of the function at x; this is the
    /// check that the formulas for p, q and r fit together.
    #[test]
    fn the_approximation_has_the_value_and_gradient_of_the_function() {
        let mma = Mma::new(vec![0.0; 3], vec![2.0; 3], 0).unwrap();
        let x = [0.5, 1.0, 1.7];
        let gradient = [2.5, -0.75, 0.0];
        let lower = [-0.3, 0.2, 1.0];
        let upper = [1.4, 2.6, 2.5];
        let (p, q, r) = mma.approximation(&x, 3.25, &gradient, &lower, &upper);
        let approx = |at: &[f64]| -> f64 {
            r + (0..3)
                .map(|j| p[j] / (upper[j] - at[j]) + q[j] / (at[j] - lower[j]))
                .sum::<f64>()
        };
        assert!((approx(&x) - 3.25).abs() < 1e-14);
        for j in 0..3 {
            let slope = p[j] / (upper[j] - x[j]).powi(2) - q[j] / (x[j] - lower[j]).powi(2);
            assert!((slope - gradient[j]).abs() < 1e-14);
            // Convex: the second derivative 2p/(u-x)^3 + 2q/(x-l)^3 is positive.
            let curvature =
                2.0 * p[j] / (upper[j] - x[j]).powi(3) + 2.0 * q[j] / (x[j] - lower[j]).powi(3);
            assert!(curvature > 0.0);
        }
    }

    #[test]
    fn the_asymptotes_close_in_on_oscillation_and_open_out_on_a_steady_move() {
        let mut mma = Mma::new(vec![0.0; 2], vec![1.0; 2], 0).unwrap();
        // Two updates set x' and x''; the third uses the rule with gamma.
        mma.update(&[0.5, 0.5], 1.0, &[1.0, -1.0], &[], &[])
            .unwrap();
        mma.update(&[0.4, 0.6], 1.0, &[1.0, -1.0], &[], &[])
            .unwrap();
        let (l1, u1) = (mma.asymptotes().0.to_vec(), mma.asymptotes().1.to_vec());
        assert!((l1[0] - (0.4 - 0.5)).abs() < 1e-15 && (u1[1] - (0.6 + 0.5)).abs() < 1e-15);
        // Variable 0 turns back (0.5 -> 0.4 -> 0.45): gamma 0.7. Variable 1 goes on (0.5 -> 0.6 -> 0.7):
        // gamma 1.2.
        mma.update(&[0.45, 0.7], 1.0, &[1.0, -1.0], &[], &[])
            .unwrap();
        let (l, u) = mma.asymptotes();
        assert!((l[0] - (0.45 - 0.7 * (0.4 - l1[0]))).abs() < 1e-15);
        assert!((u[0] - (0.45 + 0.7 * (u1[0] - 0.4))).abs() < 1e-15);
        assert!((l[1] - (0.7 - 1.2 * (0.6 - l1[1]))).abs() < 1e-15);
        assert!((u[1] - (0.7 + 1.2 * (u1[1] - 0.6))).abs() < 1e-15);
        assert_eq!(mma.iteration(), 3);
    }

    #[test]
    fn a_linear_limit_is_met_after_one_update() {
        // Minimise -(x0 + 2 x1) with x0 + x1 - 1 <= 0, 0 <= x <= 1: the answer is x = (0, 1). A linear
        // limit lies below its convex approximation, so the new design meets it whatever the step.
        let mut mma = Mma::new(vec![0.0; 2], vec![1.0; 2], 1).unwrap();
        let mut x = vec![0.5, 0.25];
        for _ in 0..30 {
            let limit = x[0] + x[1] - 1.0;
            x = mma
                .update(
                    &x,
                    -(x[0] + 2.0 * x[1]),
                    &[-1.0, -2.0],
                    &[limit],
                    &[1.0, 1.0],
                )
                .unwrap();
            assert!(x[0] + x[1] - 1.0 <= 1e-9);
        }
        assert!(x[0].abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6);
    }

    /// Many variables, a goal and a mass limit whose gradients are of the order 1/n, and a second limit
    /// that is violated and is far steeper than the goal on a fifth of the variables (gradient -0.5,
    /// thousands of times the goal's) and nearly flat elsewhere. The subproblem solver must reach its
    /// tolerance (see "The units of the limits" in `mma_subproblem.rs`), and the new design must satisfy
    /// the optimality conditions of the subproblem, checked here from the approximations themselves.
    #[test]
    fn a_violated_limit_far_steeper_than_the_goal_is_solved() {
        let n = 20_000;
        let spread = |j: usize, step: f64| (j as f64 * step) % 1.0;
        let x: Vec<f64> = (0..n).map(|j| 0.15 + 0.7 * spread(j, 0.618_034)).collect();
        let d_goal: Vec<f64> = (0..n)
            .map(|j| -(0.5 + spread(j, 0.754_878)) / n as f64)
            .collect();
        let mut d_limits = vec![1.0 / n as f64; n];
        d_limits.extend((0..n).map(|j| if j < n / 5 { -0.5 } else { 1e-9 }));
        let limits = [0.0, 20.0];
        let mut mma = Mma::new(vec![0.0; n], vec![1.0; n], 2).unwrap();
        let new_x = match mma.update(&x, 1.0, &d_goal, &limits, &d_limits) {
            Ok(new_x) => new_x,
            Err(error) => panic!("the update failed: {error}"),
        };

        // The approximations of part 4 and the move limits of part 3, as the update built them.
        let (lower, upper) = (mma.asymptotes().0.to_vec(), mma.asymptotes().1.to_vec());
        let (p0, q0, _) = mma.approximation(&x, 1.0, &d_goal, &lower, &upper);
        let mut terms = vec![];
        for i in 0..2 {
            let gradient = &d_limits[i * n..(i + 1) * n];
            terms.push(mma.approximation(&x, limits[i], gradient, &lower, &upper));
        }
        let slope = |p: &[f64], q: &[f64], j: usize| {
            p[j] / (upper[j] - new_x[j]).powi(2) - q[j] / (new_x[j] - lower[j]).powi(2)
        };
        // Both limits are met (y = 0: c = 1000 makes a violation far too dear here).
        for (p, q, r) in &terms {
            let value: f64 = r
                + (0..n)
                    .map(|j| p[j] / (upper[j] - new_x[j]) + q[j] / (new_x[j] - lower[j]))
                    .sum::<f64>();
            assert!(value <= 1e-8, "a limit is not met: {value}");
        }
        // Stationarity: on the variables strictly inside [alpha, beta] the slope of
        // g_0 + lambda_1 g_1 + lambda_2 g_2 is zero, with multipliers lambda >= 0 fitted by least squares.
        let inside: Vec<usize> = (0..n)
            .filter(|&j| {
                let alpha = (x[j] - 0.45).max(0.0);
                let beta = (x[j] + 0.45).min(1.0);
                new_x[j] - alpha > 1e-3 && beta - new_x[j] > 1e-3
            })
            .collect();
        assert!(inside.len() > n / 2);
        let g = |i: usize, j: usize| slope(&terms[i].0, &terms[i].1, j);
        let (mut m00, mut m01, mut m11, mut b0, mut b1) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &j in &inside {
            let (g0, g1, g2) = (slope(&p0, &q0, j), g(0, j), g(1, j));
            m00 += g1 * g1;
            m01 += g1 * g2;
            m11 += g2 * g2;
            b0 -= g1 * g0;
            b1 -= g2 * g0;
        }
        let det = m00 * m11 - m01 * m01;
        let lambda = [(b0 * m11 - b1 * m01) / det, (m00 * b1 - m01 * b0) / det];
        assert!(lambda[0] > 0.0 && lambda[1] > 0.0, "{lambda:?}");
        // What is left is the barrier term epsilon / (x - alpha) of the solver's last epsilon.
        let goal_slope = inside
            .iter()
            .fold(0.0_f64, |m, &j| m.max(slope(&p0, &q0, j).abs()));
        for &j in &inside {
            let stationarity = slope(&p0, &q0, j) + lambda[0] * g(0, j) + lambda[1] * g(1, j);
            assert!(
                stationarity.abs() < 1e-4 * goal_slope,
                "variable {j}: {stationarity}"
            );
        }
    }

    #[test]
    fn wrong_input_is_refused_and_leaves_the_state_alone() {
        assert!(Mma::new(vec![0.0], vec![0.0], 0).is_err());
        assert!(Mma::new(vec![0.0, 0.0], vec![1.0], 0).is_err());
        assert!(Mma::new(vec![], vec![], 0).is_err());
        let mut mma = Mma::new(vec![0.0; 2], vec![1.0; 2], 1).unwrap();
        assert!(
            mma.update(&[0.5, 1.5], 1.0, &[1.0, 1.0], &[0.0], &[1.0, 1.0])
                .is_err()
        );
        assert!(
            mma.update(&[0.5, 0.5], 1.0, &[1.0, 1.0], &[0.0], &[1.0])
                .is_err()
        );
        assert!(
            mma.update(&[0.5, 0.5], f64::NAN, &[1.0, 1.0], &[0.0], &[1.0, 1.0])
                .is_err()
        );
        assert_eq!(mma.iteration(), 0);
    }
}
