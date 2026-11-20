//! The optimality-criteria (OC) update: one goal that falls as material is added (compliance) and one
//! limit that is linear in the density (mass or volume).
//!
//! The update is the one of the 88-line code (Andreassen, Clausen, Schevenels, Lazarov and Sigmund,
//! "Efficient topology optimization in MATLAB using 88 lines of code", Struct. Multidisc. Optim. 43, 2011,
//! the OC update and its bisection), going back to Bendsøe and Sigmund, "Topology Optimization" (2003):
//!
//!   B_e      = -(dg/dx_e) / (dV/dx_e) / lambda
//!   x_new_e  = clamp(x_e sqrt(B_e),  between  max(0, x_e - m)  and  min(1, x_e + m))
//!
//! with m the move limit. The multiplier lambda is found by bisection so that the limit is just met. The
//! limit is measured where it acts: on the density rho = filter(x_new), as sum_e a_e rho_e, with a_e the
//! limit's weight per element (for mass, the mass of a solid element).
//!
//! A goal that grows with material at an element (dg/dx_e > 0) gives B_e = 0 there: that element moves
//! down by the move limit.

use crate::density_filter::DensityFilter;
use crate::error::EngineError;
use crate::grid2d::Grid2d;

/// The bisection starts with the multiplier between these two values (the 88-line code).
pub const MULTIPLIER_LOW: f64 = 0.0;
pub const MULTIPLIER_HIGH: f64 = 1e9;

/// The bisection stops when (high - low) / (high + low) is no longer above this (the 88-line code).
pub const BISECTION_TOLERANCE: f64 = 1e-3;

/// The OC update on one grid.
#[derive(Clone, Debug)]
pub struct Oc {
    active: Vec<bool>,
    move_limit: f64,
}

impl Oc {
    /// `move_limit` is the largest change of one design variable in one update (0.2 in the 88-line code).
    pub fn new(grid: &Grid2d, move_limit: f64) -> Result<Oc, EngineError> {
        if !(move_limit.is_finite() && move_limit > 0.0 && move_limit <= 1.0) {
            return Err(
                EngineError::invalid_input("the move limit must be above 0 and at most 1")
                    .with_number("move_limit", move_limit),
            );
        }
        Ok(Oc {
            active: grid.active().to_vec(),
            move_limit,
        })
    }

    pub fn move_limit(&self) -> f64 {
        self.move_limit
    }

    /// The new design variables.
    ///
    /// - `x`: the design variables, in [0, 1];
    /// - `d_goal`, `d_limit`: the gradients of the goal and of the limit with respect to x (the filter
    ///   already applied backwards); `d_limit` must be positive on every active element;
    /// - `limit_weights`, `limit_bound`: the limit, sum_e a_e rho_e <= bound, written on the density;
    /// - `filter`: the density filter that turns x into rho, or `None` when rho = x.
    ///
    /// Inactive elements keep their value.
    pub fn update(
        &self,
        x: &[f64],
        d_goal: &[f64],
        d_limit: &[f64],
        limit_weights: &[f64],
        limit_bound: f64,
        filter: Option<&DensityFilter>,
    ) -> Result<Vec<f64>, EngineError> {
        self.check_inputs(x, d_goal, d_limit, limit_weights, limit_bound, filter)?;
        let mut x_new = x.to_vec();
        let mut low = MULTIPLIER_LOW;
        let mut high = MULTIPLIER_HIGH;
        while (high - low) / (low + high) > BISECTION_TOLERANCE {
            let middle = 0.5 * (high + low);
            for element in 0..x.len() {
                if !self.active[element] {
                    continue;
                }
                let b = (-d_goal[element]).max(0.0) / d_limit[element] / middle;
                let candidate = x[element] * b.sqrt();
                // max(0, max(x - m, min(1, min(x + m, candidate)))), in the order of the 88-line code.
                x_new[element] = candidate
                    .min(x[element] + self.move_limit)
                    .min(1.0)
                    .max(x[element] - self.move_limit)
                    .max(0.0);
            }
            let rho = match filter {
                Some(filter) => filter.forward(&x_new)?,
                None => x_new.clone(),
            };
            let measured: f64 = limit_weights.iter().zip(&rho).map(|(a, r)| a * r).sum();
            if measured > limit_bound {
                low = middle;
            } else {
                high = middle;
            }
        }
        Ok(x_new)
    }

    fn check_inputs(
        &self,
        x: &[f64],
        d_goal: &[f64],
        d_limit: &[f64],
        limit_weights: &[f64],
        limit_bound: f64,
        filter: Option<&DensityFilter>,
    ) -> Result<(), EngineError> {
        let n = self.active.len();
        for (values, name) in [
            (x, "x"),
            (d_goal, "d_goal"),
            (d_limit, "d_limit"),
            (limit_weights, "limit_weights"),
        ] {
            if values.len() != n {
                return Err(EngineError::invalid_input(format!(
                    "{name} needs one entry per element"
                ))
                .with_integer("expected", n)
                .with_integer("got", values.len()));
            }
            if let Some(element) = values.iter().position(|v| !v.is_finite()) {
                return Err(EngineError::invalid_input(format!(
                    "{name} holds a value that is not a finite number"
                ))
                .with_integer("element", element)
                .with_number("value", values[element]));
            }
        }
        if let Some(element) = x.iter().position(|v| !(*v >= 0.0 && *v <= 1.0)) {
            return Err(EngineError::invalid_input(
                "every design variable must lie between 0 and 1",
            )
            .with_integer("element", element)
            .with_number("x", x[element]));
        }
        let not_positive = (0..n).find(|&e| self.active[e] && d_limit[e] <= 0.0);
        if let Some(element) = not_positive {
            return Err(EngineError::invalid_input(
                "the OC update needs a limit that grows with material: d_limit must be positive on \
                 every active element",
            )
            .with_integer("element", element)
            .with_number("d_limit", d_limit[element]));
        }
        if !limit_bound.is_finite() {
            return Err(
                EngineError::invalid_input("the limit bound must be a finite number")
                    .with_number("limit_bound", limit_bound),
            );
        }
        if let Some(filter) = filter {
            if filter.active() != self.active.as_slice() {
                return Err(EngineError::invalid_input(
                    "the filter was built on a different grid or active mask than the OC update",
                )
                .with_integer("filter_elements", filter.n_elements())
                .with_integer("elements", n));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume(rho: &[f64]) -> f64 {
        rho.iter().sum()
    }

    #[test]
    fn the_limit_is_met_and_the_moves_are_bounded() {
        let grid = Grid2d::full(8, 4, 1.0, 1.0).unwrap();
        let filter = DensityFilter::new(&grid, 1.5).unwrap();
        let oc = Oc::new(&grid, 0.2).unwrap();
        let x = vec![0.5; 32];
        // A goal that gains most from material in the left half.
        let d_goal: Vec<f64> = (0..32)
            .map(|e| if e % 8 < 4 { -4.0 } else { -1.0 })
            .collect();
        let d_limit = filter.backward(&[1.0; 32]).unwrap();
        let bound = 16.0;
        let x_new = oc
            .update(&x, &d_goal, &d_limit, &[1.0; 32], bound, Some(&filter))
            .unwrap();
        let measured = volume(&filter.forward(&x_new).unwrap());
        // The bisection ends with the multiplier known to 1e-3; the volume is then close to the bound.
        assert!((measured - bound).abs() < 0.01 * bound);
        for (old, new) in x.iter().zip(&x_new) {
            assert!((new - old).abs() <= 0.2 + 1e-15);
            assert!((0.0..=1.0).contains(new));
        }
        assert!(x_new[0] > x_new[7]);
    }

    #[test]
    fn a_uniform_gradient_at_the_bound_leaves_x_nearly_unchanged() {
        let grid = Grid2d::full(5, 5, 1.0, 1.0).unwrap();
        let oc = Oc::new(&grid, 0.2).unwrap();
        let x = vec![0.4; 25];
        let x_new = oc
            .update(&x, &[-2.0; 25], &[1.0; 25], &[1.0; 25], 10.0, None)
            .unwrap();
        // x sqrt(B) = x exactly at the right multiplier; the bisection finds it to 1e-3, so x moves by
        // at most about half of that, relatively.
        assert!(x_new.iter().all(|v| (v - 0.4).abs() < 1e-3 * 0.4));
    }

    #[test]
    fn inactive_elements_keep_their_value() {
        let grid = Grid2d::new(3, 1, 1.0, 1.0, vec![true, false, true]).unwrap();
        let oc = Oc::new(&grid, 0.2).unwrap();
        let x_new = oc
            .update(
                &[0.5, 0.0, 0.5],
                &[-1.0, 0.0, -1.0],
                &[1.0, 0.0, 1.0],
                &[1.0, 1.0, 1.0],
                0.6,
                None,
            )
            .unwrap();
        assert_eq!(x_new[1], 0.0);
        assert!(x_new[0] < 0.5 && x_new[2] < 0.5);
    }

    #[test]
    fn wrong_input_is_refused() {
        let grid = Grid2d::full(2, 2, 1.0, 1.0).unwrap();
        assert!(Oc::new(&grid, 0.0).is_err());
        assert!(Oc::new(&grid, 1.5).is_err());
        let oc = Oc::new(&grid, 0.2).unwrap();
        let ones = [1.0; 4];
        // d_limit not positive
        assert!(
            oc.update(
                &[0.5; 4],
                &[-1.0; 4],
                &[1.0, 0.0, 1.0, 1.0],
                &ones,
                2.0,
                None
            )
            .is_err()
        );
        // x outside [0, 1]
        assert!(
            oc.update(&[0.5, 0.5, 1.5, 0.5], &[-1.0; 4], &ones, &ones, 2.0, None)
                .is_err()
        );
        // wrong length
        assert!(
            oc.update(&[0.5; 3], &[-1.0; 4], &ones, &ones, 2.0, None)
                .is_err()
        );
        // a filter from another grid
        let other = Grid2d::full(3, 2, 1.0, 1.0).unwrap();
        let filter = DensityFilter::new(&other, 1.5).unwrap();
        assert!(
            oc.update(&[0.5; 4], &[-1.0; 4], &ones, &ones, 2.0, Some(&filter))
                .is_err()
        );
    }
}
