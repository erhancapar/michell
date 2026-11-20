//! The projection: a smoothed step that pushes a density towards 0 or 1 around a threshold `eta`,
//!
//!   P(x) = (tanh(beta eta) + tanh(beta (x - eta))) / (tanh(beta eta) + tanh(beta (1 - eta))),
//!
//! with the sharpness `beta > 0` and the threshold `0 <= eta <= 1`. P(0) = 0 and P(1) = 1 for every beta
//! and eta; as beta grows, P approaches the step from 0 to 1 at x = eta, and for beta near 0 it approaches
//! the identity. Its derivative is
//!
//!   P'(x) = beta / cosh^2(beta (x - eta)) / (tanh(beta eta) + tanh(beta (1 - eta))),
//!
//! written with 1 / cosh^2 rather than 1 - tanh^2, which rounds to 0 far from the threshold while the
//! true value is still a positive number. Source: Wang, Lazarov and Sigmund, "On projection methods,
//! convergence and robust formulations in topology optimization", Struct. Multidisc. Optim. 43 (2011),
//! where this form is introduced; with the thresholds 0.5 + d, 0.5 and 0.5 - d it gives the eroded,
//! nominal and dilated designs of their robust formulation.
//!
//! The projection acts on each value on its own, so it needs no grid and works on a 2D or a 3D field
//! alike. The chain rule back through it multiplies each entry of the incoming gradient by P' at that
//! entry.

use crate::error::EngineError;

/// The projection with one sharpness and one threshold.
#[derive(Clone, Copy, Debug)]
pub struct Projection {
    beta: f64,
    eta: f64,
    /// tanh(beta (0 - eta)) = -tanh(beta eta), the value of tanh(beta (x - eta)) at x = 0.
    tanh_at_zero: f64,
    /// tanh(beta (1 - eta)) - tanh(beta (0 - eta)) = tanh(beta eta) + tanh(beta (1 - eta)).
    denominator: f64,
}

impl Projection {
    /// The projection with sharpness `beta` (a positive number) and threshold `eta` (between 0 and 1).
    pub fn new(beta: f64, eta: f64) -> Result<Projection, EngineError> {
        if !(beta.is_finite() && beta > 0.0) {
            return Err(
                EngineError::invalid_input("the sharpness beta must be a positive number")
                    .with_number("beta", beta),
            );
        }
        if !(eta.is_finite() && (0.0..=1.0).contains(&eta)) {
            return Err(
                EngineError::invalid_input("the threshold eta must lie between 0 and 1")
                    .with_number("eta", eta),
            );
        }
        let tanh_at_zero = (beta * (0.0 - eta)).tanh();
        // Positive: tanh(beta eta) and tanh(beta (1 - eta)) are at least 0, and they cannot both be 0,
        // since eta and 1 - eta are not both 0.
        let denominator = (beta * (1.0 - eta)).tanh() - tanh_at_zero;
        Ok(Projection {
            beta,
            eta,
            tanh_at_zero,
            denominator,
        })
    }

    /// The sharpness beta.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// The threshold eta.
    pub fn eta(&self) -> f64 {
        self.eta
    }

    /// P(x) of each entry.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, EngineError> {
        check_finite(x, "x")?;
        Ok(x.iter().map(|&value| self.value(value)).collect())
    }

    /// The chain rule back through the projection: entry e of the result is `grad[e] * P'(x[e])`.
    pub fn backward(&self, x: &[f64], grad: &[f64]) -> Result<Vec<f64>, EngineError> {
        check_finite(x, "x")?;
        check_finite(grad, "grad")?;
        if grad.len() != x.len() {
            return Err(
                EngineError::invalid_input("grad needs one entry per entry of x")
                    .with_integer("expected", x.len())
                    .with_integer("got", grad.len()),
            );
        }
        Ok(x.iter()
            .zip(grad)
            .map(|(&value, &g)| g * self.derivative(value))
            .collect())
    }

    /// P(x) of one value, written as (t(x) - t(0)) / (t(1) - t(0)) with t(x) = tanh(beta (x - eta)): the
    /// same function, since tanh is odd. In this form the numerator at x = 0 is a number minus itself and
    /// at x = 1 the denominator itself, so P(0) = 0 and P(1) = 1 hold exactly, whichever maths library
    /// computes tanh.
    pub fn value(&self, x: f64) -> f64 {
        ((self.beta * (x - self.eta)).tanh() - self.tanh_at_zero) / self.denominator
    }

    /// P'(x) of one value.
    pub fn derivative(&self, x: f64) -> f64 {
        let c = (self.beta * (x - self.eta)).cosh();
        self.beta / (c * c) / self.denominator
    }
}

fn check_finite(values: &[f64], name: &str) -> Result<(), EngineError> {
    if let Some(index) = values.iter().position(|v| !v.is_finite()) {
        return Err(EngineError::invalid_input(format!(
            "{name} holds a value that is not a finite number"
        ))
        .with_integer("entry", index)
        .with_number("value", values[index]));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_one_are_kept_exactly() {
        for &beta in &[0.1, 1.0, 8.0, 32.0, 500.0] {
            for &eta in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                let projection = Projection::new(beta, eta).unwrap();
                let ends = projection.forward(&[0.0, 1.0]).unwrap();
                assert_eq!(ends, vec![0.0, 1.0], "beta {beta}, eta {eta}");
            }
        }
    }

    #[test]
    fn the_threshold_is_the_midpoint_for_eta_one_half() {
        // With eta = 0.5 the projection is point-symmetric about (0.5, 0.5).
        let projection = Projection::new(8.0, 0.5).unwrap();
        assert!((projection.value(0.5) - 0.5).abs() < 1e-15);
        let (low, high) = (projection.value(0.3), projection.value(0.7));
        assert!((low + high - 1.0).abs() < 1e-15);
    }

    #[test]
    fn a_large_beta_approaches_the_step() {
        let projection = Projection::new(500.0, 0.25).unwrap();
        let values = projection.forward(&[0.2, 0.3]).unwrap();
        assert!(values[0] < 1e-10);
        assert!(values[1] > 1.0 - 1e-10);
    }

    #[test]
    fn a_small_beta_approaches_the_identity() {
        // P(x) = x + O(beta^2) as beta -> 0.
        let projection = Projection::new(1e-4, 0.3).unwrap();
        for &x in &[0.1, 0.45, 0.9] {
            assert!((projection.value(x) - x).abs() < 1e-8);
        }
    }

    #[test]
    fn the_derivative_matches_central_differences() {
        let projection = Projection::new(8.0, 0.75).unwrap();
        let step = 1e-6;
        for k in 1..20 {
            let x = k as f64 / 20.0;
            let difference =
                (projection.value(x + step) - projection.value(x - step)) / (2.0 * step);
            let exact = projection.derivative(x);
            assert!((difference - exact).abs() < 1e-7 * exact.max(1.0), "x {x}");
        }
    }

    #[test]
    fn backward_multiplies_by_the_derivative() {
        let projection = Projection::new(4.0, 0.4).unwrap();
        let x = [0.1, 0.4, 0.8];
        let grad = [2.0, -1.0, 0.5];
        let result = projection.backward(&x, &grad).unwrap();
        for e in 0..3 {
            assert_eq!(result[e], grad[e] * projection.derivative(x[e]));
        }
    }

    #[test]
    fn the_derivative_stays_positive_far_from_the_threshold() {
        // 1 - tanh^2 would round to 0 here; 1 / cosh^2 keeps the small positive value.
        let projection = Projection::new(64.0, 0.5).unwrap();
        let far = projection.derivative(1.0);
        assert!(far > 0.0 && far < 1e-20);
    }

    #[test]
    fn wrong_input_is_refused() {
        assert!(Projection::new(0.0, 0.5).is_err());
        assert!(Projection::new(-1.0, 0.5).is_err());
        assert!(Projection::new(f64::INFINITY, 0.5).is_err());
        assert!(Projection::new(8.0, -0.1).is_err());
        assert!(Projection::new(8.0, 1.5).is_err());
        assert!(Projection::new(8.0, f64::NAN).is_err());
        let projection = Projection::new(8.0, 0.5).unwrap();
        assert!(projection.forward(&[0.5, f64::NAN]).is_err());
        assert!(projection.backward(&[0.5, 0.5], &[1.0]).is_err());
        assert!(projection.backward(&[0.5], &[f64::INFINITY]).is_err());
    }
}
