//! The material law shared by the 2D and the 3D analysis: an isotropic linear-elastic material whose
//! modulus follows the density by modified SIMP.
//!
//! E(rho) = E_min + rho^p (E_0 - E_min) (Sigmund, "Morphology-based black and white filters for topology
//! optimization", Struct. Multidisc. Optim. 33 (2007); the form used by the 88-line code of Andreassen et
//! al., 2011).

use crate::error::EngineError;
use crate::quad4;

/// The material constants and the SIMP penalty.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// E_0, the modulus of solid material, MPa.
    pub youngs_modulus: f64,
    pub poisson_ratio: f64,
    /// E_min, the modulus left in an empty element (keeps K invertible), MPa. Must be above 0.
    pub e_min: f64,
    /// p, the SIMP penalty.
    pub penalty: f64,
}

impl Material {
    /// Refuses constants that give no valid material: E_0 and E_min positive with E_min below E_0, nu
    /// strictly between -1 and 0.5 (the same range in plane stress and in 3D), a penalty of at least 1.
    pub(crate) fn check(&self) -> Result<(), EngineError> {
        quad4::check_elastic_constants(self.youngs_modulus, self.poisson_ratio)?;
        let e0 = self.youngs_modulus;
        if !(self.e_min.is_finite() && self.e_min > 0.0 && self.e_min < e0) {
            return Err(EngineError::invalid_input(
                "e_min must be above 0 and below youngs_modulus",
            )
            .with_number("e_min", self.e_min)
            .with_number("youngs_modulus", e0));
        }
        if !(self.penalty.is_finite() && self.penalty >= 1.0) {
            return Err(
                EngineError::invalid_input("penalty must be a number of at least 1")
                    .with_number("penalty", self.penalty),
            );
        }
        Ok(())
    }

    /// E(rho) = E_min + rho^p (E_0 - E_min).
    pub fn modulus(&self, rho: f64) -> f64 {
        self.e_min + rho.powf(self.penalty) * (self.youngs_modulus - self.e_min)
    }

    /// dE/drho = p rho^(p-1) (E_0 - E_min).
    pub fn modulus_derivative(&self, rho: f64) -> f64 {
        self.penalty * rho.powf(self.penalty - 1.0) * (self.youngs_modulus - self.e_min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steel() -> Material {
        Material {
            youngs_modulus: 210000.0,
            poisson_ratio: 0.3,
            e_min: 1e-9 * 210000.0,
            penalty: 3.0,
        }
    }

    #[test]
    fn modulus_runs_from_e_min_to_e_0() {
        let material = steel();
        assert_eq!(material.modulus(0.0), material.e_min);
        assert!((material.modulus(1.0) - 210000.0).abs() < 1e-9);
        // p rho^(p-1) (E_0 - E_min) at rho = 0.5: 3 * 0.25 * (E_0 - E_min)
        let expected = 0.75 * (210000.0 - material.e_min);
        assert!((material.modulus_derivative(0.5) - expected).abs() < 1e-9);
    }

    #[test]
    fn wrong_constants_are_refused() {
        assert!(steel().check().is_ok());
        for broken in [
            Material {
                poisson_ratio: 0.5,
                ..steel()
            },
            Material {
                e_min: 0.0,
                ..steel()
            },
            Material {
                penalty: 0.5,
                ..steel()
            },
        ] {
            assert!(broken.check().is_err());
        }
    }
}
