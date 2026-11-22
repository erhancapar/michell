//! The minimum member size as a limit: the geometric constraint on the solid phase of Zhou, Lazarov, Wang
//! and Sigmund, "Minimum length scale in topology optimization by geometric constraints", Comput. Methods
//! Appl. Mech. Engrg. 293 (2015) 266-282 (equation numbers below are those of that paper).
//!
//! The paper works on three fields: the design variables, the filtered field rho~ (a cone filter, eq. 6)
//! and the projected field rho_bar = P(rho~) (the tanh projection at the threshold eta_i, eq. 7). Inside a
//! solid member the filtered field has a ridge, where its gradient is zero. If the filtered field reaches
//! at least eta_e on every such ridge, every member is at least as wide as a member whose filtered field
//! just reaches eta_e (section 1, condition (i), eq. 1). The ridges are found by the indicator (eq. 8)
//!
//!   I_e = rho_bar_e exp(-c |grad rho~|_e^2),
//!
//! which is near 1 on solid elements where the filtered field is flat and falls off where it slopes, and
//! the constraint (eqs. 10 and 12) is
//!
//!   g = (1/n) sum_e I_e [min(rho~_e - eta_e, 0)]^2  <=  eps,
//!
//! with n the number of elements. Here the sum and n run over the active elements, and eta_e is given per
//! element (the paper's eta_e is one number for the whole domain): an element whose threshold is 0 adds
//! nothing, since rho~ is never negative. The paper relaxes the
//! equality g = 0 of eq. 10 to g <= eps (eq. 12), because with a finite c the indicator is not exactly 0
//! away from the ridges; it gives eps for each filter radius in its figure 8.
//!
//! The constant c and the units of the gradient. Section 2.2 and the appendix measure the gradient per
//! element side (the element size is 1 there): the numerical error of the gradient on a ridge is about
//! 1 / r^2 with r = R / h the filter radius in elements, and c = r^4 is the paper's choice. This file
//! therefore takes the gradient per element side, whatever the element size in mm: for element e and one
//! direction, with e- and e+ its neighbours before and after it in that direction,
//!
//!   (rho~_{e+} - rho~_{e-}) / 2   when both are active elements of the grid (central difference),
//!   rho~_{e+} - rho~_e            when only e+ is,
//!   rho~_e - rho~_{e-}            when only e- is,
//!   0                             when neither is,
//!
//! and |grad rho~|^2 is the sum of the squares over the directions (x and y on a 2D grid, x, y and z on a
//! 3D grid). The paper says only that the gradient is taken by finite differences; the stencil above is
//! this library's. A 2D grid is a grid one element thick in z, so it is the same code with no third term.
//!
//! The derivative with respect to the filtered field. With G_e = |grad rho~|_e^2, E_e = exp(-c G_e) and
//! m_e = min(rho~_e - eta_e, 0),
//!
//!   dg/drho~_k = (1/n) (P'(rho~_k) E_k m_k^2 + 2 rho_bar_k E_k m_k)
//!              + sum_e (1/n) rho_bar_e m_e^2 E_e (-2 c) sum_d (D_d rho~)_e d(D_d rho~)_e/drho~_k,
//!
//! the last line over the elements e whose difference stencil contains k. It is accumulated element by
//! element on one thread (the work is a few operations per element), so the result is the same on every
//! run. Elements with rho~_e >= eta_e add nothing to the value or to the derivative (m_e = 0, and m_e^2 has
//! a zero derivative there).
//!
//! The constraint on the void phase (eqs. 9 and 11, a minimum size of holes) is not built.

use crate::error::EngineError;
use crate::grid2d::Grid2d;
use crate::grid3d::Grid3d;
use crate::projection::Projection;

/// The solid-phase member-size constraint on one grid.
#[derive(Clone, Debug)]
pub struct MemberSize {
    nx: usize,
    ny: usize,
    nz: usize,
    active: Vec<bool>,
    /// The projection that gives rho_bar from the filtered field (eq. 7).
    projection: Projection,
    /// eta_e, one per element: the value the filtered field must reach on the ridge of every solid member.
    thresholds: Vec<f64>,
    /// c of the indicator (eq. 8).
    c: f64,
}

impl MemberSize {
    /// The constraint on a 2D grid.
    pub fn on_grid2d(
        grid: &Grid2d,
        projection: Projection,
        thresholds: Vec<f64>,
        c: f64,
    ) -> Result<MemberSize, EngineError> {
        MemberSize::new(
            grid.nx(),
            grid.ny(),
            1,
            grid.active().to_vec(),
            projection,
            thresholds,
            c,
        )
    }

    /// The constraint on a 3D grid.
    pub fn on_grid3d(
        grid: &Grid3d,
        projection: Projection,
        thresholds: Vec<f64>,
        c: f64,
    ) -> Result<MemberSize, EngineError> {
        MemberSize::new(
            grid.nx(),
            grid.ny(),
            grid.nz(),
            grid.active().to_vec(),
            projection,
            thresholds,
            c,
        )
    }

    fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        active: Vec<bool>,
        projection: Projection,
        thresholds: Vec<f64>,
        c: f64,
    ) -> Result<MemberSize, EngineError> {
        if thresholds.len() != active.len() {
            return Err(EngineError::invalid_input(
                "the thresholds eta_e need one entry per element",
            )
            .with_integer("expected", active.len())
            .with_integer("got", thresholds.len()));
        }
        if let Some(element) = thresholds
            .iter()
            .position(|t| !(t.is_finite() && (0.0..=1.0).contains(t)))
        {
            return Err(EngineError::invalid_input(
                "each threshold eta_e must lie between 0 and 1",
            )
            .with_integer("element", element)
            .with_number("threshold", thresholds[element]));
        }
        if !(c.is_finite() && c >= 0.0) {
            return Err(
                EngineError::invalid_input("c must be a number at least 0").with_number("c", c)
            );
        }
        Ok(MemberSize {
            nx,
            ny,
            nz,
            active,
            projection,
            thresholds,
            c,
        })
    }

    pub fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }

    pub fn c(&self) -> f64 {
        self.c
    }

    pub fn n_elements(&self) -> usize {
        self.active.len()
    }

    /// g and dg/drho~ (one value per element, 0 on inactive elements) at the filtered field `filtered`.
    /// A grid without active elements gives 0.
    pub fn evaluate(&self, filtered: &[f64]) -> Result<(f64, Vec<f64>), EngineError> {
        self.check_field(filtered)?;
        let n_active = self.active.iter().filter(|&&a| a).count();
        let mut gradient = vec![0.0; self.n_elements()];
        if n_active == 0 {
            return Ok((0.0, gradient));
        }
        let per_element = 1.0 / n_active as f64;
        let mut sum = 0.0;
        for element in 0..self.n_elements() {
            if !self.active[element] {
                continue;
            }
            let value = filtered[element];
            let m = (value - self.thresholds[element]).min(0.0);
            if m == 0.0 {
                continue;
            }
            let differences = self.differences(element);
            let mut squared = 0.0;
            for (plus, minus, factor) in differences.iter().flatten() {
                let d = factor * (filtered[*plus] - filtered[*minus]);
                squared += d * d;
            }
            let decay = (-self.c * squared).exp();
            let rho_bar = self.projection.value(value);
            sum += rho_bar * decay * m * m;
            gradient[element] += per_element
                * (self.projection.derivative(value) * decay * m * m + 2.0 * rho_bar * decay * m);
            let a = per_element * rho_bar * m * m * decay * (-2.0 * self.c);
            for (plus, minus, factor) in differences.iter().flatten() {
                let d = factor * (filtered[*plus] - filtered[*minus]);
                gradient[*plus] += a * d * factor;
                gradient[*minus] -= a * d * factor;
            }
        }
        Ok((sum * per_element, gradient))
    }

    /// For each direction (x, y, z), the difference of the file header as (e+, e-, factor), the
    /// difference being factor (rho~_{e+} - rho~_{e-}); `None` when the element has no active neighbour
    /// in that direction.
    fn differences(&self, element: usize) -> [Option<(usize, usize, f64)>; 3] {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let i = element % nx;
        let j = (element / nx) % ny;
        let k = element / (nx * ny);
        let steps = [(i, nx, 1), (j, ny, nx), (k, nz, nx * ny)];
        steps.map(|(position, count, step)| {
            let before = position > 0 && self.active[element - step];
            let after = position + 1 < count && self.active[element + step];
            match (before, after) {
                (true, true) => Some((element + step, element - step, 0.5)),
                (false, true) => Some((element + step, element, 1.0)),
                (true, false) => Some((element, element - step, 1.0)),
                (false, false) => None,
            }
        })
    }

    fn check_field(&self, values: &[f64]) -> Result<(), EngineError> {
        if values.len() != self.n_elements() {
            return Err(EngineError::invalid_input(
                "the filtered field needs one entry per element",
            )
            .with_integer("expected", self.n_elements())
            .with_integer("got", values.len()));
        }
        if let Some(element) = values.iter().position(|v| !v.is_finite()) {
            return Err(EngineError::invalid_input(
                "the filtered field holds a value that is not a finite number",
            )
            .with_integer("element", element)
            .with_number("value", values[element]));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smooth_field(nx: usize, ny: usize, nz: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(nx * ny * nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y, z) = (i as f64, j as f64, k as f64);
                    values.push(
                        0.55 + 0.3 * (0.21 * x + 0.1).sin() * (0.17 * y + 0.4).cos() + 0.02 * z,
                    );
                }
            }
        }
        values
    }

    /// Central differences of g against the derivative, relative to the derivative's largest entry.
    fn derivative_error(constraint: &MemberSize, field: &[f64]) -> f64 {
        let (_, gradient) = constraint.evaluate(field).unwrap();
        let step = 1e-6;
        let largest = gradient.iter().fold(0.0_f64, |a, g| a.max(g.abs()));
        assert!(largest > 0.0);
        let mut worst = 0.0_f64;
        for e in 0..field.len() {
            let mut up = field.to_vec();
            let mut down = field.to_vec();
            up[e] += step;
            down[e] -= step;
            let difference = (constraint.evaluate(&up).unwrap().0
                - constraint.evaluate(&down).unwrap().0)
                / (2.0 * step);
            worst = worst.max((difference - gradient[e]).abs() / largest);
        }
        worst
    }

    #[test]
    fn a_uniform_field_has_the_closed_form_value() {
        // No slope anywhere: I = P(v), g = P(v) (v - eta_e)^2 for v below eta_e, 0 at or above it.
        let grid = Grid3d::full(5, 4, 3, 2.0).unwrap();
        let projection = Projection::new(8.0, 0.5).unwrap();
        let constraint = MemberSize::on_grid3d(&grid, projection, vec![0.8; 60], 4096.0).unwrap();
        let (g, gradient) = constraint.evaluate(&[0.6; 60]).unwrap();
        let expected = projection.value(0.6) * 0.2 * 0.2;
        assert!((g - expected).abs() < 1e-15);
        let sum: f64 = gradient.iter().sum();
        let expected_sum = projection.derivative(0.6) * 0.04 - 2.0 * projection.value(0.6) * 0.2;
        assert!((sum - expected_sum).abs() < 1e-13);
        assert_eq!(constraint.evaluate(&[0.85; 60]).unwrap().0, 0.0);
    }

    #[test]
    fn each_element_is_judged_against_its_own_threshold() {
        // A uniform field 0.6 with the threshold 0.8 on every element but 0 on a third of them: those add
        // nothing, so g is two thirds of the value with 0.8 everywhere, and their derivative is 0.
        let grid = Grid3d::full(5, 4, 3, 1.0).unwrap();
        let projection = Projection::new(8.0, 0.5).unwrap();
        let mut thresholds = vec![0.8; 60];
        for t in thresholds.iter_mut().step_by(3) {
            *t = 0.0;
        }
        let some = MemberSize::on_grid3d(&grid, projection, thresholds, 4096.0).unwrap();
        let every = MemberSize::on_grid3d(&grid, projection, vec![0.8; 60], 4096.0).unwrap();
        let (g, gradient) = some.evaluate(&[0.6; 60]).unwrap();
        assert!((g - 2.0 / 3.0 * every.evaluate(&[0.6; 60]).unwrap().0).abs() < 1e-15);
        assert!(gradient.iter().step_by(3).all(|&d| d == 0.0));
        assert!(gradient.iter().skip(1).step_by(3).all(|&d| d != 0.0));
        let mut thresholds = smooth_field(9, 7, 5);
        for t in thresholds.iter_mut() {
            *t = (*t + 0.1).min(1.0);
        }
        let varying = MemberSize::on_grid3d(
            &Grid3d::full(9, 7, 5, 1.0).unwrap(),
            projection,
            thresholds,
            16.0,
        )
        .unwrap();
        assert!(derivative_error(&varying, &smooth_field(9, 7, 5)) < 1e-6);
    }

    #[test]
    fn the_derivative_matches_central_differences_in_2d_and_3d() {
        let projection = Projection::new(8.0, 0.5).unwrap();
        let (nx, ny, nz) = (9, 7, 5);
        let mut active = vec![true; nx * ny * nz];
        for k in 1..3 {
            for j in 2..4 {
                for i in 3..5 {
                    active[i + nx * (j + ny * k)] = false;
                }
            }
        }
        let solid = Grid3d::new(nx, ny, nz, 1.0, active).unwrap();
        let constraint =
            MemberSize::on_grid3d(&solid, projection, vec![0.7; nx * ny * nz], 16.0).unwrap();
        assert!(derivative_error(&constraint, &smooth_field(nx, ny, nz)) < 1e-6);

        let mut active = vec![true; nx * ny];
        active[4 + nx * 3] = false;
        let plane = Grid2d::new(nx, ny, 1.0, 1.0, active).unwrap();
        let constraint =
            MemberSize::on_grid2d(&plane, projection, vec![0.7; nx * ny], 16.0).unwrap();
        assert!(derivative_error(&constraint, &smooth_field(nx, ny, 1)) < 1e-6);
    }

    #[test]
    fn a_2d_grid_equals_a_3d_grid_one_element_thick() {
        let projection = Projection::new(16.0, 0.5).unwrap();
        let plane = Grid2d::full(8, 6, 1.0, 1.0).unwrap();
        let flat = Grid3d::full(8, 6, 1, 1.0).unwrap();
        let field = smooth_field(8, 6, 1);
        let a = MemberSize::on_grid2d(&plane, projection, vec![0.75; 48], 81.0).unwrap();
        let b = MemberSize::on_grid3d(&flat, projection, vec![0.75; 48], 81.0).unwrap();
        assert_eq!(a.evaluate(&field).unwrap(), b.evaluate(&field).unwrap());
    }

    #[test]
    fn differences_are_one_sided_at_the_edge_of_the_active_region() {
        let projection = Projection::new(8.0, 0.5).unwrap();
        // A row of four with the last outside: element 0 looks forward, 1 both ways, 2 backward.
        let grid = Grid2d::new(4, 1, 1.0, 1.0, vec![true, true, true, false]).unwrap();
        let constraint = MemberSize::on_grid2d(&grid, projection, vec![1.0; 4], 1.0).unwrap();
        assert_eq!(constraint.differences(0)[0], Some((1, 0, 1.0)));
        assert_eq!(constraint.differences(1)[0], Some((2, 0, 0.5)));
        assert_eq!(constraint.differences(2)[0], Some((2, 1, 1.0)));
        assert_eq!(constraint.differences(1)[1], None);
        assert_eq!(constraint.differences(1)[2], None);
    }

    #[test]
    fn wrong_input_is_refused() {
        let projection = Projection::new(8.0, 0.5).unwrap();
        let grid = Grid2d::full(3, 2, 1.0, 1.0).unwrap();
        assert!(MemberSize::on_grid2d(&grid, projection, vec![1.2; 6], 1.0).is_err());
        assert!(MemberSize::on_grid2d(&grid, projection, vec![0.8; 6], -1.0).is_err());
        assert!(MemberSize::on_grid2d(&grid, projection, vec![0.8; 6], f64::NAN).is_err());
        assert!(MemberSize::on_grid2d(&grid, projection, vec![0.8; 5], 1.0).is_err());
        let mut one_wrong = vec![0.8; 6];
        one_wrong[3] = -0.1;
        assert!(MemberSize::on_grid2d(&grid, projection, one_wrong, 1.0).is_err());
        let constraint = MemberSize::on_grid2d(&grid, projection, vec![0.8; 6], 1.0).unwrap();
        assert!(constraint.evaluate(&[0.5; 5]).is_err());
        assert!(
            constraint
                .evaluate(&[0.5, 0.5, f64::NAN, 0.5, 0.5, 0.5])
                .is_err()
        );
    }
}
