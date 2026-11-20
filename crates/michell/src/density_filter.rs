//! The density filter: the physical density of an element is a weighted average of the design variables
//! around it, with weights that fall linearly to zero at the filter radius (a cone).
//!
//!   rho_e = sum_j w_ej x_j / sum_j w_ej,      w_ej = max(0, r - |c_e - c_j|)
//!
//! where c_e is the centre of element e and r the radius in mm. Source: Bruns and Tortorelli, "Topology
//! optimization of non-linear elastic structures and compliant mechanisms", CMAME 190 (2001), and Bourdin,
//! "Filters in topology optimization", IJNME 50 (2001); written as in the density filter of the 88-line code
//! (Andreassen, Clausen, Schevenels, Lazarov and Sigmund, Struct. Multidisc. Optim. 43, 2011).
//!
//! The filter is linear, so its reverse (the chain rule back to x) is its transpose:
//!
//!   dg/dx_j = sum_e (w_ej / sum_k w_ek) dg/drho_e.
//!
//! Only active elements take part: an inactive element is never averaged in, and its own density is 0.

use crate::error::EngineError;
use crate::grid2d::Grid2d;

/// The density filter on one grid, with its weights computed once.
///
/// The weights are stored row by row: the neighbours of element `e` (itself included) are
/// `neighbours[row_start[e]..row_start[e + 1]]`, with their weights at the same positions in `weights`.
#[derive(Clone, Debug)]
pub struct DensityFilter {
    radius: f64,
    active: Vec<bool>,
    row_start: Vec<usize>,
    neighbours: Vec<usize>,
    weights: Vec<f64>,
    /// sum_j w_ej for each element (0 for an inactive one).
    weight_sums: Vec<f64>,
}

impl DensityFilter {
    /// The filter of radius `radius` mm on `grid`. A radius below the element size leaves every element
    /// with itself as its only neighbour, so the filter changes nothing.
    pub fn new(grid: &Grid2d, radius: f64) -> Result<DensityFilter, EngineError> {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(
                EngineError::invalid_input("the filter radius must be a positive number")
                    .with_number("radius", radius),
            );
        }
        let (nx, ny, h) = (grid.nx(), grid.ny(), grid.h());
        // A neighbour d elements away along x or y can only have a positive weight if d h < r, that is
        // d <= ceil(r / h) - 1. Never more than the grid itself.
        let reach_f = ((radius / h).ceil() - 1.0).max(0.0);
        let reach = reach_f.min(nx.max(ny) as f64) as usize;

        let n_elements = grid.n_elements();
        let mut row_start = Vec::with_capacity(n_elements + 1);
        let mut neighbours = Vec::new();
        let mut weights = Vec::new();
        let mut weight_sums = Vec::with_capacity(n_elements);
        row_start.push(0);
        for element in 0..n_elements {
            let mut sum = 0.0;
            if grid.is_active(element) {
                let (i, j) = grid.element_position(element);
                for j2 in j.saturating_sub(reach)..=(j + reach).min(ny - 1) {
                    for i2 in i.saturating_sub(reach)..=(i + reach).min(nx - 1) {
                        let other = grid.element_index(i2, j2);
                        if !grid.is_active(other) {
                            continue;
                        }
                        let (di, dj) = (i.abs_diff(i2) as f64, j.abs_diff(j2) as f64);
                        let weight = radius - h * (di * di + dj * dj).sqrt();
                        if weight > 0.0 {
                            neighbours.push(other);
                            weights.push(weight);
                            sum += weight;
                        }
                    }
                }
            }
            weight_sums.push(sum);
            row_start.push(neighbours.len());
        }
        Ok(DensityFilter {
            radius,
            active: grid.active().to_vec(),
            row_start,
            neighbours,
            weights,
            weight_sums,
        })
    }

    /// The radius, mm.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    pub fn n_elements(&self) -> usize {
        self.active.len()
    }

    /// The active mask of the grid the filter was built on.
    pub fn active(&self) -> &[bool] {
        &self.active
    }

    /// rho = the filtered field of `x`, one value per element; 0 on inactive elements.
    ///
    /// Each density is summed in the same order as its weight sum, so a field of values in [0, 1] gives
    /// densities in [0, 1] exactly, without a rounding error pushing one past 1.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.check_field(x, "x")?;
        let mut rho = vec![0.0; self.n_elements()];
        for (element, value) in rho.iter_mut().enumerate() {
            if self.active[element] {
                let mut sum = 0.0;
                for k in self.row_start[element]..self.row_start[element + 1] {
                    sum += self.weights[k] * x[self.neighbours[k]];
                }
                *value = sum / self.weight_sums[element];
            }
        }
        Ok(rho)
    }

    /// The chain rule back to x: given dg/drho, returns dg/dx = H^T (dg/drho / weight sums).
    pub fn backward(&self, grad_rho: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.check_field(grad_rho, "grad")?;
        let mut grad_x = vec![0.0; self.n_elements()];
        for element in 0..self.n_elements() {
            if !self.active[element] {
                continue;
            }
            let share = grad_rho[element] / self.weight_sums[element];
            for k in self.row_start[element]..self.row_start[element + 1] {
                grad_x[self.neighbours[k]] += self.weights[k] * share;
            }
        }
        Ok(grad_x)
    }

    fn check_field(&self, values: &[f64], name: &str) -> Result<(), EngineError> {
        if values.len() != self.n_elements() {
            return Err(
                EngineError::invalid_input(format!("{name} needs one entry per element"))
                    .with_integer("expected", self.n_elements())
                    .with_integer("got", values.len()),
            );
        }
        if let Some(element) = values.iter().position(|v| !v.is_finite()) {
            return Err(EngineError::invalid_input(format!(
                "{name} holds a value that is not a finite number"
            ))
            .with_integer("element", element)
            .with_number("value", values[element]));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_on_a_row_of_three_elements() {
        // r = 1.5 h: an element's own weight is 1.5, a direct neighbour's 0.5, two elements away 0.
        let grid = Grid2d::full(3, 1, 2.0, 1.0).unwrap();
        let filter = DensityFilter::new(&grid, 3.0).unwrap();
        let rho = filter.forward(&[1.0, 0.0, 0.0]).unwrap();
        assert!((rho[0] - 1.5 / 2.0).abs() < 1e-15);
        assert!((rho[1] - 0.5 / 2.5).abs() < 1e-15);
        assert_eq!(rho[2], 0.0);
    }

    #[test]
    fn a_uniform_field_stays_uniform() {
        let grid = Grid2d::full(9, 5, 1.0, 1.0).unwrap();
        let filter = DensityFilter::new(&grid, 2.4).unwrap();
        let rho = filter.forward(&[0.37; 45]).unwrap();
        assert!(rho.iter().all(|r| (r - 0.37).abs() < 1e-14));
        let ones = filter.forward(&[1.0; 45]).unwrap();
        assert!(ones.iter().all(|&r| r <= 1.0));
    }

    #[test]
    fn a_radius_below_the_element_size_changes_nothing() {
        let grid = Grid2d::full(4, 3, 2.0, 1.0).unwrap();
        let filter = DensityFilter::new(&grid, 1.9).unwrap();
        let x: Vec<f64> = (0..12).map(|e| e as f64 / 12.0).collect();
        // Equal up to the rounding of (w x) / w.
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(p, q)| (p - q).abs() <= 1e-15);
        assert!(close(&filter.forward(&x).unwrap(), &x));
        assert!(close(&filter.backward(&x).unwrap(), &x));
    }

    #[test]
    fn backward_is_the_transpose_of_forward() {
        // <F x, g> = <x, F^T g> for any x and g.
        let grid = Grid2d::full(7, 4, 1.5, 1.0).unwrap();
        let filter = DensityFilter::new(&grid, 3.3).unwrap();
        let x: Vec<f64> = (0..28).map(|e| ((e * 7) % 11) as f64 / 11.0).collect();
        let g: Vec<f64> = (0..28).map(|e| ((e * 5) % 13) as f64 - 6.0).collect();
        let fx = filter.forward(&x).unwrap();
        let ftg = filter.backward(&g).unwrap();
        let left: f64 = fx.iter().zip(&g).map(|(a, b)| a * b).sum();
        let right: f64 = x.iter().zip(&ftg).map(|(a, b)| a * b).sum();
        assert!((left - right).abs() < 1e-12 * left.abs());
    }

    #[test]
    fn inactive_elements_are_not_averaged_in() {
        // Row of three, the middle one outside the part: the outer two only see themselves.
        let grid = Grid2d::new(3, 1, 1.0, 1.0, vec![true, false, true]).unwrap();
        let filter = DensityFilter::new(&grid, 1.5).unwrap();
        let rho = filter.forward(&[0.2, 1.0, 0.6]).unwrap();
        let grad = filter.backward(&[1.0, 1.0, 1.0]).unwrap();
        let expected_rho = [0.2, 0.0, 0.6];
        let expected_grad = [1.0, 0.0, 1.0];
        for e in 0..3 {
            assert!((rho[e] - expected_rho[e]).abs() <= 1e-15);
            assert!((grad[e] - expected_grad[e]).abs() <= 1e-15);
        }
    }

    #[test]
    fn wrong_input_is_refused() {
        let grid = Grid2d::full(3, 2, 1.0, 1.0).unwrap();
        assert!(DensityFilter::new(&grid, 0.0).is_err());
        assert!(DensityFilter::new(&grid, f64::NAN).is_err());
        let filter = DensityFilter::new(&grid, 1.5).unwrap();
        assert!(filter.forward(&[0.5; 5]).is_err());
        assert!(
            filter
                .backward(&[0.5, 0.5, 0.5, 0.5, 0.5, f64::INFINITY])
                .is_err()
        );
    }
}
