//! The density filter on the 3D grid: the filtered density of an element is a weighted average of the
//! design variables around it, with weights that fall linearly to zero at the filter radius (a cone),
//!
//!   rho_e = sum_j w_ej x_j / W_e,      w_ej = max(0, R - |c_e - c_j|),      W_e = sum_j w_ej,
//!
//! where c_e is the centre of element e, R the radius in mm, and both sums run over the active elements
//! inside the grid. The filter of `density_filter` (the 2D grid), with the same sources: Bruns and
//! Tortorelli, "Topology optimization of non-linear elastic structures and compliant mechanisms", CMAME
//! 190 (2001), and Bourdin, "Filters in topology optimization", IJNME 50 (2001).
//!
//! The weights are not stored per element. The grid is regular, so w_ej depends only on the offset
//! (di, dj, dk) from e to j: one stencil of weights by offset, `(2 reach + 1)^3` numbers, serves every
//! element. Per element only W_e is stored, computed once from the stencil and the active mask.
//!
//! The reverse (the chain rule back to x) is the transpose,
//!
//!   dg/dx_j = sum_e w_ej (dg/drho_e / W_e),
//!
//! and since the weight of an offset equals the weight of the opposite offset (w_ej = w_je), it too is a
//! gather over the stencil around j: first s_e = dg/drho_e / W_e for every element, then each x_j sums the
//! stencil's weights times s over its active neighbours.
//!
//! Threads. Both directions write one element per piece (`Threads::for_each_piece`), each summed over the
//! stencil in a fixed order (k, then j, then i, each ascending), so the numbers do not depend on the
//! thread count. W_e is summed in the same order as the forward sum, so a field of ones gives exactly 1
//! and a field in [0, 1] stays in [0, 1] without a rounding error past 1.
//!
//! Only active elements take part: an inactive element is never averaged in, its own density is 0, and
//! its entry of the reverse is 0.
//!
//! Faces and inactive elements can be closed (the default, as above) or empty (`FilterEdges`, see
//! `density_filter`): the weights that fall on an empty face's padding or on empty inactive elements are
//! added to W_e after the sum over the active elements, in a fixed order per element, so the closed
//! filter's numbers are the same whether or not the choice is made.

use crate::density_filter::{FilterEdges, padding_reach};
use crate::error::EngineError;
use crate::grid3d::Grid3d;
use crate::threads::Threads;

/// The density filter on one 3D grid.
#[derive(Clone, Debug)]
pub struct DensityFilter3d {
    grid: Grid3d,
    /// The radius R, mm.
    radius: f64,
    /// The largest offset along one direction with a positive weight, in elements.
    reach: usize,
    /// The weight of offset (di, dj, dk), each from -reach to reach, at index
    /// `(di + reach) + side (dj + reach) + side^2 (dk + reach)` with `side = 2 reach + 1`; 0 where the
    /// offset lies at or beyond the radius.
    stencil: Vec<f64>,
    /// W_e = sum_j w_ej over the active neighbours inside the grid, plus the weights on void when a face
    /// or the inactive elements are empty; 0 for an inactive element.
    weight_sums: Vec<f64>,
    threads: Threads,
}

impl DensityFilter3d {
    /// The filter of radius `radius` mm on `grid`, on `threads` threads, every face and the inactive
    /// elements closed. A radius at or below the element size leaves every element with itself as its
    /// only neighbour, so the filter changes nothing.
    pub fn new(grid: &Grid3d, radius: f64, threads: usize) -> Result<DensityFilter3d, EngineError> {
        DensityFilter3d::with_edges(grid, radius, threads, FilterEdges::closed())
    }

    /// The filter of radius `radius` mm on `grid`, on `threads` threads, with `edges` saying what lies
    /// beyond each face and in the inactive elements.
    pub fn with_edges(
        grid: &Grid3d,
        radius: f64,
        threads: usize,
        edges: FilterEdges,
    ) -> Result<DensityFilter3d, EngineError> {
        if !(radius.is_finite() && radius > 0.0) {
            return Err(
                EngineError::invalid_input("the filter radius must be a positive number")
                    .with_number("radius", radius),
            );
        }
        let threads = Threads::new(threads)?;
        let h = grid.h();
        // An offset d elements along one direction can only have a positive weight if d h < R, that is
        // d <= ceil(R / h) - 1. Never more than the grid itself.
        let largest = grid.nx().max(grid.ny()).max(grid.nz());
        let reach = ((radius / h).ceil() - 1.0).max(0.0).min(largest as f64) as usize;
        let side = 2 * reach + 1;
        let mut stencil = vec![0.0; side * side * side];
        for dk in 0..side {
            for dj in 0..side {
                for di in 0..side {
                    let (a, b, c) = (
                        di as f64 - reach as f64,
                        dj as f64 - reach as f64,
                        dk as f64 - reach as f64,
                    );
                    let weight = radius - h * (a * a + b * b + c * c).sqrt();
                    if weight > 0.0 {
                        stencil[di + side * (dj + side * dk)] = weight;
                    }
                }
            }
        }
        let mut filter = DensityFilter3d {
            grid: grid.clone(),
            radius,
            reach,
            stencil,
            weight_sums: Vec::new(),
            threads,
        };
        let ones = vec![1.0; grid.n_elements()];
        filter.weight_sums = filter.gather(&ones);
        if !edges.all_closed() {
            let void_reach = padding_reach(radius, h, largest)?;
            let size = [grid.nx(), grid.ny(), grid.nz()];
            let weight = |di: isize, dj: isize, dk: isize| {
                radius - h * ((di * di + dj * dj + dk * dk) as f64).sqrt()
            };
            let is_active =
                |i: usize, j: usize, k: usize| grid.is_active(grid.element_index(i, j, k));
            filter
                .threads
                .update(&mut filter.weight_sums, |element, sum| {
                    if grid.is_active(element) {
                        let (i, j, k) = grid.element_position(element);
                        *sum += edges.void_weight([i, j, k], size, void_reach, weight, is_active);
                    }
                });
        }
        Ok(filter)
    }

    /// The radius, mm.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// The side of one element, mm.
    pub fn element_size(&self) -> f64 {
        self.grid.h()
    }

    /// The number of threads every call runs on.
    pub fn threads(&self) -> usize {
        self.threads.count()
    }

    pub fn n_elements(&self) -> usize {
        self.grid.n_elements()
    }

    /// rho = the filtered field of `x`, one value per element; 0 on inactive elements.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.check_field(x, "x")?;
        let mut rho = self.gather(x);
        let weight_sums = &self.weight_sums;
        self.threads.update(&mut rho, |element, value| {
            if weight_sums[element] > 0.0 {
                *value /= weight_sums[element];
            }
        });
        Ok(rho)
    }

    /// The chain rule back to x: given dg/drho (`grad`), returns dg/dx_j = sum_e w_ej grad_e / W_e.
    pub fn backward(&self, grad: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.check_field(grad, "grad")?;
        let mut shares = vec![0.0; self.n_elements()];
        let weight_sums = &self.weight_sums;
        self.threads.update(&mut shares, |element, share| {
            if weight_sums[element] > 0.0 {
                *share = grad[element] / weight_sums[element];
            }
        });
        Ok(self.gather(&shares))
    }

    /// out_e = sum over the active elements j within the radius of e of w_ej field_j, for every active
    /// element e (0 for an inactive one), summed in the order k, j, i, each ascending.
    fn gather(&self, field: &[f64]) -> Vec<f64> {
        let grid = &self.grid;
        let (nx, ny, nz) = (grid.nx(), grid.ny(), grid.nz());
        let reach = self.reach;
        let side = 2 * reach + 1;
        let mut out = vec![0.0; grid.n_elements()];
        self.threads.for_each_piece(&mut out, 1, |element, value| {
            if !grid.is_active(element) {
                return;
            }
            let (i, j, k) = grid.element_position(element);
            let mut sum = 0.0;
            for k2 in k.saturating_sub(reach)..=(k + reach).min(nz - 1) {
                let dk = k2 + reach - k;
                for j2 in j.saturating_sub(reach)..=(j + reach).min(ny - 1) {
                    let dj = j2 + reach - j;
                    for i2 in i.saturating_sub(reach)..=(i + reach).min(nx - 1) {
                        let weight = self.stencil[(i2 + reach - i) + side * (dj + side * dk)];
                        let other = grid.element_index(i2, j2, k2);
                        if weight > 0.0 && grid.is_active(other) {
                            sum += weight * field[other];
                        }
                    }
                }
            }
            value[0] = sum;
        });
        out
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
    use crate::density_filter::DensityFilter;
    use crate::grid2d::Grid2d;

    fn field(n: usize, seed: usize) -> Vec<f64> {
        (0..n)
            .map(|e| (((e + seed) * 7919) % 1009) as f64 / 1009.0)
            .collect()
    }

    fn grid_with_a_pocket() -> Grid3d {
        let (nx, ny, nz) = (9, 7, 6);
        let mut active = vec![true; nx * ny * nz];
        for k in 2..4 {
            for j in 1..4 {
                for i in 3..6 {
                    active[i + nx * (j + ny * k)] = false;
                }
            }
        }
        Grid3d::new(nx, ny, nz, 1.5, active).unwrap()
    }

    #[test]
    fn weights_on_a_row_of_three_elements() {
        // R = 1.5 h. In units of h an element's own weight is 1.5, a direct neighbour's 0.5.
        let grid = Grid3d::full(3, 1, 1, 2.0).unwrap();
        let filter = DensityFilter3d::new(&grid, 3.0, 1).unwrap();
        let rho = filter.forward(&[1.0, 0.0, 0.0]).unwrap();
        assert!((rho[0] - 1.5 / 2.0).abs() < 1e-15);
        assert!((rho[1] - 0.5 / 2.5).abs() < 1e-15);
        assert_eq!(rho[2], 0.0);
    }

    #[test]
    fn one_layer_equals_the_2d_filter() {
        let (nx, ny) = (11, 8);
        let mut active = vec![true; nx * ny];
        active[3 + nx * 4] = false;
        active[4 + nx * 4] = false;
        let flat = Grid3d::new(nx, ny, 1, 1.0, active.clone()).unwrap();
        let plane = Grid2d::new(nx, ny, 1.0, 1.0, active).unwrap();
        let x = field(nx * ny, 3);
        let a = DensityFilter3d::new(&flat, 3.3, 1).unwrap();
        let b = DensityFilter::new(&plane, 3.3).unwrap();
        let (fa, fb) = (a.forward(&x).unwrap(), b.forward(&x).unwrap());
        let (ga, gb) = (a.backward(&x).unwrap(), b.backward(&x).unwrap());
        for e in 0..nx * ny {
            assert!((fa[e] - fb[e]).abs() < 1e-14);
            assert!((ga[e] - gb[e]).abs() < 1e-14);
        }
    }

    #[test]
    fn a_uniform_field_stays_uniform_and_ones_stay_at_most_one() {
        let grid = grid_with_a_pocket();
        let filter = DensityFilter3d::new(&grid, 3.7, 1).unwrap();
        let n = grid.n_elements();
        let rho = filter.forward(&vec![0.37; n]).unwrap();
        let ones = filter.forward(&vec![1.0; n]).unwrap();
        for e in 0..n {
            if grid.is_active(e) {
                assert!((rho[e] - 0.37).abs() < 1e-14);
                assert_eq!(ones[e], 1.0);
            } else {
                assert_eq!(rho[e], 0.0);
            }
        }
    }

    #[test]
    fn backward_is_the_transpose_of_forward() {
        let grid = grid_with_a_pocket();
        let filter = DensityFilter3d::new(&grid, 3.7, 1).unwrap();
        let n = grid.n_elements();
        let x: Vec<f64> = field(n, 1)
            .iter()
            .enumerate()
            .map(|(e, v)| if grid.is_active(e) { *v } else { 0.0 })
            .collect();
        let g: Vec<f64> = field(n, 5).iter().map(|v| v - 0.5).collect();
        let fx = filter.forward(&x).unwrap();
        let ftg = filter.backward(&g).unwrap();
        let left: f64 = fx.iter().zip(&g).map(|(a, b)| a * b).sum();
        let right: f64 = x.iter().zip(&ftg).map(|(a, b)| a * b).sum();
        assert!((left - right).abs() < 1e-12 * left.abs());
    }

    fn mixed_edges() -> FilterEdges {
        use crate::density_filter::FilterEdge::{Closed, Empty};
        FilterEdges {
            faces: [[Closed, Empty], [Empty, Empty], [Closed, Closed]],
            inactive: Empty,
        }
    }

    #[test]
    fn with_closed_z_faces_one_layer_equals_the_2d_filter_with_the_same_edges() {
        let (nx, ny) = (11, 8);
        let mut active = vec![true; nx * ny];
        active[3 + nx * 4] = false;
        active[4 + nx * 4] = false;
        let flat = Grid3d::new(nx, ny, 1, 1.0, active.clone()).unwrap();
        let plane = Grid2d::new(nx, ny, 1.0, 1.0, active).unwrap();
        let x = field(nx * ny, 3);
        let a = DensityFilter3d::with_edges(&flat, 3.3, 1, mixed_edges()).unwrap();
        let b = DensityFilter::with_edges(&plane, 3.3, mixed_edges()).unwrap();
        let (fa, fb) = (a.forward(&x).unwrap(), b.forward(&x).unwrap());
        let (ga, gb) = (a.backward(&x).unwrap(), b.backward(&x).unwrap());
        for e in 0..nx * ny {
            assert!((fa[e] - fb[e]).abs() < 1e-14);
            assert!((ga[e] - gb[e]).abs() < 1e-14);
        }
    }

    #[test]
    fn with_empty_edges_backward_is_the_transpose_and_threads_change_nothing() {
        let grid = grid_with_a_pocket();
        let mut edges = mixed_edges();
        edges.faces[2][0] = crate::density_filter::FilterEdge::Empty;
        let filter = DensityFilter3d::with_edges(&grid, 3.7, 1, edges).unwrap();
        let n = grid.n_elements();
        let x = field(n, 1);
        let g: Vec<f64> = field(n, 5).iter().map(|v| v - 0.5).collect();
        let (fx, ftg) = (filter.forward(&x).unwrap(), filter.backward(&g).unwrap());
        let left: f64 = fx.iter().zip(&g).map(|(a, b)| a * b).sum();
        let right: f64 = x.iter().zip(&ftg).map(|(a, b)| a * b).sum();
        assert!((left - right).abs() < 1e-12 * left.abs());
        for count in [2, 5] {
            let many = DensityFilter3d::with_edges(&grid, 3.7, count, edges).unwrap();
            assert_eq!(many.forward(&x).unwrap(), fx);
            assert_eq!(many.backward(&g).unwrap(), ftg);
        }
        let ones = filter.forward(&vec![1.0; n]).unwrap();
        assert!(ones.iter().all(|&r| r <= 1.0));
    }

    #[test]
    fn the_numbers_do_not_depend_on_the_thread_count() {
        let grid = Grid3d::full(17, 13, 11, 1.0).unwrap();
        let x = field(grid.n_elements(), 2);
        let one = DensityFilter3d::new(&grid, 3.2, 1).unwrap();
        let (f1, b1) = (one.forward(&x).unwrap(), one.backward(&x).unwrap());
        for count in [2, 5, 12] {
            let many = DensityFilter3d::new(&grid, 3.2, count).unwrap();
            assert_eq!(many.forward(&x).unwrap(), f1);
            assert_eq!(many.backward(&x).unwrap(), b1);
        }
    }

    #[test]
    fn wrong_input_is_refused() {
        let grid = Grid3d::full(3, 2, 2, 1.0).unwrap();
        assert!(DensityFilter3d::new(&grid, 0.0, 1).is_err());
        assert!(DensityFilter3d::new(&grid, f64::NAN, 1).is_err());
        assert!(DensityFilter3d::new(&grid, 1.5, 0).is_err());
        let filter = DensityFilter3d::new(&grid, 1.5, 1).unwrap();
        assert!(filter.forward(&[0.5; 5]).is_err());
        let mut bad = vec![0.5; 12];
        bad[4] = f64::INFINITY;
        assert!(filter.backward(&bad).is_err());
    }
}
