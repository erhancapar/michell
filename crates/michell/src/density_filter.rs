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
//!
//! What lies beyond a face of the grid, and in the inactive elements, is a choice (`FilterEdges`). By
//! default both are closed: the sums above run over the active elements inside the grid only, so a face
//! acts like a mirror. A face or the inactive region can instead be empty: its weights stay in the sum
//! W_e = sum_j w_ej with a density of 0, as if void elements lay there. This is the void padding of the
//! filter domain discussed by Clausen and Andreassen, "On filter boundary conditions in topology
//! optimization", Struct. Multidisc. Optim. 56 (2017), section 3.2.2: only the filter sees the padding;
//! the analysis does not.

use crate::error::EngineError;
use crate::grid2d::Grid2d;

/// What the filter takes to lie beyond one face of the grid, or in the inactive elements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterEdge {
    /// Nothing is averaged in from there: only the active elements inside the grid count, and each
    /// average is normalised over them. At a face this acts like a mirror (homogeneous Neumann), right
    /// for a symmetry plane and for a face that continues into a wall or a support. A uniform field
    /// stays uniform next to a closed face.
    #[default]
    Closed,
    /// Void lies there: its weights stay in the normalising sum with a density of 0, so material next to
    /// it is averaged with nothing. Right for a free face of the part. A uniform field drops next to it.
    Empty,
}

/// The choice for each face of the grid and for the inactive elements; all closed by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilterEdges {
    /// `faces[axis][0]` is the face at the low end of the axis (x-, y-, z-), `faces[axis][1]` the face at
    /// the high end (x+, y+, z+); axis 0 is x, 1 is y, 2 is z.
    pub faces: [[FilterEdge; 2]; 3],
    /// The inactive elements inside the grid.
    pub inactive: FilterEdge,
}

impl FilterEdges {
    /// Every face and the inactive elements closed: the filter of the sources, as it was before the
    /// choice existed.
    pub fn closed() -> FilterEdges {
        FilterEdges::default()
    }

    /// Whether everything is closed, so no weight falls on void.
    pub fn all_closed(&self) -> bool {
        *self == FilterEdges::default()
    }

    /// The weight that falls on void around the element at `position` of a grid of `size` elements per
    /// axis: the weights of the offsets that lie beyond an empty face and beyond no closed one (the
    /// padding reaches out across each empty face, and stops at the plane of each closed one), and, when
    /// the inactive elements are empty, of the offsets on inactive elements. `reach` is the largest offset
    /// along one axis with a positive weight, `weight` the weight of an offset in elements, `is_active`
    /// tells an active element from its position. Summed in the order k, j, i, each ascending.
    pub(crate) fn void_weight(
        &self,
        position: [usize; 3],
        size: [usize; 3],
        reach: usize,
        weight: impl Fn(isize, isize, isize) -> f64,
        is_active: impl Fn(usize, usize, usize) -> bool,
    ) -> f64 {
        let reach = reach as isize;
        let range = |axis: usize| {
            let (at, n) = (position[axis] as isize, size[axis] as isize);
            let low = match self.faces[axis][0] {
                FilterEdge::Empty => at - reach,
                FilterEdge::Closed => (at - reach).max(0),
            };
            let high = match self.faces[axis][1] {
                FilterEdge::Empty => at + reach,
                FilterEdge::Closed => (at + reach).min(n - 1),
            };
            (low, high)
        };
        let ([i0, i1], [j0, j1], [k0, k1]) = {
            let (x, y, z) = (range(0), range(1), range(2));
            ([x.0, x.1], [y.0, y.1], [z.0, z.1])
        };
        let beyond = |axis: usize, value: isize| value < 0 || value >= size[axis] as isize;
        let padding = beyond(0, i0) || beyond(0, i1) || beyond(1, j0) || beyond(1, j1);
        let padding = padding || beyond(2, k0) || beyond(2, k1);
        if !padding && self.inactive == FilterEdge::Closed {
            return 0.0;
        }
        let [i, j, k] = position.map(|p| p as isize);
        let mut sum = 0.0;
        for k2 in k0..=k1 {
            for j2 in j0..=j1 {
                for i2 in i0..=i1 {
                    let outside = beyond(0, i2) || beyond(1, j2) || beyond(2, k2);
                    let on_void = outside
                        || (self.inactive == FilterEdge::Empty
                            && !is_active(i2 as usize, j2 as usize, k2 as usize));
                    if on_void {
                        let w = weight(i2 - i, j2 - j, k2 - k);
                        if w > 0.0 {
                            sum += w;
                        }
                    }
                }
            }
        }
        sum
    }
}

/// The largest offset along one axis with a positive weight, `ceil(radius / h) - 1`, for the padding
/// beyond an empty face. A radius that reaches further than `limit` elements is refused.
pub(crate) fn padding_reach(radius: f64, h: f64, limit: usize) -> Result<usize, EngineError> {
    let reach = ((radius / h).ceil() - 1.0).max(0.0);
    if reach > limit as f64 {
        return Err(EngineError::invalid_input(
            "with an empty face or empty inactive elements the filter radius must not reach further than \
             the grid is long",
        )
        .with_number("radius", radius)
        .with_integer("largest number of elements along one axis", limit));
    }
    Ok(reach as usize)
}

/// The density filter on one grid, with its weights computed once.
///
/// The weights are stored row by row: the neighbours of element `e` (itself included) are
/// `neighbours[row_start[e]..row_start[e + 1]]`, with their weights at the same positions in `weights`.
#[derive(Clone, Debug)]
pub struct DensityFilter {
    radius: f64,
    /// The side of one element, mm.
    h: f64,
    active: Vec<bool>,
    row_start: Vec<usize>,
    neighbours: Vec<usize>,
    weights: Vec<f64>,
    /// W_e = sum_j w_ej for each element, the weights on void included (0 for an inactive one).
    weight_sums: Vec<f64>,
}

impl DensityFilter {
    /// The filter of radius `radius` mm on `grid`, every face and the inactive elements closed. A radius
    /// below the element size leaves every element with itself as its only neighbour, so the filter
    /// changes nothing.
    pub fn new(grid: &Grid2d, radius: f64) -> Result<DensityFilter, EngineError> {
        DensityFilter::with_edges(grid, radius, FilterEdges::closed())
    }

    /// The filter of radius `radius` mm on `grid`, with `edges` saying what lies beyond each face and in
    /// the inactive elements. The grid has no z faces, so those must be closed.
    pub fn with_edges(
        grid: &Grid2d,
        radius: f64,
        edges: FilterEdges,
    ) -> Result<DensityFilter, EngineError> {
        if edges.faces[2] != [FilterEdge::Closed; 2] {
            return Err(EngineError::invalid_input(
                "a 2D grid has no z faces; they cannot be empty",
            ));
        }
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
        let void_reach = if edges.all_closed() {
            0
        } else {
            padding_reach(radius, h, nx.max(ny))?
        };
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
            if grid.is_active(element) && !edges.all_closed() {
                let (i, j) = grid.element_position(element);
                sum += edges.void_weight(
                    [i, j, 0],
                    [nx, ny, 1],
                    void_reach,
                    |di, dj, _| radius - h * ((di * di + dj * dj) as f64).sqrt(),
                    |i2, j2, _| grid.is_active(grid.element_index(i2, j2)),
                );
            }
            weight_sums.push(sum);
            row_start.push(neighbours.len());
        }
        Ok(DensityFilter {
            radius,
            h,
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

    /// The side of one element, mm.
    pub fn element_size(&self) -> f64 {
        self.h
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
        // r = 1.5 h. In units of h an element's own weight is 1.5, a direct neighbour's 0.5, two elements
        // away 0 (in mm, with h = 2: 3 and 1; only the ratios enter the result).
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

    fn edges(x_minus: FilterEdge, y_minus: FilterEdge, inactive: FilterEdge) -> FilterEdges {
        let mut edges = FilterEdges::closed();
        edges.faces[0][0] = x_minus;
        edges.faces[1][0] = y_minus;
        edges.inactive = inactive;
        edges
    }

    #[test]
    fn all_closed_edges_give_the_same_numbers_bit_for_bit() {
        let mut active = vec![true; 9 * 7];
        active[3 + 9 * 3] = false;
        let grid = Grid2d::new(9, 7, 1.5, 1.0, active).unwrap();
        let x: Vec<f64> = (0..63).map(|e| ((e * 7) % 11) as f64 / 11.0).collect();
        let a = DensityFilter::new(&grid, 3.7).unwrap();
        let b = DensityFilter::with_edges(&grid, 3.7, FilterEdges::closed()).unwrap();
        assert_eq!(a.forward(&x).unwrap(), b.forward(&x).unwrap());
        assert_eq!(a.backward(&x).unwrap(), b.backward(&x).unwrap());
    }

    #[test]
    fn an_empty_face_keeps_its_weights_in_the_sum() {
        // r = 1.5 h on a row of three: beyond the empty x- face lies one void neighbour of weight 0.5.
        use FilterEdge::{Closed, Empty};
        let grid = Grid2d::full(3, 1, 1.0, 1.0).unwrap();
        let filter = DensityFilter::with_edges(&grid, 1.5, edges(Empty, Closed, Closed)).unwrap();
        let rho = filter.forward(&[1.0, 0.0, 0.0]).unwrap();
        assert!((rho[0] - 1.5 / 2.5).abs() < 1e-15);
        assert!((rho[1] - 0.5 / 2.5).abs() < 1e-15);
        // The x+ face is closed: the last element still sees only two.
        let rho = filter.forward(&[0.0, 0.0, 1.0]).unwrap();
        assert!((rho[2] - 1.5 / 2.0).abs() < 1e-15);
    }

    #[test]
    fn a_corner_counts_as_void_only_beyond_empty_faces() {
        // r = 1.5 h, element (0, 0), x- closed and y- empty: the void neighbours are (0, -1) and (1, -1);
        // (-1, -1) lies beyond the closed face too and does not count.
        use FilterEdge::{Closed, Empty};
        let grid = Grid2d::full(3, 3, 1.0, 1.0).unwrap();
        let ones = [1.0; 9];
        let filter = DensityFilter::with_edges(&grid, 1.5, edges(Closed, Empty, Closed)).unwrap();
        let inside = 1.5 + 2.0 * 0.5 + (1.5 - 2f64.sqrt());
        let void = 0.5 + (1.5 - 2f64.sqrt());
        let rho = filter.forward(&ones).unwrap();
        assert!((rho[0] - inside / (inside + void)).abs() < 1e-15);
        // Both faces empty: the diagonal neighbour beyond both counts as well.
        let filter = DensityFilter::with_edges(&grid, 1.5, edges(Empty, Empty, Closed)).unwrap();
        let void = 2.0 * 0.5 + 3.0 * (1.5 - 2f64.sqrt());
        let rho = filter.forward(&ones).unwrap();
        assert!((rho[0] - inside / (inside + void)).abs() < 1e-15);
    }

    #[test]
    fn empty_inactive_elements_are_averaged_in_as_void() {
        use FilterEdge::{Closed, Empty};
        let grid = Grid2d::new(3, 1, 1.0, 1.0, vec![true, false, true]).unwrap();
        let filter = DensityFilter::with_edges(&grid, 1.5, edges(Closed, Closed, Empty)).unwrap();
        let rho = filter.forward(&[0.2, 1.0, 0.6]).unwrap();
        assert!((rho[0] - 0.2 * 1.5 / 2.0).abs() < 1e-15);
        assert_eq!(rho[1], 0.0);
        assert!((rho[2] - 0.6 * 1.5 / 2.0).abs() < 1e-15);
    }

    #[test]
    fn with_empty_edges_a_uniform_field_drops_only_near_them_and_backward_is_the_transpose() {
        use FilterEdge::{Closed, Empty};
        let (nx, ny) = (12, 9);
        let mut active = vec![true; nx * ny];
        active[8 + nx * 6] = false;
        let grid = Grid2d::new(nx, ny, 1.0, 1.0, active).unwrap();
        let filter = DensityFilter::with_edges(&grid, 2.5, edges(Empty, Closed, Empty)).unwrap();
        let rho = filter.forward(&vec![0.37; nx * ny]).unwrap();
        for (e, &value) in rho.iter().enumerate() {
            let (i, j) = grid.element_position(e);
            let (di, dj) = (i.abs_diff(8), j.abs_diff(6));
            let near_void = i < 2 || di * di + dj * dj < 6;
            if !grid.is_active(e) {
                assert_eq!(value, 0.0);
            } else if near_void {
                assert!(value < 0.37 - 1e-3);
            } else {
                assert!((value - 0.37).abs() < 1e-14);
            }
        }
        let x: Vec<f64> = (0..nx * ny).map(|e| ((e * 7) % 11) as f64 / 11.0).collect();
        let g: Vec<f64> = (0..nx * ny).map(|e| ((e * 5) % 13) as f64 - 6.0).collect();
        let left: f64 = filter
            .forward(&x)
            .unwrap()
            .iter()
            .zip(&g)
            .map(|(a, b)| a * b)
            .sum();
        let right: f64 = x
            .iter()
            .zip(&filter.backward(&g).unwrap())
            .map(|(a, b)| a * b)
            .sum();
        assert!((left - right).abs() < 1e-12 * left.abs());
    }

    #[test]
    fn empty_z_faces_and_a_radius_past_the_grid_are_refused() {
        let grid = Grid2d::full(3, 2, 1.0, 1.0).unwrap();
        let mut edges = FilterEdges::closed();
        edges.faces[2][1] = FilterEdge::Empty;
        assert!(DensityFilter::with_edges(&grid, 1.5, edges).is_err());
        let mut edges = FilterEdges::closed();
        edges.faces[0][1] = FilterEdge::Empty;
        assert!(DensityFilter::with_edges(&grid, 10.0, edges).is_err());
        assert!(DensityFilter::new(&grid, 10.0).is_ok());
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
