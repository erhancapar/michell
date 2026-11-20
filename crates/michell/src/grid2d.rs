//! The 2D grid: `nx` by `ny` square elements of side `h` mm and thickness `t` mm, with y pointing up.
//!
//! Numbering (fixed in `docs/design/engine-boundary.md`):
//! - element `(i, j)` has index `i + nx * j`: x runs fastest;
//! - node `(i, j)` has index `i + (nx + 1) * j`, on the `(nx + 1)` by `(ny + 1)` points;
//! - node `n` has its x displacement at degree of freedom `2n` and its y displacement at `2n + 1`.
//!
//! Element `(i, j)` covers `i*h <= x <= (i+1)*h` and `j*h <= y <= (j+1)*h`.

use crate::error::EngineError;

/// The grid and which of its elements belong to the part.
#[derive(Clone, Debug)]
pub struct Grid2d {
    nx: usize,
    ny: usize,
    h: f64,
    t: f64,
    /// One entry per element: `false` means the element is outside the part and takes part in no equation.
    active: Vec<bool>,
}

impl Grid2d {
    /// A grid of `nx` by `ny` elements. `active` has one entry per element, in element order.
    pub fn new(
        nx: usize,
        ny: usize,
        h: f64,
        t: f64,
        active: Vec<bool>,
    ) -> Result<Grid2d, EngineError> {
        if nx == 0 || ny == 0 {
            return Err(EngineError::invalid_input(
                "the grid needs at least one element in x and in y",
            )
            .with_integer("nx", nx)
            .with_integer("ny", ny));
        }
        if !(h.is_finite() && h > 0.0) {
            return Err(
                EngineError::invalid_input("the element size h must be a positive number")
                    .with_number("h", h),
            );
        }
        if !(t.is_finite() && t > 0.0) {
            return Err(
                EngineError::invalid_input("the thickness t must be a positive number")
                    .with_number("t", t),
            );
        }
        let n_elements = element_count(nx, ny)?;
        if active.len() != n_elements {
            return Err(
                EngineError::invalid_input("the active mask needs one entry per element")
                    .with_integer("expected", n_elements)
                    .with_integer("got", active.len()),
            );
        }
        Ok(Grid2d {
            nx,
            ny,
            h,
            t,
            active,
        })
    }

    /// A grid whose elements are all active.
    pub fn full(nx: usize, ny: usize, h: f64, t: f64) -> Result<Grid2d, EngineError> {
        let n_elements = element_count(nx, ny)?;
        Grid2d::new(nx, ny, h, t, vec![true; n_elements])
    }

    pub fn nx(&self) -> usize {
        self.nx
    }

    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Side of one square element, in mm.
    pub fn h(&self) -> f64 {
        self.h
    }

    /// Thickness of the plate, in mm.
    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn n_elements(&self) -> usize {
        self.nx * self.ny
    }

    pub fn n_nodes(&self) -> usize {
        (self.nx + 1) * (self.ny + 1)
    }

    /// Two degrees of freedom (x and y displacement) per node.
    pub fn n_dofs(&self) -> usize {
        2 * self.n_nodes()
    }

    /// One entry per element: whether it belongs to the part.
    pub fn active(&self) -> &[bool] {
        &self.active
    }

    pub fn is_active(&self, element: usize) -> bool {
        self.active[element]
    }

    pub fn element_index(&self, i: usize, j: usize) -> usize {
        i + self.nx * j
    }

    pub fn node_index(&self, i: usize, j: usize) -> usize {
        i + (self.nx + 1) * j
    }

    /// The grid position `(i, j)` of an element.
    pub fn element_position(&self, element: usize) -> (usize, usize) {
        (element % self.nx, element / self.nx)
    }

    /// The four nodes of an element, counter-clockwise from the lower left:
    /// lower left, lower right, upper right, upper left. `quad4` uses the same order.
    pub fn element_nodes(&self, element: usize) -> [usize; 4] {
        let (i, j) = self.element_position(element);
        [
            self.node_index(i, j),
            self.node_index(i + 1, j),
            self.node_index(i + 1, j + 1),
            self.node_index(i, j + 1),
        ]
    }

    /// The eight degrees of freedom of an element: x and y of each node, nodes in `element_nodes` order.
    pub fn element_dofs(&self, element: usize) -> [usize; 8] {
        let nodes = self.element_nodes(element);
        [
            2 * nodes[0],
            2 * nodes[0] + 1,
            2 * nodes[1],
            2 * nodes[1] + 1,
            2 * nodes[2],
            2 * nodes[2] + 1,
            2 * nodes[3],
            2 * nodes[3] + 1,
        ]
    }
}

/// nx * ny, refused when the grid is so large that its degree-of-freedom count, 2 (nx+1)(ny+1), does
/// not fit in a machine integer.
fn element_count(nx: usize, ny: usize) -> Result<usize, EngineError> {
    let dofs = nx
        .checked_add(1)
        .zip(ny.checked_add(1))
        .and_then(|(nodes_x, nodes_y)| nodes_x.checked_mul(nodes_y))
        .and_then(|nodes| nodes.checked_mul(2));
    match dofs {
        Some(_) => Ok(nx * ny),
        None => Err(
            EngineError::invalid_input("the grid is too large to number")
                .with_integer("nx", nx)
                .with_integer("ny", ny),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbering_runs_x_fastest() {
        let grid = Grid2d::full(3, 2, 1.0, 1.0).unwrap();
        assert_eq!(grid.n_elements(), 6);
        assert_eq!(grid.n_nodes(), 12);
        assert_eq!(grid.n_dofs(), 24);
        assert_eq!(grid.element_index(2, 1), 5);
        assert_eq!(grid.element_position(5), (2, 1));
        // Element (2, 1): nodes (2,1), (3,1), (3,2), (2,2) on a grid 4 nodes wide.
        assert_eq!(grid.element_nodes(5), [6, 7, 11, 10]);
        assert_eq!(grid.element_dofs(5), [12, 13, 14, 15, 22, 23, 20, 21]);
    }

    #[test]
    fn wrong_input_is_refused() {
        assert!(Grid2d::full(0, 2, 1.0, 1.0).is_err());
        assert!(Grid2d::full(2, 2, -1.0, 1.0).is_err());
        assert!(Grid2d::full(2, 2, 1.0, f64::NAN).is_err());
        assert!(Grid2d::new(2, 2, 1.0, 1.0, vec![true; 3]).is_err());
        assert!(Grid2d::full(1 << 40, 1 << 40, 1.0, 1.0).is_err());
    }
}
