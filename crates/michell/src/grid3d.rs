//! The 3D grid: `nx` by `ny` by `nz` cube elements (voxels) of side `h` mm.
//!
//! Numbering (fixed in `docs/design/engine-boundary.md`):
//! - element `(i, j, k)` has index `i + nx * (j + ny * k)`: x runs fastest, then y, then z;
//! - node `(i, j, k)` has index `i + (nx + 1) * (j + (ny + 1) * k)`, on the `(nx + 1)` by `(ny + 1)` by
//!   `(nz + 1)` points;
//! - node `n` has its x, y and z displacement at degrees of freedom `3n`, `3n + 1` and `3n + 2`.
//!
//! Element `(i, j, k)` covers `i*h <= x <= (i+1)*h`, `j*h <= y <= (j+1)*h` and `k*h <= z <= (k+1)*h`.

use crate::error::EngineError;

/// The position of each of an element's eight nodes, as steps (0 or 1) in x, y and z from the element's
/// lowest corner. First the four nodes of the face at the lower z, counter-clockwise seen from above
/// (the order of the 2D element), then the four of the face at the upper z in the same order. `hex8` uses
/// the same order.
pub const NODE_STEPS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

/// The grid and which of its elements belong to the part.
#[derive(Clone, Debug)]
pub struct Grid3d {
    nx: usize,
    ny: usize,
    nz: usize,
    h: f64,
    /// One entry per element: `false` means the element is outside the part and takes part in no equation.
    active: Vec<bool>,
}

impl Grid3d {
    /// A grid of `nx` by `ny` by `nz` elements. `active` has one entry per element, in element order.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        h: f64,
        active: Vec<bool>,
    ) -> Result<Grid3d, EngineError> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(EngineError::invalid_input(
                "the grid needs at least one element in x, in y and in z",
            )
            .with_integer("nx", nx)
            .with_integer("ny", ny)
            .with_integer("nz", nz));
        }
        if !(h.is_finite() && h > 0.0) {
            return Err(
                EngineError::invalid_input("the element size h must be a positive number")
                    .with_number("h", h),
            );
        }
        let n_elements = element_count(nx, ny, nz)?;
        if active.len() != n_elements {
            return Err(
                EngineError::invalid_input("the active mask needs one entry per element")
                    .with_integer("expected", n_elements)
                    .with_integer("got", active.len()),
            );
        }
        Ok(Grid3d {
            nx,
            ny,
            nz,
            h,
            active,
        })
    }

    /// A grid whose elements are all active.
    pub fn full(nx: usize, ny: usize, nz: usize, h: f64) -> Result<Grid3d, EngineError> {
        let n_elements = element_count(nx, ny, nz)?;
        Grid3d::new(nx, ny, nz, h, vec![true; n_elements])
    }

    pub fn nx(&self) -> usize {
        self.nx
    }

    pub fn ny(&self) -> usize {
        self.ny
    }

    pub fn nz(&self) -> usize {
        self.nz
    }

    /// Side of one cube element, in mm.
    pub fn h(&self) -> f64 {
        self.h
    }

    /// Volume of one element, h^3, in mm^3.
    pub fn element_volume(&self) -> f64 {
        self.h * self.h * self.h
    }

    pub fn n_elements(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    pub fn n_nodes(&self) -> usize {
        (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
    }

    /// Three degrees of freedom (x, y and z displacement) per node.
    pub fn n_dofs(&self) -> usize {
        3 * self.n_nodes()
    }

    /// One entry per element: whether it belongs to the part.
    pub fn active(&self) -> &[bool] {
        &self.active
    }

    pub fn is_active(&self, element: usize) -> bool {
        self.active[element]
    }

    pub fn element_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    pub fn node_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + (self.nx + 1) * (j + (self.ny + 1) * k)
    }

    /// The grid position `(i, j, k)` of an element.
    pub fn element_position(&self, element: usize) -> (usize, usize, usize) {
        let i = element % self.nx;
        let j = (element / self.nx) % self.ny;
        let k = element / (self.nx * self.ny);
        (i, j, k)
    }

    /// The grid position `(i, j, k)` of a node.
    pub fn node_position(&self, node: usize) -> (usize, usize, usize) {
        let i = node % (self.nx + 1);
        let j = (node / (self.nx + 1)) % (self.ny + 1);
        let k = node / ((self.nx + 1) * (self.ny + 1));
        (i, j, k)
    }

    /// The eight nodes of an element, in the order of `NODE_STEPS`.
    pub fn element_nodes(&self, element: usize) -> [usize; 8] {
        let (i, j, k) = self.element_position(element);
        NODE_STEPS.map(|[di, dj, dk]| self.node_index(i + di, j + dj, k + dk))
    }

    /// The 24 degrees of freedom of an element: x, y and z of each node, nodes in `element_nodes` order.
    pub fn element_dofs(&self, element: usize) -> [usize; 24] {
        let nodes = self.element_nodes(element);
        let mut dofs = [0; 24];
        for (a, node) in nodes.iter().enumerate() {
            for c in 0..3 {
                dofs[3 * a + c] = 3 * node + c;
            }
        }
        dofs
    }

    /// The elements around a node: entry `place` (0 to 7) is the element in which this node is node
    /// `place` of `NODE_STEPS`, or `None` where that element would lie outside the grid. Inactive
    /// elements are included; the caller decides what to do with them.
    pub fn node_elements(&self, node: usize) -> [Option<usize>; 8] {
        let (i, j, k) = self.node_position(node);
        let mut around = [None; 8];
        for (place, [di, dj, dk]) in NODE_STEPS.iter().enumerate() {
            // The element in which this node is node `place` starts `di, dj, dk` steps lower.
            if i < *di || j < *dj || k < *dk {
                continue;
            }
            let (ei, ej, ek) = (i - di, j - dj, k - dk);
            if ei < self.nx && ej < self.ny && ek < self.nz {
                around[place] = Some(self.element_index(ei, ej, ek));
            }
        }
        around
    }
}

/// nx * ny * nz, refused when the grid is so large that its degree-of-freedom count,
/// 3 (nx+1)(ny+1)(nz+1), does not fit in a machine integer.
fn element_count(nx: usize, ny: usize, nz: usize) -> Result<usize, EngineError> {
    let dofs = nx
        .checked_add(1)
        .zip(ny.checked_add(1))
        .and_then(|(nodes_x, nodes_y)| nodes_x.checked_mul(nodes_y))
        .zip(nz.checked_add(1))
        .and_then(|(nodes_xy, nodes_z)| nodes_xy.checked_mul(nodes_z))
        .and_then(|nodes| nodes.checked_mul(3));
    match dofs {
        Some(_) => Ok(nx * ny * nz),
        None => Err(
            EngineError::invalid_input("the grid is too large to number")
                .with_integer("nx", nx)
                .with_integer("ny", ny)
                .with_integer("nz", nz),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbering_runs_x_then_y_then_z() {
        let grid = Grid3d::full(3, 2, 4, 1.0).unwrap();
        assert_eq!(grid.n_elements(), 24);
        assert_eq!(grid.n_nodes(), 4 * 3 * 5);
        assert_eq!(grid.n_dofs(), 180);
        assert_eq!(grid.element_index(2, 1, 3), 2 + 3 * (1 + 2 * 3));
        assert_eq!(grid.element_position(23), (2, 1, 3));
        assert_eq!(grid.node_position(grid.node_index(3, 2, 4)), (3, 2, 4));
        // Element (0, 0, 0): bottom face 0, 1, 5, 4 (4 nodes wide), top face one layer of 12 nodes up.
        assert_eq!(grid.element_nodes(0), [0, 1, 5, 4, 12, 13, 17, 16]);
        assert_eq!(grid.element_dofs(0)[..6], [0, 1, 2, 3, 4, 5]);
        assert_eq!(grid.element_dofs(0)[21..], [48, 49, 50]);
    }

    #[test]
    fn each_node_finds_the_elements_around_it() {
        let grid = Grid3d::full(3, 2, 2, 1.0).unwrap();
        let count = |node: usize| grid.node_elements(node).iter().flatten().count();
        // A corner node has one element (in which it is node 0), an inside node eight.
        assert_eq!(grid.node_elements(0)[0], Some(0));
        assert_eq!(count(0), 1);
        assert_eq!(count(grid.node_index(1, 1, 1)), 8);
        // Every element found must name the node back at that place.
        for node in 0..grid.n_nodes() {
            for (place, element) in grid.node_elements(node).iter().enumerate() {
                if let Some(element) = element {
                    assert_eq!(grid.element_nodes(*element)[place], node);
                }
            }
        }
        // And every element's node finds that element.
        for element in 0..grid.n_elements() {
            for (place, node) in grid.element_nodes(element).iter().enumerate() {
                assert_eq!(grid.node_elements(*node)[place], Some(element));
            }
        }
    }

    #[test]
    fn wrong_input_is_refused() {
        assert!(Grid3d::full(0, 2, 2, 1.0).is_err());
        assert!(Grid3d::full(2, 2, 2, 0.0).is_err());
        assert!(Grid3d::full(2, 2, 2, f64::INFINITY).is_err());
        assert!(Grid3d::new(2, 2, 2, 1.0, vec![true; 7]).is_err());
        assert!(Grid3d::full(1 << 30, 1 << 30, 1 << 30, 1.0).is_err());
    }
}
