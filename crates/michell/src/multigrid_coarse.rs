//! The matrices of the coarse levels of the multigrid, by the Galerkin rule A_coarse = P^T A_fine P, and
//! the product with them.
//!
//! The rule is applied element by element. The fine elements nest in the coarse ones (each fine element
//! lies inside one coarse element), and inside a coarse element the interpolation P is the same small
//! matrix for every child at the same position. So the coarse element matrix
//!
//!   A_C = sum over the children e of C of  P_p(e)^T A_e P_p(e),
//!
//! with P_p the 24 by 24 interpolation from the coarse element's nodes to those of a child at position p
//! (`multigrid_transfer::local_weights`), is exact: summed over the coarse elements, these matrices are
//! P^T A P. Each active coarse element keeps its own 24 by 24 matrix, and the product with the coarse
//! matrix is computed node by node from them, like the fine product of `elasticity3d`.
//!
//! Fixed degrees of freedom (`multigrid_levels`) take no part: a fine element matrix enters the product
//! with the rows and columns of its fixed degrees of freedom set to zero, which also makes the rows and
//! columns of the fixed coarse degrees of freedom come out zero (no free fine degree of freedom takes a
//! value from them). In the product they act as identity rows and columns.
//!
//! On the first coarse level the fine element matrices are E_e K_unit, so a child that touches no fixed
//! degree of freedom contributes E_e (P_p^T K_unit P_p): the eight matrices P_p^T K_unit P_p (one per
//! position) are computed once per build and scaled by each child's modulus. Only children that touch a
//! fixed degree of freedom need the product with their own reduced matrix.
//!
//! The coarse matrices may be positive semi-definite, not definite: at the far edge of an odd direction a
//! coarse node lies outside the fine grid, and next to an inactive region a coarse node may have no free
//! fine node of its own, so the interpolation P can map some combination of coarse values to zero. That
//! combination is in the null space of P^T A P whatever the density. The smoother needs only positive
//! diagonals, and the coarsest solve takes those combinations out (`multigrid_coarsest`).
//!
//! Source: Trottenberg, Oosterlee and Schueller, "Multigrid" (2001), section 2.7.2 (the Galerkin coarse
//! operator); the element-by-element form for nested hexahedral grids as in Amir, Aage and Lazarov, "On
//! multigrid-CG for efficient topology optimization", Struct. Multidisc. Optim. 49 (2014).

use crate::grid3d::Grid3d;
use crate::multigrid_levels::{Level, children};
use crate::multigrid_transfer::{halved_directions, local_weights};
use crate::threads::Threads;

/// Entries of one element matrix, 24 by 24, stored row by row.
pub const ENTRIES: usize = 24 * 24;

/// No matrix: the element is inactive.
const NONE: usize = usize::MAX;

/// Where the element matrices of a level come from.
#[derive(Clone, Copy)]
pub enum LevelMatrices<'a> {
    /// The finest level: element e has E_e K_unit (`moduli` is 0 for inactive elements).
    Fine {
        moduli: &'a [f64],
        k_unit: &'a [[f64; 24]; 24],
    },
    /// A coarse level, with its stored element matrices.
    Coarse(&'a ElementMatrices),
}

impl LevelMatrices<'_> {
    /// The matrix of `element` on `level`, with the rows and columns of fixed degrees of freedom set to
    /// zero, row by row; `None` when the element adds nothing (inactive).
    pub fn element_matrix(&self, level: &Level, element: usize) -> Option<[f64; ENTRIES]> {
        let dofs = level.grid.element_dofs(element);
        let mut matrix = [0.0; ENTRIES];
        match self {
            LevelMatrices::Fine { moduli, k_unit } => {
                let modulus = moduli[element];
                if modulus == 0.0 {
                    return None;
                }
                for (row, &dof_row) in dofs.iter().enumerate() {
                    if level.fixed[dof_row] {
                        continue;
                    }
                    for (col, &dof_col) in dofs.iter().enumerate() {
                        if !level.fixed[dof_col] {
                            matrix[24 * row + col] = modulus * k_unit[row][col];
                        }
                    }
                }
            }
            LevelMatrices::Coarse(stored) => {
                // Stored matrices already have zero rows and columns for the fixed degrees of freedom.
                matrix.copy_from_slice(stored.matrix(element)?);
            }
        }
        Some(matrix)
    }
}

/// The element matrices of one coarse level: one 24 by 24 matrix per active element.
#[derive(Clone, Debug)]
pub struct ElementMatrices {
    /// For each element of the level, where its matrix starts in `values` (in matrices), or `NONE`.
    slot: Vec<usize>,
    /// The matrices of the active elements, one after the other, each row by row.
    values: Vec<f64>,
}

impl ElementMatrices {
    /// The matrices of `coarse` by the Galerkin rule from those of the next finer level.
    pub fn galerkin(
        finer: &Level,
        finer_matrices: LevelMatrices,
        coarse: &Level,
        threads: &Threads,
    ) -> ElementMatrices {
        let halved = halved_directions(&finer.grid);
        // The interpolation of each child position, and on the first coarse level the matrices
        // P_p^T K_unit P_p for children that touch no fixed degree of freedom.
        let weights: Vec<[[f64; 8]; 8]> = (0..8)
            .map(|p| local_weights([p & 1, (p >> 1) & 1, (p >> 2) & 1], halved))
            .collect();
        let unit_products: Option<Vec<[f64; ENTRIES]>> = match finer_matrices {
            LevelMatrices::Fine { k_unit, .. } => {
                let mut flat = [0.0; ENTRIES];
                for (row, values) in k_unit.iter().enumerate() {
                    flat[24 * row..24 * row + 24].copy_from_slice(values);
                }
                Some(
                    weights
                        .iter()
                        .map(|w| {
                            let mut product = [0.0; ENTRIES];
                            add_galerkin_product(&mut product, &flat, w, 1.0);
                            product
                        })
                        .collect(),
                )
            }
            LevelMatrices::Coarse(_) => None,
        };

        let grid = &coarse.grid;
        let mut slot = vec![NONE; grid.n_elements()];
        let mut elements = Vec::new();
        for (element, place) in slot.iter_mut().enumerate() {
            if grid.is_active(element) {
                *place = elements.len();
                elements.push(element);
            }
        }
        let mut values = vec![0.0; elements.len() * ENTRIES];
        threads.for_each_piece(&mut values, ENTRIES, |piece, target| {
            let (found, count) = children(&finer.grid, grid, elements[piece]);
            for &(child, position) in &found[..count] {
                let p = position[0] + 2 * position[1] + 4 * position[2];
                if let (Some(products), LevelMatrices::Fine { moduli, .. }) =
                    (&unit_products, finer_matrices)
                {
                    let touches_fixed = finer
                        .grid
                        .element_dofs(child)
                        .iter()
                        .any(|&d| finer.fixed[d]);
                    if !touches_fixed {
                        if moduli[child] != 0.0 {
                            for (t, u) in target.iter_mut().zip(&products[p]) {
                                *t += moduli[child] * u;
                            }
                        }
                        continue;
                    }
                }
                if let Some(matrix) = finer_matrices.element_matrix(finer, child) {
                    add_galerkin_product(target, &matrix, &weights[p], 1.0);
                }
            }
        });
        ElementMatrices { slot, values }
    }

    /// The matrix of `element`, row by row, or `None` if the element is inactive.
    pub fn matrix(&self, element: usize) -> Option<&[f64]> {
        let place = self.slot[element];
        if place == NONE {
            return None;
        }
        Some(&self.values[place * ENTRIES..(place + 1) * ENTRIES])
    }

    /// Every stored matrix, row by row, one after the other.
    pub fn all(&self) -> &[f64] {
        &self.values
    }

    /// out = A x on `level`, node by node: the three entries of a node gather the rows of its (at most
    /// eight) active elements. Fixed degrees of freedom act as identity rows and columns.
    pub fn product(&self, level: &Level, threads: &Threads, x: &[f64], out: &mut [f64]) {
        let grid: &Grid3d = &level.grid;
        threads.for_each_piece(out, 3, |node, values| {
            let mut block = [0.0; 3];
            for (place, element) in grid.node_elements(node).iter().enumerate() {
                let Some(element) = *element else { continue };
                let Some(matrix) = self.matrix(element) else {
                    continue;
                };
                let dofs = grid.element_dofs(element);
                for (c, entry) in block.iter_mut().enumerate() {
                    let row = &matrix[24 * (3 * place + c)..24 * (3 * place + c) + 24];
                    let mut sum = 0.0;
                    for (a, &dof) in row.iter().zip(&dofs) {
                        sum += a * x[dof];
                    }
                    *entry += sum;
                }
            }
            for (c, value) in values.iter_mut().enumerate() {
                let dof = 3 * node + c;
                *value = if level.fixed[dof] { x[dof] } else { block[c] };
            }
        });
    }

    /// The diagonal of the assembled matrix on `level`, 1 on the fixed degrees of freedom.
    pub fn diagonal(&self, level: &Level, threads: &Threads) -> Vec<f64> {
        let grid = &level.grid;
        let mut diagonal = vec![1.0; grid.n_dofs()];
        threads.for_each_piece(&mut diagonal, 3, |node, values| {
            for (c, value) in values.iter_mut().enumerate() {
                if level.fixed[3 * node + c] {
                    continue;
                }
                let mut sum = 0.0;
                for (place, element) in grid.node_elements(node).iter().enumerate() {
                    if let Some(matrix) = element.and_then(|e| self.matrix(e)) {
                        let row = 3 * place + c;
                        sum += matrix[24 * row + row];
                    }
                }
                *value = sum;
            }
        });
        diagonal
    }
}

/// target += scale * P^T A P for one child, where P (24 by 24) is the node-level interpolation `weights`
/// (`[a][b]`: weight of coarse node b at fine node a), the same for x, y and z, and A is row by row.
///
/// Written with the 3 by 3 blocks of node pairs: (P^T A P)_(b, b') = sum over a, a' of
/// w[a][b] w[a'][b'] A_(a, a'), first T_(a, b') = sum over a' of A_(a, a') w[a'][b'], then the sum over a.
fn add_galerkin_product(target: &mut [f64], a: &[f64], weights: &[[f64; 8]; 8], scale: f64) {
    // t[(3 a + i) * 24 + 3 b' + j] = sum over a' of A[3 a + i][3 a' + j] w[a'][b']
    let mut t = [0.0; ENTRIES];
    for row in 0..24 {
        for (a2, weights_a2) in weights.iter().enumerate() {
            for (b2, &w) in weights_a2.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                for j in 0..3 {
                    t[24 * row + 3 * b2 + j] += a[24 * row + 3 * a2 + j] * w;
                }
            }
        }
    }
    for (a1, weights_a1) in weights.iter().enumerate() {
        for (b1, &w) in weights_a1.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            for i in 0..3 {
                let (from, to) = (24 * (3 * a1 + i), 24 * (3 * b1 + i));
                for col in 0..24 {
                    target[to + col] += scale * w * t[from + col];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex8;
    use crate::multigrid_levels::build_levels;
    use crate::multigrid_transfer::Transfer;

    /// A dense matrix from a product function that maps `n_in` entries to `n_out`, column by column.
    fn dense(n_in: usize, n_out: usize, product: impl Fn(&[f64], &mut [f64])) -> Vec<Vec<f64>> {
        let mut columns = vec![vec![0.0; n_out]; n_in];
        let mut e = vec![0.0; n_in];
        for (col, column) in columns.iter_mut().enumerate() {
            e[col] = 1.0;
            product(&e, column);
            e[col] = 0.0;
        }
        columns
    }

    #[test]
    fn the_element_matrices_add_up_to_p_transpose_k_p() {
        // 5 x 4 x 3 fine elements (odd and even counts), a corner outside the part, uneven moduli with a
        // near-void region, the face x = 0 held and a patch of z held on the far face. Small enough to
        // compare the coarse level with a dense P^T K P; levels are forced by building them by hand.
        let (nx, ny, nz) = (5, 4, 3);
        let mut active = vec![true; nx * ny * nz];
        active[nx * ny * nz - 1] = false;
        active[nx - 1] = false;
        let fine = Grid3d::new(nx, ny, nz, 1.0, active).unwrap();
        let mut fixed = vec![true; fine.n_dofs()];
        for element in 0..fine.n_elements() {
            if fine.is_active(element) {
                for dof in fine.element_dofs(element) {
                    fixed[dof] = false;
                }
            }
        }
        for node in 0..fine.n_nodes() {
            let (i, j, _) = fine.node_position(node);
            if i == 0 {
                fixed[3 * node..3 * node + 3].fill(true);
            }
            if i == nx && j <= 2 {
                fixed[3 * node + 2] = true;
            }
        }
        let moduli: Vec<f64> = (0..fine.n_elements())
            .map(|e| match (fine.is_active(e), e % 4) {
                (false, _) => 0.0,
                (true, 0) => 1e-9,
                (true, r) => 0.3 + 0.2 * r as f64,
            })
            .collect();
        let k_unit = hex8::stiffness(1.0, 0.3, 1.0);
        let threads = Threads::new(1).unwrap();
        let levels = build_levels(&fine, &fixed).unwrap();
        assert_eq!(levels.len(), 1, "small grid: one level of its own");
        // Build the next two levels by hand with the same rules.
        let coarse_grid = {
            let grid = Grid3d::full(3, 2, 2, 1.0).unwrap();
            let active = (0..grid.n_elements())
                .map(|e| {
                    let (found, count) = children(&fine, &grid, e);
                    found[..count].iter().any(|(c, _)| fine.is_active(*c))
                })
                .collect();
            Grid3d::new(3, 2, 2, 1.0, active).unwrap()
        };
        let coarse_fixed = (0..coarse_grid.n_dofs())
            .map(|d| {
                !crate::multigrid_transfer::reaches_a_free_fine_dof(&fine, &fixed, &coarse_grid, d)
            })
            .collect();
        let finest = Level {
            grid: fine.clone(),
            fixed: fixed.clone(),
        };
        let coarse = Level {
            grid: coarse_grid,
            fixed: coarse_fixed,
        };
        let source = LevelMatrices::Fine {
            moduli: &moduli,
            k_unit: &k_unit,
        };
        let matrices = ElementMatrices::galerkin(&finest, source, &coarse, &threads);

        // Dense K (fine, fixed rows and columns zero) and P from the transfer, then P^T K P.
        let n_fine = fine.n_dofs();
        let n_coarse = coarse.grid.n_dofs();
        let mut k = vec![vec![0.0; n_fine]; n_fine];
        for element in 0..fine.n_elements() {
            if let Some(m) = source.element_matrix(&finest, element) {
                let dofs = fine.element_dofs(element);
                for a in 0..24 {
                    for b in 0..24 {
                        k[dofs[a]][dofs[b]] += m[24 * a + b];
                    }
                }
            }
        }
        let transfer = Transfer {
            fine: &fine,
            fine_fixed: &fixed,
            coarse: &coarse.grid,
            coarse_fixed: &coarse.fixed,
        };
        let p = dense(n_coarse, n_fine, |x, out| {
            transfer.prolong(&threads, x, out)
        }); // columns of P
        let ours = dense(n_coarse, n_coarse, |x, out| {
            matrices.product(&coarse, &threads, x, out)
        });
        let mut largest = 0.0f64;
        let mut worst = 0.0f64;
        for col in 0..n_coarse {
            let kp: Vec<f64> = (0..n_fine)
                .map(|r| (0..n_fine).map(|c| k[r][c] * p[col][c]).sum())
                .collect();
            for row in 0..n_coarse {
                let mut expected: f64 = (0..n_fine).map(|f| p[row][f] * kp[f]).sum();
                if coarse.fixed[row] && row == col {
                    expected = 1.0;
                }
                largest = largest.max(expected.abs());
                worst = worst.max((ours[col][row] - expected).abs());
            }
        }
        assert!(worst < 1e-14 * largest, "worst {worst:e} of {largest:e}");
        // The diagonal of the stored matrices is that of the product.
        let diagonal = matrices.diagonal(&coarse, &threads);
        for dof in 0..n_coarse {
            assert!((diagonal[dof] - ours[dof][dof]).abs() < 1e-14 * largest);
        }
        // Coarse node i = 3 lies outside the fine grid (x = 6 > 5); it takes part through fine node 5
        // wherever that one is free.
        let outside = coarse.grid.node_index(3, 1, 1);
        assert!(!coarse.fixed[3 * outside]);
    }
}
