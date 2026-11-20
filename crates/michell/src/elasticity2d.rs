//! Linear elasticity on the 2D grid: assemble K(rho), solve K u = f, and the per-element quantities an
//! optimisation needs (sensitivity, strain energy, stress).
//!
//! Material law: modified SIMP, E(rho) = E_min + rho^p (E_0 - E_min) (Sigmund, "Morphology-based black
//! and white filters for topology optimization", 2007; the form used by the 88-line code of Andreassen et
//! al., 2011), kept in `material`, which the 3D analysis shares. Every element has the stiffness
//! E(rho_e) * K_unit, where K_unit is the `quad4` matrix of modulus 1.
//!
//! Held degrees of freedom: their row and column are left out of K and a 1 is put on the diagonal, so
//! the solve returns the held value there. A degree of freedom that touches no active element is held
//! at zero in the same way (it belongs to no equation).
//!
//! Is the part held? Every element keeps a stiffness of at least E_min > 0, and the null space of a
//! sum of element matrices with positive weights does not depend on the weights. So whether K(rho) is
//! singular does not depend on rho: `new` checks it once, with every active element at modulus 1. That
//! check is reliable where a check on K(rho) is not: with E_min = 1e-9 E_0 a real near-void pivot and
//! the rounding noise of a true zero pivot can be of the same size (see `HELD_CHECK_PIVOT_RATIO`).

use crate::banded_cholesky::BandedMatrix;
use crate::error::{EngineError, ErrorCode};
use crate::grid2d::Grid2d;
use crate::material::Material;
use crate::quad4;

/// The check that the part is held (in `new`, every element at modulus 1) calls a pivot zero when it is
/// not above this fraction of its row's original diagonal.
///
/// Measured with this solver, plane stress, nu = 0.3: a part that is not held leaves rounding noise of
/// 2e-13 (8 x 4 elements), 7e-13 (150 x 50) and 1.1e-11 (600 x 200) of the diagonal. Held parts keep
/// their smallest pivot far above it: 1.6e-2 (600 x 200 cantilever), 2.4e-6 (600 x 10), 3.3e-8 (1000 x 4).
/// Only a beam one element deep and 2000 long gets close (9.4e-11), and would be called singular.
pub const HELD_CHECK_PIVOT_RATIO: f64 = 1e-10;

/// Elasticity on one grid with one set of held degrees of freedom.
#[derive(Clone, Debug)]
pub struct Elasticity2d {
    grid: Grid2d,
    material: Material,
    /// One entry per degree of freedom: held by the user, or touching no active element.
    fixed: Vec<bool>,
    /// The degrees of freedom that touch no active element (a subset of `fixed`).
    orphan: Vec<bool>,
    /// The element stiffness matrix for a modulus of 1.
    k_unit: [[f64; 8]; 8],
    /// For each degree of freedom, its number inside the solver (see `solver_numbering`).
    solver_dof: Vec<usize>,
    half_band: usize,
}

impl Elasticity2d {
    /// `held` has one entry per degree of freedom (2 per node, x then y); `true` holds it.
    pub fn new(
        grid: Grid2d,
        material: Material,
        held: Vec<bool>,
    ) -> Result<Elasticity2d, EngineError> {
        material.check()?;
        if held.len() != grid.n_dofs() {
            return Err(EngineError::invalid_input(
                "held needs one entry per degree of freedom (2 per node)",
            )
            .with_integer("expected", grid.n_dofs())
            .with_integer("got", held.len()));
        }
        let mut orphan = vec![true; grid.n_dofs()];
        for element in 0..grid.n_elements() {
            if grid.is_active(element) {
                for dof in grid.element_dofs(element) {
                    orphan[dof] = false;
                }
            }
        }
        let fixed: Vec<bool> = held.iter().zip(&orphan).map(|(h, o)| *h || *o).collect();
        let k_unit = quad4::stiffness(1.0, material.poisson_ratio, grid.h(), grid.t());
        let (solver_dof, half_band) = solver_numbering(&grid);
        let model = Elasticity2d {
            grid,
            material,
            fixed,
            orphan,
            k_unit,
            solver_dof,
            half_band,
        };

        // Is the part held? K with every active element at modulus 1 (see the note at the top).
        let mut k_check = model.assemble(&vec![1.0; model.grid.n_elements()]);
        if let Err(failure) = k_check.factor(HELD_CHECK_PIVOT_RATIO) {
            return Err(model.singular_error(failure.row, failure.pivot, failure.diagonal));
        }
        Ok(model)
    }

    pub fn grid(&self) -> &Grid2d {
        &self.grid
    }

    pub fn material(&self) -> &Material {
        &self.material
    }

    /// Solves K(rho) u = loads. Loads on held degrees of freedom are ignored (they become reactions).
    /// `held_values`, if given, has one entry per degree of freedom and sets the displacement of each
    /// held one (the others are ignored); without it every held degree of freedom stays at zero.
    pub fn solve(
        &self,
        rho: &[f64],
        loads: &[f64],
        held_values: Option<&[f64]>,
    ) -> Result<Vec<f64>, EngineError> {
        self.check_density(rho)?;
        self.check_dof_vector(loads, "loads")?;
        if let Some(values) = held_values {
            self.check_dof_vector(values, "held_values")?;
        }
        self.solve_system(rho, loads, held_values)
    }

    /// Solves K(rho) lambda = rhs with every held degree of freedom at zero: the adjoint system.
    /// K is symmetric, so this is the same solve as `solve`; it exists so that the call says what it is for.
    pub fn adjoint(&self, rho: &[f64], rhs: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.check_density(rho)?;
        self.check_dof_vector(rhs, "rhs")?;
        self.solve_system(rho, rhs, None)
    }

    /// Per element, the term -lambda_e^T (dK/drho_e) u_e = -dE/drho(rho_e) * lambda_e^T K_unit u_e.
    ///
    /// This is the part of dg/drho_e that comes through the state (modules.md):
    /// dg/drho_e = partial g / partial rho_e + sensitivity_e. For compliance (lambda = u) it is the whole
    /// gradient, -p rho^(p-1) (E_0 - E_min) u_e^T K_unit u_e. Inactive elements give 0.
    pub fn sensitivity(
        &self,
        rho: &[f64],
        u: &[f64],
        lambda: &[f64],
    ) -> Result<Vec<f64>, EngineError> {
        self.check_density(rho)?;
        self.check_dof_vector(u, "u")?;
        self.check_dof_vector(lambda, "lambda")?;
        let mut result = vec![0.0; self.grid.n_elements()];
        for (element, value) in result.iter_mut().enumerate() {
            if self.grid.is_active(element) {
                let dofs = self.grid.element_dofs(element);
                let product = self.unit_energy_product(&dofs, lambda, u);
                *value = -self.material.modulus_derivative(rho[element]) * product;
            }
        }
        Ok(result)
    }

    /// Per element, the strain energy 1/2 u_e^T K_e u_e = 1/2 E(rho_e) u_e^T K_unit u_e, in N mm.
    /// The compliance f.u is twice their sum. Inactive elements give 0.
    pub fn element_strain_energy(&self, rho: &[f64], u: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.check_density(rho)?;
        self.check_dof_vector(u, "u")?;
        let mut result = vec![0.0; self.grid.n_elements()];
        for (element, value) in result.iter_mut().enumerate() {
            if self.grid.is_active(element) {
                let dofs = self.grid.element_dofs(element);
                *value = 0.5
                    * self.material.modulus(rho[element])
                    * self.unit_energy_product(&dofs, u, u);
            }
        }
        Ok(result)
    }

    /// Per element, the stress at its centre `[sx, sy, txy]` in MPa, computed with the element's own
    /// modulus E(rho_e): the stress the model actually carries. Returned flat, three numbers per element
    /// in element order. Inactive elements give zeros.
    pub fn element_stress(&self, rho: &[f64], u: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.check_density(rho)?;
        self.check_dof_vector(u, "u")?;
        let mut result = vec![0.0; 3 * self.grid.n_elements()];
        for element in 0..self.grid.n_elements() {
            if self.grid.is_active(element) {
                let dofs = self.grid.element_dofs(element);
                let u_e = dofs.map(|dof| u[dof]);
                let modulus = self.material.modulus(rho[element]);
                let stress =
                    quad4::centre_stress(modulus, self.material.poisson_ratio, self.grid.h(), &u_e);
                result[3 * element..3 * element + 3].copy_from_slice(&stress);
            }
        }
        Ok(result)
    }

    /// a_e^T K_unit b_e for one element.
    fn unit_energy_product(&self, dofs: &[usize; 8], a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;
        for (row, &dof_row) in dofs.iter().enumerate() {
            for (col, &dof_col) in dofs.iter().enumerate() {
                sum += a[dof_row] * self.k_unit[row][col] * b[dof_col];
            }
        }
        sum
    }

    /// The global stiffness matrix in solver numbering, for the given modulus of each element.
    /// Rows and columns of fixed degrees of freedom are left out, with a 1 on their diagonal.
    fn assemble(&self, moduli: &[f64]) -> BandedMatrix {
        let mut matrix = BandedMatrix::zeros(self.grid.n_dofs(), self.half_band);
        for (dof, &fixed) in self.fixed.iter().enumerate() {
            if fixed {
                let s = self.solver_dof[dof];
                matrix.add(s, s, 1.0);
            }
        }
        for (element, &modulus) in moduli.iter().enumerate() {
            if !self.grid.is_active(element) {
                continue;
            }
            let dofs = self.grid.element_dofs(element);
            for (row, &dof_row) in dofs.iter().enumerate() {
                for (col, &dof_col) in dofs.iter().enumerate() {
                    if self.fixed[dof_row] || self.fixed[dof_col] {
                        continue;
                    }
                    let (s_row, s_col) = (self.solver_dof[dof_row], self.solver_dof[dof_col]);
                    // Only the lower half is stored: each symmetric pair is added once.
                    if s_col <= s_row {
                        matrix.add(s_row, s_col, modulus * self.k_unit[row][col]);
                    }
                }
            }
        }
        matrix
    }

    /// Assembles K(rho), moves the held displacements to the right-hand side, factors and solves.
    fn solve_system(
        &self,
        rho: &[f64],
        rhs: &[f64],
        held_values: Option<&[f64]>,
    ) -> Result<Vec<f64>, EngineError> {
        let n_dofs = self.grid.n_dofs();
        for dof in 0..n_dofs {
            if self.orphan[dof] && rhs[dof] != 0.0 {
                let node = dof / 2;
                return Err(EngineError::invalid_input(format!(
                    "a load is put on node {node}, which no active element touches"
                ))
                .with_integer("dof", dof)
                .with_integer("node", node)
                .with_number("load", rhs[dof]));
            }
        }
        // The held displacement of each degree of freedom (zero where nothing is held).
        let mut held = vec![0.0; n_dofs];
        if let Some(values) = held_values {
            for dof in 0..n_dofs {
                if self.fixed[dof] && !self.orphan[dof] {
                    held[dof] = values[dof];
                }
            }
        }
        let moduli: Vec<f64> = rho.iter().map(|&r| self.material.modulus(r)).collect();

        // Right-hand side in solver numbering: the loads on free dofs, the held values on fixed ones.
        let mut b = vec![0.0; n_dofs];
        for dof in 0..n_dofs {
            let s = self.solver_dof[dof];
            b[s] = if self.fixed[dof] { held[dof] } else { rhs[dof] };
        }
        // A free row r meets a held column c inside an element: K_rc * held_c is known, so it moves to
        // the right-hand side, b_r -= K_rc * held_c (the "lifting" of prescribed displacements).
        for (element, &modulus) in moduli.iter().enumerate() {
            if !self.grid.is_active(element) {
                continue;
            }
            let dofs = self.grid.element_dofs(element);
            for (row, &dof_row) in dofs.iter().enumerate() {
                for (col, &dof_col) in dofs.iter().enumerate() {
                    if !self.fixed[dof_row] && self.fixed[dof_col] {
                        b[self.solver_dof[dof_row]] -=
                            modulus * self.k_unit[row][col] * held[dof_col];
                    }
                }
            }
        }

        let mut matrix = self.assemble(&moduli);
        // `new` has shown that the part is held, so K(rho) is positive definite in exact arithmetic; only
        // a pivot that rounding made zero or negative stops the factorisation here.
        if let Err(failure) = matrix.factor(0.0) {
            return Err(self.singular_error(failure.row, failure.pivot, failure.diagonal));
        }
        matrix.solve_factored(&mut b);

        // Back to the outside numbering.
        Ok(self.solver_dof.iter().map(|&s| b[s]).collect())
    }

    /// The `singular_system` error for a factorisation that stopped at `solver_row`.
    fn singular_error(&self, solver_row: usize, pivot: f64, diagonal: f64) -> EngineError {
        let dof = self
            .solver_dof
            .iter()
            .position(|&s| s == solver_row)
            .unwrap_or(solver_row);
        let node = dof / 2;
        let (i, j) = (node % (self.grid.nx() + 1), node / (self.grid.nx() + 1));
        EngineError::new(
            ErrorCode::SingularSystem,
            format!(
                "the stiffness matrix is singular at degree of freedom {dof} (node {node}, at i={i}, j={j}): \
                 the part is not held against every rigid motion"
            ),
        )
        .with_integer("dof", dof)
        .with_integer("node", node)
        .with_number("pivot", pivot)
        .with_number("diagonal", diagonal)
    }

    fn check_density(&self, rho: &[f64]) -> Result<(), EngineError> {
        if rho.len() != self.grid.n_elements() {
            return Err(
                EngineError::invalid_input("rho needs one entry per element")
                    .with_integer("expected", self.grid.n_elements())
                    .with_integer("got", rho.len()),
            );
        }
        if let Some(element) = rho.iter().position(|r| !(*r >= 0.0 && *r <= 1.0)) {
            return Err(
                EngineError::invalid_input("every density must lie between 0 and 1")
                    .with_integer("element", element)
                    .with_number("rho", rho[element]),
            );
        }
        Ok(())
    }

    fn check_dof_vector(&self, values: &[f64], name: &str) -> Result<(), EngineError> {
        if values.len() != self.grid.n_dofs() {
            return Err(EngineError::invalid_input(format!(
                "{name} needs one entry per degree of freedom (2 per node)"
            ))
            .with_integer("expected", self.grid.n_dofs())
            .with_integer("got", values.len()));
        }
        if let Some(dof) = values.iter().position(|v| !v.is_finite()) {
            return Err(EngineError::invalid_input(format!(
                "{name} holds a value that is not a finite number"
            ))
            .with_integer("dof", dof)
            .with_number("value", values[dof]));
        }
        Ok(())
    }
}

/// The numbering of the degrees of freedom inside the solver, and the half bandwidth it gives.
///
/// The band of K is set by the largest difference between two node numbers in one element. Numbering
/// the nodes along the shorter grid direction first keeps that difference small: with m elements across
/// the short direction, an element's nodes are n, n+1, n+m+1, n+m+2, so its degrees of freedom span
/// 2(m+2)+1. The outside numbering (x fastest) does not change; only the solver sees this one.
fn solver_numbering(grid: &Grid2d) -> (Vec<usize>, usize) {
    let (nx, ny) = (grid.nx(), grid.ny());
    let y_first = ny < nx;
    let mut solver_dof = vec![0; grid.n_dofs()];
    for j in 0..=ny {
        for i in 0..=nx {
            let node = grid.node_index(i, j);
            let solver_node = if y_first { j + (ny + 1) * i } else { node };
            solver_dof[2 * node] = 2 * solver_node;
            solver_dof[2 * node + 1] = 2 * solver_node + 1;
        }
    }
    let short = nx.min(ny);
    (solver_dof, 2 * (short + 2) + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_material() -> Material {
        Material {
            youngs_modulus: 1.0,
            poisson_ratio: 0.3,
            e_min: 1e-9,
            penalty: 3.0,
        }
    }

    /// Holds every degree of freedom on the left edge (a clamped cantilever).
    fn clamped_left(grid: &Grid2d) -> Vec<bool> {
        let mut held = vec![false; grid.n_dofs()];
        for j in 0..=grid.ny() {
            let node = grid.node_index(0, j);
            held[2 * node] = true;
            held[2 * node + 1] = true;
        }
        held
    }

    #[test]
    fn solver_numbering_is_the_same_either_way_round() {
        // The same cantilever as 6 x 2 and as 2 x 6 (turned by 90 degrees) must have the same compliance.
        let long = Grid2d::full(6, 2, 1.0, 1.0).unwrap();
        let held = clamped_left(&long);
        let model = Elasticity2d::new(long.clone(), unit_material(), held).unwrap();
        let mut loads = vec![0.0; long.n_dofs()];
        loads[2 * long.node_index(6, 1) + 1] = -1.0;
        let u = model.solve(&[1.0; 12], &loads, None).unwrap();
        let compliance_long: f64 = loads.iter().zip(&u).map(|(f, u)| f * u).sum();

        let tall = Grid2d::full(2, 6, 1.0, 1.0).unwrap();
        let mut held = vec![false; tall.n_dofs()];
        for i in 0..=2 {
            let node = tall.node_index(i, 0);
            held[2 * node] = true;
            held[2 * node + 1] = true;
        }
        let model = Elasticity2d::new(tall.clone(), unit_material(), held).unwrap();
        let mut loads = vec![0.0; tall.n_dofs()];
        loads[2 * tall.node_index(1, 6)] = 1.0;
        let u = model.solve(&[1.0; 12], &loads, None).unwrap();
        let compliance_tall: f64 = loads.iter().zip(&u).map(|(f, u)| f * u).sum();
        assert!((compliance_long - compliance_tall).abs() < 1e-10 * compliance_long);
    }

    #[test]
    fn a_part_that_is_not_held_is_singular() {
        let grid = Grid2d::full(8, 4, 1.0, 1.0).unwrap();
        let error = Elasticity2d::new(grid.clone(), unit_material(), vec![false; grid.n_dofs()])
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::SingularSystem);
        // Held in x along the left edge only: it can still slide up and down.
        let mut held = vec![false; grid.n_dofs()];
        for j in 0..=4 {
            held[2 * grid.node_index(0, j)] = true;
        }
        let error = Elasticity2d::new(grid, unit_material(), held).unwrap_err();
        assert_eq!(error.code, ErrorCode::SingularSystem);
    }

    #[test]
    fn a_block_hanging_on_one_corner_node_is_singular() {
        // Two active elements of a 2 x 2 grid, (0, 0) and (1, 1), share only the centre node: the second
        // can turn about it like a hinge. The first is clamped on the left.
        let grid = Grid2d::new(2, 2, 1.0, 1.0, vec![true, false, false, true]).unwrap();
        let held = clamped_left(&grid);
        let error = Elasticity2d::new(grid, unit_material(), held).unwrap_err();
        assert_eq!(error.code, ErrorCode::SingularSystem);
    }

    #[test]
    fn a_solid_block_hanging_on_near_void_elements_still_solves() {
        // A cantilever 12 x 4 whose middle column is at rho = 0, so E = E_min = 1e-9 there: the solid
        // tip hangs on near-void material only. It must solve, not be called singular.
        let grid = Grid2d::full(12, 4, 1.0, 1.0).unwrap();
        let held = clamped_left(&grid);
        let model = Elasticity2d::new(grid.clone(), unit_material(), held).unwrap();
        let mut rho = vec![1.0; 48];
        for j in 0..4 {
            rho[grid.element_index(6, j)] = 0.0;
        }
        let mut loads = vec![0.0; grid.n_dofs()];
        loads[2 * grid.node_index(12, 2) + 1] = -1.0;
        let u = model.solve(&rho, &loads, None).unwrap();
        assert!(u.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn inactive_elements_leave_no_singular_nodes() {
        // A 4 x 4 plate clamped on the left with its top-right 2 x 2 elements outside the part: the node
        // at the top-right corner touches no active element and must simply stay at zero.
        let mut active = vec![true; 16];
        for (i, j) in [(2, 2), (3, 2), (2, 3), (3, 3)] {
            active[i + 4 * j] = false;
        }
        let grid = Grid2d::new(4, 4, 1.0, 1.0, active).unwrap();
        let held = clamped_left(&grid);
        let model = Elasticity2d::new(grid.clone(), unit_material(), held).unwrap();
        let mut loads = vec![0.0; grid.n_dofs()];
        loads[2 * grid.node_index(4, 0) + 1] = -1.0;
        let u = model.solve(&[1.0; 16], &loads, None).unwrap();
        let corner = grid.node_index(4, 4);
        assert_eq!(u[2 * corner], 0.0);
        assert!(u[2 * grid.node_index(4, 0) + 1] < 0.0);
        // A load on that corner is refused.
        loads[2 * corner] = 1.0;
        let error = model.solve(&[1.0; 16], &loads, None).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidInput);
    }

    #[test]
    fn compliance_is_twice_the_strain_energy_and_sensitivity_matches_differences() {
        let grid = Grid2d::full(5, 3, 2.0, 1.5).unwrap();
        let held = clamped_left(&grid);
        let model = Elasticity2d::new(grid.clone(), unit_material(), held).unwrap();
        let rho: Vec<f64> = (0..15).map(|e| 0.3 + 0.04 * e as f64).collect();
        let mut loads = vec![0.0; grid.n_dofs()];
        loads[2 * grid.node_index(5, 3) + 1] = -1.0;
        let compliance = |rho: &[f64]| -> f64 {
            let u = model.solve(rho, &loads, None).unwrap();
            loads.iter().zip(&u).map(|(f, u)| f * u).sum()
        };
        let u = model.solve(&rho, &loads, None).unwrap();
        let energy: f64 = model.element_strain_energy(&rho, &u).unwrap().iter().sum();
        assert!((compliance(&rho) - 2.0 * energy).abs() < 1e-10 * compliance(&rho));

        let sensitivity = model.sensitivity(&rho, &u, &u).unwrap();
        let step = 1e-6;
        for element in [0, 7, 14] {
            let mut up = rho.clone();
            let mut down = rho.clone();
            up[element] += step;
            down[element] -= step;
            let difference = (compliance(&up) - compliance(&down)) / (2.0 * step);
            assert!((difference - sensitivity[element]).abs() < 1e-6 * difference.abs());
        }
    }
}
