//! Linear elasticity on the 3D grid: the product K(rho) x without assembling K, the solve K u = f by
//! conjugate gradients, and the per-element quantities an optimisation needs (sensitivity, strain
//! energy, stress).
//!
//! Material law: modified SIMP from `material`, as in 2D. Every element has the stiffness
//! E(rho_e) * K_unit, where K_unit is the `hex8` matrix of modulus 1 on a cube of side h, computed once.
//!
//! The product is computed node by node: the three entries of K x that belong to a node gather the
//! contributions of the (at most eight) active elements around it. Each node writes only its own three
//! entries, so the nodes are shared out between threads without two threads writing the same entry.
//!
//! Threads: the object runs on the number of threads it was built with (`threads`). The product, the
//! diagonal, the right-hand side, the per-element quantities, the solver's sums and vector updates and the
//! multigrid V-cycle (all but its coarsest solve, which runs on one thread) run on them, by the rules of `threads`: every node or element is computed by one thread with the same
//! arithmetic whichever thread it is, and every sum runs over fixed blocks. No result depends on the
//! thread count.
//!
//! Held degrees of freedom, and those that touch no active element, act as identity rows and columns:
//! the product leaves them as they are and no other row sees them. The solve moves the held
//! displacements to the right-hand side (b_free = f_free - K_free,held u_held), solves for the free
//! degrees of freedom with the held ones at zero, and puts the held values back.
//!
//! Solver: conjugate gradients (`conjugate_gradient`) preconditioned by one multigrid V-cycle
//! (`multigrid`), stopped at the relative residual `tolerance` of the system for the free degrees of
//! freedom. The parts of the multigrid that do not depend on the density (the levels, and the set the
//! coarsest solve leaves out) are built with the object; the coarse matrices are built at the start of
//! every solve from that solve's density and dropped at its end, so no call depends on an earlier one.
//!
//! Is the part held? `new` checks that every body of face-joined active elements is held against all six
//! rigid motions (`held_check`), and refuses one that is not with `singular_system`. Like in 2D this does
//! not depend on the density: every active element keeps a modulus of at least E_min > 0.

use crate::conjugate_gradient::{self, SolveReport};
use crate::error::{DetailValue, EngineError, ErrorCode};
use crate::grid3d::Grid3d;
use crate::held_check::check_held;
use crate::hex8;
use crate::material::Material;
use crate::multigrid::MultigridShape;
use crate::multigrid_smoother::SmootherSettings;
use crate::threads::Threads;

/// Which columns of K a gather over a node's elements takes.
#[derive(Clone, Copy, PartialEq)]
enum Columns {
    /// Only free degrees of freedom: the product of the system that is solved.
    Free,
    /// Only fixed degrees of freedom: K_free,held times the held displacements.
    Fixed,
}

/// Elasticity on one 3D grid with one set of held degrees of freedom.
#[derive(Clone, Debug)]
pub struct Elasticity3d {
    grid: Grid3d,
    material: Material,
    /// One entry per degree of freedom: held by the user, or touching no active element.
    fixed: Vec<bool>,
    /// The degrees of freedom that touch no active element (a subset of `fixed`).
    orphan: Vec<bool>,
    /// The element stiffness matrix for a modulus of 1 on a cube of side h.
    k_unit: [[f64; 24]; 24],
    /// The solve stops when |r| / |b| is at most this.
    tolerance: f64,
    /// More steps than this give `solver_not_converged`.
    max_steps: usize,
    /// The threads every call runs on.
    threads: Threads,
    /// The parts of the multigrid preconditioner that do not depend on the density.
    multigrid: MultigridShape,
    /// The damping and sweep count of the multigrid smoother.
    smoother: SmootherSettings,
}

impl Elasticity3d {
    /// `held` has one entry per degree of freedom (3 per node: x, y, z); `true` holds it. `tolerance` is
    /// the relative residual at which a solve stops, `max_steps` the number of conjugate-gradient steps
    /// allowed, `threads` the number of threads every call runs on (at least 1).
    ///
    /// Gives `singular_system` if a body of face-joined active elements is not held against every rigid
    /// motion (see `held_check`).
    pub fn new(
        grid: Grid3d,
        material: Material,
        held: Vec<bool>,
        tolerance: f64,
        max_steps: usize,
        threads: usize,
    ) -> Result<Elasticity3d, EngineError> {
        material.check()?;
        conjugate_gradient::check_tolerance(tolerance)?;
        let threads = Threads::new(threads)?;
        if held.len() != grid.n_dofs() {
            return Err(EngineError::invalid_input(
                "held needs one entry per degree of freedom (3 per node)",
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
        check_held(&grid, &held)?;
        let k_unit = hex8::stiffness(1.0, material.poisson_ratio, grid.h());
        let multigrid = MultigridShape::new(&grid, &fixed, &k_unit, &threads)?;
        Ok(Elasticity3d {
            grid,
            material,
            fixed,
            orphan,
            k_unit,
            tolerance,
            max_steps,
            threads,
            multigrid,
            smoother: SmootherSettings::default(),
        })
    }

    /// The same analysis with other smoother settings: `damping` (c in omega = c * 2 / L, strictly
    /// between 0 and 1) and `sweeps` (1 to 3). For the study of the settings (`smoother_study`); the
    /// analysis always uses those of `multigrid_smoother`.
    #[cfg(test)]
    pub(crate) fn with_smoother(
        mut self,
        damping: f64,
        sweeps: usize,
    ) -> Result<Elasticity3d, EngineError> {
        if !(damping > 0.0 && damping < 1.0) || !(1..=3).contains(&sweeps) {
            return Err(EngineError::invalid_input(
                "the smoother needs a damping between 0 and 1 (both excluded) and 1 to 3 sweeps",
            )
            .with_number("damping", damping)
            .with_integer("sweeps", sweeps));
        }
        self.smoother = SmootherSettings { damping, sweeps };
        Ok(self)
    }

    /// The number of multigrid levels, the finest included (1: the grid is solved directly).
    pub fn multigrid_levels(&self) -> usize {
        self.multigrid.levels().len()
    }

    /// The number of degrees of freedom the coarsest multigrid solve leaves out (see
    /// `multigrid_coarsest`).
    pub fn multigrid_dropped(&self) -> usize {
        self.multigrid.dropped_count()
    }

    pub fn grid(&self) -> &Grid3d {
        &self.grid
    }

    pub fn material(&self) -> &Material {
        &self.material
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    pub fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// The number of threads every call runs on.
    pub fn threads(&self) -> usize {
        self.threads.count()
    }

    /// Solves K(rho) u = loads. Loads on held degrees of freedom are ignored (they become reactions).
    /// `held_values`, if given, has one entry per degree of freedom and sets the displacement of each
    /// held one (the others are ignored); without it every held degree of freedom stays at zero.
    /// `start`, if given, is the first guess of u (its held entries are ignored); without it the solve
    /// starts from zero. Returns u and how the solve ended.
    pub fn solve(
        &self,
        rho: &[f64],
        loads: &[f64],
        held_values: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<(Vec<f64>, SolveReport), EngineError> {
        self.check_density(rho)?;
        self.check_dof_vector(loads, "loads")?;
        if let Some(values) = held_values {
            self.check_dof_vector(values, "held_values")?;
        }
        if let Some(values) = start {
            self.check_dof_vector(values, "start")?;
        }
        self.solve_system(rho, loads, held_values, start)
    }

    /// Solves K(rho) lambda = rhs with every held degree of freedom at zero: the adjoint system.
    /// K is symmetric, so this is the same solve as `solve`; it exists so that the call says what it is for.
    pub fn adjoint(
        &self,
        rho: &[f64],
        rhs: &[f64],
        start: Option<&[f64]>,
    ) -> Result<(Vec<f64>, SolveReport), EngineError> {
        self.check_density(rho)?;
        self.check_dof_vector(rhs, "rhs")?;
        if let Some(values) = start {
            self.check_dof_vector(values, "start")?;
        }
        self.solve_system(rho, rhs, None, start)
    }

    /// K(rho) x, computed node by node without assembling K. Held degrees of freedom and those that touch
    /// no active element act as identity rows and columns: their entries of x pass through unchanged and
    /// take no part in the other rows. This is the product the solver uses.
    pub fn product(&self, rho: &[f64], x: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.check_density(rho)?;
        self.check_dof_vector(x, "x")?;
        let moduli = self.moduli(rho);
        let mut out = vec![0.0; self.grid.n_dofs()];
        self.multiply(&moduli, x, &mut out);
        Ok(out)
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
        self.threads.update(&mut result, |element, value| {
            if self.grid.is_active(element) {
                let dofs = self.grid.element_dofs(element);
                let product = self.unit_energy_product(&dofs, lambda, u);
                *value = -self.material.modulus_derivative(rho[element]) * product;
            }
        });
        Ok(result)
    }

    /// Per element, the strain energy 1/2 u_e^T K_e u_e = 1/2 E(rho_e) u_e^T K_unit u_e, in N mm.
    /// The compliance f.u is twice their sum. Inactive elements give 0.
    pub fn element_strain_energy(&self, rho: &[f64], u: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.check_density(rho)?;
        self.check_dof_vector(u, "u")?;
        let mut result = vec![0.0; self.grid.n_elements()];
        self.threads.update(&mut result, |element, value| {
            if self.grid.is_active(element) {
                let dofs = self.grid.element_dofs(element);
                *value = 0.5
                    * self.material.modulus(rho[element])
                    * self.unit_energy_product(&dofs, u, u);
            }
        });
        Ok(result)
    }

    /// Per element, the stress at its centre `[sxx, syy, szz, syz, szx, sxy]` in MPa, computed with the
    /// element's own modulus E(rho_e): the stress the model actually carries. Returned flat, six numbers
    /// per element in element order. Inactive elements give zeros.
    pub fn element_stress(&self, rho: &[f64], u: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.check_density(rho)?;
        self.check_dof_vector(u, "u")?;
        let mut result = vec![0.0; 6 * self.grid.n_elements()];
        self.threads
            .for_each_piece(&mut result, 6, |element, values| {
                if self.grid.is_active(element) {
                    let dofs = self.grid.element_dofs(element);
                    let u_e = dofs.map(|dof| u[dof]);
                    let modulus = self.material.modulus(rho[element]);
                    let stress = hex8::centre_stress(
                        modulus,
                        self.material.poisson_ratio,
                        self.grid.h(),
                        &u_e,
                    );
                    values.copy_from_slice(&stress);
                }
            });
        Ok(result)
    }

    /// The modulus E(rho_e) of every element, 0 for inactive elements (they add nothing to K).
    fn moduli(&self, rho: &[f64]) -> Vec<f64> {
        let mut moduli = vec![0.0; rho.len()];
        self.threads.update(&mut moduli, |element, modulus| {
            if self.grid.is_active(element) {
                *modulus = self.material.modulus(rho[element]);
            }
        });
        moduli
    }

    /// The three entries of K x that belong to `node`, summed over its active elements and, in each, over
    /// the element's eight nodes, taking only the columns that `columns` names:
    /// (K x)_(3n+c) = sum over elements e around n of E_e * sum over b, d of K_unit[3a+c][3b+d] x_(3m_b+d),
    /// where a is the node's place in element e and m_b the element's node b.
    fn gather_node(&self, moduli: &[f64], node: usize, x: &[f64], columns: Columns) -> [f64; 3] {
        let mut block = [0.0; 3];
        for (place, element) in self.grid.node_elements(node).iter().enumerate() {
            let Some(element) = *element else { continue };
            let modulus = moduli[element];
            if modulus == 0.0 {
                continue;
            }
            let dofs = self.grid.element_dofs(element);
            for (c, entry) in block.iter_mut().enumerate() {
                let row = &self.k_unit[3 * place + c];
                let mut sum = 0.0;
                for (col, &dof) in dofs.iter().enumerate() {
                    let take = match columns {
                        Columns::Free => !self.fixed[dof],
                        Columns::Fixed => self.fixed[dof],
                    };
                    if take {
                        sum += row[col] * x[dof];
                    }
                }
                *entry += modulus * sum;
            }
        }
        block
    }

    /// out = K x with identity rows and columns for the fixed degrees of freedom. Each node writes its
    /// own three entries.
    fn multiply(&self, moduli: &[f64], x: &[f64], out: &mut [f64]) {
        self.threads.for_each_piece(out, 3, |node, values| {
            let block = self.gather_node(moduli, node, x, Columns::Free);
            for (c, value) in values.iter_mut().enumerate() {
                let dof = 3 * node + c;
                *value = if self.fixed[dof] { x[dof] } else { block[c] };
            }
        });
    }

    /// The diagonal of K (1 on fixed degrees of freedom), for the smoother of the finest level.
    fn diagonal(&self, moduli: &[f64]) -> Vec<f64> {
        let mut diagonal_values = vec![1.0; self.grid.n_dofs()];
        self.threads
            .for_each_piece(&mut diagonal_values, 3, |node, values| {
                for (c, value) in values.iter_mut().enumerate() {
                    if self.fixed[3 * node + c] {
                        continue;
                    }
                    let mut diagonal = 0.0;
                    for (place, element) in self.grid.node_elements(node).iter().enumerate() {
                        if let Some(element) = *element {
                            diagonal += moduli[element] * self.k_unit[3 * place + c][3 * place + c];
                        }
                    }
                    // A free degree of freedom touches at least one active element, whose modulus is at
                    // least E_min > 0, so the diagonal is positive.
                    *value = diagonal;
                }
            });
        diagonal_values
    }

    /// Moves the held displacements to the right-hand side, solves for the free degrees of freedom by
    /// conjugate gradients preconditioned by the multigrid built for this density, and puts the held
    /// values back.
    fn solve_system(
        &self,
        rho: &[f64],
        rhs: &[f64],
        held_values: Option<&[f64]>,
        start: Option<&[f64]>,
    ) -> Result<(Vec<f64>, SolveReport), EngineError> {
        let n_dofs = self.grid.n_dofs();
        for dof in 0..n_dofs {
            if self.orphan[dof] && rhs[dof] != 0.0 {
                let node = dof / 3;
                let (i, j, k) = self.grid.node_position(node);
                return Err(EngineError::invalid_input(format!(
                    "a load is put on node {node} (i={i}, j={j}, k={k}), which no active element touches"
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
        let moduli = self.moduli(rho);

        // b = f - K_free,held * held on the free degrees of freedom, 0 on the fixed ones.
        let mut b = vec![0.0; n_dofs];
        self.threads.for_each_piece(&mut b, 3, |node, values| {
            let lifted = self.gather_node(&moduli, node, &held, Columns::Fixed);
            for (c, value) in values.iter_mut().enumerate() {
                let dof = 3 * node + c;
                if !self.fixed[dof] {
                    *value = rhs[dof] - lifted[c];
                }
            }
        });
        // The first guess on the free degrees of freedom; the fixed ones start (and stay) at zero.
        let mut x = vec![0.0; n_dofs];
        if let Some(values) = start {
            for dof in 0..n_dofs {
                if !self.fixed[dof] {
                    x[dof] = values[dof];
                }
            }
        }

        let diagonal = self.diagonal(&moduli);
        let multigrid = self.multigrid.build(
            &moduli,
            &self.k_unit,
            &diagonal,
            self.smoother,
            &self.threads,
        )?;
        let fine_product = |p: &[f64], q: &mut [f64]| self.multiply(&moduli, p, q);
        let mut cycles = 0;
        let mut report = conjugate_gradient::solve(
            &self.threads,
            fine_product,
            |r, z| {
                cycles += 1;
                multigrid.apply(&fine_product, r, z);
            },
            &b,
            &mut x,
            self.tolerance,
            self.max_steps,
        )
        .map_err(|error| self.name_the_usual_cause(error))?;
        report.products += cycles * multigrid.fine_products_per_cycle();
        for dof in 0..n_dofs {
            if self.fixed[dof] {
                x[dof] = held[dof];
            }
        }
        Ok((x, report))
    }

    /// A solve that used up its `max_steps` with a finite residual gets the usual cause added to its
    /// message; its code and details stay as they are.
    fn name_the_usual_cause(&self, mut error: EngineError) -> EngineError {
        let detail = |name: &str| {
            error
                .details
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
        };
        let used_up = match detail("steps") {
            Some(DetailValue::Integer(steps)) => usize::try_from(steps) == Ok(self.max_steps),
            _ => false,
        };
        let finite = match detail("relative_residual") {
            Some(DetailValue::Number(residual)) => residual.is_finite(),
            _ => false,
        };
        if error.code == ErrorCode::SolverNotConverged && used_up && finite {
            error.message.push_str(
                ". The usual cause is material that is not connected to the supports: solid joined to the \
                 rest only through void, whose stiffness is e_min (detached islands, or a loaded piece \
                 that floats); check the density for it, or allow more steps (max_steps; \
                 solver_max_steps on a Problem)",
            );
        }
        error
    }

    /// a_e^T K_unit b_e for one element.
    fn unit_energy_product(&self, dofs: &[usize; 24], a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;
        for (row, &dof_row) in dofs.iter().enumerate() {
            let mut row_sum = 0.0;
            for (col, &dof_col) in dofs.iter().enumerate() {
                row_sum += self.k_unit[row][col] * b[dof_col];
            }
            sum += a[dof_row] * row_sum;
        }
        sum
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
                "{name} needs one entry per degree of freedom (3 per node)"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorCode;
    use crate::multigrid_smoother::SWEEPS;

    fn unit_material() -> Material {
        Material {
            youngs_modulus: 1.0,
            poisson_ratio: 0.3,
            e_min: 1e-9,
            penalty: 3.0,
        }
    }

    /// Holds every degree of freedom on the face x = 0 (a clamped cantilever).
    fn clamped_left(grid: &Grid3d) -> Vec<bool> {
        let mut held = vec![false; grid.n_dofs()];
        for k in 0..=grid.nz() {
            for j in 0..=grid.ny() {
                let node = grid.node_index(0, j, k);
                for c in 0..3 {
                    held[3 * node + c] = true;
                }
            }
        }
        held
    }

    /// Densities between 0.2 and 1 that differ from element to element, without a random generator.
    fn uneven_density(n: usize) -> Vec<f64> {
        (0..n)
            .map(|e| 0.2 + 0.8 * (0.5 + 0.5 * (1.7 * e as f64).sin()))
            .collect()
    }

    #[test]
    fn the_product_equals_a_dense_assembled_matrix() {
        // 3 x 2 x 2 elements, one inactive, the face x = 0 clamped, uneven densities.
        let mut active = vec![true; 12];
        active[11] = false;
        let grid = Grid3d::new(3, 2, 2, 1.5, active).unwrap();
        let held = clamped_left(&grid);
        let model = Elasticity3d::new(grid.clone(), unit_material(), held, 1e-10, 100, 1).unwrap();
        let rho = uneven_density(12);
        let n = grid.n_dofs();

        // Dense K: element matrices added in, then fixed rows and columns replaced by the identity.
        let mut k = vec![vec![0.0; n]; n];
        for (element, &rho_e) in rho.iter().enumerate() {
            if !grid.is_active(element) {
                continue;
            }
            let modulus = unit_material().modulus(rho_e);
            let dofs = grid.element_dofs(element);
            for (a, &row) in dofs.iter().enumerate() {
                for (b, &col) in dofs.iter().enumerate() {
                    k[row][col] += modulus * model.k_unit[a][b];
                }
            }
        }
        for dof in 0..n {
            if model.fixed[dof] {
                for other in 0..n {
                    k[dof][other] = 0.0;
                    k[other][dof] = 0.0;
                }
                k[dof][dof] = 1.0;
            }
        }
        let x: Vec<f64> = (0..n).map(|i| (0.37 * i as f64).cos()).collect();
        let product = model.product(&rho, &x).unwrap();
        for row in 0..n {
            let expected: f64 = (0..n).map(|col| k[row][col] * x[col]).sum();
            assert!((product[row] - expected).abs() < 1e-13, "row {row}");
        }
        // The node of the inactive corner element that no other element touches is an orphan.
        let corner = grid.node_index(3, 2, 2);
        assert!(model.orphan[3 * corner]);
    }

    #[test]
    fn compliance_is_twice_the_strain_energy_and_sensitivity_matches_differences() {
        let grid = Grid3d::full(4, 2, 2, 2.0).unwrap();
        let held = clamped_left(&grid);
        let model = Elasticity3d::new(grid.clone(), unit_material(), held, 1e-13, 1000, 1).unwrap();
        let rho = uneven_density(16);
        let mut loads = vec![0.0; grid.n_dofs()];
        loads[3 * grid.node_index(4, 2, 2) + 2] = -1.0;
        let compliance = |rho: &[f64]| -> f64 {
            let (u, _) = model.solve(rho, &loads, None, None).unwrap();
            loads.iter().zip(&u).map(|(f, u)| f * u).sum()
        };
        let (u, report) = model.solve(&rho, &loads, None, None).unwrap();
        assert!(report.relative_residual <= 1e-13);
        let energy: f64 = model.element_strain_energy(&rho, &u).unwrap().iter().sum();
        assert!((compliance(&rho) - 2.0 * energy).abs() < 1e-10 * compliance(&rho));

        let sensitivity = model.sensitivity(&rho, &u, &u).unwrap();
        let step = 1e-6;
        for element in [0, 7, 15] {
            let mut up = rho.clone();
            let mut down = rho.clone();
            up[element] += step;
            down[element] -= step;
            let difference = (compliance(&up) - compliance(&down)) / (2.0 * step);
            assert!((difference - sensitivity[element]).abs() < 1e-6 * difference.abs());
        }
    }

    #[test]
    fn held_values_are_lifted_and_reproduce_a_uniform_stretch() {
        // Every boundary node held at u = eps x: the inside nodes must follow, with uniform stress.
        let grid = Grid3d::full(3, 3, 3, 1.0).unwrap();
        let eps = 1e-3;
        let mut held = vec![false; grid.n_dofs()];
        let mut values = vec![0.0; grid.n_dofs()];
        for node in 0..grid.n_nodes() {
            let (i, j, k) = grid.node_position(node);
            values[3 * node] = eps * i as f64;
            let on_boundary = [i, j, k].contains(&0) || i == 3 || j == 3 || k == 3;
            if on_boundary {
                held[3 * node..3 * node + 3].fill(true);
            }
        }
        let model = Elasticity3d::new(grid.clone(), unit_material(), held, 1e-12, 100, 1).unwrap();
        let rho = vec![1.0; 27];
        let zeros = vec![0.0; grid.n_dofs()];
        let (u, _) = model.solve(&rho, &zeros, Some(&values), None).unwrap();
        for dof in 0..grid.n_dofs() {
            assert!((u[dof] - values[dof]).abs() < 1e-12, "dof {dof}");
        }
        let stress = model.element_stress(&rho, &u).unwrap();
        let lambda = 0.3 / (1.3 * 0.4);
        assert!((stress[0] - (lambda + 1.0 / 1.3) * eps).abs() < 1e-12);
    }

    #[test]
    fn an_exact_start_takes_no_step_and_too_few_steps_fail() {
        // 12 x 6 x 6 elements (1911 degrees of freedom): one level, the preconditioner is the direct
        // solve and one step reaches the solution.
        let grid = Grid3d::full(12, 6, 6, 1.0).unwrap();
        let held = clamped_left(&grid);
        let model = Elasticity3d::new(grid.clone(), unit_material(), held, 1e-8, 500, 1).unwrap();
        assert_eq!(model.multigrid_levels(), 1);
        let rho = vec![1.0; grid.n_elements()];
        let mut loads = vec![0.0; grid.n_dofs()];
        loads[3 * grid.node_index(12, 3, 3) + 1] = 1.0;
        let (u, report) = model.solve(&rho, &loads, None, None).unwrap();
        assert_eq!((report.steps, report.products), (1, 2));
        let (_, again) = model.solve(&rho, &loads, None, Some(&u)).unwrap();
        assert_eq!(again.steps, 0);

        // 18 x 8 x 8 (4617 degrees of freedom): two levels, one V-cycle per step with SWEEPS sweeps
        // before and after (2 SWEEPS fine products) besides the step's own product.
        let grid = Grid3d::full(18, 8, 8, 1.0).unwrap();
        let held = clamped_left(&grid);
        let rho = vec![1.0; grid.n_elements()];
        let mut loads = vec![0.0; grid.n_dofs()];
        loads[3 * grid.node_index(18, 4, 4) + 1] = 1.0;
        let model =
            Elasticity3d::new(grid.clone(), unit_material(), held.clone(), 1e-8, 500, 1).unwrap();
        assert_eq!(model.multigrid_levels(), 2);
        let (_, report) = model.solve(&rho, &loads, None, None).unwrap();
        assert!(report.steps > 1);
        assert_eq!(report.products, 1 + report.steps * (1 + 2 * SWEEPS));
        let short = Elasticity3d::new(grid, unit_material(), held, 1e-8, 2, 1).unwrap();
        let error = short.solve(&rho, &loads, None, None).unwrap_err();
        assert_eq!(error.code, ErrorCode::SolverNotConverged);
        assert!(error.message.contains("within 2 steps; it reached "));
        assert!(error.message.contains("not connected to the supports"));
        assert!(error.message.contains("a loaded piece that floats"));
    }

    /// The V-cycle of `model` for `rho`, applied to `r`.
    fn v_cycle(model: &Elasticity3d, rho: &[f64], r: &[f64]) -> Vec<f64> {
        let moduli = model.moduli(rho);
        let diagonal = model.diagonal(&moduli);
        let multigrid = model
            .multigrid
            .build(
                &moduli,
                &model.k_unit,
                &diagonal,
                model.smoother,
                &model.threads,
            )
            .unwrap();
        let fine_product = |p: &[f64], q: &mut [f64]| model.multiply(&moduli, p, q);
        let mut z = vec![0.0; r.len()];
        multigrid.apply(&fine_product, r, &mut z);
        z
    }

    #[test]
    fn the_v_cycle_is_symmetric_positive_and_leaves_fixed_entries_at_zero() {
        // 19 x 9 x 7 elements (odd counts: coarse nodes outside the fine grid at the far edges), an
        // inactive block, a support patch that ends one node short of the odd edge y = 9, and a 0-and-1
        // design of contrast 1e9.
        let (nx, ny, nz) = (19, 9, 7);
        let mut active = vec![true; nx * ny * nz];
        for k in 4..nz {
            for j in 0..ny {
                for i in 12..nx {
                    active[i + nx * (j + ny * k)] = false;
                }
            }
        }
        let grid = Grid3d::new(nx, ny, nz, 1.0, active).unwrap();
        let mut held = vec![false; grid.n_dofs()];
        for k in 0..=nz {
            for j in 0..ny {
                let node = grid.node_index(0, j, k);
                held[3 * node..3 * node + 3].fill(true);
            }
        }
        let rho: Vec<f64> = (0..grid.n_elements())
            .map(|e| {
                let (i, j, k) = grid.element_position(e);
                if i < 2 || (j + k) % 4 < 2 || i % 5 == 0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let model =
            Elasticity3d::new(grid.clone(), unit_material(), held.clone(), 1e-8, 500, 1).unwrap();
        assert!(model.multigrid_levels() >= 2);
        let fixed = model.fixed.clone();
        let vector = |seed: f64| -> Vec<f64> {
            (0..grid.n_dofs())
                .map(|d| {
                    if fixed[d] {
                        0.0
                    } else {
                        (seed * d as f64).sin()
                    }
                })
                .collect()
        };
        let (a, b) = (vector(0.37), vector(1.13));
        let (ma, mb) = (v_cycle(&model, &rho, &a), v_cycle(&model, &rho, &b));
        let dot = |x: &[f64], y: &[f64]| -> f64 { x.iter().zip(y).map(|(p, q)| p * q).sum() };
        let (amb, bma) = (dot(&a, &mb), dot(&b, &ma));
        assert!(
            (amb - bma).abs() < 1e-10 * amb.abs().max(bma.abs()),
            "{amb} {bma}"
        );
        assert!(dot(&a, &ma) > 0.0 && dot(&b, &mb) > 0.0);
        assert!(fixed.iter().zip(&ma).all(|(f, v)| !*f || *v == 0.0));
        // The same numbers on 5 threads.
        let threaded = Elasticity3d::new(grid, unit_material(), held, 1e-8, 500, 5).unwrap();
        let threaded_ma = v_cycle(&threaded, &rho, &a);
        assert!(
            threaded_ma
                .iter()
                .zip(&ma)
                .all(|(p, q)| p.to_bits() == q.to_bits())
        );
    }

    #[test]
    fn a_part_that_is_not_held_is_refused_when_built() {
        let grid = Grid3d::full(4, 2, 2, 1.0).unwrap();
        let free = vec![false; grid.n_dofs()];
        let error = Elasticity3d::new(grid, unit_material(), free, 1e-8, 10, 1).unwrap_err();
        assert_eq!(error.code, ErrorCode::SingularSystem);
    }

    #[test]
    fn nodes_outside_the_part_stay_at_zero_and_take_no_load() {
        // A 2 x 2 x 2 block clamped at x = 0 with its far upper corner element outside the part.
        let mut active = vec![true; 8];
        active[7] = false;
        let grid = Grid3d::new(2, 2, 2, 1.0, active).unwrap();
        let model = Elasticity3d::new(
            grid.clone(),
            unit_material(),
            clamped_left(&grid),
            1e-10,
            200,
            1,
        )
        .unwrap();
        let rho = vec![0.5; 8];
        let mut loads = vec![0.0; grid.n_dofs()];
        loads[3 * grid.node_index(2, 0, 0) + 2] = -1.0;
        let (u, _) = model.solve(&rho, &loads, None, None).unwrap();
        let corner = grid.node_index(2, 2, 2);
        assert_eq!(&u[3 * corner..3 * corner + 3], &[0.0; 3]);
        assert!(u[3 * grid.node_index(2, 0, 0) + 2] < 0.0);
        loads[3 * corner] = 1.0;
        let error = model.solve(&rho, &loads, None, None).unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidInput);
    }

    #[test]
    fn results_do_not_depend_on_the_thread_count() {
        // 24 x 12 x 12 elements (12675 degrees of freedom, more than twelve blocks of a sum), a corner
        // outside the part, uneven densities, the face x = 0 held at a prescribed displacement. The
        // verification table checks the same on a solve to 1e-10 (row 15).
        let (nx, ny, nz) = (24, 12, 12);
        let mut active = vec![true; nx * ny * nz];
        for k in 0..nz {
            for j in 6..ny {
                for i in 18..nx {
                    active[i + nx * (j + ny * k)] = false;
                }
            }
        }
        let grid = Grid3d::new(nx, ny, nz, 1.0, active).unwrap();
        let held = clamped_left(&grid);
        let values: Vec<f64> = (0..grid.n_dofs())
            .map(|d| if d % 3 == 2 { 1e-3 } else { 0.0 })
            .collect();
        let mut loads = vec![0.0; grid.n_dofs()];
        loads[3 * grid.node_index(nx, 0, nz) + 2] = -1.0;
        let rho = uneven_density(grid.n_elements());
        let x: Vec<f64> = (0..grid.n_dofs())
            .map(|d| (0.37 * d as f64).cos())
            .collect();
        let run = |threads: usize| {
            let model = Elasticity3d::new(
                grid.clone(),
                unit_material(),
                held.clone(),
                1e-3,
                500,
                threads,
            )
            .unwrap();
            let product = model.product(&rho, &x).unwrap();
            // A solve stopped at a loose tolerance: a few steps, so that the test stays short in a
            // debug build, but every step's sums, updates, products and V-cycles run on the threads.
            let (u, report) = model.solve(&rho, &loads, Some(&values), None).unwrap();
            let stress = model.element_stress(&rho, &u).unwrap();
            let sensitivity = model.sensitivity(&rho, &u, &x).unwrap();
            let bits = |v: &[f64]| v.iter().map(|value| value.to_bits()).collect::<Vec<u64>>();
            let residual = report.relative_residual.to_bits();
            (
                bits(&product),
                bits(&u),
                bits(&stress),
                bits(&sensitivity),
                report.steps,
                residual,
            )
        };
        let one = run(1);
        assert!(one.4 >= 3);
        for threads in [2, 5, 12] {
            assert!(run(threads) == one, "{threads} threads");
        }
    }

    #[test]
    fn wrong_input_is_refused() {
        let grid = Grid3d::full(2, 1, 1, 1.0).unwrap();
        let held = clamped_left(&grid);
        assert!(
            Elasticity3d::new(grid.clone(), unit_material(), vec![false; 5], 1e-6, 10, 1).is_err()
        );
        assert!(
            Elasticity3d::new(grid.clone(), unit_material(), held.clone(), 0.0, 10, 1).is_err()
        );
        let no_threads =
            Elasticity3d::new(grid.clone(), unit_material(), held.clone(), 1e-6, 10, 0);
        assert_eq!(no_threads.unwrap_err().code, ErrorCode::InvalidInput);
        let model = Elasticity3d::new(grid.clone(), unit_material(), held, 1e-6, 10, 1).unwrap();
        let loads = vec![0.0; grid.n_dofs()];
        assert!(model.solve(&[1.0, 2.0], &loads, None, None).is_err());
        assert!(model.solve(&[1.0, 1.0], &loads[1..], None, None).is_err());
        assert!(
            model
                .solve(&[1.0, 1.0], &loads, None, Some(&[f64::NAN; 36]))
                .is_err()
        );
    }
}
