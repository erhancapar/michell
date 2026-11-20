//! The multigrid preconditioner of the 3D analysis: one V-cycle, z = B r, an approximation of K^-1 r that
//! the conjugate-gradient method of `conjugate_gradient` uses as its preconditioner.
//!
//! One V-cycle on level l, for the residual r of that level (Trottenberg, Oosterlee and Schueller,
//! "Multigrid" (2001), section 2.4; as a preconditioner of conjugate gradients in topology optimisation:
//! Amir, Aage and Lazarov, "On multigrid-CG for efficient topology optimization", Struct. Multidisc.
//! Optim. 49 (2014)):
//!
//! 1. smooth from x = 0: `SWEEPS` damped Jacobi sweeps (`multigrid_smoother`);
//! 2. restrict the residual r - A x to the next coarser level (`multigrid_transfer`);
//! 3. on the coarsest level solve directly (`multigrid_coarsest`), on any other run this V-cycle there;
//! 4. interpolate that correction back and add it to x;
//! 5. smooth again with the same number of sweeps.
//!
//! With the same smoother before and after, B is symmetric; with the damping below 2 / lambda_max on every
//! level it is positive definite, which conjugate gradients need. A grid small enough to be its own
//! coarsest level has one level, and B is then the direct solve.
//!
//! What is built when. `MultigridShape` (the levels, their fixed degrees of freedom and the set the
//! coarsest solve takes out) does not depend on the density and is built once, with the analysis
//! object. `Multigrid` (the coarse matrices, the diagonals and bounds of the smoothers, the coarsest
//! factor) depends on the density and is built at the start of each solve; nothing of it is kept after.
//!
//! Threads: every step writes separate pieces of its output (the entries of one node, the matrix of one
//! coarse element) by the rules of `threads`; the coarsest factorisation and solve run on one thread. No
//! result depends on the thread count.

use crate::error::EngineError;
use crate::grid3d::Grid3d;
use crate::multigrid_coarse::{ElementMatrices, LevelMatrices};
use crate::multigrid_coarsest::{Coarsest, CoarsestShape};
use crate::multigrid_levels::{Level, build_levels};
use crate::multigrid_smoother::{Jacobi, SmootherSettings, element_bound, largest_bound};
use crate::multigrid_transfer::Transfer;
use crate::threads::Threads;

/// The parts of the multigrid that do not depend on the density.
#[derive(Clone, Debug)]
pub struct MultigridShape {
    levels: Vec<Level>,
    coarsest: CoarsestShape,
    /// The bound of lambda_max(D^-1 K) on the finest level, the same for every density.
    fine_bound: f64,
}

/// The multigrid for one density.
pub struct Multigrid<'a> {
    shape: &'a MultigridShape,
    /// The element matrices of the levels 1, 2, ... (the finest level is never stored).
    coarse: Vec<ElementMatrices>,
    /// The smoothers of every level but the coarsest.
    smoothers: Vec<Jacobi>,
    coarsest: Coarsest<'a>,
    sweeps: usize,
    threads: &'a Threads,
}

impl MultigridShape {
    /// The levels for the analysis grid `grid` with its fixed set `fixed`, and the set the coarsest
    /// solve takes out, found from the matrices at full density (every active element at modulus 1).
    pub fn new(
        grid: &Grid3d,
        fixed: &[bool],
        k_unit: &[[f64; 24]; 24],
        threads: &Threads,
    ) -> Result<MultigridShape, EngineError> {
        let levels = build_levels(grid, fixed)?;
        let full: Vec<f64> = grid
            .active()
            .iter()
            .map(|a| if *a { 1.0 } else { 0.0 })
            .collect();
        let coarse = coarse_matrices(&levels, &full, k_unit, threads);
        let last = levels.len() - 1;
        let source = match coarse.last() {
            Some(matrices) => LevelMatrices::Coarse(matrices),
            None => LevelMatrices::Fine {
                moduli: &full,
                k_unit,
            },
        };
        let coarsest = CoarsestShape::new(&levels[last], source);
        let unit: Vec<f64> = k_unit.iter().flatten().copied().collect();
        Ok(MultigridShape {
            fine_bound: element_bound(&unit),
            levels,
            coarsest,
        })
    }

    /// The levels, finest first.
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// The number of degrees of freedom the coarsest solve takes out.
    pub fn dropped_count(&self) -> usize {
        self.coarsest.dropped_count()
    }

    /// The multigrid for the moduli `moduli` of the fine elements (0 for inactive ones), with
    /// `fine_diagonal` the diagonal of the fine K (1 on fixed degrees of freedom).
    pub fn build<'a>(
        &'a self,
        moduli: &[f64],
        k_unit: &[[f64; 24]; 24],
        fine_diagonal: &[f64],
        settings: SmootherSettings,
        threads: &'a Threads,
    ) -> Result<Multigrid<'a>, EngineError> {
        let coarse = coarse_matrices(&self.levels, moduli, k_unit, threads);
        let last = self.levels.len() - 1;
        let mut smoothers = Vec::with_capacity(last);
        for (l, level) in self.levels[..last].iter().enumerate() {
            let smoother = if l == 0 {
                Jacobi::new(
                    fine_diagonal,
                    &level.fixed,
                    self.fine_bound,
                    settings.damping,
                    threads,
                )
            } else {
                let matrices = &coarse[l - 1];
                let bound = largest_bound(matrices.all(), threads);
                let diagonal = matrices.diagonal(level, threads);
                Jacobi::new(&diagonal, &level.fixed, bound, settings.damping, threads)
            };
            smoothers.push(smoother);
        }
        let source = match coarse.last() {
            Some(matrices) => LevelMatrices::Coarse(matrices),
            None => LevelMatrices::Fine { moduli, k_unit },
        };
        let coarsest = self.coarsest.factor(&self.levels[last], source)?;
        Ok(Multigrid {
            shape: self,
            coarse,
            smoothers,
            coarsest,
            sweeps: settings.sweeps,
            threads,
        })
    }
}

/// The element matrices of the levels 1, 2, ..., each by the Galerkin rule from the one before.
fn coarse_matrices(
    levels: &[Level],
    moduli: &[f64],
    k_unit: &[[f64; 24]; 24],
    threads: &Threads,
) -> Vec<ElementMatrices> {
    let mut coarse: Vec<ElementMatrices> = Vec::with_capacity(levels.len().saturating_sub(1));
    for l in 1..levels.len() {
        let source = match coarse.last() {
            Some(matrices) => LevelMatrices::Coarse(matrices),
            None => LevelMatrices::Fine { moduli, k_unit },
        };
        let matrices = ElementMatrices::galerkin(&levels[l - 1], source, &levels[l], threads);
        coarse.push(matrices);
    }
    coarse
}

impl Multigrid<'_> {
    /// The number of products with the fine matrix one V-cycle makes: `sweeps` - 1 in the first
    /// smoothing (the first sweep starts from 0), 1 for the residual, `sweeps` in the second smoothing;
    /// none when the finest level is the coarsest.
    pub fn fine_products_per_cycle(&self) -> usize {
        if self.shape.levels.len() > 1 {
            2 * self.sweeps
        } else {
            0
        }
    }

    /// z = B r, one V-cycle from the finest level. `fine_product(x, out)` writes out = K x on the finest
    /// level, with identity rows for its fixed degrees of freedom.
    pub fn apply(
        &self,
        fine_product: &(dyn Fn(&[f64], &mut [f64]) + Sync),
        r: &[f64],
        z: &mut [f64],
    ) {
        self.cycle(0, fine_product, r, z);
    }

    /// out = A x on level `l`.
    fn product(
        &self,
        l: usize,
        fine_product: &(dyn Fn(&[f64], &mut [f64]) + Sync),
        x: &[f64],
        out: &mut [f64],
    ) {
        if l == 0 {
            fine_product(x, out);
        } else {
            self.coarse[l - 1].product(&self.shape.levels[l], self.threads, x, out);
        }
    }

    /// x = B_l r on level `l` (x is overwritten).
    fn cycle(
        &self,
        l: usize,
        fine_product: &(dyn Fn(&[f64], &mut [f64]) + Sync),
        r: &[f64],
        x: &mut [f64],
    ) {
        let levels = &self.shape.levels;
        if l == levels.len() - 1 {
            self.coarsest.solve(&levels[l].fixed, r, x);
            return;
        }
        let threads = self.threads;
        let smoother = &self.smoothers[l];
        let mut ax = vec![0.0; r.len()];

        // 1. Smooth from zero.
        smoother.first_sweep(threads, r, x);
        for _ in 1..self.sweeps {
            self.product(l, fine_product, x, &mut ax);
            smoother.sweep(threads, r, &ax, x);
        }

        // 2. The residual, restricted to the coarser level.
        self.product(l, fine_product, x, &mut ax);
        let mut residual = vec![0.0; r.len()];
        threads.update(&mut residual, |dof, value| *value = r[dof] - ax[dof]);
        let transfer = Transfer {
            fine: &levels[l].grid,
            fine_fixed: &levels[l].fixed,
            coarse: &levels[l + 1].grid,
            coarse_fixed: &levels[l + 1].fixed,
        };
        let n_coarse = levels[l + 1].grid.n_dofs();
        let mut coarse_residual = vec![0.0; n_coarse];
        transfer.restrict(threads, &residual, &mut coarse_residual);

        // 3. and 4. The coarse correction, interpolated back.
        let mut coarse_correction = vec![0.0; n_coarse];
        self.cycle(
            l + 1,
            fine_product,
            &coarse_residual,
            &mut coarse_correction,
        );
        let mut correction = residual; // reuse the storage
        transfer.prolong(threads, &coarse_correction, &mut correction);
        threads.update(x, |dof, x_dof| *x_dof += correction[dof]);

        // 5. Smooth again.
        for _ in 0..self.sweeps {
            self.product(l, fine_product, x, &mut ax);
            smoother.sweep(threads, r, &ax, x);
        }
    }
}
