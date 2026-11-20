//! `michell.Elasticity3d` and `michell.element_stiffness_3d`: linear elasticity on a `Grid3d`.

use std::sync::Mutex;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::arrays::{as_slice, copy_vector, read_mask, read_vector};
use crate::errors::{invalid_input, run_engine, run_in_place};
use crate::grid::non_negative;
use crate::grid3d::Grid3d;

/// Linear elasticity on one 3D grid with one set of held degrees of freedom, solved by conjugate
/// gradients with a diagonal (Jacobi) preconditioner on a stiffness matrix that is never assembled.
///
/// Material law: E(rho) = e_min + rho^penalty (youngs_modulus - e_min), isotropic.
#[pyclass(frozen, subclass, module = "michell", name = "Elasticity3d")]
pub struct Elasticity3d {
    inner: michell::Elasticity3d,
    /// How the last `solve` or `adjoint` ended; `None` before the first and after one that failed.
    last_report: Mutex<Option<michell::SolveReport>>,
}

impl Elasticity3d {
    fn remember(&self, report: Option<michell::SolveReport>) {
        // A poisoned lock only means another thread panicked while holding it; the value is still usable.
        let mut last = self
            .last_report
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *last = report;
    }

    /// Runs a solve with Python's lock released and remembers how it ended.
    fn run_solve<'py, F>(&self, py: Python<'py>, work: F) -> PyResult<Bound<'py, PyArray1<f64>>>
    where
        F: FnOnce() -> Result<(Vec<f64>, michell::SolveReport), michell::EngineError> + Send,
    {
        match run_engine(py, work) {
            Ok((solution, report)) => {
                self.remember(Some(report));
                Ok(PyArray1::from_vec(py, solution))
            }
            Err(error) => {
                self.remember(None);
                Err(error)
            }
        }
    }
}

/// An optional array argument, copied for a call that releases Python's lock.
fn copy_optional(value: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<Option<Vec<f64>>> {
    match value {
        Some(values) => Ok(Some(copy_vector(values, name)?)),
        None => Ok(None),
    }
}

#[pymethods]
impl Elasticity3d {
    /// `held` is a float64 array with one entry per degree of freedom (3 per node): 1.0 holds it.
    /// `tolerance` is the relative residual |b - K u| / |b| at which a solve stops; more than `max_steps`
    /// steps give `solver_not_converged`.
    #[new]
    #[pyo3(signature = (
        grid, held, *, youngs_modulus, poisson_ratio, e_min, penalty, tolerance = 1e-6, max_steps = 500
    ))]
    #[allow(clippy::too_many_arguments)] // the signature Python sees; each argument is one setting
    fn new(
        py: Python<'_>,
        grid: &Bound<'_, Grid3d>,
        held: &Bound<'_, PyAny>,
        youngs_modulus: f64,
        poisson_ratio: f64,
        e_min: f64,
        penalty: f64,
        tolerance: f64,
        max_steps: i64,
    ) -> PyResult<Elasticity3d> {
        let held = read_mask(held, "held")?;
        let max_steps = non_negative(py, max_steps, "max_steps")?;
        let grid = grid.get().inner.clone();
        let material = michell::Material {
            youngs_modulus,
            poisson_ratio,
            e_min,
            penalty,
        };
        let inner = run_engine(py, || {
            michell::Elasticity3d::new(grid, material, held, tolerance, max_steps)
        })?;
        Ok(Elasticity3d {
            inner,
            last_report: Mutex::new(None),
        })
    }

    /// Displacements u with K(rho) u = loads. Loads on held degrees of freedom are ignored.
    /// `held_values` (one entry per degree of freedom) sets the displacement of the held ones; `start`
    /// is the first guess of u (zero without it).
    ///
    /// A long call: the inputs are copied, then Python's lock is released for the solve.
    #[pyo3(signature = (rho, loads, held_values = None, start = None))]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        rho: &Bound<'py, PyAny>,
        loads: &Bound<'py, PyAny>,
        held_values: Option<&Bound<'py, PyAny>>,
        start: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho = copy_vector(rho, "rho")?;
        let loads = copy_vector(loads, "loads")?;
        let held_values = copy_optional(held_values, "held_values")?;
        let start = copy_optional(start, "start")?;
        self.run_solve(py, || {
            self.inner
                .solve(&rho, &loads, held_values.as_deref(), start.as_deref())
        })
    }

    /// lambda with K(rho) lambda = rhs, every held degree of freedom at zero; `start` is the first guess.
    ///
    /// A long call: the inputs are copied, then Python's lock is released for the solve.
    #[pyo3(signature = (rho, rhs, start = None))]
    fn adjoint<'py>(
        &self,
        py: Python<'py>,
        rho: &Bound<'py, PyAny>,
        rhs: &Bound<'py, PyAny>,
        start: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho = copy_vector(rho, "rho")?;
        let rhs = copy_vector(rhs, "rhs")?;
        let start = copy_optional(start, "start")?;
        self.run_solve(py, || self.inner.adjoint(&rho, &rhs, start.as_deref()))
    }

    /// How the last `solve` or `adjoint` on this object ended: a dict with `steps` and
    /// `relative_residual`, or None before the first one and after one that raised an error.
    fn last_solve_report<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let last = *self
            .last_report
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match last {
            None => Ok(None),
            Some(report) => {
                let dict = PyDict::new(py);
                dict.set_item("steps", report.steps)?;
                dict.set_item("relative_residual", report.relative_residual)?;
                Ok(Some(dict))
            }
        }
    }

    /// K(rho) x without assembling K. Held degrees of freedom and those that touch no active element
    /// act as identity rows and columns. The product the solver uses.
    fn product<'py>(
        &self,
        py: Python<'py>,
        rho: &Bound<'py, PyAny>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho = read_vector(rho, "rho")?;
        let x = read_vector(x, "x")?;
        let (rho, x) = (as_slice(py, &rho, "rho")?, as_slice(py, &x, "x")?);
        let result = run_in_place(py, || self.inner.product(rho, x))?;
        Ok(PyArray1::from_vec(py, result))
    }

    /// Per element, -lambda_e^T (dK/drho_e) u_e: the part of dg/drho that comes through the state.
    /// For compliance (lambda = u) it is the whole gradient.
    fn sensitivity<'py>(
        &self,
        py: Python<'py>,
        rho: &Bound<'py, PyAny>,
        u: &Bound<'py, PyAny>,
        lam: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho = read_vector(rho, "rho")?;
        let u = read_vector(u, "u")?;
        let lam = read_vector(lam, "lam")?;
        let rho = as_slice(py, &rho, "rho")?;
        let u = as_slice(py, &u, "u")?;
        let lam = as_slice(py, &lam, "lam")?;
        let result = run_in_place(py, || self.inner.sensitivity(rho, u, lam))?;
        Ok(PyArray1::from_vec(py, result))
    }

    /// Per element, the strain energy 1/2 u_e^T K_e(rho) u_e in N mm (compliance = twice the sum).
    fn element_strain_energy<'py>(
        &self,
        py: Python<'py>,
        rho: &Bound<'py, PyAny>,
        u: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho = read_vector(rho, "rho")?;
        let u = read_vector(u, "u")?;
        let (rho, u) = (as_slice(py, &rho, "rho")?, as_slice(py, &u, "u")?);
        let result = run_in_place(py, || self.inner.element_strain_energy(rho, u))?;
        Ok(PyArray1::from_vec(py, result))
    }

    /// Stress at each element's centre with modulus E(rho_e), MPa: an array of shape (n_elements, 6)
    /// holding sxx, syy, szz, syz, szx, sxy.
    fn element_stress<'py>(
        &self,
        py: Python<'py>,
        rho: &Bound<'py, PyAny>,
        u: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rho = read_vector(rho, "rho")?;
        let u = read_vector(u, "u")?;
        let (rho, u) = (as_slice(py, &rho, "rho")?, as_slice(py, &u, "u")?);
        let flat = run_in_place(py, || self.inner.element_stress(rho, u))?;
        let n_elements = flat.len() / 6;
        PyArray1::from_vec(py, flat).reshape([n_elements, 6])
    }

    /// The relative residual at which a solve stops.
    #[getter]
    fn tolerance(&self) -> f64 {
        self.inner.tolerance()
    }

    /// The number of conjugate-gradient steps a solve may take.
    #[getter]
    fn max_steps(&self) -> usize {
        self.inner.max_steps()
    }

    /// The grid this model was built on (a new handle with the same contents).
    #[getter]
    fn grid(&self) -> Grid3d {
        Grid3d {
            inner: self.inner.grid().clone(),
        }
    }
}

/// The 24 x 24 stiffness matrix of one cube element of side h (8 nodes, 2 x 2 x 2 Gauss points).
///
/// Nodes: the face z = 0 counter-clockwise from (0, 0, 0) seen from above, then the face z = h in the
/// same order; degrees of freedom u, v, w of node 0, then of node 1, and so on.
#[pyfunction]
pub fn element_stiffness_3d<'py>(
    py: Python<'py>,
    youngs_modulus: f64,
    poisson_ratio: f64,
    h: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let k = run_engine(py, || {
        michell::hex8::checked_stiffness(youngs_modulus, poisson_ratio, h)
    })?;
    let rows: Vec<Vec<f64>> = k.iter().map(|row| row.to_vec()).collect();
    PyArray2::from_vec2(py, &rows)
        .map_err(|_| invalid_input(py, "could not build the 24 x 24 array"))
}
