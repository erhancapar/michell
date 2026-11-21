//! `michell.HelmholtzFilter`: the PDE (Helmholtz) filter on a `Grid3d`.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::arrays::copy_vector;
use crate::errors::{invalid_input, run_engine};
use crate::grid::non_negative;
use crate::grid3d::Grid3d;

/// The Helmholtz filter (Lazarov and Sigmund 2011): rho solves -r^2 lap(psi) + psi = x on the active
/// elements with a zero-flux boundary, r = radius / (2 sqrt 3), on 8-node elements of the same grid with
/// a lumped mass matrix;
/// element values go to the nodes, node values are averaged back to the elements. The radius is in mm.
/// Solved by conjugate gradients with a diagonal preconditioner to the relative residual `tolerance`, on
/// `threads` threads; the numbers do not depend on the thread count.
#[pyclass(frozen, subclass, module = "michell", name = "HelmholtzFilter")]
pub struct HelmholtzFilter {
    pub inner: michell::HelmholtzFilter,
}

#[pymethods]
impl HelmholtzFilter {
    #[new]
    #[pyo3(signature = (grid, radius, threads = 1, *, tolerance = michell::helmholtz_filter::DEFAULT_TOLERANCE))]
    fn new(
        py: Python<'_>,
        grid: &Bound<'_, Grid3d>,
        radius: f64,
        threads: i64,
        tolerance: f64,
    ) -> PyResult<HelmholtzFilter> {
        let threads = non_negative(py, threads, "threads")?;
        let grid = grid.get().inner.clone();
        let inner = run_engine(py, || {
            michell::HelmholtzFilter::new(&grid, radius, tolerance, threads)
        })?;
        Ok(HelmholtzFilter { inner })
    }

    /// The radius R, mm.
    #[getter]
    fn radius(&self) -> f64 {
        self.inner.radius()
    }

    /// The length scale r = R / (2 sqrt 3) of the equation, mm.
    #[getter]
    fn length(&self) -> f64 {
        self.inner.length()
    }

    /// The side of one element of the grid the filter was built on, mm.
    #[getter]
    fn element_size(&self) -> f64 {
        self.inner.element_size()
    }

    /// The relative residual at which a solve stops.
    #[getter]
    fn tolerance(&self) -> f64 {
        self.inner.tolerance()
    }

    /// The number of threads every call runs on.
    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }

    /// rho = the filtered field of x, one value per element (0 on inactive elements). The map keeps a
    /// field in [0, 1] inside [0, 1] for any radius (the mass matrix is lumped); a value outside by no
    /// more than solver round-off (1e-6) is put on the bound.
    ///
    /// A long call: the input is copied, then Python's lock is released for the solve.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = copy_vector(x, "x")?;
        let rho = run_engine(py, || self.inner.forward(&x))?;
        Ok(PyArray1::from_vec(py, rho))
    }

    /// dg/dx from dg/drho (`grad`): the chain rule back through the filter. The map is linear and
    /// symmetric, so this is the same solve as `forward` applied to `grad`; `x` (the field the filter was
    /// applied to) is there because every filter takes it, and only its length is checked.
    ///
    /// A long call: the input is copied, then Python's lock is released for the solve.
    fn backward<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        grad: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n_elements = self.inner.n_elements();
        if copy_vector(x, "x")?.len() != n_elements {
            return Err(invalid_input(
                py,
                format!("x needs one entry per element ({n_elements})"),
            ));
        }
        let grad = copy_vector(grad, "grad")?;
        let result = run_engine(py, || self.inner.backward(&grad))?;
        Ok(PyArray1::from_vec(py, result))
    }
}
