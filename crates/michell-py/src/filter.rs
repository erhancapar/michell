//! `michell.DensityFilter`: the density filter with cone weights.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::arrays::{as_slice, read_vector};
use crate::errors::{invalid_input, run_engine, run_in_place};
use crate::grid::Grid;

/// The density filter: rho_e = sum_j w_ej x_j / sum_j w_ej with w_ej = max(0, radius - distance between
/// the centres of elements e and j), over the active elements. The radius is in mm.
#[pyclass(frozen, subclass, module = "michell", name = "DensityFilter")]
pub struct DensityFilter {
    pub inner: michell::DensityFilter,
}

#[pymethods]
impl DensityFilter {
    #[new]
    fn new(py: Python<'_>, grid: &Bound<'_, Grid>, radius: f64) -> PyResult<DensityFilter> {
        let grid = grid.get().inner.clone();
        let inner = run_engine(py, || michell::DensityFilter::new(&grid, radius))?;
        Ok(DensityFilter { inner })
    }

    /// The radius, mm.
    #[getter]
    fn radius(&self) -> f64 {
        self.inner.radius()
    }

    /// rho = the filtered field of x, one value per element (0 on inactive elements).
    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = read_vector(x, "x")?;
        let x = as_slice(py, &x, "x")?;
        let rho = run_in_place(py, || self.inner.forward(x))?;
        Ok(PyArray1::from_vec(py, rho))
    }

    /// dg/dx from dg/drho (`grad`): the chain rule back through the filter.
    ///
    /// `x` is the field the filter was applied to. The filter is linear, so its reverse does not depend
    /// on x; the argument is there because every filter takes it (see `docs/design/modules.md`), and only
    /// its length is checked.
    fn backward<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        grad: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = read_vector(x, "x")?;
        let n_elements = self.inner.n_elements();
        if as_slice(py, &x, "x")?.len() != n_elements {
            return Err(invalid_input(
                py,
                format!("x needs one entry per element ({n_elements})"),
            ));
        }
        let grad = read_vector(grad, "grad")?;
        let grad = as_slice(py, &grad, "grad")?;
        let result = run_in_place(py, || self.inner.backward(grad))?;
        Ok(PyArray1::from_vec(py, result))
    }
}
