//! `michell.Projection`: the smoothed step towards 0 or 1 around a threshold.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::arrays::{as_slice, read_vector};
use crate::errors::{run_engine, run_in_place};

/// The projection (Wang, Lazarov and Sigmund 2011):
/// P(x) = (tanh(beta eta) + tanh(beta (x - eta))) / (tanh(beta eta) + tanh(beta (1 - eta))), applied to
/// each value on its own, with the sharpness beta > 0 and the threshold 0 <= eta <= 1. It needs no grid.
#[pyclass(frozen, subclass, module = "michell", name = "Projection")]
pub struct Projection {
    pub inner: michell::Projection,
}

#[pymethods]
impl Projection {
    #[new]
    #[pyo3(signature = (beta, eta = 0.5))]
    fn new(py: Python<'_>, beta: f64, eta: f64) -> PyResult<Projection> {
        let inner = run_engine(py, || michell::Projection::new(beta, eta))?;
        Ok(Projection { inner })
    }

    /// The sharpness beta.
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta()
    }

    /// The threshold eta.
    #[getter]
    fn eta(&self) -> f64 {
        self.inner.eta()
    }

    /// P(x) of each entry of x, as a new array.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = read_vector(x, "x")?;
        let x = as_slice(py, &x, "x")?;
        let result = run_in_place(py, || self.inner.forward(x))?;
        Ok(PyArray1::from_vec(py, result))
    }

    /// dg/dx from dg/dP (`grad`): each entry of `grad` times the derivative P' at the same entry of `x`,
    /// the field the projection was applied to.
    fn backward<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        grad: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = read_vector(x, "x")?;
        let x = as_slice(py, &x, "x")?;
        let grad = read_vector(grad, "grad")?;
        let grad = as_slice(py, &grad, "grad")?;
        let result = run_in_place(py, || self.inner.backward(x, grad))?;
        Ok(PyArray1::from_vec(py, result))
    }
}
