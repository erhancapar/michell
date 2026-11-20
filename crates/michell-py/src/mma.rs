//! `michell.Mma`: the method of moving asymptotes.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::arrays::{copy_matrix, copy_vector};
use crate::errors::{invalid_input, run_engine};

/// The MMA update for variables between `x_min` and `x_max` and `n_limits` limits written f_i(x) <= 0.
///
/// Unlike the other engine objects this one changes with every update (it remembers the two previous
/// designs and the asymptotes), so it is not frozen. Calling `update` from a second Python thread while
/// one update runs raises an error instead of mixing the two.
#[pyclass(module = "michell", name = "Mma")]
pub struct Mma {
    inner: michell::Mma,
}

#[pymethods]
impl Mma {
    #[new]
    fn new(
        py: Python<'_>,
        x_min: &Bound<'_, PyAny>,
        x_max: &Bound<'_, PyAny>,
        n_limits: usize,
    ) -> PyResult<Mma> {
        let x_min = copy_vector(x_min, "x_min")?;
        let x_max = copy_vector(x_max, "x_max")?;
        let inner = run_engine(py, || michell::Mma::new(x_min, x_max, n_limits))?;
        Ok(Mma { inner })
    }

    #[getter]
    fn n_variables(&self) -> usize {
        self.inner.n_variables()
    }

    #[getter]
    fn n_limits(&self) -> usize {
        self.inner.n_limits()
    }

    /// How many updates have been made.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// The asymptotes (lower, upper) of the last update.
    fn asymptotes<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let (lower, upper) = self.inner.asymptotes();
        (
            PyArray1::from_slice(py, lower),
            PyArray1::from_slice(py, upper),
        )
    }

    /// The next design. `goal` and `d_goal` are f_0(x) and its gradient; `limits` holds f_i(x) for each
    /// limit (a 1-D array of length n_limits) and `d_limits` their gradients (a 2-D array, one row per
    /// limit).
    ///
    /// A long call: the inputs are copied, then Python's lock is released for the subproblem.
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        goal: f64,
        d_goal: &Bound<'py, PyAny>,
        limits: &Bound<'py, PyAny>,
        d_limits: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = copy_vector(x, "x")?;
        let d_goal = copy_vector(d_goal, "d_goal")?;
        let limits = copy_vector(limits, "limits")?;
        let (rows, columns, d_limits) = copy_matrix(d_limits, "d_limits")?;
        let n = self.inner.n_variables();
        let m = self.inner.n_limits();
        if rows != m || columns != n {
            return Err(invalid_input(
                py,
                format!("d_limits must have shape ({m}, {n}); got ({rows}, {columns})"),
            ));
        }
        let inner = &mut self.inner;
        let x_new = run_engine(py, || inner.update(&x, goal, &d_goal, &limits, &d_limits))?;
        Ok(PyArray1::from_vec(py, x_new))
    }
}
