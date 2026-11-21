//! `michell.Oc`: the optimality-criteria update of the 88-line code.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::arrays::copy_vector;
use crate::errors::{invalid_input, run_engine};
use crate::filter::{DensityFilter, Inner};
use crate::grid::Grid;

/// The optimality-criteria update for one goal and one limit that is linear in the density.
#[pyclass(frozen, module = "michell", name = "Oc")]
pub struct Oc {
    inner: michell::Oc,
}

#[pymethods]
impl Oc {
    /// `move_limit`: the largest change of one design variable in one update.
    #[new]
    #[pyo3(signature = (grid, move_limit = 0.2))]
    fn new(py: Python<'_>, grid: &Bound<'_, Grid>, move_limit: f64) -> PyResult<Oc> {
        let grid = grid.get().inner.clone();
        let inner = run_engine(py, || michell::Oc::new(&grid, move_limit))?;
        Ok(Oc { inner })
    }

    #[getter]
    fn move_limit(&self) -> f64 {
        self.inner.move_limit()
    }

    /// The new design variables. `d_goal` and `d_limit` are gradients with respect to x; the limit is
    /// sum(limit_weights * rho) <= limit_bound, measured on rho = filter.forward(x_new) (rho = x_new
    /// without a filter).
    ///
    /// A long call: the inputs are copied, then Python's lock is released for the bisection.
    #[pyo3(signature = (x, d_goal, d_limit, limit_weights, limit_bound, filter = None))]
    #[allow(clippy::too_many_arguments)] // the signature Python sees; each argument is one array
    fn update<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        d_goal: &Bound<'py, PyAny>,
        d_limit: &Bound<'py, PyAny>,
        limit_weights: &Bound<'py, PyAny>,
        limit_bound: f64,
        filter: Option<&Bound<'py, DensityFilter>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = copy_vector(x, "x")?;
        let d_goal = copy_vector(d_goal, "d_goal")?;
        let d_limit = copy_vector(d_limit, "d_limit")?;
        let limit_weights = copy_vector(limit_weights, "limit_weights")?;
        // The filter is frozen (it never changes after it is built), so it can be read without the lock.
        let filter = match filter.map(|f| &f.get().inner) {
            None => None,
            Some(Inner::Plane(filter)) => Some(filter),
            Some(Inner::Solid(_)) => {
                return Err(invalid_input(
                    py,
                    "OC takes the density filter on a 2D grid only",
                ));
            }
        };
        let x_new = run_engine(py, || {
            self.inner
                .update(&x, &d_goal, &d_limit, &limit_weights, limit_bound, filter)
        })?;
        Ok(PyArray1::from_vec(py, x_new))
    }
}
