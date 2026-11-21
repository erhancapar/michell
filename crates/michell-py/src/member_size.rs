//! `michell.member_size`: the solid-phase member-size constraint and its derivative.

use numpy::PyArray1;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::arrays::{as_slice, read_vector};
use crate::errors::run_in_place;
use crate::grid::Grid;
use crate::grid3d::Grid3d;
use crate::projection::Projection;

/// g = (1/n) sum_e I_e [min(f_e - threshold, 0)]^2 with the indicator I_e = P(f_e) exp(-c |grad f|_e^2)
/// (Zhou, Lazarov, Wang and Sigmund 2015, eqs. 8 and 10), for the filtered field `filtered` on `grid` (a
/// `Grid` or a `Grid3d`), with P the `projection`, the gradient per element side by differences between
/// neighbouring active elements, and n the number of active elements. Returns `(g, dg/dfiltered)`.
///
/// A short call: one pass over the elements on one thread, reading the array in place.
#[pyfunction]
#[pyo3(signature = (grid, filtered, projection, threshold, c))]
pub fn member_size<'py>(
    py: Python<'py>,
    grid: &Bound<'py, PyAny>,
    filtered: &Bound<'py, PyAny>,
    projection: &Bound<'py, Projection>,
    threshold: f64,
    c: f64,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    let projection = projection.get().inner;
    let constraint = if let Ok(grid) = grid.cast::<Grid>() {
        let grid = &grid.get().inner;
        run_in_place(py, || {
            michell::MemberSize::on_grid2d(grid, projection, threshold, c)
        })?
    } else if let Ok(grid) = grid.cast::<Grid3d>() {
        let grid = &grid.get().inner;
        run_in_place(py, || {
            michell::MemberSize::on_grid3d(grid, projection, threshold, c)
        })?
    } else {
        return Err(PyTypeError::new_err(
            "the member-size limit needs a michell.Grid or a michell.Grid3d",
        ));
    };
    let filtered = read_vector(filtered, "filtered")?;
    let filtered = as_slice(py, &filtered, "filtered")?;
    let (value, gradient) = run_in_place(py, || constraint.evaluate(filtered))?;
    Ok((value, PyArray1::from_vec(py, gradient)))
}
