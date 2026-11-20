//! `michell.dot`: the dot product of two vectors, summed in index order.

use pyo3::prelude::*;

use crate::arrays::{as_slice, read_vector};
use crate::errors::run_in_place;

/// a . b of two 1-D float64 arrays of the same length, summed in index order on one thread.
///
/// A short call: it keeps Python's lock and reads both arrays in place.
#[pyfunction]
pub fn dot(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
    let a = read_vector(a, "a")?;
    let a = as_slice(py, &a, "a")?;
    let b = read_vector(b, "b")?;
    let b = as_slice(py, &b, "b")?;
    run_in_place(py, || michell::dot(a, b))
}
