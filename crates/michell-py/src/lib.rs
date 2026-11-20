//! Python bindings for the michell engine, built as the extension module `michell._engine`.
//!
//! This crate only converts between Python and Rust; the work itself lives in the `michell` crate.

use pyo3::pymodule;

/// The `michell._engine` module that Python imports.
#[pymodule]
mod _engine {
    use numpy::PyReadonlyArray1;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// Return the version of the Rust engine crate.
    #[pyfunction]
    fn engine_version() -> &'static str {
        michell::VERSION
    }

    /// Sum a 1-D float64 NumPy array in Rust, reading its memory without copying it.
    #[pyfunction]
    fn sum_f64(values: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        // A view such as `a[::2]` skips elements in memory, so it is not one plain slice.
        let slice = values.as_slice().map_err(|_| {
            PyValueError::new_err(
                "sum_f64 needs a contiguous array; pass numpy.ascontiguousarray(values)",
            )
        })?;
        Ok(michell::sum(slice))
    }
}
