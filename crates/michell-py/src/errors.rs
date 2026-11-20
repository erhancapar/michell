//! `michell.EngineError`, and running an engine call safely: with any panic caught, so that no panic ever
//! reaches Python, and with Python's lock released for long calls.

use std::panic::{AssertUnwindSafe, catch_unwind};

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use michell::{DetailValue, ErrorCode};

create_exception!(
    michell,
    EngineError,
    PyException,
    "An error from the michell engine, with `code` (stable, for programs), `message` (for people) and \
     `details` (a dict of the numbers involved)."
);

/// Turns an engine error into a Python `EngineError` with its three fields set.
pub fn to_py_err(py: Python<'_>, error: michell::EngineError) -> PyErr {
    let py_error = EngineError::new_err(error.to_string());
    let filled = fill_fields(py, &py_error, &error);
    match filled {
        Ok(()) => py_error,
        Err(failure) => failure,
    }
}

fn fill_fields(py: Python<'_>, py_error: &PyErr, error: &michell::EngineError) -> PyResult<()> {
    let details = PyDict::new(py);
    for (name, value) in &error.details {
        match value {
            DetailValue::Integer(number) => details.set_item(name, number)?,
            DetailValue::Number(number) => details.set_item(name, number)?,
        }
    }
    let value = py_error.value(py);
    value.setattr("code", error.code.as_str())?;
    value.setattr("message", &error.message)?;
    value.setattr("details", details)?;
    Ok(())
}

/// An `invalid_input` error, for problems the binding itself finds (wrong array type, wrong values).
pub fn invalid_input(py: Python<'_>, message: impl Into<String>) -> PyErr {
    to_py_err(py, michell::EngineError::invalid_input(message))
}

/// Runs `work` with Python's lock released and turns its result into a Python result.
///
/// For long calls (a solve, an optimiser update). While the lock is released, other Python threads run
/// and could change a NumPy array, so `work` must only read owned copies of its input arrays, never the
/// arrays themselves (rule 2 of "What crosses the boundary" in `docs/design/engine-boundary.md`).
pub fn run_engine<T, F>(py: Python<'_>, work: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> Result<T, michell::EngineError> + Send,
{
    let outcome = py.detach(|| catch_unwind(AssertUnwindSafe(work)));
    finish(py, outcome)
}

/// Runs `work` while keeping Python's lock, so it may read NumPy arrays in place.
///
/// For short calls (one pass over the elements), where copying the inputs would cost as much as the work.
pub fn run_in_place<T, F>(py: Python<'_>, work: F) -> PyResult<T>
where
    F: FnOnce() -> Result<T, michell::EngineError>,
{
    let outcome = catch_unwind(AssertUnwindSafe(work));
    finish(py, outcome)
}

/// Turns the outcome of an engine call into a Python result. A panic inside the call is a bug in michell:
/// it is reported as `internal_error` instead of crossing into Python.
fn finish<T>(
    py: Python<'_>,
    outcome: std::thread::Result<Result<T, michell::EngineError>>,
) -> PyResult<T> {
    match outcome {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(to_py_err(py, error)),
        Err(panic) => {
            let text = if let Some(text) = panic.downcast_ref::<&str>() {
                text.to_string()
            } else if let Some(text) = panic.downcast_ref::<String>() {
                text.clone()
            } else {
                "unknown panic".to_string()
            };
            let error = michell::EngineError::new(
                ErrorCode::InternalError,
                format!("the engine stopped on a bug (please report it): {text}"),
            );
            Err(to_py_err(py, error))
        }
    }
}
