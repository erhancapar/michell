//! Reading NumPy arrays from Python, by the rules of `docs/design/engine-boundary.md`: a 1-D,
//! contiguous float64 array is read in place; anything else is refused, never converted.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::errors::invalid_input;

/// Borrows `value` as a 1-D contiguous float64 array. `name` is the argument's name, for the message.
pub fn read_vector<'py>(
    value: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    let py = value.py();
    let Ok(array) = value.cast::<PyArray1<f64>>() else {
        return Err(invalid_input(
            py,
            format!(
                "{name} must be a 1-D NumPy array of float64; got {}",
                describe(value)
            ),
        ));
    };
    let Ok(readonly) = array.try_readonly() else {
        return Err(invalid_input(
            py,
            format!("{name} is being written elsewhere and cannot be read now"),
        ));
    };
    if readonly.as_slice().is_err() {
        return Err(invalid_input(
            py,
            format!("{name} must be contiguous in memory; pass numpy.ascontiguousarray({name})"),
        ));
    }
    Ok(readonly)
}

/// Reads a mask: a float64 array whose entries are all 0.0 (no) or 1.0 (yes).
pub fn read_mask(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<bool>> {
    let py = value.py();
    let array = read_vector(value, name)?;
    let values = as_slice(py, &array, name)?;
    let mut mask = Vec::with_capacity(values.len());
    for (index, &entry) in values.iter().enumerate() {
        if entry == 0.0 {
            mask.push(false);
        } else if entry == 1.0 {
            mask.push(true);
        } else {
            return Err(invalid_input(
                py,
                format!("{name} must hold only 0.0 and 1.0; entry {index} is {entry}"),
            ));
        }
    }
    Ok(mask)
}

/// A short description of a wrong argument: "a 2-D array of int64", or the Python type's name.
fn describe(value: &Bound<'_, PyAny>) -> String {
    if let Ok(array) = value.cast::<PyUntypedArray>() {
        let dtype = array.dtype().to_string();
        format!("a {}-D array of {dtype}", array.ndim())
    } else {
        match value.get_type().name() {
            Ok(name) => format!("a {name}"),
            Err(_) => "another kind of object".to_string(),
        }
    }
}

/// The numbers of an array that `read_vector` accepted.
pub fn as_slice<'a>(
    py: Python<'_>,
    array: &'a PyReadonlyArray1<'_, f64>,
    name: &str,
) -> PyResult<&'a [f64]> {
    array
        .as_slice()
        .map_err(|_| invalid_input(py, format!("{name} must be contiguous in memory")))
}
