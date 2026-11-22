//! `michell._engine.iso_surface`: the closed surface where a field sampled on a lattice equals a level.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::arrays::copy_vector;
use crate::errors::run_engine;
use crate::grid::non_negative;

/// The vertices and the triangles of a surface, as two NumPy arrays of shape (n, 3) and (m, 3).
type Arrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

/// The surface where `values` equal `level` on a lattice of `counts` (nx, ny, nz) points `spacing_mm` apart,
/// point (0, 0, 0) at `origin_mm`: `values` is a contiguous float64 array with one sample per point, x
/// fastest. Returns the vertices, a float64 array of shape (n, 3) in mm, and the triangles, a float64 array
/// of shape (m, 3) holding whole-number vertex indices, counter-clockwise seen from outside (the side of
/// larger values is inside). Runs on `threads` threads; the result does not depend on the thread count.
///
/// A long call: the samples are copied, then Python's lock is released.
#[pyfunction]
#[pyo3(signature = (counts, spacing_mm, origin_mm, values, level, threads = 1))]
pub fn iso_surface<'py>(
    py: Python<'py>,
    counts: (i64, i64, i64),
    spacing_mm: f64,
    origin_mm: (f64, f64, f64),
    values: &Bound<'py, PyAny>,
    level: f64,
    threads: i64,
) -> PyResult<Arrays<'py>> {
    let lattice = michell::Lattice {
        counts: [
            non_negative(py, counts.0, "nx")?,
            non_negative(py, counts.1, "ny")?,
            non_negative(py, counts.2, "nz")?,
        ],
        spacing: spacing_mm,
        origin: [origin_mm.0, origin_mm.1, origin_mm.2],
    };
    let threads = non_negative(py, threads, "threads")?;
    let values = copy_vector(values, "values")?;
    let surface = run_engine(py, || {
        michell::iso_surface(&lattice, &values, level, threads)
    })?;
    let n_vertices = surface.vertices.len();
    let n_triangles = surface.triangles.len();
    let vertices: Vec<f64> = surface.vertices.into_iter().flatten().collect();
    // Vertex indices are below 2^53, so they are exact in float64.
    let triangles: Vec<f64> = surface
        .triangles
        .into_iter()
        .flatten()
        .map(|v| v as f64)
        .collect();
    Ok((
        PyArray1::from_vec(py, vertices).reshape([n_vertices, 3])?,
        PyArray1::from_vec(py, triangles).reshape([n_triangles, 3])?,
    ))
}
