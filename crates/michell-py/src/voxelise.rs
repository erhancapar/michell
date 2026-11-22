//! `michell.voxel_occupancy`: the share of every voxel of a grid inside a combination of closed meshes.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::arrays::copy_matrix;
use crate::errors::{invalid_input, run_engine};
use crate::grid::non_negative;

/// The occupancy of every voxel of a grid of `counts` (nx, ny, nz) voxels of side `voxel_mm`, with the corner
/// of voxel (0, 0, 0) at `origin_mm` (x, y, z) in the meshes' coordinates: the share of the voxel inside
/// the union of the `adding` bodies minus the union of the `subtracting` bodies, one float64 per voxel in the
/// order of `Grid3d`. Each body is a pair (vertices, triangles): a C-contiguous float64 array of shape (n, 3)
/// in mm and one of shape (m, 3) holding whole-number vertex indices. Rays along x, `rays_per_side` x
/// `rays_per_side` per voxel column; `threads` threads, the numbers do not depend on the thread count.
///
/// A long call: the inputs are copied, then Python's lock is released.
#[pyfunction]
#[pyo3(signature = (counts, voxel_mm, origin_mm, adding, subtracting = Vec::new(), rays_per_side = 3, threads = 1))]
#[allow(clippy::too_many_arguments)]
pub fn voxel_occupancy<'py>(
    py: Python<'py>,
    counts: (i64, i64, i64),
    voxel_mm: f64,
    origin_mm: (f64, f64, f64),
    adding: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    subtracting: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    rays_per_side: i64,
    threads: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let grid = michell::VoxelGrid {
        counts: [
            non_negative(py, counts.0, "nx")?,
            non_negative(py, counts.1, "ny")?,
            non_negative(py, counts.2, "nz")?,
        ],
        h: voxel_mm,
        origin: [origin_mm.0, origin_mm.1, origin_mm.2],
    };
    let rays_per_side = non_negative(py, rays_per_side, "rays_per_side")?;
    let threads = non_negative(py, threads, "threads")?;
    let adding = read_bodies(py, &adding, "adding")?;
    let subtracting = read_bodies(py, &subtracting, "subtracting")?;
    let values = run_engine(py, || {
        let adding = build(adding, "adding")?;
        let subtracting = build(subtracting, "subtracting")?;
        michell::occupancy(&grid, &adding, &subtracting, rays_per_side, threads)
    })?;
    Ok(PyArray1::from_vec(py, values))
}

type Copied = (Vec<[f64; 3]>, Vec<[usize; 3]>);

/// Copies the bodies' arrays into memory the engine owns, checking shapes and indices.
fn read_bodies(
    py: Python<'_>,
    bodies: &[(Bound<'_, PyAny>, Bound<'_, PyAny>)],
    list: &str,
) -> PyResult<Vec<Copied>> {
    let mut copied = Vec::with_capacity(bodies.len());
    for (index, (vertices, triangles)) in bodies.iter().enumerate() {
        let name = format!("the vertices of {list} body {index}");
        let (rows, columns, values) = copy_matrix(vertices, &name)?;
        if columns != 3 && rows > 0 {
            return Err(invalid_input(
                py,
                format!("{name} must have shape (n, 3); got (n, {columns})"),
            ));
        }
        let vertices = values.chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
        let name = format!("the triangles of {list} body {index}");
        let (rows, columns, values) = copy_matrix(triangles, &name)?;
        if columns != 3 && rows > 0 {
            return Err(invalid_input(
                py,
                format!("{name} must have shape (m, 3); got (m, {columns})"),
            ));
        }
        let mut indices = Vec::with_capacity(values.len());
        for &value in &values {
            // Whole numbers from 0 to 2^53 are exact in float64.
            if !(value >= 0.0 && value.fract() == 0.0 && value <= 9_007_199_254_740_992.0) {
                return Err(invalid_input(
                    py,
                    format!("{name} must hold whole-number vertex indices; found {value}"),
                ));
            }
            indices.push(value as usize);
        }
        let triangles = indices
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]])
            .collect();
        copied.push((vertices, triangles));
    }
    Ok(copied)
}

/// The engine's bodies; a refusal names the body.
fn build(bodies: Vec<Copied>, list: &str) -> Result<Vec<michell::Body>, michell::EngineError> {
    bodies
        .into_iter()
        .enumerate()
        .map(|(index, (vertices, triangles))| {
            michell::Body::new(vertices, triangles).map_err(|mut error| {
                error.message = format!("{list} body {index}: {}", error.message);
                error.with_integer("body", index)
            })
        })
        .collect()
}
