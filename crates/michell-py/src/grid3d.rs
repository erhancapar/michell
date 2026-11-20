//! `michell.Grid3d`: the 3D grid of cube elements (voxels).

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::arrays::read_mask;
use crate::errors::{invalid_input, run_engine};
use crate::grid::non_negative;

/// A grid of `nx` by `ny` by `nz` cube elements of side `h` mm.
///
/// Element (i, j, k) has index `i + nx * (j + ny * k)`; node (i, j, k) has index
/// `i + (nx + 1) * (j + (ny + 1) * k)`; node n has its x, y and z displacement at degrees of freedom
/// `3n`, `3n + 1` and `3n + 2`.
#[pyclass(frozen, subclass, module = "michell", name = "Grid3d")]
pub struct Grid3d {
    pub inner: michell::Grid3d,
}

#[pymethods]
impl Grid3d {
    /// `active`, if given, is a float64 array with one entry per element: 1.0 inside the part, 0.0 outside.
    #[new]
    #[pyo3(signature = (nx, ny, nz, h, active = None))]
    fn new(
        py: Python<'_>,
        nx: i64,
        ny: i64,
        nz: i64,
        h: f64,
        active: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Grid3d> {
        let nx = non_negative(py, nx, "nx")?;
        let ny = non_negative(py, ny, "ny")?;
        let nz = non_negative(py, nz, "nz")?;
        let inner = match active {
            Some(mask) => {
                let mask = read_mask(mask, "active")?;
                run_engine(py, || michell::Grid3d::new(nx, ny, nz, h, mask))?
            }
            None => run_engine(py, || michell::Grid3d::full(nx, ny, nz, h))?,
        };
        Ok(Grid3d { inner })
    }

    #[getter]
    fn nx(&self) -> usize {
        self.inner.nx()
    }

    #[getter]
    fn ny(&self) -> usize {
        self.inner.ny()
    }

    #[getter]
    fn nz(&self) -> usize {
        self.inner.nz()
    }

    /// Side of one cube element, mm.
    #[getter]
    fn h(&self) -> f64 {
        self.inner.h()
    }

    /// Volume of one element, h^3, mm^3.
    #[getter]
    fn voxel_volume_mm3(&self) -> f64 {
        self.inner.element_volume()
    }

    #[getter]
    fn n_elements(&self) -> usize {
        self.inner.n_elements()
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    #[getter]
    fn n_dofs(&self) -> usize {
        self.inner.n_dofs()
    }

    /// The active mask as a new float64 array (1.0 inside the part).
    #[getter]
    fn active<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values: Vec<f64> = self
            .inner
            .active()
            .iter()
            .map(|&a| if a { 1.0 } else { 0.0 })
            .collect();
        PyArray1::from_vec(py, values)
    }

    /// Index of node (i, j, k), 0 <= i <= nx, 0 <= j <= ny, 0 <= k <= nz.
    fn node(&self, py: Python<'_>, i: i64, j: i64, k: i64) -> PyResult<usize> {
        let (i, j) = (non_negative(py, i, "i")?, non_negative(py, j, "j")?);
        let k = non_negative(py, k, "k")?;
        if i > self.inner.nx() || j > self.inner.ny() || k > self.inner.nz() {
            return Err(invalid_input(
                py,
                format!("node ({i}, {j}, {k}) is outside the grid"),
            ));
        }
        Ok(self.inner.node_index(i, j, k))
    }

    /// Index of element (i, j, k), 0 <= i < nx, 0 <= j < ny, 0 <= k < nz.
    fn element(&self, py: Python<'_>, i: i64, j: i64, k: i64) -> PyResult<usize> {
        let (i, j) = (non_negative(py, i, "i")?, non_negative(py, j, "j")?);
        let k = non_negative(py, k, "k")?;
        if i >= self.inner.nx() || j >= self.inner.ny() || k >= self.inner.nz() {
            return Err(invalid_input(
                py,
                format!("element ({i}, {j}, {k}) is outside the grid"),
            ));
        }
        Ok(self.inner.element_index(i, j, k))
    }

    fn __repr__(&self) -> String {
        format!(
            "Grid3d(nx={}, ny={}, nz={}, h={})",
            self.inner.nx(),
            self.inner.ny(),
            self.inner.nz(),
            self.inner.h()
        )
    }
}
