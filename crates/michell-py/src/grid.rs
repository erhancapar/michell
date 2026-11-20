//! `michell.Grid`: the 2D grid of square elements.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::arrays::read_mask;
use crate::errors::{invalid_input, run_engine};

/// A grid of `nx` by `ny` square elements of side `h` mm, thickness `t` mm, y up.
///
/// Element (i, j) has index `i + nx * j`; node (i, j) has index `i + (nx + 1) * j`; node n has its x
/// displacement at degree of freedom `2n` and its y displacement at `2n + 1`.
#[pyclass(frozen, subclass, module = "michell", name = "Grid")]
pub struct Grid {
    pub inner: michell::Grid2d,
}

/// A whole number that must not be negative, for counts and grid positions.
fn non_negative(py: Python<'_>, value: i64, name: &str) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| invalid_input(py, format!("{name} must not be negative; got {value}")))
}

#[pymethods]
impl Grid {
    /// `active`, if given, is a float64 array with one entry per element: 1.0 inside the part, 0.0 outside.
    #[new]
    #[pyo3(signature = (nx, ny, h, t, active = None))]
    fn new(
        py: Python<'_>,
        nx: i64,
        ny: i64,
        h: f64,
        t: f64,
        active: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Grid> {
        let nx = non_negative(py, nx, "nx")?;
        let ny = non_negative(py, ny, "ny")?;
        let inner = match active {
            Some(mask) => {
                let mask = read_mask(mask, "active")?;
                run_engine(py, || michell::Grid2d::new(nx, ny, h, t, mask))?
            }
            None => run_engine(py, || michell::Grid2d::full(nx, ny, h, t))?,
        };
        Ok(Grid { inner })
    }

    #[getter]
    fn nx(&self) -> usize {
        self.inner.nx()
    }

    #[getter]
    fn ny(&self) -> usize {
        self.inner.ny()
    }

    /// Side of one element, mm.
    #[getter]
    fn h(&self) -> f64 {
        self.inner.h()
    }

    /// Thickness, mm.
    #[getter]
    fn t(&self) -> f64 {
        self.inner.t()
    }

    /// Volume of one element, h * h * t, mm^3.
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

    /// Index of node (i, j), 0 <= i <= nx, 0 <= j <= ny.
    fn node(&self, py: Python<'_>, i: i64, j: i64) -> PyResult<usize> {
        let (i, j) = (non_negative(py, i, "i")?, non_negative(py, j, "j")?);
        if i > self.inner.nx() || j > self.inner.ny() {
            return Err(invalid_input(
                py,
                format!("node ({i}, {j}) is outside the grid"),
            ));
        }
        Ok(self.inner.node_index(i, j))
    }

    /// Index of element (i, j), 0 <= i < nx, 0 <= j < ny.
    fn element(&self, py: Python<'_>, i: i64, j: i64) -> PyResult<usize> {
        let (i, j) = (non_negative(py, i, "i")?, non_negative(py, j, "j")?);
        if i >= self.inner.nx() || j >= self.inner.ny() {
            return Err(invalid_input(
                py,
                format!("element ({i}, {j}) is outside the grid"),
            ));
        }
        Ok(self.inner.element_index(i, j))
    }

    fn __repr__(&self) -> String {
        format!(
            "Grid(nx={}, ny={}, h={}, t={})",
            self.inner.nx(),
            self.inner.ny(),
            self.inner.h(),
            self.inner.t()
        )
    }
}
