//! `michell.DensityFilter`: the density filter with cone weights, on a 2D or a 3D grid.

use michell::{FilterEdge, FilterEdges};
use numpy::PyArray1;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::arrays::{as_slice, copy_vector, read_vector};
use crate::errors::{invalid_input, run_engine, run_in_place};
use crate::grid::{Grid, non_negative};
use crate::grid3d::Grid3d;

/// The engine's filter for the grid it was built on.
pub enum Inner {
    /// On a 2D grid: weights stored per element, one thread, a short call.
    Plane(michell::DensityFilter),
    /// On a 3D grid: one stencil of weights, on threads, a long call.
    Solid(michell::DensityFilter3d),
}

/// The density filter: rho_e = sum_j w_ej x_j / sum_j w_ej with w_ej = max(0, radius - distance between
/// the centres of elements e and j), over the active elements. The radius is in mm. On a `Grid` it runs on
/// one thread (`threads` must be 1); on a `Grid3d` on `threads` threads, with numbers that do not depend
/// on the thread count. `outside` ("closed" or "empty", one word for every face or a mapping by face
/// "x-", "x+", "y-", "y+", and on a 3D grid "z-", "z+"; faces left out are closed) and `inactive`
/// ("closed" or "empty") say what lies beyond each face and in the inactive elements.
#[pyclass(frozen, subclass, module = "michell", name = "DensityFilter")]
pub struct DensityFilter {
    pub inner: Inner,
    edges: FilterEdges,
}

/// The faces by name, with their axis and end.
const FACES: [(&str, usize, usize); 6] = [
    ("x-", 0, 0),
    ("x+", 0, 1),
    ("y-", 1, 0),
    ("y+", 1, 1),
    ("z-", 2, 0),
    ("z+", 2, 1),
];

fn edge_word(edge: FilterEdge) -> &'static str {
    match edge {
        FilterEdge::Closed => "closed",
        FilterEdge::Empty => "empty",
    }
}

fn read_edge(py: Python<'_>, value: &Bound<'_, PyAny>, what: &str) -> PyResult<FilterEdge> {
    let Ok(word) = value.cast::<PyString>() else {
        return Err(PyTypeError::new_err(format!(
            "{what} must be \"closed\" or \"empty\""
        )));
    };
    match word.to_str()? {
        "closed" => Ok(FilterEdge::Closed),
        "empty" => Ok(FilterEdge::Empty),
        other => Err(invalid_input(
            py,
            format!("{what} must be \"closed\" or \"empty\" (got {other:?})"),
        )),
    }
}

/// The edges from the `outside` and `inactive` arguments, for a grid with `axes` axes (2 or 3).
fn read_edges(
    py: Python<'_>,
    outside: Option<&Bound<'_, PyAny>>,
    inactive: &Bound<'_, PyAny>,
    axes: usize,
) -> PyResult<FilterEdges> {
    let mut edges = FilterEdges::closed();
    edges.inactive = read_edge(py, inactive, "inactive")?;
    let faces = &FACES[..2 * axes];
    match outside {
        None => {}
        Some(value) if value.cast::<PyString>().is_ok() => {
            let edge = read_edge(py, value, "outside")?;
            for &(_, axis, end) in faces {
                edges.faces[axis][end] = edge;
            }
        }
        Some(value) => {
            let Ok(mapping) = value.cast::<PyDict>() else {
                return Err(PyTypeError::new_err(
                    "outside must be \"closed\", \"empty\" or a dict from face to one of them",
                ));
            };
            for (key, edge) in mapping.iter() {
                let name: String = key.extract().map_err(|_| {
                    PyTypeError::new_err("the faces in outside are named by strings such as \"x-\"")
                })?;
                let Some(&(_, axis, end)) = faces.iter().find(|(face, _, _)| *face == name) else {
                    let names: Vec<&str> = faces.iter().map(|(face, _, _)| *face).collect();
                    return Err(invalid_input(
                        py,
                        format!(
                            "outside names a face {name:?} this grid does not have; its faces are {}",
                            names.join(", ")
                        ),
                    ));
                };
                edges.faces[axis][end] = read_edge(py, &edge, &format!("outside[{name:?}]"))?;
            }
        }
    }
    Ok(edges)
}

#[pymethods]
impl DensityFilter {
    #[new]
    #[pyo3(signature = (grid, radius, threads = 1, *, outside = None, inactive = None))]
    fn new(
        py: Python<'_>,
        grid: &Bound<'_, PyAny>,
        radius: f64,
        threads: i64,
        outside: Option<&Bound<'_, PyAny>>,
        inactive: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<DensityFilter> {
        let threads = non_negative(py, threads, "threads")?;
        let closed = PyString::new(py, "closed");
        let inactive = inactive.unwrap_or(closed.as_any());
        if let Ok(grid) = grid.cast::<Grid>() {
            if threads != 1 {
                return Err(invalid_input(
                    py,
                    format!(
                        "the density filter on a 2D grid runs on one thread; threads must be 1 (got {threads})"
                    ),
                ));
            }
            let edges = read_edges(py, outside, inactive, 2)?;
            let grid = grid.get().inner.clone();
            let inner = run_engine(py, || {
                michell::DensityFilter::with_edges(&grid, radius, edges)
            })?;
            return Ok(DensityFilter {
                inner: Inner::Plane(inner),
                edges,
            });
        }
        if let Ok(grid) = grid.cast::<Grid3d>() {
            let edges = read_edges(py, outside, inactive, 3)?;
            let grid = grid.get().inner.clone();
            let inner = run_engine(py, || {
                michell::DensityFilter3d::with_edges(&grid, radius, threads, edges)
            })?;
            return Ok(DensityFilter {
                inner: Inner::Solid(inner),
                edges,
            });
        }
        Err(PyTypeError::new_err(
            "the density filter needs a michell.Grid or a michell.Grid3d",
        ))
    }

    /// The radius, mm.
    #[getter]
    fn radius(&self) -> f64 {
        match &self.inner {
            Inner::Plane(filter) => filter.radius(),
            Inner::Solid(filter) => filter.radius(),
        }
    }

    /// The side of one element of the grid the filter was built on, mm.
    #[getter]
    fn element_size(&self) -> f64 {
        match &self.inner {
            Inner::Plane(filter) => filter.element_size(),
            Inner::Solid(filter) => filter.element_size(),
        }
    }

    /// What lies beyond each face of the grid: a dict from face ("x-", "x+", ...) to "closed" or "empty".
    #[getter]
    fn outside<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let axes = if matches!(self.inner, Inner::Solid(_)) {
            3
        } else {
            2
        };
        let faces = PyDict::new(py);
        for &(name, axis, end) in &FACES[..2 * axes] {
            faces.set_item(name, edge_word(self.edges.faces[axis][end]))?;
        }
        Ok(faces)
    }

    /// What the inactive elements are to the filter: "closed" or "empty".
    #[getter]
    fn inactive(&self) -> &'static str {
        edge_word(self.edges.inactive)
    }

    /// Whether the filter was built on a 3D grid.
    #[getter]
    fn is_3d(&self) -> bool {
        matches!(self.inner, Inner::Solid(_))
    }

    /// The number of threads every call runs on (1 on a 2D grid).
    #[getter]
    fn threads(&self) -> usize {
        match &self.inner {
            Inner::Plane(_) => 1,
            Inner::Solid(filter) => filter.threads(),
        }
    }

    /// rho = the filtered field of x, one value per element (0 on inactive elements).
    ///
    /// On a 3D grid a long call: the input is copied, then Python's lock is released.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho = match &self.inner {
            Inner::Plane(filter) => {
                let x = read_vector(x, "x")?;
                let x = as_slice(py, &x, "x")?;
                run_in_place(py, || filter.forward(x))?
            }
            Inner::Solid(filter) => {
                let x = copy_vector(x, "x")?;
                run_engine(py, || filter.forward(&x))?
            }
        };
        Ok(PyArray1::from_vec(py, rho))
    }

    /// dg/dx from dg/drho (`grad`): the chain rule back through the filter.
    ///
    /// `x` is the field the filter was applied to. The filter is linear, so its reverse does not depend
    /// on x; the argument is there because every filter takes it (see `docs/design/modules.md`), and only
    /// its length is checked. On a 3D grid a long call, like `forward`.
    fn backward<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        grad: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let n_elements = match &self.inner {
            Inner::Plane(filter) => filter.n_elements(),
            Inner::Solid(filter) => filter.n_elements(),
        };
        let x = read_vector(x, "x")?;
        if as_slice(py, &x, "x")?.len() != n_elements {
            return Err(invalid_input(
                py,
                format!("x needs one entry per element ({n_elements})"),
            ));
        }
        let result = match &self.inner {
            Inner::Plane(filter) => {
                let grad = read_vector(grad, "grad")?;
                let grad = as_slice(py, &grad, "grad")?;
                run_in_place(py, || filter.backward(grad))?
            }
            Inner::Solid(filter) => {
                let grad = copy_vector(grad, "grad")?;
                run_engine(py, || filter.backward(&grad))?
            }
        };
        Ok(PyArray1::from_vec(py, result))
    }
}
