//! `michell.Elasticity` and `michell.element_stiffness`: plane-stress elasticity on a `Grid`.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::arrays::{as_slice, copy_vector, read_mask, read_vector};
use crate::errors::{invalid_input, run_engine, run_in_place};
use crate::grid::Grid;

/// Linear elasticity on one grid with one set of held degrees of freedom.
///
/// Material law: E(rho) = e_min + rho^penalty (youngs_modulus - e_min), plane stress.
#[pyclass(frozen, subclass, module = "michell", name = "Elasticity")]
pub struct Elasticity {
    inner: michell::Elasticity2d,
}

#[pymethods]
impl Elasticity {
    /// `held` is a float64 array with one entry per degree of freedom: 1.0 holds it at zero.
    /// Refuses (code `singular_system`) a part that is not held against every rigid motion.
    #[new]
    #[pyo3(signature = (grid, held, *, youngs_modulus, poisson_ratio, e_min, penalty))]
    fn new(
        py: Python<'_>,
        grid: &Bound<'_, Grid>,
        held: &Bound<'_, PyAny>,
        youngs_modulus: f64,
        poisson_ratio: f64,
        e_min: f64,
        penalty: f64,
    ) -> PyResult<Elasticity> {
        let held = read_mask(held, "held")?;
        let grid = grid.get().inner.clone();
        let material = michell::Material {
            youngs_modulus,
            poisson_ratio,
            e_min,
            penalty,
        };
        let inner = run_engine(py, || michell::Elasticity2d::new(grid, material, held))?;
        Ok(Elasticity { inner })
    }

    /// Displacements u with K(rho) u = loads. Loads on held degrees of freedom are ignored.
    /// `held_values` (one entry per degree of freedom) sets the displacement of the held ones.
    ///
    /// A long call: the inputs are copied, then Python's lock is released for the solve.
    #[pyo3(signature = (rho, loads, held_values = None))]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        rho: &Bound<'py, PyAny>,
        loads: &Bound<'py, PyAny>,
        held_values: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho = copy_vector(rho, "rho")?;
        let loads = copy_vector(loads, "loads")?;
        let held_values = match held_values {
            Some(values) => Some(copy_vector(values, "held_values")?),
            None => None,
        };
        let u = run_engine(py, || {
            self.inner.solve(&rho, &loads, held_values.as_deref())
        })?;
        Ok(PyArray1::from_vec(py, u))
    }

    /// lambda with K(rho) lambda = rhs, every held degree of freedom at zero.
    ///
    /// A long call: the inputs are copied, then Python's lock is released for the solve.
    fn adjoint<'py>(
        &self,
        py: Python<'py>,
        rho: &Bound<'py, PyAny>,
        rhs: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho = copy_vector(rho, "rho")?;
        let rhs = copy_vector(rhs, "rhs")?;
        let lambda = run_engine(py, || self.inner.adjoint(&rho, &rhs))?;
        Ok(PyArray1::from_vec(py, lambda))
    }

    /// Per element, -lambda_e^T (dK/drho_e) u_e: the part of dg/drho that comes through the state.
    /// For compliance (lambda = u) it is the whole gradient.
    fn sensitivity<'py>(
        &self,
        py: Python<'py>,
        rho: &Bound<'py, PyAny>,
        u: &Bound<'py, PyAny>,
        lam: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho = read_vector(rho, "rho")?;
        let u = read_vector(u, "u")?;
        let lam = read_vector(lam, "lam")?;
        let rho = as_slice(py, &rho, "rho")?;
        let u = as_slice(py, &u, "u")?;
        let lam = as_slice(py, &lam, "lam")?;
        let result = run_in_place(py, || self.inner.sensitivity(rho, u, lam))?;
        Ok(PyArray1::from_vec(py, result))
    }

    /// Per element, the strain energy 1/2 u_e^T K_e(rho) u_e in N mm (compliance = twice the sum).
    fn element_strain_energy<'py>(
        &self,
        py: Python<'py>,
        rho: &Bound<'py, PyAny>,
        u: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rho = read_vector(rho, "rho")?;
        let u = read_vector(u, "u")?;
        let (rho, u) = (as_slice(py, &rho, "rho")?, as_slice(py, &u, "u")?);
        let result = run_in_place(py, || self.inner.element_strain_energy(rho, u))?;
        Ok(PyArray1::from_vec(py, result))
    }

    /// Stress at each element's centre with modulus E(rho_e), MPa: an array of shape
    /// (n_elements, 3) holding sx, sy, txy.
    fn element_stress<'py>(
        &self,
        py: Python<'py>,
        rho: &Bound<'py, PyAny>,
        u: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rho = read_vector(rho, "rho")?;
        let u = read_vector(u, "u")?;
        let (rho, u) = (as_slice(py, &rho, "rho")?, as_slice(py, &u, "u")?);
        let flat = run_in_place(py, || self.inner.element_stress(rho, u))?;
        let n_elements = flat.len() / 3;
        PyArray1::from_vec(py, flat).reshape([n_elements, 3])
    }

    /// The grid this model was built on (a new handle with the same contents).
    #[getter]
    fn grid(&self) -> Grid {
        Grid {
            inner: self.inner.grid().clone(),
        }
    }
}

/// The 8 x 8 stiffness matrix of one square plane-stress element of side h and thickness t, nodes
/// counter-clockwise from the lower left, degrees of freedom u0, v0, u1, v1, u2, v2, u3, v3.
#[pyfunction]
pub fn element_stiffness<'py>(
    py: Python<'py>,
    youngs_modulus: f64,
    poisson_ratio: f64,
    h: f64,
    t: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let k = run_engine(py, || {
        michell::quad4::checked_stiffness(youngs_modulus, poisson_ratio, h, t)
    })?;
    let rows: Vec<Vec<f64>> = k.iter().map(|row| row.to_vec()).collect();
    PyArray2::from_vec2(py, &rows).map_err(|_| invalid_input(py, "could not build the 8 x 8 array"))
}
