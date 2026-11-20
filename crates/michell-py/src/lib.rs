//! Python bindings for the michell engine, built as the extension module `michell._engine`.
//!
//! This crate only converts between Python and Rust; the work itself lives in the `michell` crate.
//!
//! - `errors`: `michell.EngineError` and the safe way to run an engine call;
//! - `arrays`: reading NumPy arrays by the boundary rules;
//! - `grid`: `michell.Grid`;
//! - `elasticity`: `michell.Elasticity` and `michell.element_stiffness`;
//! - `filter`: `michell.DensityFilter`;
//! - `oc`: `michell.Oc`, the optimality-criteria update;
//! - `mma`: `michell.Mma`, the method of moving asymptotes.

mod arrays;
mod elasticity;
mod errors;
mod filter;
mod grid;
mod mma;
mod oc;

use pyo3::pymodule;

/// The `michell._engine` module that Python imports.
#[pymodule]
mod _engine {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::elasticity::{Elasticity, element_stiffness};
    #[pymodule_export]
    use super::errors::EngineError;
    #[pymodule_export]
    use super::filter::DensityFilter;
    #[pymodule_export]
    use super::grid::Grid;
    #[pymodule_export]
    use super::mma::Mma;
    #[pymodule_export]
    use super::oc::Oc;

    /// Return the version of the Rust engine crate.
    #[pyfunction]
    fn engine_version() -> &'static str {
        michell::VERSION
    }
}
