//! The michell engine: plain Rust, no Python inside, so other programs can use it too.
//!
//! It holds the 2D analysis (the grid, the 4-node plane-stress element, a banded Cholesky solver and
//! linear elasticity) and the pieces of the optimisation loop that run over every element (the density
//! filter, the OC update and the method of moving asymptotes). Each file has one job:
//!
//! - `grid2d`: the grid of square elements and its numbering;
//! - `quad4`: the element stiffness matrix and the stress at an element's centre;
//! - `banded_cholesky`: the linear solver;
//! - `elasticity2d`: assembly, solve, adjoint, sensitivity, strain energy, stress;
//! - `density_filter`: the density filter with cone weights, forward and backward;
//! - `oc`: the optimality-criteria update;
//! - `mma`: the method of moving asymptotes (asymptotes, approximations, move limits);
//! - `mma_subproblem`: the interior-point solver of the MMA subproblem;
//! - `error`: the error type every call returns.

pub mod banded_cholesky;
pub mod density_filter;
pub mod elasticity2d;
pub mod error;
pub mod grid2d;
pub mod mma;
pub mod mma_subproblem;
pub mod oc;
pub mod quad4;

pub use density_filter::DensityFilter;
pub use elasticity2d::{Elasticity2d, Material};
pub use error::{DetailValue, EngineError, ErrorCode};
pub use grid2d::Grid2d;
pub use mma::Mma;
pub use oc::Oc;

/// The version of this crate, as written in `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_cargo_toml() {
        assert_eq!(VERSION, "0.0.1");
    }
}
