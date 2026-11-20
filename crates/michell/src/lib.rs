//! The michell engine: plain Rust, no Python inside, so other programs can use it too.
//!
//! It holds the 2D analysis (the grid, the 4-node plane-stress element, a banded Cholesky solver and
//! linear elasticity), the 3D analysis (the grid of cube voxels, the 8-node element, a matrix-free
//! product solved by conjugate gradients) and the pieces of the optimisation loop that run over every
//! element (the density filter, the OC update and the method of moving asymptotes), with the dot product
//! the loop needs. The crate contains no unsafe code.
//! Each file has one job:
//!
//! - `grid2d`: the grid of square elements and its numbering;
//! - `quad4`: the element stiffness matrix and the stress at an element's centre;
//! - `banded_cholesky`: the linear solver;
//! - `elasticity2d`: assembly, solve, adjoint, sensitivity, strain energy, stress;
//! - `grid3d`: the grid of cube elements and its numbering;
//! - `hex8`: the 8-node cube element: stiffness matrix and the stress at an element's centre;
//! - `conjugate_gradient`: the preconditioned conjugate-gradient method, given a product and a
//!   preconditioner;
//! - `elasticity3d`: the matrix-free product, solve, adjoint, sensitivity, strain energy, stress;
//! - `material`: the material law both analyses share (modified SIMP);
//! - `density_filter`: the density filter with cone weights, forward and backward;
//! - `oc`: the optimality-criteria update;
//! - `mma`: the method of moving asymptotes (asymptotes, approximations, move limits);
//! - `mma_subproblem`: the interior-point solver of the MMA subproblem;
//! - `dot`: the dot product of two vectors, summed in a fixed order;
//! - `error`: the error type every call returns.

#![forbid(unsafe_code)]

pub mod banded_cholesky;
pub mod conjugate_gradient;
pub mod density_filter;
pub mod dot;
pub mod elasticity2d;
pub mod elasticity3d;
pub mod error;
pub mod grid2d;
pub mod grid3d;
pub mod hex8;
pub mod material;
pub mod mma;
pub mod mma_subproblem;
pub mod oc;
pub mod quad4;

pub use conjugate_gradient::SolveReport;
pub use density_filter::DensityFilter;
pub use dot::dot;
pub use elasticity2d::Elasticity2d;
pub use elasticity3d::Elasticity3d;
pub use error::{DetailValue, EngineError, ErrorCode};
pub use grid2d::Grid2d;
pub use grid3d::Grid3d;
pub use material::Material;
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
