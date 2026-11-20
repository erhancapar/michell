//! The michell engine: plain Rust, no Python inside, so other programs can use it too.
//!
//! It currently holds the 2D analysis: the grid, the 4-node plane-stress element, a banded Cholesky
//! solver and linear elasticity. Each file has one job:
//!
//! - `grid2d`: the grid of square elements and its numbering;
//! - `quad4`: the element stiffness matrix and the stress at an element's centre;
//! - `banded_cholesky`: the linear solver;
//! - `elasticity2d`: assembly, solve, adjoint, sensitivity, strain energy, stress;
//! - `error`: the error type every call returns.

pub mod banded_cholesky;
pub mod elasticity2d;
pub mod error;
pub mod grid2d;
pub mod quad4;

pub use elasticity2d::{Elasticity2d, Material};
pub use error::{DetailValue, EngineError, ErrorCode};
pub use grid2d::Grid2d;

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
