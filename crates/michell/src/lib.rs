//! The michell engine: plain Rust, no Python inside, so other programs can use it too.
//!
//! Nothing of the engine exists yet. The two items below only prove that the Python
//! binding (the `michell-py` crate) can reach this crate.

/// The version of this crate, as written in `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Adds up all the numbers in `values`; an empty slice gives 0.0.
pub fn sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_adds_all_values() {
        assert_eq!(sum(&[1.0, 2.0, 3.5]), 6.5);
    }

    #[test]
    fn sum_of_nothing_is_zero() {
        assert_eq!(sum(&[]), 0.0);
    }

    #[test]
    fn version_matches_cargo_toml() {
        assert_eq!(VERSION, "0.0.1");
    }
}
