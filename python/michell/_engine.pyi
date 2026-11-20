"""Type hints for the compiled Rust module (source: crates/michell-py/src/lib.rs)."""

import numpy as np
import numpy.typing as npt

def engine_version() -> str:
    """Return the version of the Rust engine crate."""

def sum_f64(values: npt.NDArray[np.float64], /) -> float:
    """Sum a 1-D float64 NumPy array in Rust, reading its memory without copying it.

    Raises ValueError for a non-contiguous array such as ``a[::2]``.
    """
