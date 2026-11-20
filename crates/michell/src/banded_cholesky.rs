//! A symmetric banded matrix and its Cholesky factorisation, K = L L^T.
//!
//! A stiffness matrix is mostly zeros: row i has non-zeros only in the columns of the nodes next to
//! node i. If the unknowns are numbered so that neighbours have close numbers, every non-zero sits
//! within a distance `half_band` of the diagonal, and L has the same band (Cholesky creates no
//! entries outside it). The work is then about n * half_band^2 instead of n^3.
//!
//! Source: Golub and Van Loan, "Matrix Computations", 4th ed. (2013), section 4.3.5 (band Cholesky).
//!
//! Storage: only the lower half is kept. Row i holds columns `i - half_band ..= i` in this order, so
//! entry (i, j) with j <= i lives at `i * (half_band + 1) + (j + half_band - i)`. For the first rows
//! the columns before 0 do not exist; their places stay zero.

/// A pivot was too small to be trusted: the matrix is singular (or as good as singular).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZeroPivot {
    /// The row (in the matrix's own numbering) where the factorisation stopped.
    pub row: usize,
    /// The value left on the diagonal when its turn came.
    pub pivot: f64,
    /// The original diagonal entry of that row.
    pub diagonal: f64,
}

/// A symmetric positive definite matrix stored by its lower band; after `factor` it holds L instead.
#[derive(Clone, Debug)]
pub struct BandedMatrix {
    n: usize,
    half_band: usize,
    values: Vec<f64>,
}

impl BandedMatrix {
    /// An `n` by `n` matrix of zeros whose non-zeros will lie within `half_band` of the diagonal.
    pub fn zeros(n: usize, half_band: usize) -> BandedMatrix {
        BandedMatrix {
            n,
            half_band,
            values: vec![0.0; n * (half_band + 1)],
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    /// Where entry (i, j), j <= i, is stored.
    fn position(&self, i: usize, j: usize) -> usize {
        assert!(
            j <= i && i - j <= self.half_band && i < self.n,
            "entry outside the band"
        );
        i * (self.half_band + 1) + (j + self.half_band - i)
    }

    /// Adds `value` to entry (i, j) and, by symmetry, to (j, i). Either order of i and j is accepted.
    pub fn add(&mut self, i: usize, j: usize, value: f64) {
        let (row, col) = if j <= i { (i, j) } else { (j, i) };
        let at = self.position(row, col);
        self.values[at] += value;
    }

    /// Entry (i, j); either order of i and j is accepted.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let (row, col) = if j <= i { (i, j) } else { (j, i) };
        self.values[self.position(row, col)]
    }

    /// Entries (i, from) .. (i, to - 1) of row i, as one slice; every column must lie in the band.
    fn row_part(&self, i: usize, from: usize, to: usize) -> &[f64] {
        let row_start = i * (self.half_band + 1) + self.half_band - i;
        &self.values[row_start + from..row_start + to]
    }

    /// Replaces the matrix by its Cholesky factor L (lower triangle), row by row:
    ///
    ///   L[i][j] = (A[i][j] - sum_k L[i][k] L[j][k]) / L[j][j]   for j < i,
    ///   L[i][i] = sqrt(A[i][i] - sum_k L[i][k]^2),
    ///
    /// where k runs over the columns both rows have inside the band.
    ///
    /// A pivot (the value under the square root) counts as zero, and the factorisation stops, when it is
    /// not above `min_pivot_ratio` times its row's original diagonal. Pass 0.0 to stop only at pivots
    /// that are zero, negative or not a number.
    pub fn factor(&mut self, min_pivot_ratio: f64) -> Result<(), ZeroPivot> {
        for i in 0..self.n {
            let first = i.saturating_sub(self.half_band);
            let diagonal = self.get(i, i);
            for j in first..=i {
                // The sum over k = first .. j-1 of L[i][k] * L[j][k].
                let dot: f64 = self
                    .row_part(i, first, j)
                    .iter()
                    .zip(self.row_part(j, first, j))
                    .map(|(a, b)| a * b)
                    .sum();
                let s = self.get(i, j) - dot;
                let value = if j < i {
                    s / self.get(j, j)
                } else {
                    // A comparison with NaN is false, so a NaN pivot is not trusted either.
                    let trusted = s > min_pivot_ratio * diagonal && diagonal > 0.0;
                    if !trusted {
                        return Err(ZeroPivot {
                            row: i,
                            pivot: s,
                            diagonal,
                        });
                    }
                    s.sqrt()
                };
                let at = self.position(i, j);
                self.values[at] = value;
            }
        }
        Ok(())
    }

    /// Solves A x = rhs in place, after `factor`: first L y = rhs (forward), then L^T x = y (backward).
    pub fn solve_factored(&self, rhs: &mut [f64]) {
        assert_eq!(rhs.len(), self.n, "right-hand side of the wrong length");
        // Forward: y_i = (rhs_i - sum_{k<i} L[i][k] y_k) / L[i][i]
        for i in 0..self.n {
            let first = i.saturating_sub(self.half_band);
            let known: f64 = self
                .row_part(i, first, i)
                .iter()
                .zip(&rhs[first..i])
                .map(|(l, y)| l * y)
                .sum();
            rhs[i] = (rhs[i] - known) / self.get(i, i);
        }
        // Backward, row by row from the bottom: once x_i is known, remove its share from the rows above:
        // x_k -= L[i][k] x_i for the columns k < i of row i.
        for i in (0..self.n).rev() {
            rhs[i] /= self.get(i, i);
            let x_i = rhs[i];
            let first = i.saturating_sub(self.half_band);
            for (x_k, l) in rhs[first..i].iter_mut().zip(self.row_part(i, first, i)) {
                *x_k -= l * x_i;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The matrix of a chain of n nodes joined by springs of stiffness 1; with `fixed`, one more spring
    /// ties the first node to a wall.
    fn spring_chain(n: usize, fixed: bool) -> BandedMatrix {
        let mut a = BandedMatrix::zeros(n, 1);
        for i in 1..n {
            // the spring between node i-1 and node i
            a.add(i, i, 1.0);
            a.add(i - 1, i - 1, 1.0);
            a.add(i, i - 1, -1.0);
        }
        if fixed {
            a.add(0, 0, 1.0);
        }
        a
    }

    #[test]
    fn solves_a_spring_chain() {
        // A unit pull on the free end stretches every spring by 1: x_i = i + 1.
        let n = 50;
        let mut a = spring_chain(n, true);
        a.factor(0.0).unwrap();
        let mut x = vec![0.0; n];
        x[n - 1] = 1.0;
        a.solve_factored(&mut x);
        for (i, xi) in x.iter().enumerate() {
            assert!((xi - (i as f64 + 1.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn solves_a_wide_band_against_a_known_answer() {
        // A banded matrix whose diagonal outweighs the rest of its row, so it is positive definite;
        // check that solving A x = A x_true gives back x_true.
        let (n, half_band) = (40, 5);
        let mut a = BandedMatrix::zeros(n, half_band);
        let mut dense = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i.saturating_sub(half_band)..=i {
                let value = if i == j {
                    30.0 + i as f64
                } else {
                    ((i * 7 + j * 3) % 5) as f64 - 2.0
                };
                a.add(i, j, value);
                dense[i][j] = value;
                dense[j][i] = value;
            }
        }
        let x_true: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let mut x: Vec<f64> = dense
            .iter()
            .map(|row| row.iter().zip(&x_true).map(|(a, b)| a * b).sum())
            .collect();
        a.factor(0.0).unwrap();
        a.solve_factored(&mut x);
        for (xi, ti) in x.iter().zip(&x_true) {
            assert!((xi - ti).abs() < 1e-12);
        }
    }

    #[test]
    fn a_free_chain_is_singular() {
        // Not fixed to the wall: it can slide as a whole, the last pivot is zero (to rounding).
        let mut a = spring_chain(30, false);
        let failure = a.factor(1e-12).unwrap_err();
        assert_eq!(failure.row, 29);
        assert!(failure.pivot.abs() < 1e-14 * failure.diagonal);
    }
}
