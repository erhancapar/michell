//! The 8-node cube element of 3D linear elasticity (often called H8 or "trilinear hexahedron").
//!
//! The element is a cube of side `h`. Its nodes sit at the natural coordinates (xi, eta, zeta), each -1 or
//! 1, in the order of `grid3d::NODE_STEPS`: first the face zeta = -1, counter-clockwise seen from above,
//! (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1); then the face zeta = 1 in the same order. The degrees of
//! freedom are ordered `u0, v0, w0, u1, v1, w1, ..., u7, v7, w7` (u along x, v along y, w along z).
//!
//! Strains and stresses are written as six numbers in the order xx, yy, zz, yz, zx, xy, with engineering
//! shear strains (gamma_yz = dv/dz + dw/dy, and so on), so that the shear entries of the material matrix
//! are the shear modulus G = E / (2 (1 + nu)).
//!
//! Source of the formulas: Cook, Malkus, Plesha and Witt, "Concepts and Applications of Finite Element
//! Analysis", 4th ed. (2002), sections 1.5 (the isotropic material matrix in 3D), 6.3 (Gauss quadrature)
//! and 6.5 (the 8-node solid element). Verification row 11 checks this integration against an exact one.

use crate::error::EngineError;
use crate::grid3d::NODE_STEPS;
use crate::quad4;

/// The isotropic material matrix D, so that `stress = D * strain` in the order xx, yy, zz, yz, zx, xy,
/// with engineering shear strains.
///
/// With the Lame constants lambda = E nu / ((1 + nu)(1 - 2 nu)) and G = E / (2 (1 + nu)): the normal
/// block has lambda + 2G on the diagonal and lambda off it; the shear block is G on the diagonal.
pub fn material_matrix(e: f64, nu: f64) -> [[f64; 6]; 6] {
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let g = e / (2.0 * (1.0 + nu));
    let (normal, off) = (lambda + 2.0 * g, lambda);
    [
        [normal, off, off, 0.0, 0.0, 0.0],
        [off, normal, off, 0.0, 0.0, 0.0],
        [off, off, normal, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, g, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, g, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, g],
    ]
}

/// The natural coordinate (-1 or 1) of node `a` in direction `direction` (0 = xi, 1 = eta, 2 = zeta).
fn node_natural(a: usize, direction: usize) -> f64 {
    2.0 * NODE_STEPS[a][direction] as f64 - 1.0
}

/// Strain-displacement matrix B at the natural point (xi, eta, zeta), so that `strain = B * u_e`.
///
/// Shape function of node a: N_a = (1 + xi xi_a)(1 + eta eta_a)(1 + zeta zeta_a) / 8.
/// On a cube of side h, x = h xi / 2 + const, so d/dx = (2/h) d/dxi, and the same in y and z.
pub fn strain_displacement(h: f64, xi: f64, eta: f64, zeta: f64) -> [[f64; 24]; 6] {
    let mut b = [[0.0; 24]; 6];
    for a in 0..8 {
        let (xa, ya, za) = (node_natural(a, 0), node_natural(a, 1), node_natural(a, 2));
        let dn_dxi = xa * (1.0 + eta * ya) * (1.0 + zeta * za) / 8.0;
        let dn_deta = ya * (1.0 + xi * xa) * (1.0 + zeta * za) / 8.0;
        let dn_dzeta = za * (1.0 + xi * xa) * (1.0 + eta * ya) / 8.0;
        let (dn_dx, dn_dy, dn_dz) = (dn_dxi * 2.0 / h, dn_deta * 2.0 / h, dn_dzeta * 2.0 / h);
        let (u, v, w) = (3 * a, 3 * a + 1, 3 * a + 2);
        // exx = du/dx, eyy = dv/dy, ezz = dw/dz
        b[0][u] = dn_dx;
        b[1][v] = dn_dy;
        b[2][w] = dn_dz;
        // gyz = dv/dz + dw/dy, gzx = dw/dx + du/dz, gxy = du/dy + dv/dx
        b[3][v] = dn_dz;
        b[3][w] = dn_dy;
        b[4][w] = dn_dx;
        b[4][u] = dn_dz;
        b[5][u] = dn_dy;
        b[5][v] = dn_dx;
    }
    b
}

/// Element stiffness matrix K_e = integral of B^T D B over the cube.
///
/// Integrated with 2 x 2 x 2 Gauss points at xi, eta, zeta = +-1/sqrt(3), each of weight 1; the volume in
/// natural coordinates is 8 and in real coordinates h^3, so dV = (h / 2)^3 dxi deta dzeta. The rule is
/// exact here: each entry of B^T D B is at most quadratic in each natural coordinate. Unlike the square
/// plane element, the result grows in proportion to h (B scales with 1/h, the volume with h^3).
pub fn stiffness(e: f64, nu: f64, h: f64) -> [[f64; 24]; 24] {
    let d = material_matrix(e, nu);
    let g = 1.0 / 3.0_f64.sqrt();
    let volume_factor = (h / 2.0).powi(3);

    let mut k = [[0.0; 24]; 24];
    for a in 0..8 {
        // The Gauss points lie in the same pattern as the nodes, at +-g instead of +-1.
        let (xi, eta, zeta) = (
            g * node_natural(a, 0),
            g * node_natural(a, 1),
            g * node_natural(a, 2),
        );
        let b = strain_displacement(h, xi, eta, zeta);
        // db = D * B, a 6 x 24 matrix
        let mut db = [[0.0; 24]; 6];
        for row in 0..6 {
            for col in 0..24 {
                let mut sum = 0.0;
                for m in 0..6 {
                    sum += d[row][m] * b[m][col];
                }
                db[row][col] = sum;
            }
        }
        // k += B^T * (D * B) * dV
        for row in 0..24 {
            for col in 0..24 {
                let mut btdb = 0.0;
                for m in 0..6 {
                    btdb += b[m][row] * db[m][col];
                }
                k[row][col] += btdb * volume_factor;
            }
        }
    }
    k
}

/// `stiffness`, after refusing invalid constants or a size that is not a positive number. The valid range
/// of the constants is the same as in plane stress (E positive, -1 < nu < 0.5).
pub fn checked_stiffness(e: f64, nu: f64, h: f64) -> Result<[[f64; 24]; 24], EngineError> {
    quad4::check_elastic_constants(e, nu)?;
    if !(h.is_finite() && h > 0.0) {
        return Err(EngineError::invalid_input("h must be a positive number").with_number("h", h));
    }
    Ok(stiffness(e, nu, h))
}

/// Stress `[sxx, syy, szz, syz, szx, sxy]` at the centre of the element (xi = eta = zeta = 0) for the
/// element displacements `u_e`.
pub fn centre_stress(e: f64, nu: f64, h: f64, u_e: &[f64; 24]) -> [f64; 6] {
    let d = material_matrix(e, nu);
    let b = strain_displacement(h, 0.0, 0.0, 0.0);
    let mut strain = [0.0; 6];
    for row in 0..6 {
        for col in 0..24 {
            strain[row] += b[row][col] * u_e[col];
        }
    }
    let mut stress = [0.0; 6];
    for row in 0..6 {
        for m in 0..6 {
            stress[row] += d[row][m] * strain[m];
        }
    }
    stress
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Node positions of a cube of side h with its lowest corner at the origin, in element order.
    fn node_positions(h: f64) -> [[f64; 3]; 8] {
        NODE_STEPS.map(|[i, j, k]| [h * i as f64, h * j as f64, h * k as f64])
    }

    #[test]
    fn stiffness_is_symmetric_and_grows_with_h() {
        let k1 = stiffness(210000.0, 0.3, 1.0);
        let k2 = stiffness(210000.0, 0.3, 2.5);
        for row in 0..24 {
            for col in 0..24 {
                assert!((k1[row][col] - k1[col][row]).abs() < 1e-9);
                assert!((2.5 * k1[row][col] - k2[row][col]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn rigid_motions_need_no_force() {
        let h = 2.0;
        let k = stiffness(1.0, 0.3, h);
        let p = node_positions(h);
        // Three shifts and three rotations (about x: v = -z, w = y; about y: w = -x, u = z;
        // about z: u = -y, v = x).
        let motions: [fn([f64; 3]) -> [f64; 3]; 6] = [
            |_| [1.0, 0.0, 0.0],
            |_| [0.0, 1.0, 0.0],
            |_| [0.0, 0.0, 1.0],
            |[_, y, z]| [0.0, -z, y],
            |[x, _, z]| [z, 0.0, -x],
            |[x, y, _]| [-y, x, 0.0],
        ];
        for motion in motions {
            let mut u = [0.0; 24];
            for a in 0..8 {
                u[3 * a..3 * a + 3].copy_from_slice(&motion(p[a]));
            }
            for row in k {
                let force: f64 = row.iter().zip(u).map(|(kij, uj)| kij * uj).sum();
                assert!(force.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pure_shear_gives_g_times_gamma_in_each_plane() {
        // u = gamma * y gives gxy = gamma; v = gamma * z gives gyz; w = gamma * x gives gzx.
        let (e, nu, gamma, h) = (100.0, 0.25, 0.01, 3.0);
        let g = e / (2.0 * (1.0 + nu));
        let p = node_positions(h);
        // (displaced component, coordinate it follows, stress entry that must equal G gamma)
        for (component, coordinate, entry) in [(0, 1, 5), (1, 2, 3), (2, 0, 4)] {
            let mut u_e = [0.0; 24];
            for a in 0..8 {
                u_e[3 * a + component] = gamma * p[a][coordinate];
            }
            let stress = centre_stress(e, nu, h, &u_e);
            for (index, value) in stress.iter().enumerate() {
                let expected = if index == entry { g * gamma } else { 0.0 };
                assert!((value - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn uniaxial_strain_gives_lambda_and_lambda_plus_2g() {
        // u = eps * x: sxx = (lambda + 2G) eps, syy = szz = lambda eps.
        let (e, nu, eps) = (1.0, 0.3, 1e-3);
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let g = e / (2.0 * (1.0 + nu));
        let p = node_positions(1.0);
        let mut u_e = [0.0; 24];
        for a in 0..8 {
            u_e[3 * a] = eps * p[a][0];
        }
        let stress = centre_stress(e, nu, 1.0, &u_e);
        let expected = [
            (lambda + 2.0 * g) * eps,
            lambda * eps,
            lambda * eps,
            0.0,
            0.0,
            0.0,
        ];
        for (value, expected) in stress.iter().zip(expected) {
            assert!((value - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn wrong_constants_are_refused() {
        assert!(checked_stiffness(1.0, 0.5, 1.0).is_err());
        assert!(checked_stiffness(-1.0, 0.3, 1.0).is_err());
        assert!(checked_stiffness(1.0, 0.3, 0.0).is_err());
        assert!(checked_stiffness(1.0, 0.3, 1.0).is_ok());
    }
}
