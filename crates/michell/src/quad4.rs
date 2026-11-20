//! The 4-node square element in plane stress (often called Q4 or "bilinear quad").
//!
//! The element is a square of side `h` and thickness `t`. Its nodes, counter-clockwise from the lower
//! left, sit at the natural coordinates (xi, eta) = (-1,-1), (1,-1), (1,1), (-1,1). The degrees of
//! freedom are ordered `u0, v0, u1, v1, u2, v2, u3, v3` (u along x, v along y).
//!
//! Source of the formulas: any finite-element textbook, e.g. Cook, Malkus, Plesha and Witt, "Concepts
//! and Applications of Finite Element Analysis", 4th ed. (2002), sections 3.6 (bilinear element) and
//! 6.3 (Gauss quadrature). The closed-form result is in Bendsøe and Sigmund, "Topology Optimization"
//! (2003); a test checks this integration against it.

use crate::error::EngineError;

/// Refuses elastic constants that give no valid plane-stress material: E must be positive and nu must
/// lie strictly between -1 and 0.5.
pub fn check_elastic_constants(e: f64, nu: f64) -> Result<(), EngineError> {
    if !(e.is_finite() && e > 0.0) {
        return Err(
            EngineError::invalid_input("youngs_modulus must be a positive number")
                .with_number("youngs_modulus", e),
        );
    }
    if !(nu.is_finite() && nu > -1.0 && nu < 0.5) {
        return Err(EngineError::invalid_input(
            "poisson_ratio must lie between -1 and 0.5 (both excluded)",
        )
        .with_number("poisson_ratio", nu));
    }
    Ok(())
}

/// Plane-stress material matrix D, so that `[sx, sy, txy] = D * [ex, ey, gxy]`.
///
/// `gxy` is the engineering shear strain (du/dy + dv/dx), so the last diagonal entry is the shear
/// modulus G = E / (2 (1 + nu)).
pub fn plane_stress_matrix(e: f64, nu: f64) -> [[f64; 3]; 3] {
    let c = e / (1.0 - nu * nu);
    [
        [c, c * nu, 0.0],
        [c * nu, c, 0.0],
        [0.0, 0.0, c * (1.0 - nu) / 2.0],
    ]
}

/// The natural coordinates of the four nodes, in element order.
const NODE_XI: [f64; 4] = [-1.0, 1.0, 1.0, -1.0];
const NODE_ETA: [f64; 4] = [-1.0, -1.0, 1.0, 1.0];

/// Strain-displacement matrix B at the natural point (xi, eta), so that `[ex, ey, gxy] = B * u_e`.
///
/// Shape function of node a: N_a = (1 + xi xi_a)(1 + eta eta_a) / 4.
/// On a square of side h, x = h xi / 2 + const, so d/dx = (2/h) d/dxi and d/dy = (2/h) d/deta.
pub fn strain_displacement(h: f64, xi: f64, eta: f64) -> [[f64; 8]; 3] {
    let mut b = [[0.0; 8]; 3];
    for a in 0..4 {
        let dn_dxi = NODE_XI[a] * (1.0 + eta * NODE_ETA[a]) / 4.0;
        let dn_deta = NODE_ETA[a] * (1.0 + xi * NODE_XI[a]) / 4.0;
        let dn_dx = dn_dxi * 2.0 / h;
        let dn_dy = dn_deta * 2.0 / h;
        // ex = du/dx, ey = dv/dy, gxy = du/dy + dv/dx
        b[0][2 * a] = dn_dx;
        b[1][2 * a + 1] = dn_dy;
        b[2][2 * a] = dn_dy;
        b[2][2 * a + 1] = dn_dx;
    }
    b
}

/// Element stiffness matrix K_e = integral of B^T D B over the element, times the thickness t.
///
/// Integrated with 2 x 2 Gauss points at xi, eta = +-1/sqrt(3), each of weight 1. The area of the
/// element in natural coordinates is 4 and in real coordinates h^2, so dA = (h^2 / 4) dxi deta.
/// For a square element the result does not depend on h (B scales with 1/h, the area with h^2).
pub fn stiffness(e: f64, nu: f64, h: f64, t: f64) -> [[f64; 8]; 8] {
    let d = plane_stress_matrix(e, nu);
    let g = 1.0 / 3.0_f64.sqrt();
    let gauss_points = [(-g, -g), (g, -g), (g, g), (-g, g)];
    let area_factor = h * h / 4.0;

    let mut k = [[0.0; 8]; 8];
    for (xi, eta) in gauss_points {
        let b = strain_displacement(h, xi, eta);
        // db = D * B, a 3 x 8 matrix
        let mut db = [[0.0; 8]; 3];
        for row in 0..3 {
            for col in 0..8 {
                db[row][col] =
                    d[row][0] * b[0][col] + d[row][1] * b[1][col] + d[row][2] * b[2][col];
            }
        }
        // k += B^T * (D * B) * t * dA
        for row in 0..8 {
            for col in 0..8 {
                let btdb = b[0][row] * db[0][col] + b[1][row] * db[1][col] + b[2][row] * db[2][col];
                k[row][col] += btdb * t * area_factor;
            }
        }
    }
    k
}

/// `stiffness`, after refusing invalid constants or a size or thickness that is not a positive number.
pub fn checked_stiffness(e: f64, nu: f64, h: f64, t: f64) -> Result<[[f64; 8]; 8], EngineError> {
    check_elastic_constants(e, nu)?;
    for (name, value) in [("h", h), ("t", t)] {
        if !(value.is_finite() && value > 0.0) {
            return Err(
                EngineError::invalid_input(format!("{name} must be a positive number"))
                    .with_number(name, value),
            );
        }
    }
    Ok(stiffness(e, nu, h, t))
}

/// Stress `[sx, sy, txy]` at the centre of the element (xi = eta = 0) for the element displacements `u_e`.
pub fn centre_stress(e: f64, nu: f64, h: f64, u_e: &[f64; 8]) -> [f64; 3] {
    let d = plane_stress_matrix(e, nu);
    let b = strain_displacement(h, 0.0, 0.0);
    let mut strain = [0.0; 3];
    for row in 0..3 {
        for col in 0..8 {
            strain[row] += b[row][col] * u_e[col];
        }
    }
    let mut stress = [0.0; 3];
    for row in 0..3 {
        stress[row] = d[row][0] * strain[0] + d[row][1] * strain[1] + d[row][2] * strain[2];
    }
    stress
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stiffness_is_symmetric_and_does_not_depend_on_h() {
        let k1 = stiffness(210000.0, 0.3, 1.0, 2.0);
        let k2 = stiffness(210000.0, 0.3, 7.5, 2.0);
        for row in 0..8 {
            for col in 0..8 {
                assert!((k1[row][col] - k1[col][row]).abs() < 1e-9);
                assert!((k1[row][col] - k2[row][col]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn rigid_motions_need_no_force() {
        let h = 2.0;
        let k = stiffness(1.0, 0.3, h, 1.0);
        // Node positions, element order, for a rigid rotation u = -y, v = x.
        let x = [0.0, h, h, 0.0];
        let y = [0.0, 0.0, h, h];
        let shift_x = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let shift_y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let rotation = [-y[0], x[0], -y[1], x[1], -y[2], x[2], -y[3], x[3]];
        for motion in [shift_x, shift_y, rotation] {
            for row in k {
                let force: f64 = row.iter().zip(motion).map(|(kij, uj)| kij * uj).sum();
                assert!(force.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pure_shear_gives_g_times_gamma() {
        // u = gamma * y, v = 0 on an element of side 3: engineering shear strain gamma.
        let (e, nu, gamma) = (100.0, 0.25, 0.01);
        let y = [0.0, 0.0, 3.0, 3.0];
        let mut u_e = [0.0; 8];
        for a in 0..4 {
            u_e[2 * a] = gamma * y[a];
        }
        let stress = centre_stress(e, nu, 3.0, &u_e);
        let g = e / (2.0 * (1.0 + nu));
        assert!(stress[0].abs() < 1e-12 && stress[1].abs() < 1e-12);
        assert!((stress[2] - g * gamma).abs() < 1e-12);
    }
}
