//! The measured study behind the smoother settings of `multigrid_smoother` (`DAMPING`, `SWEEPS`): steps to
//! a relative residual of 1e-8 and products with the fine stiffness matrix per solve, for c in 0.6, 0.75,
//! 0.9, 0.95 and 1 to 3 sweeps, on the four cases of verification row 16 (their set-ups are written out
//! again here from `python/michell/verification/designs3d.py`).
//!
//! It takes about half a minute in a release build on 16 threads and is left out of the test run; run it
//! with
//!
//!     cargo test --release -p michell smoother_study -- --ignored --nocapture

use std::time::Instant;

use crate::elasticity3d::Elasticity3d;
use crate::grid3d::Grid3d;
use crate::material::Material;

const THREADS: usize = 16;

fn material() -> Material {
    Material {
        youngs_modulus: 1.0,
        poisson_ratio: 0.3,
        e_min: 1e-9,
        penalty: 3.0,
    }
}

struct Case {
    name: &'static str,
    grid: Grid3d,
    held: Vec<bool>,
    loads: Vec<f64>,
    rho: Vec<f64>,
}

/// Held: every displacement of the nodes of the face x = 0 with `on(j, k)`.
fn clamped(grid: &Grid3d, on: impl Fn(usize, usize) -> bool) -> Vec<bool> {
    let mut held = vec![false; grid.n_dofs()];
    for k in 0..=grid.nz() {
        for j in 0..=grid.ny() {
            if on(j, k) {
                let node = grid.node_index(0, j, k);
                held[3 * node..3 * node + 3].fill(true);
            }
        }
    }
    held
}

/// A total load of 1 in -z over the rectangle of nodes of the face x = nx with `on(j, k)`, by area share.
fn face_load(grid: &Grid3d, on: impl Fn(usize, usize) -> bool) -> Vec<f64> {
    let nodes: Vec<(usize, usize)> = (0..=grid.nz())
        .flat_map(|k| (0..=grid.ny()).map(move |j| (j, k)))
        .filter(|&(j, k)| on(j, k))
        .collect();
    let (j_low, j_high) = (
        nodes.iter().map(|n| n.0).min().unwrap(),
        nodes.iter().map(|n| n.0).max().unwrap(),
    );
    let (k_low, k_high) = (
        nodes.iter().map(|n| n.1).min().unwrap(),
        nodes.iter().map(|n| n.1).max().unwrap(),
    );
    let mut loads = vec![0.0; grid.n_dofs()];
    let mut total = 0.0;
    for &(j, k) in &nodes {
        let share = if j == j_low || j == j_high { 0.5 } else { 1.0 }
            * if k == k_low || k == k_high { 0.5 } else { 1.0 };
        loads[3 * grid.node_index(grid.nx(), j, k) + 2] = -share;
        total += share;
    }
    loads.iter_mut().for_each(|f| *f /= total);
    loads
}

fn cantilever(name: &'static str, nx: usize, ny: usize, nz: usize, rho: Vec<f64>) -> Case {
    let grid = Grid3d::full(nx, ny, nz, 1.0).unwrap();
    let held = clamped(&grid, |_, _| true);
    let loads = face_load(&grid, |_, _| true);
    Case {
        name,
        grid,
        held,
        loads,
        rho,
    }
}

fn slab(nx: usize, i: usize) -> bool {
    i < 2 || i >= nx - 2
}

fn lattice(nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    (0..nx * ny * nz)
        .map(|e| {
            let (i, j, k) = (e % nx, (e / nx) % ny, e / (nx * ny));
            let on = [i % 8 < 2, j % 8 < 2, k % 8 < 2];
            let solid = (on[1] && on[2]) || (on[0] && on[2]) || (on[0] && on[1]) || slab(nx, i);
            if solid { 1.0 } else { 0.0 }
        })
        .collect()
}

/// The face-joined group of `solid` that holds element 0.
fn group_of_first(grid: &Grid3d, solid: &[bool]) -> Vec<bool> {
    let mut group = vec![false; solid.len()];
    if !solid[0] {
        return group;
    }
    group[0] = true;
    let mut stack = vec![0];
    while let Some(e) = stack.pop() {
        let (i, j, k) = grid.element_position(e);
        let mut try_add = |n: usize| {
            if solid[n] && !group[n] {
                group[n] = true;
                stack.push(n);
            }
        };
        if i > 0 {
            try_add(grid.element_index(i - 1, j, k));
        }
        if i + 1 < grid.nx() {
            try_add(grid.element_index(i + 1, j, k));
        }
        if j > 0 {
            try_add(grid.element_index(i, j - 1, k));
        }
        if j + 1 < grid.ny() {
            try_add(grid.element_index(i, j + 1, k));
        }
        if k > 0 {
            try_add(grid.element_index(i, j, k - 1));
        }
        if k + 1 < grid.nz() {
            try_add(grid.element_index(i, j, k + 1));
        }
    }
    group
}

fn irregular(nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let grid = Grid3d::full(nx, ny, nz, 1.0).unwrap();
    let n = nx * ny * nz;
    let a = [3.1, 2.3, 2.9, 4.3];
    let field: Vec<f64> = (0..n)
        .map(|e| {
            let (i, j, k) = grid.element_position(e);
            let (x, y, z) = (i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5);
            (x / a[0] + 0.3).sin()
                + (y / a[1] + x / 7.7).sin()
                + (z / a[2] + 1.1).sin()
                + ((x + y + z) / a[3]).sin()
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&p, &q| field[q].total_cmp(&field[p])); // largest first, ties in index order
    let group_at = |percent: usize| {
        let mut solid = vec![false; n];
        let count = (percent as f64 / 100.0 * n as f64).round() as usize;
        for &e in &order[..count] {
            solid[e] = true;
        }
        for (e, s) in solid.iter_mut().enumerate() {
            if slab(nx, e % nx) {
                *s = true;
            }
        }
        group_of_first(&grid, &solid)
    };
    let joined = |group: &[bool]| (0..n).any(|e| e % nx == nx - 1 && group[e]);
    let mut percent = 40;
    while !joined(&group_at(percent)) {
        percent += 1;
    }
    group_at(percent)
        .iter()
        .map(|&g| if g { 1.0 } else { 0.0 })
        .collect()
}

fn l_shaped(nx: usize, ny: usize, nz: usize) -> Case {
    let (i_cut, k_cut) = (nx.div_ceil(2), nz.div_ceil(2));
    let active = (0..nx * ny * nz)
        .map(|e| !(e % nx >= i_cut && e / (nx * ny) >= k_cut))
        .collect();
    let grid = Grid3d::new(nx, ny, nz, 1.0, active).unwrap();
    // Held on the face x = nx at the nodes j <= ny - 1, k <= k_cut; loaded on the face x = 0.
    let mut held = vec![false; grid.n_dofs()];
    for k in 0..=k_cut {
        for j in 0..ny {
            let node = grid.node_index(nx, j, k);
            held[3 * node..3 * node + 3].fill(true);
        }
    }
    let mut loads = vec![0.0; grid.n_dofs()];
    let mut total = 0.0;
    for k in 0..=nz {
        for j in 0..=ny {
            let share = if j == 0 || j == ny { 0.5 } else { 1.0 }
                * if k == 0 || k == nz { 0.5 } else { 1.0 };
            loads[3 * grid.node_index(0, j, k) + 2] = -share;
            total += share;
        }
    }
    loads.iter_mut().for_each(|f| *f /= total);
    Case {
        name: "L-shaped 35x25x27",
        rho: vec![1.0; grid.n_elements()],
        grid,
        held,
        loads,
    }
}

#[test]
#[ignore]
fn smoother_study() {
    let cases = [
        cantilever("uniform 50x20x20", 50, 20, 20, vec![1.0; 20000]),
        cantilever("lattice 50x20x20", 50, 20, 20, lattice(50, 20, 20)),
        cantilever("irregular 50x20x20", 50, 20, 20, irregular(50, 20, 20)),
        l_shaped(35, 25, 27),
    ];
    for case in &cases {
        let solid: f64 = case.rho.iter().sum();
        let model = Elasticity3d::new(
            case.grid.clone(),
            material(),
            case.held.clone(),
            1e-8,
            20000,
            THREADS,
        )
        .unwrap();
        println!(
            "{}: {} solid elements, {} levels, {} coarsest dofs left out",
            case.name,
            solid,
            model.multigrid_levels(),
            model.multigrid_dropped()
        );
        for damping in [0.6, 0.75, 0.9, 0.95] {
            for sweeps in 1..=3 {
                let model = model.clone().with_smoother(damping, sweeps).unwrap();
                let start = Instant::now();
                let result = model.solve(&case.rho, &case.loads, None, None);
                let seconds = start.elapsed().as_secs_f64();
                match result {
                    Ok((_, report)) => println!(
                        "  c {damping} sweeps {sweeps}: {} steps, {} products, {seconds:.2} s",
                        report.steps, report.products
                    ),
                    Err(error) => println!("  c {damping} sweeps {sweeps}: {}", error.message),
                }
            }
        }
    }
}
