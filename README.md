# michell

Generative design by topology optimisation: a Rust engine with a Python interface.

michell computes the material layout of a structural part from its design space, supports, loads and
design objectives. It takes a 3D model of the design space together with a problem definition and returns an
optimised 3D model and a report of the results. The method is density-based topology optimisation: a
finite-element analysis coupled with a gradient-based optimiser. Results are reproducible: on a given machine
and thread count, identical inputs produce identical outputs.

The library is named after A. G. M. Michell, whose 1904 paper on minimum-weight frame structures is the
classical starting point of structural layout optimisation.

## Status

Pre-alpha. The repository currently contains the package structure (the engine crate, the Python bindings and
the continuous-integration configuration) and the design documents. Analysis and optimisation are not yet
implemented, and no release has been published.

## Planned scope

- Stiffness-driven design in 2D and 3D under mass, stress and deflection constraints
- Multiple load cases, keep-in and keep-out regions, symmetry and minimum member size
- STEP and STL input, STL and 3MF output
- Further physics (heat conduction, natural frequency, buckling) and manufacturing constraints as separate
  modules, each with its own verification cases
- Three interfaces to one engine: a command-line tool, a Python API, and a structured interface (a published
  schema, machine-readable results and errors) for integration into other software and automated workflows
- A public verification table in which every supported feature is checked against closed-form solutions and
  against other codes

## Design documents

- [`docs/design/modules.md`](docs/design/modules.md): how objectives, constraints and shape rules are defined
- [`docs/design/engine-boundary.md`](docs/design/engine-boundary.md): the division of work between the Rust
  engine and Python
- [`docs/design/problem-file.md`](docs/design/problem-file.md): the draft format of the problem definition

## Building from source

Requirements: Rust 1.85 or later ([rustup.rs](https://rustup.rs)) and Python 3.10 or later.

    pip install .
    michell --version

To run the tests:

    cargo test
    pip install pytest
    pytest

## Repository layout

| Path | Contents |
|---|---|
| `crates/michell` | The engine: a Rust library with no Python dependency |
| `crates/michell-py` | The Python bindings, compiled into the module `michell._engine` |
| `python/michell` | The Python package |
| `tests` | The Python test suite |
| `schema` | The schema of the problem definition (draft) |

## Licence

Licensed under either of the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT licence](LICENSE-MIT),
at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work by
you, as defined in the Apache-2.0 licence, shall be dual licensed as above, without any additional terms or
conditions.
