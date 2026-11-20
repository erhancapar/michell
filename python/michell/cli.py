"""The `michell` command."""

import argparse

from michell import __version__, engine_version


def main(argv: list[str] | None = None) -> int:
    """Run the `michell` command; `michell --version` prints the package and engine versions."""
    parser = argparse.ArgumentParser(
        prog="michell",
        description="Generative design by topology optimisation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"michell {__version__} (engine {engine_version()})",
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
