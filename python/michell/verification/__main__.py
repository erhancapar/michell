"""``python -m michell.verification``: runs every verification row and writes the table.

python -m michell.verification --reference tests/reference --out docs/verification.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from michell import __version__, engine_version
from michell.verification import all_rows
from michell.verification.page import exit_status, render, result_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m michell.verification",
        description="Run the verification rows of the engine and write the verification table.",
    )
    parser.add_argument("--reference", type=Path, help="the folder that holds top88/ and calculix/ (tests/reference)")
    parser.add_argument(
        "--examples", type=Path, help="the folder that holds bridge_file/ (default: examples beside tests/)"
    )
    parser.add_argument("--out", type=Path, help="write the Markdown page here (default: print it)")
    parser.add_argument("--commit", help="the commit to name on the page")
    parser.add_argument("--require-all", action="store_true", help="count a row that was not run as a failure")
    args = parser.parse_args(argv)

    rows = all_rows(args.reference, args.examples)
    page = render(rows, __version__, engine_version(), args.commit)
    if args.out is None:
        print(page)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(page, encoding="utf-8")
        for row in rows:
            measured = "-" if row.measured is None else f"{row.measured:.3g}"
            print(f"row {row.number:>2}  {result_text(row):8}  {measured:>10}  {row.name}")
        print(f"written to {args.out}")
    return exit_status(rows, args.require_all)


if __name__ == "__main__":
    raise SystemExit(main())
