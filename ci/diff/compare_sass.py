#!/usr/bin/env python3
"""
Compare two cuobjdump -sass dumps by normalizing them and reporting
per-function instruction count differences in a table.

Usage:
    compare_sass.py BASELINE.sass MODIFIED.sass

The script reuses normalize_sass.py's processing and demangling pipeline,
then pairs functions positionally (in ELF order) and prints a table of
those whose instruction count changed.
"""

import argparse
import re
import sys
from pathlib import Path

# Import shared normalization logic from normalize_sass (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_sass import demangle_lines, find_cufilt, process


def extract_functions(lines: list[str]) -> list[tuple[str, int]]:
    """Return [(function_name, instruction_count), ...] in order."""
    functions: list[tuple[str, int]] = []
    current_name: str | None = None
    count = 0

    for line in lines:
        if line.startswith("Function: "):
            if current_name is not None:
                functions.append((current_name, count))
            current_name = line[len("Function: ") :]
            count = 0
        elif line.startswith("  ") and current_name is not None:
            count += 1

    if current_name is not None:
        functions.append((current_name, count))

    return functions


def truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def main():
    parser = argparse.ArgumentParser(
        description="Compare two cuobjdump -sass dumps by per-function instruction count."
    )
    parser.add_argument("baseline", help="Baseline SASS file")
    parser.add_argument("modified", help="Modified SASS file")
    args = parser.parse_args()

    cufilt = find_cufilt()
    base_lines = demangle_lines(process(args.baseline), cufilt)
    mod_lines = demangle_lines(process(args.modified), cufilt)

    base_funcs = extract_functions(base_lines)
    mod_funcs = extract_functions(mod_lines)

    base_total = sum(c for _, c in base_funcs)
    mod_total = sum(c for _, c in mod_funcs)

    # Pair by position (ELF order)
    max_len = max(len(base_funcs), len(mod_funcs))
    diffs: list[tuple[str, int, int, int, float]] = []  # name, base, mod, delta, pct
    identical = 0
    NAME_W = 90

    for i in range(max_len):
        if i < len(base_funcs) and i < len(mod_funcs):
            bname, bc = base_funcs[i]
            mname, mc = mod_funcs[i]
            name = (
                bname
                if bname == mname
                else f"{truncate(bname, 40)} -> {truncate(mname, 40)}"
            )
            if bc == mc:
                identical += 1
            else:
                delta = mc - bc
                pct = (delta / bc * 100) if bc else float("inf")
                diffs.append((name, bc, mc, delta, pct))
        elif i < len(base_funcs):
            bname, bc = base_funcs[i]
            diffs.append((bname, bc, 0, -bc, -100.0))
        else:
            mname, mc = mod_funcs[i]
            diffs.append((mname, 0, mc, mc, float("inf")))

    # Sort by absolute delta descending
    diffs.sort(key=lambda x: -abs(x[3]))

    # Print summary
    print(
        f"Baseline:  {len(base_funcs)} functions, {base_total:,} instructions  ({args.baseline})"
    )
    print(
        f"Modified:  {len(mod_funcs)} functions, {mod_total:,} instructions  ({args.modified})"
    )
    delta_total = mod_total - base_total
    pct_total = (delta_total / base_total * 100) if base_total else 0
    print(f"Delta:     {delta_total:+,} instructions ({pct_total:+.2f}%)")
    print(f"Identical: {identical} / {max_len} functions")
    print()

    if not diffs:
        print("All functions have identical instruction counts.")
        return

    # Table
    hdr_name = "Function"
    hdr = f"{'':2s} {hdr_name:<{NAME_W}s}  {'Base':>7s}  {'Mod':>7s}  {'Delta':>7s}  {'%':>7s}"
    print(hdr)
    print("-" * len(hdr))

    for rank, (name, bc, mc, delta, pct) in enumerate(diffs, 1):
        pct_str = f"{pct:+.1f}%" if abs(pct) < 1e6 else ("new" if bc == 0 else "gone")
        print(
            f"{rank:2d} {truncate(name, NAME_W):<{NAME_W}s}  {bc:7,d}  {mc:7,d}  {delta:+7,d}  {pct_str:>7s}"
        )

    print("-" * len(hdr))
    pct_total_str = f"{pct_total:+.1f}%"
    print(
        f"{'':2s} {'TOTAL':<{NAME_W}s}  {base_total:7,d}  {mod_total:7,d}  {delta_total:+7,d}  {pct_total_str:>7s}"
    )


if __name__ == "__main__":
    main()
