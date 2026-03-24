#!/usr/bin/env python3
"""
Compare two cuobjdump -ptx dumps by normalizing them and reporting
per-function instruction count differences in a table.

Usage:
    compare_ptx.py BASELINE.ptx MODIFIED.ptx

The script reuses normalize_ptx.py's processing and demangling pipeline,
then pairs functions by name and prints a table of those whose instruction
count changed.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_ptx import find_cufilt, process


RE_ENTRY = re.compile(r"^\.(?:visible|weak)\s+\.(?:entry|func)\s+(\S+)")


def extract_functions(lines: list[str]) -> list[tuple[str, int]]:
    """Return [(mangled_function_name, instruction_count), ...] in order.
    Must be called on lines BEFORE demangling (mangled names have no spaces)."""
    functions: list[tuple[str, int]] = []
    current_name: str | None = None
    count = 0
    in_body = False
    brace_depth = 0

    for line in lines:
        entry_m = RE_ENTRY.match(line)
        if entry_m:
            if current_name is not None:
                functions.append((current_name, count))
            current_name = entry_m.group(1).rstrip("(")
            count = 0
            in_body = False
            brace_depth = 0
            continue

        if current_name is not None:
            if "{" in line:
                brace_depth += line.count("{") - line.count("}")
                in_body = True
                continue
            if "}" in line:
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    in_body = False
                continue

            if in_body:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("."):
                    continue
                if stripped.endswith(":"):
                    continue
                count += 1

    if current_name is not None:
        functions.append((current_name, count))

    return functions


def demangle_names(names: list[str], cufilt: str) -> dict[str, str]:
    """Demangle a list of mangled names, returning mangled->demangled map."""
    text = "\n".join(names)
    result = subprocess.run(
        [cufilt], input=text, capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        demangled = result.stdout.strip().split("\n")
        if len(demangled) == len(names):
            return dict(zip(names, demangled))
    return {n: n for n in names}


def truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def main():
    parser = argparse.ArgumentParser(
        description="Compare two cuobjdump -ptx dumps by per-function instruction count."
    )
    parser.add_argument("baseline", help="Baseline PTX file")
    parser.add_argument("modified", help="Modified PTX file")
    args = parser.parse_args()

    base_funcs = extract_functions(process(args.baseline))
    mod_funcs = extract_functions(process(args.modified))

    # Demangle all unique names
    cufilt = find_cufilt()
    all_mangled = list(
        dict.fromkeys([n for n, _ in base_funcs] + [n for n, _ in mod_funcs])
    )
    name_map = demangle_names(all_mangled, cufilt)

    base_total = sum(c for _, c in base_funcs)
    mod_total = sum(c for _, c in mod_funcs)

    # Build name->count maps (keyed by mangled name for matching)
    base_map = {n: c for n, c in base_funcs}
    mod_map = {n: c for n, c in mod_funcs}

    diffs: list[tuple[str, int, int, int, float]] = []
    identical = 0
    NAME_W = 90

    for mangled in all_mangled:
        bc = base_map.get(mangled, 0)
        mc = mod_map.get(mangled, 0)
        display = name_map.get(mangled, mangled)
        if bc == mc:
            identical += 1
        else:
            delta = mc - bc
            pct = (delta / bc * 100) if bc else float("inf")
            diffs.append((display, bc, mc, delta, pct))

    diffs.sort(key=lambda x: -abs(x[3]))

    print(
        f"Baseline:  {len(base_funcs)} functions, {base_total:,} instructions  ({args.baseline})"
    )
    print(
        f"Modified:  {len(mod_funcs)} functions, {mod_total:,} instructions  ({args.modified})"
    )
    delta_total = mod_total - base_total
    pct_total = (delta_total / base_total * 100) if base_total else 0
    print(f"Delta:     {delta_total:+,} instructions ({pct_total:+.2f}%)")
    print(f"Identical: {identical} / {len(all_mangled)} functions")
    print()

    if not diffs:
        print("All functions have identical instruction counts.")
        return

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
