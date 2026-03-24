#!/usr/bin/env python3
"""
Normalize a cuobjdump -ptx dump for diffing.

- Strips Fatbin ELF sections (only keeps PTX)
- Strips blank lines
- Normalizes registers: %r1->%rxx, %rd3->%rdxx, %p2->%pxx, %f1->%fxx, %fd1->%fdxx
- Normalizes branch labels: $L__BB1_2 -> $LABEL
- Normalizes .loc debug directives (stripped entirely)
- Collapses whitespace
- Demangles symbol names via cu++filt (auto-detected)

Usage:
    normalize_ptx.py INPUT.ptx [-o OUTPUT.ptx]

    If -o is omitted, writes to stdout.
"""

import argparse
import re
import shutil
import subprocess
import sys


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

RE_FATBIN = re.compile(r"^Fatbin (ptx|elf) code")
RE_SKIP = re.compile(r"(^=+$|^\s*$)")

# .entry or .func header (may span multiple lines, detected by .visible/.weak prefix)
RE_ENTRY = re.compile(r"^\.(?:visible|weak)\s+\.(?:entry|func)\s+(\S+)")

# PTX instruction: indented line with an opcode (not a directive, label, or brace)
RE_INSTR = re.compile(r"^\s+(\S+)")

# PTX label
RE_LABEL = re.compile(r"^\s*(\$\S+)\s*:")

# Debug .loc directive
RE_LOC = re.compile(r"^\s*\.loc\s")

# Register patterns (order matters: longer prefixes first)
RE_REGS = [
    (re.compile(r"%fd(\d+)"), "%fdxx"),
    (re.compile(r"%rd(\d+)"), "%rdxx"),
    (re.compile(r"%rs(\d+)"), "%rsxx"),
    (re.compile(r"%f(\d+)"), "%fxx"),
    (re.compile(r"%r(\d+)"), "%rxx"),
    (re.compile(r"%p(\d+)"), "%pxx"),
    (re.compile(r"%hh(\d+)"), "%hhxx"),
    (re.compile(r"%h(\d+)"), "%hxx"),
]

# Branch label references
RE_BRANCH_LABEL = re.compile(r"\$\w+")


def normalize_registers(text: str) -> str:
    for pat, repl in RE_REGS:
        text = pat.sub(repl, text)
    return text


def normalize_labels(text: str) -> str:
    return RE_BRANCH_LABEL.sub("$LABEL", text)


def find_cufilt() -> str:
    path = shutil.which("cu++filt")
    if path:
        return path
    candidate = "/usr/local/cuda/bin/cu++filt"
    if shutil.which(candidate):
        return candidate
    print(
        "error: cu++filt not found on PATH or at /usr/local/cuda/bin/cu++filt",
        file=sys.stderr,
    )
    sys.exit(1)


def demangle_lines(lines: list[str], cufilt: str) -> list[str]:
    text = "\n".join(lines)
    result = subprocess.run(
        [cufilt], input=text, capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        return result.stdout.split("\n")
    return lines


def process(input_path: str) -> list[str]:
    with open(input_path) as f:
        raw_lines = f.readlines()

    out: list[str] = []
    in_ptx = False

    for line in raw_lines:
        stripped = line.rstrip("\n")

        if RE_FATBIN.match(stripped):
            in_ptx = "ptx" in stripped
            continue
        if not in_ptx:
            continue

        if RE_SKIP.match(stripped):
            continue

        # Strip .loc debug directives
        if RE_LOC.match(stripped):
            continue

        # Normalize and emit
        text = stripped.rstrip()
        text = normalize_registers(text)
        text = normalize_labels(text)
        text = " ".join(text.split())
        out.append(text)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Normalize cuobjdump -ptx output for diffing."
    )
    parser.add_argument("input", help="Input PTX file from cuobjdump -ptx")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    lines = process(args.input)
    lines = demangle_lines(lines, find_cufilt())

    text = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
