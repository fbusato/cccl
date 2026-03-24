#!/usr/bin/env python3
"""
Normalize a cuobjdump -sass dump for diffing.

- Strips Fatbin PTX sections (only keeps ELF/SASS)
- Strips instruction address prefixes (/*0000*/)
- Strips instruction encoding hex (/* 0x00000a00ff017b82 */)
- Strips scheduling/control-word lines (the standalone /* 0x... */ lines)
- Strips .reuse register hints
- Normalizes branch/call targets (BRA 0x880 -> BRA ADDR)
- Normalizes memory offsets ([R1+0x8] -> [R1+OFF])
- Collapses whitespace
- Demangles function names via cu++filt (auto-detected)
- Normalizes registers: R0->Rxx, P0->Pxx, UR0->URxx, B0->Bxx (preserves RZ, URZ, PT)

Usage:
    normalize_sass.py INPUT.sass [-o OUTPUT.sass]

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

# Instruction line: /*addr*/  [@pred] OPCODE operands ;  /* encoding */
RE_INSTR = re.compile(
    r"^\s+/\*[0-9a-fA-F]+\*/\s+(.*?)\s*;\s*/\*\s*0x[0-9a-fA-F]+\s*\*/\s*$"
)

# Scheduling / control-word line: only /* 0x… */ on the line
RE_SCHED = re.compile(r"^\s+/\*\s*0x[0-9a-fA-F]+\s*\*/\s*$")

# Function header
RE_FUNC = re.compile(r"^\s*Function\s*:\s*(.*)")

# Fatbin section headers
RE_FATBIN = re.compile(r"^Fatbin (ptx|elf) code")

# Lines to skip entirely (Fatbin section separators and blank lines)
RE_SKIP = re.compile(r"(^=+$|^\s*$)")


def normalize_registers(text: str) -> str:
    """R0-R255 -> Rxx, P0-P6 -> Pxx, UR0-UR63 -> URxx, UP0-UP6 -> UPxx, B0-B15 -> Bxx.
    Preserves RZ, URZ, PT, SB (special barrier)."""
    text = re.sub(r"\bUR(\d+)\b", "URxx", text)
    text = re.sub(r"\bUP(\d+)\b", "UPxx", text)
    text = re.sub(r"\bR(\d+)\b", "Rxx", text)
    text = re.sub(r"\bP(\d+)\b", "Pxx", text)
    text = re.sub(r"\bB(\d+)\b", "Bxx", text)
    return text


def normalize_addresses(text: str) -> str:
    """Absolute branch/barrier targets -> ADDR.
    Handles: BRA 0x880, BSSY B0 0x10c0, CALL.* 0x..., etc.
    Also normalizes +0xNN memory offset immediates."""
    text = re.sub(
        r"\b(BRA|BRX|BSSY|CALL\S*|JMP|JMX)\s+(.*?,\s*)?0x[0-9a-fA-F]+",
        lambda m: m.group(1) + " " + (m.group(2) or "") + "ADDR",
        text,
    )
    text = re.sub(r"(\[.*?)\+0x[0-9a-fA-F]+\]", r"\1+OFF]", text)
    return text


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
    """Pipe all lines through cu++filt for demangling."""
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
    in_elf = False

    for line in raw_lines:
        stripped = line.rstrip("\n")

        if RE_FATBIN.match(stripped):
            in_elf = "elf" in stripped
            continue
        if not in_elf:
            continue

        if RE_SKIP.match(stripped):
            continue

        func_m = RE_FUNC.match(stripped)
        if func_m:
            out.append("")
            out.append(f"Function: {func_m.group(1).strip()}")
            continue

        if RE_SCHED.match(stripped):
            continue

        instr_m = RE_INSTR.match(stripped)
        if instr_m:
            instr = " ".join(instr_m.group(1).split())
            instr = normalize_registers(instr)
            instr = instr.replace(".reuse", "")
            instr = normalize_addresses(instr)
            out.append(f"  {instr}")
            continue

        # Metadata (arch, code version, .headerflags, .target, etc.)
        out.append(stripped.rstrip())

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Normalize cuobjdump -sass output for diffing."
    )
    parser.add_argument("input", help="Input SASS file from cuobjdump -sass")
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
