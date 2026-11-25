"""Service-tree Ascend NPU memory (MB) and summed chip utilization samplers."""

import math
import os
import re
import subprocess
import sys

from .process_metrics import SamplingError, read_process, resolve_roots, service_tree


_HEADER = ("NPU Chip", "Process id", "Process name", "Process memory(MB)")


def _cells(line):
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def parse_process_table(text):
    """Parse the process section only, keyed by (card, chip, host PID)."""
    found = False
    empty = False
    records = {}
    for line in text.splitlines():
        cells = _cells(line)
        if tuple(" ".join(cell.split()) for cell in cells) == _HEADER:
            found = True
            continue
        if re.fullmatch(r"\s*\|?\s*No running processes(?: found)?(?: in NPU \d+)?[.!]?\s*\|?\s*", line):
            empty = True
            continue
        if not found or not line.strip() or re.fullmatch(r"[+|=\-\s]+", line):
            continue
        if "|" not in line:
            raise SamplingError(f"malformed NPU process table line: {line.strip()}")
        if len(cells) != 4:
            raise SamplingError(f"malformed NPU process table row: {line.strip()}")
        pair = cells[0].split()
        if (len(pair) != 2 or not all(re.fullmatch(r"\d+", item) for item in pair)
                or not re.fullmatch(r"\d+", cells[1]) or not cells[2]):
            raise SamplingError(f"malformed NPU process table row: {line.strip()}")
        try:
            memory = float(cells[3])
        except ValueError as exc:
            raise SamplingError(f"invalid NPU process memory: {cells[3]}") from exc
        if not math.isfinite(memory) or memory < 0:
            raise SamplingError(f"invalid NPU process memory: {cells[3]}")
        key = (int(pair[0]), int(pair[1]), int(cells[1]))
        if key in records and records[key] != memory:
            raise SamplingError(f"conflicting NPU process memory for {key}")
        records[key] = memory
    if not found and not empty:
        raise SamplingError("npu-smi info is missing the NPU process table")
    return records


def _query(args):
    try:
        result = subprocess.run(["npu-smi", "info"] + args, check=True,
                                capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise SamplingError(f"npu-smi query failed: {detail}") from exc
    return result.stdout


def _device_filter(device_info):
    if not device_info or not device_info.strip():
        return None
    pairs = set()
    for token in device_info.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", token)
        if match is None:
            raise SamplingError("device_info must contain comma-separated card:chip pairs")
        pairs.add((int(match[1]), int(match[2])))
    return pairs


def _survivors(selected, proc_root):
    surviving = set()
    for pid, original in selected.items():
        try:
            current = read_process(proc_root, pid)
        except (OSError, ValueError, IndexError):
            continue
        if current.starttime == original.starttime:
            surviving.add(pid)
    if not surviving:
        raise SamplingError("all sampled service processes exited or became unavailable")
    return surviving


def parse_usage(text):
    values = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*Aicore Usage Rate\s*\(%\)\s*:\s*(\S+)\s*", line)
        if match is not None:
            try:
                value = float(match[1])
            except ValueError as exc:
                raise SamplingError(f"invalid Aicore Usage Rate: {match[1]}") from exc
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise SamplingError(f"invalid Aicore Usage Rate: {match[1]}")
            values.append(value)
    if len(values) != 1:
        raise SamplingError("npu-smi usages must contain exactly one Aicore Usage Rate(%)")
    return values[0]


def _sample(roots, proc_root, device_info, usage):
    allowed = _device_filter(device_info)
    selected = service_tree(roots, proc_root)
    records = parse_process_table(_query([]))
    surviving = _survivors(selected, proc_root)
    records = {key: value for key, value in records.items()
               if key[2] in surviving and (allowed is None or key[:2] in allowed)}
    if not usage:
        return sum(records.values())
    rates = {}
    for card, chip in sorted({key[:2] for key in records}):
        rates[(card, chip)] = parse_usage(_query(
            ["-t", "usages", "-i", str(card), "-c", str(chip)]))
    # Querying the devices can take time: discard associations of exited/reused
    # PIDs once more before counting each surviving service chip exactly once.
    surviving = _survivors(selected, proc_root)
    pairs = {key[:2] for key in records if key[2] in surviving}
    return sum(rates[pair] for pair in pairs)


def sample_memory(roots, proc_root="/proc", device_info=None):
    return _sample(roots, proc_root, device_info, usage=False)


def sample_usage(roots, proc_root="/proc", device_info=None):
    return _sample(roots, proc_root, device_info, usage=True)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if not args or args[0] not in ("mem", "usage") or len(args) > 4:
            raise SamplingError("usage: python3 -m autoperf.npu_metrics mem|usage [container [workers [device_info]]]")
        roots = resolve_roots(args[1] if len(args) > 1 else None)
        sampler = sample_memory if args[0] == "mem" else sample_usage
        value = sampler(roots, os.environ.get("NPU_PROC_ROOT", "/proc"),
                        args[3] if len(args) > 3 else None)
        print(f"{value:.2f}")
        return 0
    except (SamplingError, ValueError) as exc:
        print(f"npu_metrics: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
