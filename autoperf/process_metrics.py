"""Linux service-tree CPU (one core = 100%) and summed RSS samplers."""

import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from dataclasses import dataclass


class SamplingError(RuntimeError):
    """No reliable service measurement is available."""


def select_service_pids(rows):
    """Select topmost gunicorn processes, or outermost processes if absent.

    ``rows`` contains host ``(pid, ppid, command)`` tuples. Input order is
    preserved; duplicate PIDs and overlapping ancestor roots are removed.
    """
    processes = {int(pid): (int(ppid), command) for pid, ppid, command in rows}
    gunicorn = {
        pid for pid, (_, command) in processes.items()
        if re.search(r"\bgunicorn\b", command, re.IGNORECASE)
    }
    candidates = gunicorn or set(processes)
    selected = []
    for pid in processes:
        if pid not in candidates:
            continue
        parent = processes[pid][0]
        seen = {pid}
        while parent in processes and parent not in seen:
            if parent in candidates:
                break
            seen.add(parent)
            parent = processes[parent][0]
        else:
            selected.append(pid)
    return selected


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    ticks: int
    starttime: int


def read_process(proc_root, pid):
    """Read stat using the last ')' because comm may contain spaces and ')'."""
    text = (Path(proc_root) / str(pid) / "stat").read_text()
    head, separator, tail = text.rpartition(")")
    fields = tail.split()
    if not separator or "(" not in head or len(fields) < 22:
        raise ValueError("malformed process stat")
    if int(head.split("(", 1)[0]) != pid:
        raise ValueError("process stat PID mismatch")
    return Process(pid, int(fields[1]), int(fields[11]) + int(fields[12]),
                   int(fields[19]))


def _processes(proc_root):
    processes = {}
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError as exc:
        raise SamplingError(f"cannot read proc root {proc_root}: {exc}") from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            process = read_process(proc_root, int(entry.name))
        except (OSError, ValueError, IndexError):
            continue  # Exit, permission failure, or a partially read stat.
        processes[process.pid] = process
    return processes


def _subtree(processes, roots):
    children = {}
    for process in processes.values():
        children.setdefault(process.ppid, []).append(process.pid)
    pending = list(roots)
    selected = {}
    while pending:
        pid = pending.pop()
        if pid in selected or pid not in processes:
            continue
        selected[pid] = processes[pid]
        pending.extend(children.get(pid, ()))
    return selected


def service_tree(roots, proc_root="/proc"):
    """Return a fresh, deduplicated snapshot of all service descendants."""
    selected = _subtree(_processes(proc_root), roots)
    if not selected:
        raise SamplingError("all service roots are unavailable")
    return selected


def resolve_roots(container=None):
    """Use explicit host PIDs, otherwise discover service roots via docker top."""
    if "AUTOPERF_SERVICE_PIDS" in os.environ:
        explicit = os.environ["AUTOPERF_SERVICE_PIDS"].strip()
        if not explicit:
            raise SamplingError("explicit service PIDs are empty")
    else:
        explicit = os.environ.get("AUTOPERF_MONITOR_PIDS", "").strip()
    if explicit:
        try:
            roots = list(dict.fromkeys(int(pid) for pid in re.split(r"[,\s]+", explicit)))
        except ValueError as exc:
            raise SamplingError("explicit service PIDs must be positive integers") from exc
        if not roots or any(pid <= 0 for pid in roots):
            raise SamplingError("explicit service PIDs must be positive integers")
        return roots
    if not container:
        raise SamplingError("container or explicit service PIDs are required")
    try:
        result = subprocess.run(
            ["docker", "top", container, "-eo", "pid,ppid,args"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SamplingError(f"docker top failed: {exc}") from exc
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    roots = select_service_pids(rows)
    if not roots:
        raise SamplingError("docker top returned no service processes")
    return roots


def sample_cpu(roots, proc_root="/proc", seconds=1.0):
    if not math.isfinite(seconds) or seconds <= 0:
        raise SamplingError("CPU_SAMPLE_SECONDS must be a positive finite number")
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    first = _subtree(_processes(proc_root), roots)
    if not first:
        raise SamplingError("all service roots are unavailable")
    try:
        boot_start = float((Path(proc_root) / "uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError) as exc:
        raise SamplingError(f"cannot read proc uptime: {exc}") from exc
    # Timestamp completed snapshots consistently: initial proc discovery is not
    # part of the interval over which we measure CPU deltas.
    start = time.monotonic()
    time.sleep(seconds)
    processes = _processes(proc_root)
    # Keep surviving descendants as anchors if their parent exited/reparented.
    # A reused root PID must not pull an unrelated process tree into the sample.
    anchors = [pid for pid, old in first.items()
               if pid in processes and processes[pid].starttime == old.starttime]
    second = _subtree(processes, anchors)
    elapsed = time.monotonic() - start
    if not second:
        raise SamplingError("all sampled service processes exited or became unavailable")
    if elapsed <= 0:
        raise SamplingError("non-positive CPU sampling interval")
    ticks = 0
    for pid, current in second.items():
        previous = first.get(pid)
        if previous is not None and previous.starttime == current.starttime:
            ticks += max(0, current.ticks - previous.ticks)
        elif boot_start * ticks_per_second <= current.starttime:
            # A child first observed now contributes all of its own lifetime CPU
            # only if it was born within this sample, including a reused child PID.
            ticks += max(0, current.ticks)
    return ticks / ticks_per_second / elapsed * 100.0


def sample_memory(roots, proc_root="/proc"):
    selected = _subtree(_processes(proc_root), roots)
    if not selected:
        raise SamplingError("all service roots are unavailable")
    rss_kib = 0
    measured = 0
    for pid, process in selected.items():
        try:
            status = (Path(proc_root) / str(pid) / "status").read_text()
            match = re.search(r"^VmRSS:\s+(\d+)\s+kB\s*$", status, re.MULTILINE)
            if match is None:
                continue
            # Ensure status did not belong to a replacement with the same PID.
            if read_process(proc_root, pid).starttime != process.starttime:
                continue
            rss_kib += int(match.group(1))
            measured += 1
        except (OSError, ValueError, IndexError):
            continue
    if not measured:
        raise SamplingError("RSS is unavailable for all service processes")
    return rss_kib / (1024.0 * 1024.0)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if not args or args[0] not in ("cpu", "mem"):
            raise SamplingError("usage: python3 -m autoperf.process_metrics cpu|mem [container [workers]]")
        metric = args[0]
        roots = resolve_roots(args[1] if len(args) > 1 else None)
        if metric == "cpu":
            value = sample_cpu(roots, os.environ.get("CPU_PROC_ROOT", "/proc"),
                               float(os.environ.get("CPU_SAMPLE_SECONDS", "1")))
            print(f"{value:.2f}")
        else:
            value = sample_memory(roots, os.environ.get("MEM_PROC_ROOT", "/proc"))
            print(f"{value:.4f}")
        return 0
    except (SamplingError, ValueError) as exc:
        print(f"process_metrics: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
