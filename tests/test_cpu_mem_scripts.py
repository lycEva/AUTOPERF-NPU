import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from autoperf import process_metrics as metrics


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class ProcessMetricsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.proc = Path(self.temp.name)
        (self.proc / "uptime").write_text("100.00 0.00\n")
        self.hz = os.sysconf("SC_CLK_TCK")

    def process(self, pid, ppid=1, ticks=0, start=10, rss=1048576,
                child_ticks=0):
        directory = self.proc / str(pid)
        directory.mkdir(exist_ok=True)
        fields = ["S"] + ["0"] * 21
        fields[1] = str(ppid)
        fields[11] = str(ticks)
        fields[12] = "0"
        fields[13] = fields[14] = str(child_ticks)
        fields[19] = str(start)
        fields[21] = "256"
        (directory / "stat").write_text(
            f"{pid} (worker with (nested) spaces)) " + " ".join(fields) + "\n")
        (directory / "status").write_text(f"Name:\tworker\nVmRSS:\t{rss} kB\n")

    def cpu(self, roots, update):
        with patch.object(metrics.time, "sleep", side_effect=lambda _: update()), \
                patch.object(metrics.time, "monotonic", side_effect=[10.0, 11.0]):
            return metrics.sample_cpu(roots, self.proc, 1)

    def test_stat_parser_handles_spaces_and_parentheses_and_ignores_child_ticks(self):
        self.process(2, ticks=17, child_ticks=99999)
        process = metrics.read_process(self.proc, 2)
        self.assertEqual((process.ppid, process.ticks, process.starttime), (1, 17, 10))

    def test_cpu_aggregates_all_descendants_above_one_core_without_duplicates(self):
        for pid, ppid in [(2, 1), (10, 2), (100, 10)]:
            self.process(pid, ppid, ticks=100, child_ticks=10000)
        def update():
            for pid, ppid in [(2, 1), (10, 2), (100, 10)]:
                self.process(pid, ppid, ticks=100 + self.hz, child_ticks=99999)
        self.assertAlmostEqual(self.cpu([2, 10, 2], update), 300.0)

    def test_cpu_discovers_newborn_but_excludes_preexisting_newly_attached_child(self):
        self.process(2, ticks=100)
        def update():
            self.process(2, ticks=100 + self.hz)
            self.process(3, 2, ticks=self.hz // 2, start=100 * self.hz)
            self.process(4, 3, ticks=50000, start=99 * self.hz)
        self.assertAlmostEqual(self.cpu([2], update), 150.0)

    def test_cpu_reused_child_born_in_interval_counts_only_new_lifetime(self):
        self.process(2)
        self.process(3, 2, ticks=10000)
        def update():
            self.process(3, 2, ticks=self.hz, start=100 * self.hz)
        self.assertAlmostEqual(self.cpu([2], update), 100.0)

    def test_reused_root_does_not_include_unrelated_process_tree(self):
        self.process(2)
        self.process(3, 2)
        def update():
            self.process(2, ticks=99999, start=100 * self.hz)
            self.process(3, 1, ticks=self.hz)
            self.process(4, 2, ticks=99999, start=100 * self.hz)
        self.assertAlmostEqual(self.cpu([2], update), 100.0)

    def test_cpu_keeps_surviving_orphan_and_skips_exited_child(self):
        self.process(2)
        self.process(3, 2)
        self.process(4, 2)
        def update():
            shutil.rmtree(self.proc / "2")
            shutil.rmtree(self.proc / "4")
            self.process(3, 1, ticks=self.hz)
        self.assertAlmostEqual(self.cpu([2], update), 100.0)

    def test_memory_sums_tree_in_gib_and_deduplicates_overlapping_roots(self):
        self.process(2)
        self.process(10, 2)
        self.process(100, 10)
        self.process(999)
        self.assertEqual(metrics.sample_memory([2, 10, 2], self.proc), 3.0)

    def test_memory_skips_races_and_reused_pid(self):
        self.process(2)
        self.process(3, 2)
        self.process(4, 2)
        (self.proc / "4" / "status").unlink()
        original = metrics.read_process
        calls = {}
        def read(root, pid):
            calls[pid] = calls.get(pid, 0) + 1
            if pid == 3 and calls[pid] > 1:
                return metrics.Process(3, 2, 0, 999)
            return original(root, pid)
        with patch.object(metrics, "read_process", side_effect=read):
            self.assertEqual(metrics.sample_memory([2], self.proc), 1.0)

    def test_unavailable_or_malformed_roots_and_missing_rss_fail(self):
        self.process(2)
        (self.proc / "2" / "stat").write_text("malformed\n")
        with self.assertRaises(metrics.SamplingError):
            metrics.sample_memory([2], self.proc)
        with self.assertRaises(metrics.SamplingError):
            metrics.sample_cpu([2], self.proc, 0.01)
        self.process(2)
        (self.proc / "2" / "status").write_text("Name: worker\n")
        with self.assertRaises(metrics.SamplingError):
            metrics.sample_memory([2], self.proc)
        with self.assertRaises(metrics.SamplingError):
            metrics.sample_memory([2], self.proc / "missing")

    def test_all_exited_or_reused_roots_fail_cpu(self):
        for reuse in (False, True):
            self.process(2)
            def update():
                if reuse:
                    self.process(2, start=100 * self.hz)
                else:
                    shutil.rmtree(self.proc / "2")
            with self.subTest(reuse=reuse), self.assertRaises(metrics.SamplingError):
                self.cpu([2], update)

    def test_invalid_sample_intervals_fail(self):
        for interval in (0, -1, float("nan"), float("inf")):
            with self.subTest(interval=interval), self.assertRaises(metrics.SamplingError):
                metrics.sample_cpu([2], self.proc, interval)

    def test_selector_nested_idle_gunicorns_and_multiple_masters(self):
        rows = [(1, 0, "init"), (2, 1, "gunicorn: master [app]"),
                (3, 2, "helper"), (4, 3, "gunicorn: worker [app]"),
                (5, 1, "/bin/gunicorn app:app"), (6, 5, "python busy.py")]
        self.assertEqual(metrics.select_service_pids(rows), [2, 5])

    def test_selector_non_gunicorn_fallback_deduplication_and_order(self):
        rows = [(2, 1, "python service.py"), (4, 3, "helper"),
                (3, 2, "python worker.py"), (9, 8, "other service"),
                (2, 1, "python service.py")]
        self.assertEqual(metrics.select_service_pids(rows), [2, 9])
        self.assertEqual(metrics.select_service_pids([]), [])

    def test_explicit_roots_precedence_and_no_docker(self):
        with patch.dict(os.environ, {"AUTOPERF_SERVICE_PIDS": "2, 3 2\n4",
                                     "AUTOPERF_MONITOR_PIDS": "99"}, clear=True), \
                patch.object(metrics.subprocess, "run") as docker:
            self.assertEqual(metrics.resolve_roots(), [2, 3, 4])
            docker.assert_not_called()
        with patch.dict(os.environ, {"AUTOPERF_MONITOR_PIDS": "7"}, clear=True):
            self.assertEqual(metrics.resolve_roots(), [7])
        for invalid in ("", "  ", "0", "-2", "2,nope"):
            with patch.dict(os.environ, {"AUTOPERF_SERVICE_PIDS": invalid,
                                         "AUTOPERF_MONITOR_PIDS": "99"}, clear=True), \
                    patch.object(metrics.subprocess, "run") as docker, \
                    self.assertRaises(metrics.SamplingError):
                metrics.resolve_roots("service")
            docker.assert_not_called()

    def test_docker_discovery_arguments_fallback_and_errors(self):
        for rows, expected in [("PID PPID COMMAND\n2 1 gunicorn: master\n3 2 gunicorn: worker\n", [2]),
                               ("PID PPID COMMAND\n2 1 python app.py\n3 2 python helper.py\n", [2])]:
            result = subprocess.CompletedProcess([], 0, stdout=rows)
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(metrics.subprocess, "run", return_value=result) as docker:
                self.assertEqual(metrics.resolve_roots("service"), expected)
                self.assertEqual(docker.call_args.args[0],
                                 ["docker", "top", "service", "-eo", "pid,ppid,args"])
        for outcome in (FileNotFoundError("docker"),
                        subprocess.CalledProcessError(1, "docker"),
                        subprocess.TimeoutExpired("docker", 15)):
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(metrics.subprocess, "run", side_effect=outcome), \
                    self.assertRaises(metrics.SamplingError):
                metrics.resolve_roots("service")
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(metrics.SamplingError):
            metrics.resolve_roots()
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(metrics.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="PID PPID COMMAND\n")), \
                self.assertRaises(metrics.SamplingError):
            metrics.resolve_roots("service")

    @unittest.skipUnless(shutil.which("bash"), "requires bash")
    def test_wrappers_from_outside_checkout_use_installed_package_and_fixtures(self):
        self.process(2)
        self.process(3, 2)
        env = os.environ.copy()
        env.update(AUTOPERF_SERVICE_PIDS="2", AUTOPERF_MONITOR_PIDS="999",
                   MEM_PROC_ROOT=str(self.proc), CPU_PROC_ROOT=str(self.proc),
                   CPU_SAMPLE_SECONDS="0.01")
        for name, expected in (("run_mem.sh", 2.0), ("run_cpu.sh", 0.0)):
            result = subprocess.run([str(SCRIPTS / name)], cwd=self.proc,
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertEqual(float(result.stdout), expected)
        env["AUTOPERF_SERVICE_PIDS"] = "99999999"
        for name in ("run_cpu.sh", "run_mem.sh"):
            result = subprocess.run([str(SCRIPTS / name)], cwd=self.proc,
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("unavailable", result.stderr)

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "requires Linux procfs")
    def test_live_busy_multiprocessing_subtree_without_docker_or_npu(self):
        code = '''
import multiprocessing as mp
import time

def busy():
    memory = bytearray(12 * 1024 * 1024)
    while True:
        sum(range(10000))

if __name__ == "__main__":
    children = [mp.Process(target=busy) for _ in range(2)]
    for child in children:
        child.start()
    print(" ".join(str(child.pid) for child in children), flush=True)
    try:
        time.sleep(60)
    finally:
        for child in children:
            child.terminate()
        for child in children:
            child.join()
'''
        service = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            children = [int(pid) for pid in service.stdout.readline().split()]
            self.assertEqual(len(children), 2)
            roots = [service.pid, children[0], service.pid]
            memory = metrics.sample_memory(roots)
            root_memory = metrics.sample_memory([service.pid])
            self.assertGreater(memory, 0.01)
            self.assertAlmostEqual(memory, root_memory, delta=0.03)
            self.assertGreater(metrics.sample_cpu(roots, seconds=0.25), 5.0)
        finally:
            import signal
            os.killpg(service.pid, signal.SIGTERM)
            service.wait(timeout=10)
            service.stdout.close()


if __name__ == "__main__":
    unittest.main()
