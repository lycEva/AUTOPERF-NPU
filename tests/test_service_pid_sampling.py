import logging
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from autoperf.monitor import MonitorLoop, _run_monitor_scripts, _run_script, run_monitor_once
from autoperf.process_metrics import select_service_pids


class ServicePidSamplingTest(unittest.TestCase):
    def test_idle_master_is_root_and_all_workers_belong_to_tree(self):
        rows = [
            ("1", "0", "/bin/bash entrypoint.sh"),
            ("2", "1", "python gunicorn master"),
            ("3", "2", "python gunicorn worker"),
            ("4", "3", "python multiprocessing.spawn"),
            ("5", "4", "native-engine"),
            ("6", "1", "bash"),
        ]
        self.assertEqual([str(p) for p in select_service_pids(rows)], ["2"])

    def test_cli_passes_same_complete_service_tree_to_all_samplers(self):
        rows = [
            ("1", "0", "0", "00:00:00", "/bin/bash entrypoint.sh"),
            ("2", "1", "0", "00:00:00", "python gunicorn master"),
            ("3", "2", "40", "00:01:00", "python gunicorn worker"),
            ("4", "3", "90", "00:02:00", "python multiprocessing.spawn"),
            ("5", "4", "0", "00:00:00", "native-engine"),
            ("6", "1", "90", "01:00:00", "unrelated-helper"),
        ]
        calls = {}

        def sample(path, container, workers, device, logger, timeout, worker_pids, related_pids, service_pids):
            calls[path.name] = (worker_pids, related_pids, service_pids)
            return 1.0

        with patch("autoperf.monitor._docker_top_rows", return_value=rows) as top, patch(
            "autoperf.monitor._run_script", side_effect=sample
        ):
            values = _run_monitor_scripts(Path("scripts"), "service", 1, "", logging.getLogger())
        top.assert_called_once_with("service")
        self.assertEqual(values["mem_usage"], 1.0)
        self.assertEqual(calls["run_cpu.sh"][2], ["2"])
        self.assertEqual(calls["run_mem.sh"][2], ["2"])
        for _, (roots, related, service) in calls.items():
            self.assertEqual(roots, ["2"])
            self.assertEqual(service, ["2"])
            self.assertEqual(related, {"2", "3", "4", "5"})

    def test_monitor_start_does_not_cache_discovered_chips(self):
        loop = MonitorLoop(Path("scripts"), "service", 1, "", Path("unused/monitor.csv"),
                           0.1, logging.getLogger())
        with patch("autoperf.monitor.validate_monitor_scripts"), patch(
            "autoperf.monitor.Path.mkdir"
        ), patch("autoperf.monitor.threading.Thread") as thread, patch(
            "autoperf.monitor.discover_npu_devices"
        ) as discover:
            loop.start()
        discover.assert_not_called()
        thread.return_value.start.assert_called_once()
        self.assertEqual(loop.device_info, "")

    def test_one_shot_monitor_refreshes_devices_inside_samplers(self):
        with patch("autoperf.monitor.validate_monitor_scripts"), patch(
            "autoperf.monitor.discover_npu_devices"
        ) as discover, patch("autoperf.monitor._run_monitor_scripts", return_value={}) as sample:
            run_monitor_once(Path("scripts"), "service", 1, logging.getLogger())
        discover.assert_not_called()
        self.assertEqual(sample.call_args.args[3], "")

    def test_roots_are_refreshed_between_sampling_cycles(self):
        snapshots = [
            [("2", "1", "0", "00:00:00", "gunicorn master")],
            [("20", "1", "0", "00:00:00", "gunicorn master")],
        ]
        observed = []

        def sample(*args):
            observed.append(args[-1])
            return 0.0

        with patch("autoperf.monitor._docker_top_rows", side_effect=snapshots), patch(
            "autoperf.monitor._run_script", side_effect=sample
        ):
            for _ in range(2):
                _run_monitor_scripts(Path("scripts"), "service", 1, "", logging.getLogger())
        self.assertEqual(observed[:4], [["2"]] * 4)
        self.assertEqual(observed[4:], [["20"]] * 4)

    def test_service_pid_environment_reaches_subprocess(self):
        logger = logging.getLogger("service-env")
        with patch("autoperf.monitor.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "2.5\n", "")) as run:
            value = _run_script(Path("run_mem.sh"), "service", 1, "", logger,
                                worker_pids=["9"], related_pids={"9"}, service_pids=["2", "1"])
        self.assertEqual(value, 2.5)
        self.assertEqual(run.call_args.kwargs["env"]["AUTOPERF_SERVICE_PIDS"], "1,2")

    def test_discovery_failure_is_logged_and_explicitly_empty(self):
        logger = logging.getLogger("service-discovery")
        calls = []

        def sample(*args):
            calls.append(args)
            return 0.0

        with patch("autoperf.monitor._docker_top_rows", side_effect=RuntimeError("docker unavailable")), patch(
            "autoperf.monitor._run_script", side_effect=sample
        ), self.assertLogs(logger, level="WARNING") as logs:
            _run_monitor_scripts(Path("scripts"), "service-discovery-fail", 1, "", logger)
        self.assertIn("docker unavailable", "\n".join(logs.output))
        self.assertEqual(next(c for c in calls if c[0].name == "run_cpu.sh")[-1], [])


if __name__ == "__main__":
    unittest.main()
