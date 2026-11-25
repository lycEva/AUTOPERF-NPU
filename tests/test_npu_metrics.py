import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from autoperf import npu_metrics as npu
from autoperf.process_metrics import SamplingError


HEADER = '| NPU Chip | Process id | Process name | Process memory(MB) |\n'
HARDWARE = '| NPU Name | Health | Power(W) Temp(C) |\n| 0 0 | 222 | device | 9999 |\n'


def table(*rows):
    return HARDWARE + HEADER + '\n'.join(
        f'| {card} {chip} | {pid} | yt_server | {memory} |'
        for card, chip, pid, memory in rows) + '\n'


def stat(root, pid, ppid=1, start=100, name='idle server'):
    path = Path(root) / str(pid)
    path.mkdir(exist_ok=True)
    fields = ['S', str(ppid)] + ['0'] * 22
    fields[19] = str(start)
    (path / 'stat').write_text(f'{pid} ({name}) ' + ' '.join(fields))


class NpuMetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.proc = Path(self.tmp.name)
        stat(self.proc, 100, name='idle launcher')
        stat(self.proc, 101, 100, name='python spawn')
        stat(self.proc, 102, 101, name='native engine')
        stat(self.proc, 222, name='unrelated')

    def sample(self, metric, text, device_info=None):
        with patch.object(npu, '_query', return_value=text) as query:
            value = metric([100], self.proc, device_info)
        return value, query

    def test_nested_idle_non_gunicorn_tree_and_all_memory_records(self):
        text = table((0, 0, 100, 2), (0, 0, 102, 10), (0, 0, 102, 10),
                     (0, 1, 102, 20), (1, 0, 101, 30), (9, 0, 222, 900))
        value, query = self.sample(npu.sample_memory, text)
        self.assertEqual(value, 62)
        query.assert_called_once_with([])

    def test_usage_is_unique_chip_sum_and_uses_only_service_pairs(self):
        text = table((0, 0, 101, 10), (0, 0, 102, 20), (0, 0, 102, 20),
                     (0, 1, 102, 30), (1, 0, 102, 40), (9, 0, 222, 900))
        outputs = [text, '\tAicore Usage Rate(%) : 15\n',
                   '\tAicore Usage Rate(%) : 25.5\n', 'Aicore Usage Rate(%) : 100']
        with patch.object(npu, '_query', side_effect=outputs) as query:
            self.assertEqual(npu.sample_usage([100], self.proc), 140.5)
        self.assertEqual([call.args[0] for call in query.call_args_list], [[],
            ['-t', 'usages', '-i', '0', '-c', '0'],
            ['-t', 'usages', '-i', '0', '-c', '1'],
            ['-t', 'usages', '-i', '1', '-c', '0']])

    def test_no_match_returns_zero_without_usage_queries_or_hardware_rows(self):
        value, query = self.sample(npu.sample_usage, table((0, 0, 222, 10)), '0:0,9:9')
        self.assertEqual(value, 0)
        query.assert_called_once_with([])
        value, _ = self.sample(npu.sample_memory, table((0, 0, 222, 10)))
        self.assertEqual(value, 0)

    def test_device_filter_only_intersects_observed_pairs(self):
        value, query = self.sample(npu.sample_memory,
                                  table((0, 0, 101, 10), (0, 1, 102, 20)), '0:1,9:9')
        self.assertEqual(value, 20)
        with patch.object(npu, '_query', side_effect=[table((0, 0, 102, 10)),
                                                    'Aicore Usage Rate(%) : 12']) as query:
            self.assertEqual(npu.sample_usage([100], self.proc, '0:0,8:8'), 12)
        self.assertEqual(query.call_count, 2)

    def test_tree_and_associations_refresh_each_sample(self):
        with patch.object(npu, '_query', return_value=table((0, 0, 103, 40))):
            self.assertEqual(npu.sample_memory([100], self.proc), 0)
            stat(self.proc, 103, 102)
            self.assertEqual(npu.sample_memory([100], self.proc), 40)
        with patch.object(npu, '_query', return_value=table((1, 2, 103, 50))):
            self.assertEqual(npu.sample_memory([100], self.proc, '0:0'), 0)
            self.assertEqual(npu.sample_memory([100], self.proc, '1:2'), 50)

    def test_reused_or_exited_pid_is_discarded_after_info(self):
        def query(args):
            stat(self.proc, 102, 101, start=200)
            (self.proc / '101' / 'stat').unlink()
            return table((0, 0, 101, 30), (0, 1, 102, 40))
        with patch.object(npu, '_query', side_effect=query) as run:
            self.assertEqual(npu.sample_usage([100], self.proc), 0)
        run.assert_called_once_with([])

    def test_reused_pid_after_usage_query_excludes_only_its_chip(self):
        def query(args):
            if not args:
                return table((0, 0, 101, 10), (0, 1, 102, 20))
            if args[-1] == '1':
                stat(self.proc, 102, 101, start=200)
            return 'Aicore Usage Rate(%) : 15'
        with patch.object(npu, '_query', side_effect=query):
            self.assertEqual(npu.sample_usage([100], self.proc), 15)

    def test_all_gone_after_info_or_usage_fails(self):
        for pid in (101, 102):
            stat(self.proc, pid, ppid=1)
        for after_usage in (False, True):
            with self.subTest(after_usage=after_usage):
                stat(self.proc, 100)
                def query(args):
                    if bool(args) == after_usage:
                        (self.proc / '100' / 'stat').unlink()
                    return 'Aicore Usage Rate(%) : 15' if args else table((0, 0, 100, 10))
                with patch.object(npu, '_query', side_effect=query), self.assertRaises(SamplingError):
                    npu.sample_usage([100], self.proc)

    def test_missing_roots_fails_without_query(self):
        with patch.object(npu, '_query') as query, self.assertRaises(SamplingError):
            npu.sample_memory([999], self.proc)
        query.assert_not_called()

    def test_real_padded_header_and_mixed_active_empty_cards(self):
        header = '| NPU     Chip | Process id | Process name | Process memory(MB) |\n'
        text = (HARDWARE + header + '| 0       0 | 102 | yt_server | 5263 |\n'
                '| No running processes found in NPU 32768                                  |\n')
        self.assertEqual(npu.parse_process_table(text), {(0, 0, 102): 5263})
        self.assertEqual(self.sample(npu.sample_memory, text)[0], 5263)
        with patch.object(npu, '_query', side_effect=[text, 'Aicore Usage Rate(%) : 15']) as query:
            self.assertEqual(npu.sample_usage([100], self.proc), 15)
        self.assertEqual([call.args[0] for call in query.call_args_list],
                         [[], ['-t', 'usages', '-i', '0', '-c', '0']])

    def test_valid_empty_process_tables(self):
        for text in (HEADER, HEADER + '| No running processes |\n',
                     '| No running processes found |\n',
                     '| No running processes found in NPU 32768                                  |\n',
                     HEADER + '| No running processes found in NPU 32768 |\n'):
            with self.subTest(text=text):
                self.assertEqual(npu.parse_process_table(text), {})
                self.assertEqual(self.sample(npu.sample_usage, text)[0], 0)

    def test_malformed_process_tables_fail_usefully(self):
        for text in ('', HARDWARE, 'garbage No running processes garbage',
                     HEADER + '| 0 | 101 | engine | 3 |',
                     HEADER + '| 0 0 | bad | engine | 3 |',
                     HEADER + '| 0 0 | 101 | engine | unknown |',
                     HEADER + '| 0 0 | 101 | engine | nan |',
                     HEADER + '| 0 0 | 101 | engine | -1 |',
                     HEADER + '| 0 0 | 101 | engine | 3 | extra |',
                     table((0, 0, 101, 3), (0, 0, 101, 4))):
            with self.subTest(text=text), self.assertRaises(SamplingError):
                npu.parse_process_table(text)

    def test_invalid_usage_and_device_filter_fail(self):
        for value in ('nan', 'inf', '-1', '101', 'garbage', ''):
            with self.subTest(value=value), self.assertRaises(SamplingError):
                npu.parse_usage('Aicore Usage Rate(%) : ' + value)
        for text in ('', 'CPU Usage Rate(%) : 15',
                     'Aicore Usage Rate(%) : 1\nAicore Usage Rate(%) : 2'):
            with self.subTest(text=text), self.assertRaises(SamplingError):
                npu.parse_usage(text)
        for device in ('0', '0:1,', '-1:0', '0:1:2'):
            with self.subTest(device=device), self.assertRaises(SamplingError):
                npu.sample_memory([100], self.proc, device)

    def test_query_timeout_failure_and_missing_binary_are_errors(self):
        for error in (FileNotFoundError('npu-smi'),
                      subprocess.TimeoutExpired(['npu-smi'], 15),
                      subprocess.CalledProcessError(1, ['npu-smi'], stderr='device unavailable')):
            with self.subTest(error=error), patch.object(npu.subprocess, 'run', side_effect=error), self.assertRaisesRegex(SamplingError, 'npu-smi query failed'):
                npu.sample_memory([100], self.proc)
        with patch.object(npu.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, HEADER)) as run:
            self.assertEqual(npu.sample_memory([100], self.proc), 0)
        self.assertEqual(run.call_args.args[0], ['npu-smi', 'info'])
        self.assertEqual(run.call_args.kwargs['timeout'], 15)

    def test_wrappers_outside_checkout_explicit_pid_without_docker(self):
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        binary = self.proc / 'bin'
        binary.mkdir()
        log = self.proc / 'calls'
        smi = binary / 'npu-smi'
        smi.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n'
                       'if [ "$*" = "info" ]; then\n'
                       "printf '%s\\n' '| NPU Chip | Process id | Process name | Process memory(MB) |' "
                       "'| 0 0 | 102 | yt_server | 5263 |'\n"
                       "else printf '\\tAicore Usage Rate(%%) : 15\\n'; fi\n")
        smi.chmod(0o755)
        docker = binary / 'docker'
        docker.write_text('#!/bin/sh\necho docker-called >&2\nexit 90\n')
        docker.chmod(0o755)
        env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                   AUTOPERF_SERVICE_PIDS='100', AUTOPERF_MONITOR_PIDS='222',
                   NPU_PROC_ROOT=str(self.proc), CALL_LOG=str(log))
        for metric, expected in (('mem', '5263.00\n'), ('usage', '15.00\n')):
            script = scripts / f'run_npu_{metric}.sh'
            self.assertEqual(script.stat().st_mode & 0o777, 0o755)
            result = subprocess.run([str(script), 'unused', '1', '0:0,9:9'],
                                    cwd=self.proc, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, expected)
            self.assertEqual(result.stderr, '')
        self.assertEqual(log.read_text().splitlines(), ['info', 'info', 'info -t usages -i 0 -c 0'])
        env['AUTOPERF_SERVICE_PIDS'] = ''
        result = subprocess.run([str(scripts / 'run_npu_mem.sh'), 'unused'],
                                cwd=self.proc, env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertIn('explicit service PIDs are empty', result.stderr)


if __name__ == '__main__':
    unittest.main()
