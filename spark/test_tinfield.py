#!/usr/bin/env python3
"""CPU-only recipe contract tests; no Docker daemon, network or GPU needed."""

import copy
import hashlib
import json
from pathlib import Path
import socket
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import tinfield as tf


class TinfieldTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)
        self.p = copy.deepcopy(tf.profile())

    def config(self):
        return {'model_type': 'qwen4_exp',
                'text_config': {'model_type': 'qwen4_exp_text', 'max_position_embeddings': 262144,
                                'mtp_num_hidden_layers': 1},
                'quantization_config': {'quant_method': 'exl3', 'bits': 4.05,
                                        'codebook': 'mul1', 'version': '1.5.1'}}

    def fixture(self, *, mtp=True):
        path = tf.snapshot(self.cache, self.p)
        path.mkdir(parents=True)
        shards = [f for f in self.p['files'] if f.startswith('model-') and f.endswith('.safetensors')]
        index = {f'model.layer.{i}': name for i, name in enumerate(shards)}
        if mtp:
            index.update({'mtp.fc_embedding.trellis': shards[-1], 'mtp.fc_hidden.trellis': shards[-1]})
        for name, spec in self.p['files'].items():
            data = json.dumps(self.config()).encode() if name == 'config.json' else b'fixture'
            if name == 'model.safetensors.index.json':
                data = json.dumps({'weight_map': index}).encode()
            (path / name).write_bytes(data)
            spec['size'] = len(data)
            if spec.get('sha256'):
                spec['sha256'] = hashlib.sha256(data).hexdigest()
        return path

    def table(self, *, missing=None, width=61, short=False):
        file = self.cache / 'table.safetensors'
        base = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.'
        header = {base + k: {'dtype': 'I64', 'shape': [1], 'data_offsets': [0, 8]}
                  for k in ('head_bias', 'head_offsets', 'head_vocab_sizes', 'layer_multipliers')}
        offset = 8
        for i in range(128):
            size = 2500012 * width * 2
            if i != missing:
                header[base + f'shard_{i}.trellis'] = {
                    'dtype': 'I16', 'shape': [2500012, width], 'data_offsets': [offset, offset + size]}
            offset += size
        data = json.dumps(header).encode()
        with file.open('wb') as stream:
            stream.write(struct.pack('<Q', len(data)) + data)
            if not short:
                stream.truncate(8 + len(data) + offset)  # sparse fixture, not 36 GiB of writes
        return file

    def test_profile_pin_and_inventory(self):
        self.assertEqual(self.p['revision'], '460f8565373f20e1c172f72f261f591e4f76b8a4')
        self.assertEqual(self.p['branch'], '4.05bpw_h6_ng6')
        self.assertIn('ngram_embedding.safetensors', self.p['files'])
        self.assertEqual(len([n for n in self.p['files'] if n.startswith('model-')]), 9)
        for spec in self.p['files'].values():
            if 'sha256' in spec:
                self.assertEqual(len(spec['sha256']), 64)

    def test_snapshot_never_uses_main_or_newest(self):
        target = tf.snapshot(self.cache, self.p)
        self.assertEqual(target.name, self.p['revision'])
        self.assertEqual(target.parent.name, 'snapshots')
        self.assertIn('models--khronnuz--Tinfield-1-exl3', str(target))

    def test_pinned_config(self):
        tf.check_config(self.config(), self.p)

    def test_wrong_family_is_rejected(self):
        cfg = self.config()
        cfg['model_type'] = 'qwen3_5'
        with self.assertRaisesRegex(ValueError, 'family'):
            tf.check_config(cfg, self.p)

    def test_nvfp4_is_rejected(self):
        cfg = self.config()
        cfg['quantization_config']['quant_method'] = 'modelopt'
        with self.assertRaisesRegex(ValueError, 'quantization'):
            tf.check_config(cfg, self.p)

    def test_short_context_or_absent_mtp_is_rejected(self):
        for key, value in (('max_position_embeddings', 32768), ('mtp_num_hidden_layers', 0)):
            cfg = self.config()
            cfg['text_config'][key] = value
            with self.assertRaisesRegex(ValueError, 'context or MTP'):
                tf.check_config(cfg, self.p)

    def test_complete_snapshot(self):
        path = self.fixture()
        self.assertEqual(tf.check_snapshot(self.cache, self.p), path)

    def test_missing_extra_table_is_rejected(self):
        path = self.fixture()
        (path / 'ngram_embedding.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete.*ngram'):
            tf.check_snapshot(self.cache, self.p)

    def test_partial_weight_is_rejected(self):
        path = self.fixture()
        (path / 'model-00001-of-00009.safetensors').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'incomplete.*model-00001'):
            tf.check_snapshot(self.cache, self.p)

    def test_missing_mtp_tensors_is_rejected(self):
        self.fixture(mtp=False)
        with self.assertRaisesRegex(ValueError, 'MTP tensors'):
            tf.check_snapshot(self.cache, self.p)

    def test_corrupt_same_size_metadata_is_rejected(self):
        path = self.fixture()
        (path / 'tokenizer.json').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'corrupt metadata'):
            tf.check_snapshot(self.cache, self.p)

    def test_wrong_symlink_blob_is_rejected(self):
        path = self.fixture()
        file = path / 'model-00001-of-00009.safetensors'
        blob = self.cache / 'wrong-hf-blob'
        blob.write_bytes(file.read_bytes())
        file.unlink()
        file.symlink_to(blob)
        with self.assertRaisesRegex(ValueError, 'wrong pinned HF blob'):
            tf.check_snapshot(self.cache, self.p)

    def test_progress_counts_known_partial_only(self):
        blob = tf.snapshot(self.cache, self.p).parent.parent / 'blobs'
        blob.mkdir(parents=True)
        spec = self.p['files']['ngram_embedding.safetensors']
        (blob / (spec['sha256'] + '.incomplete')).write_bytes(b'1234')
        (blob / 'unrelated.incomplete').write_bytes(b'not this model')
        status = tf.progress(self.cache, self.p)
        self.assertEqual(status['downloaded_bytes'], 4)
        self.assertEqual(status['complete_files'], 0)

    def test_actual_table_segment_geometry(self):
        self.assertEqual(tf.table_header(self.table()), 320001536)

    def test_progress_supports_hf124_random_suffix_without_double_counting(self):
        blob = tf.snapshot(self.cache, self.p).parent.parent / 'blobs'
        blob.mkdir(parents=True)
        digest = self.p['files']['ngram_embedding.safetensors']['sha256']
        (blob / (digest + '.incomplete')).write_bytes(b'1234')
        (blob / (digest + '.16496f3a.incomplete')).write_bytes(b'12345678')
        self.assertEqual(tf.progress(self.cache, self.p)['downloaded_bytes'], 8)

    def test_missing_table_segment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'segment 17'):
            tf.table_header(self.table(missing=17))

    def test_wrong_table_codec_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'six-bit'):
            tf.table_header(self.table(width=41))

    def test_truncated_table_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'past the file'):
            tf.table_header(self.table(short=True))

    def test_download_is_bounded_and_does_not_load_gpu(self):
        cmd = tf.worker_command(self.p, self.cache)
        self.assertIn('--memory=512m', cmd)
        self.assertIn('--memory-swap=512m', cmd)
        self.assertIn('--runtime=runc', cmd)
        self.assertIn('NVIDIA_VISIBLE_DEVICES=void', cmd)
        self.assertIn('HF_HUB_OFFLINE=0', cmd)
        self.assertIn('HF_HUB_DISABLE_XET=1', cmd)
        self.assertIn('--restart=no', cmd)
        self.assertNotIn('--gpus', cmd)
        self.assertNotIn('--rm', cmd)

    def test_checks_are_cpu_only_and_offline(self):
        cmd = tf.check_command(self.p, self.cache)
        self.assertIn('--network=none', cmd)
        self.assertIn('--memory=2g', cmd)
        self.assertIn(f'type=bind,src={self.cache},dst={tf.CONTAINER_HF},readonly', cmd)
        self.assertNotIn('--gpus', cmd)

    def test_launch_preserves_route_and_decode_settings(self):
        cmd = tf.launch_command(self.p, self.cache, self.cache / 'kernels')
        for flag, value in (('--port', '8888'), ('--host', '127.0.0.1'), ('--parallel', '4'),
                            ('--context', '262144'), ('--kv-dtype', 'int4'), ('--mtp-drafts', '4'),
                            ('--decode-share', f"{self.p.get('decode_share', .20):.2f}"), ('--name', self.p['container'])):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn(self.p['alias'], cmd)
        self.assertIn(self.p['image'], cmd)
        self.assertIn(str(tf.snapshot(tf.CONTAINER_HF, self.p)), cmd)
        self.assertIn('--restart=unless-stopped', cmd)
        self.assertIn('TENSORFOLD_MTP_COPY=1', cmd)
        for bad in ('--ple-on-ssd', '--prefill-fp8', '--vision', '--publish', '30000', 'main'):
            self.assertNotIn(bad, cmd)

    def test_decode_share_default_and_override(self):
        for share in (None, .30):
            p = {**self.p}
            p.pop('decode_share', None)
            if share is not None:
                p['decode_share'] = share
            cmd = tf.launch_command(p, self.cache, self.cache / 'kernels')
            self.assertEqual(cmd[cmd.index('--decode-share') + 1], '0.20' if share is None else '0.30')

    def test_invalid_decode_share_is_rejected(self):
        for share in (-.1, 1, float('nan'), '0.30'):
            with self.assertRaisesRegex(ValueError, 'decode_share'):
                tf.launch_command({**self.p, 'decode_share': share}, self.cache, self.cache / 'kernels')

    def test_busy_port_cannot_mutate_any_container(self):
        with patch.object(tf, 'refuse_busy_port', side_effect=ValueError('occupied')), \
                patch.object(tf, 'run') as run, patch.object(tf, 'inspect_container') as inspect:
            with self.assertRaisesRegex(ValueError, 'occupied'):
                tf.start(self.p, self.cache, self.cache / 'kernels')
            run.assert_not_called()
            inspect.assert_not_called()

    def test_busy_socket(self):
        with patch.object(socket.socket, 'bind', side_effect=OSError('in use')):
            with self.assertRaisesRegex(ValueError, 'never stops'):
                tf.refuse_busy_port()

    def test_port_probe_reuses_time_wait_without_reuse_port(self):
        with patch.object(tf.socket, 'socket') as factory:
            tf.refuse_busy_port()
            probe = factory.return_value.__enter__.return_value
            probe.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind.assert_called_once_with(('127.0.0.1', 8888))

    def test_preserve_existing_stopped_backend(self):
        with patch.object(tf, 'refuse_busy_port'), patch.object(tf, 'inspect_container', return_value={"exists": True}), \
                patch.object(tf, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'already exists'):
                tf.start(self.p, self.cache, self.cache / 'kernels')
            run.assert_not_called()

    def test_inspect_recognizes_both_missing_name_error_cases(self):
        for error in ('Error: No such object: trial', 'error: no such object: trial'):
            with patch.object(tf.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr=error)):
                self.assertIsNone(tf.inspect_container('trial'))

    def test_inspect_never_treats_daemon_failure_as_free_name(self):
        with patch.object(tf.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='permission denied')):
            with self.assertRaisesRegex(RuntimeError, 'permission denied'):
                tf.inspect_container('trial')

    def test_prepare_does_not_duplicate_running_worker(self):
        state = {'Config': {'Labels': {'spark.tinfield.revision': self.p['revision']}},
                 'Image': self.p['image'], 'State': {'Running': True}}
        with patch.object(tf, 'inspect_container', return_value=state), patch.object(tf, 'run') as run:
            tf.prepare(self.p, self.cache)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][:3], ['docker', 'image', 'inspect'])

    def test_prepare_refuses_blind_oom_retry(self):
        state = {'Config': {'Labels': {'spark.tinfield.revision': self.p['revision']}},
                 'Image': self.p['image'], 'State': {'Running': False, 'ExitCode': 137, 'OOMKilled': True}}
        with patch.object(tf, 'inspect_container', return_value=state), patch.object(tf, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'memory cap'):
                tf.prepare(self.p, self.cache)
            self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
