#!/usr/bin/env python3
"""Pinned Tinfield EXL3 preparation; never stops another model or changes routes."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import socket
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
GIB = 1024**3
CONTAINER_HF = Path('/root/.cache/huggingface')


def profile():
    return json.loads((HERE / 'tinfield-exl3.json').read_text())


def snapshot(cache, p):
    return Path(cache) / 'hub' / ('models--' + p['repo'].replace('/', '--')) / 'snapshots' / p['revision']


def progress(cache, p):
    """Completed snapshot files plus this revision's known unfinished HF blobs."""
    path = snapshot(cache, p)
    ready = downloaded = 0
    for name, spec in p['files'].items():
        file = path / name
        count = file.stat().st_size if file.is_file() else 0
        if count == spec['size']:
            ready += 1
        elif not count and spec.get('sha256'):
            blobs = path.parent.parent / 'blobs'
            partials = [blobs / (spec['sha256'] + '.incomplete')]
            # HF 1.24 uses <hash>.<random>.incomplete; older clients use
            # <hash>.incomplete. Never sum overlapping attempts for one blob.
            partials.extend(blobs.glob(spec['sha256'] + '.*.incomplete'))
            for partial in partials:
                try:
                    count = max(count, partial.stat().st_size)
                except FileNotFoundError:  # worker atomically finished/renamed it
                    pass
        downloaded += min(count, spec['size'])
    return {'revision': p['revision'], 'complete_files': ready, 'files': len(p['files']),
            'downloaded_bytes': downloaded, 'total_bytes': sum(f['size'] for f in p['files'].values()),
            'snapshot': str(path)}


def check_config(config, p):
    quant, text = config.get('quantization_config', {}), config.get('text_config', {})
    if config.get('model_type') != 'qwen4_exp' or text.get('model_type') != 'qwen4_exp_text':
        raise ValueError('Tinfield requires the Flash Next qwen4_exp family, not dense Qwen')
    if quant.get('quant_method') != 'exl3' or quant.get('bits') != p['bits']:
        raise ValueError('wrong quantization pack for this pinned Tinfield profile')
    if quant.get('codebook') != 'mul1' or quant.get('version') != '1.5.1':
        raise ValueError('unexpected EXL3 codec/version')
    if text.get('max_position_embeddings', 0) < p['context'] or text.get('mtp_num_hidden_layers', 0) < 1:
        raise ValueError('native context or MTP head missing')


def check_snapshot(cache, p):
    path = snapshot(cache, p)
    for name, spec in p['files'].items():
        file = path / name
        if not file.is_file() or file.stat().st_size != spec['size']:
            raise ValueError(f'incomplete pinned snapshot: {name}; run prepare/status first')
        # Small critical LFS metadata is hashed; weight payloads are checked by size
        # and HF blob identity, not re-read in full over the live backend's SSD.
        if spec.get('sha256'):
            if file.is_symlink() and file.resolve().name != spec['sha256']:
                raise ValueError(f'wrong pinned HF blob: {name}')
            if not name.endswith('.safetensors'):
                with file.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != spec['sha256']:
                    raise ValueError(f'corrupt metadata: {name}')
    check_config(json.loads((path / 'config.json').read_text()), p)
    index = json.loads((path / 'model.safetensors.index.json').read_text())['weight_map']
    shards = {name for name in p['files'] if name.startswith('model-') and name.endswith('.safetensors')}
    if set(index.values()) != shards:
        raise ValueError('model index does not name exactly the pinned nine weight shards')
    if not {'mtp.fc_embedding.trellis', 'mtp.fc_hidden.trellis'} <= index.keys():
        raise ValueError('required Tinfield MTP tensors missing')
    return path


def table_header(file):
    with file.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < length <= 64 * 1024**2:
            raise ValueError('invalid EXL3 table header length')
        header = json.loads(stream.read(length))
    base = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.'
    # This revision has 128 packed segments in ONE extra safetensors file.
    # Do not confuse those segments with 128 separate downloaded files.
    rows = 0
    for i in range(128):
        info = header.get(base + f'shard_{i}.trellis', {})
        shape = info.get('shape', [])
        if info.get('dtype') != 'I16' or len(shape) != 2 or shape[0] != 2500012:
            raise ValueError(f'expected Tinfield int16 EXL3 n-gram segment {i}')
        begin, end = info['data_offsets']
        if shape[1] != 61 or end - begin != 2 * shape[0] * shape[1]:
            raise ValueError('expected one scale plus 160 six-bit values per Tinfield n-gram row')
        if begin < 0 or 8 + length + end > file.stat().st_size:
            raise ValueError('EXL3 n-gram rows extend past the file')
        rows += shape[0]
    for suffix in ('head_bias', 'head_offsets', 'head_vocab_sizes', 'layer_multipliers'):
        if base + suffix not in header:
            raise ValueError(f'missing EXL3 n-gram lookup metadata: {suffix}')
    return rows


def runtime_check(p):
    import inspect
    import torch
    from tensorfold.families import qwen4_exp
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import NgramTable
    from tensorfold.families.qwen4_exp.cuda import multi

    if importlib.metadata.version('tensorfold') != p['tensorfold_version']:
        raise ValueError(f"this launcher is pinned to TensorFold {p['tensorfold_version']}")
    if 'exl3' not in qwen4_exp.QUANT_METHODS['cuda'] or qwen4_exp.EXL3_VARIANT != 'any':
        raise ValueError('runtime lacks Flash Next EXL3 support')
    if 'consolidated' not in inspect.getsource(NgramTable):
        raise ValueError('runtime cannot read Tinfield consolidated n-gram tables')
    if 'copy_accepted' not in inspect.getsource(multi):
        raise ValueError('runtime lacks the verified prompt-copy patch')
    if torch.cuda.is_initialized():
        raise ValueError('CPU preflight unexpectedly initialized CUDA')
    return {'tensorfold': p['tensorfold_version'], 'exl3': True, 'consolidated_ple': True,
            'prompt_copy': True, 'gpu_loaded': False}


def capacity_report(path, p):
    """The installed engine's header-only estimate, no CUDA context or weight load."""
    from tensorfold.cuda.capacity import Weights, config, estimate_weights
    from tensorfold.cuda.geometry import indexed_stream_geometry, indexed_weights
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import admission, extra_files

    text = config(path)
    transform = indexed_weights(1, True, mapped_tables=True)
    main = estimate_weights(path, transform)
    extra = estimate_weights(path, transform, files=list(extra_files(path)))
    weights = Weights(main.resident + extra.resident, max(main.staging, extra.staging), main.mapped + extra.mapped)
    each = p['mtp_drafts'] + 1
    geometry = admission(lambda t: indexed_stream_geometry(t, p['parallel'], each, 8, mtp=True, kv_bits=4))(text)
    # The stock guard initially sizes one full window; other streams grow subject
    # to the runtime memory gate. Also show four genuinely full windows, not just
    # '--parallel 4' as proof of a million tokens simultaneously resident.
    full = admission(lambda t: indexed_stream_geometry(t, p['parallel'], each, 8, mtp=True,
                                                       kv_bits=4, first=p['context'] + each))(text)
    startup = weights.resident + max(weights.staging, geometry.needed(p['context']))
    four = weights.resident + max(weights.staging, full.needed(p['context']))
    return {'weights_gib': weights.resident / GIB, 'mapped_ple_gib': weights.mapped / GIB,
            'startup_peak_gib': startup / GIB, 'four_full_windows_peak_gib': four / GIB,
            'four_full_windows_plus_ple_gib': (four + weights.mapped) / GIB,
            'note': 'estimates exclude OS reserve/other services; actual CUDA admission remains authoritative'}


def docker_base(p, cache, *, worker=False):
    args = ['docker', 'run', '--pull=never', '--runtime=runc', '--cpus=2',
            '--memory=512m' if worker else '--memory=2g',
            '--memory-swap=512m' if worker else '--memory-swap=2g',
            '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=',
            '-e', 'HF_HUB_OFFLINE=0' if worker else 'HF_HUB_OFFLINE=1',
            '-e', 'HF_HUB_DISABLE_XET=1', '-e', 'HF_HUB_ENABLE_HF_TRANSFER=0',
            '-e', 'HF_HOME=' + str(CONTAINER_HF), '-e', 'PYTHONUNBUFFERED=1',
            '--mount', f'type=bind,src={HERE},dst=/recipe,readonly',
            '--mount', f'type=bind,src={cache},dst={CONTAINER_HF}' + ('' if worker else ',readonly')]
    return args


def worker_command(p, cache):
    name = p['container'] + '-download'
    return docker_base(p, cache, worker=True) + [
        '-d', '--name', name, '--restart=no', '--label', 'spark.tinfield.revision=' + p['revision'],
        '--entrypoint', 'python3', p['image'], '/recipe/tinfield.py', '_download-worker']


def check_command(p, cache, mode='_check-worker'):
    return docker_base(p, cache) + ['--rm', '--network=none', '--entrypoint', 'python3',
                                   p['image'], '/recipe/tinfield.py', mode]


def launch_command(p, cache, kernels):
    share = p.get('decode_share', 0.20)
    if not isinstance(share, (int, float)) or not 0 <= share < 1:
        raise ValueError('decode_share must be a number in [0, 1)')
    model = snapshot(CONTAINER_HF, p)
    return ['docker', 'run', '-d', '--pull=never', '--name', p['container'], '--gpus', 'all',
            '--network=host', '--ipc=host', '--restart=unless-stopped',
            '--mount', f'type=bind,src={cache},dst={CONTAINER_HF},readonly',
            '--mount', f'type=bind,src={kernels},dst=/cache',
            '-e', 'HF_HUB_OFFLINE=1', '-e', 'TENSORFOLD_NO_UPDATE_CHECK=1',
            '-e', 'TORCH_EXTENSIONS_DIR=/cache/torch_extensions', '-e', 'TRITON_CACHE_DIR=/cache/triton',
            '-e', 'TENSORFOLD_MTP_COPY=1', '--entrypoint', 'tensorfold', p['image'],
            'serve', str(model), '--backend', 'cuda', '--tp', '1', '--host', '127.0.0.1', '--port', '8888',
            '--name', p['alias'], '--parallel', str(p['parallel']), '--context', str(p['context']),
            '--kv-dtype', 'int4', '--mtp-drafts', str(p['mtp_drafts']), '--mtp-confidence', '0.60',
            '--temperature', '0.6', '--top-p', '0.95', '--top-k', '20', '--thinking', '--decode-share', f'{share:.2f}',
            '--no-update-check']


def run(args, *, capture=False):
    return subprocess.run(args, check=True, text=True, capture_output=capture)


def inspect_container(name):
    result = subprocess.run(['docker', 'container', 'inspect', name], text=True, capture_output=True)
    if result.returncode:
        # A daemon/permission failure is not evidence that the name is free.
        error = result.stderr.lower()
        if 'no such container' not in error and 'no such object' not in error:
            raise RuntimeError(result.stderr.strip())
        return None
    return json.loads(result.stdout)[0]


def prepare(p, cache):
    run(['docker', 'image', 'inspect', p['image']], capture=True)
    cache.mkdir(parents=True, exist_ok=True)
    name = p['container'] + '-download'
    existing = inspect_container(name)
    if existing:
        labels = existing['Config'].get('Labels') or {}
        if labels.get('spark.tinfield.revision') != p['revision'] or existing['Image'] != p['image']:
            raise ValueError('download container is not this pinned profile; preserve it and inspect manually')
        if existing['State']['Running']:
            print('Download already running; use status. No second worker started.')
            return
        if progress(cache, p)['complete_files'] == len(p['files']) and existing['State']['ExitCode'] == 0:
            print('Download completed; run check before cutover.')
            return
        if existing['State'].get('OOMKilled'):
            raise ValueError('bounded download hit its memory cap; inspect logs before changing/retrying it')
        run(['docker', 'start', name])
        return
    remaining = progress(cache, p)
    if shutil.disk_usage(cache).free < remaining['total_bytes'] - remaining['downloaded_bytes'] + 10 * GIB:
        raise ValueError('not enough free disk for this pinned pack plus 10 GiB headroom')
    run(worker_command(p, cache))


def refuse_busy_port():
    with socket.socket() as probe:
        # Match the server's reusable bind: TIME_WAIT is not a live listener.
        # SO_REUSEADDR does not permit stealing a bound/listening port.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', 8888))
        except OSError as exc:
            raise ValueError('private port 8888 is occupied; launcher never stops the live backend') from exc


def start(p, cache, kernels):
    refuse_busy_port()
    if inspect_container(p['container']):
        raise ValueError('Tinfield container already exists; preserve it and inspect manually')
    run(check_command(p, cache))
    refuse_busy_port()  # the source could have restarted during the CPU preflight
    kernels.mkdir(parents=True, exist_ok=True)
    run(launch_command(p, cache, kernels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'status', 'runtime', 'check', 'command', 'start',
                                           '_download-worker', '_check-worker', '_runtime-worker'))
    parser.add_argument('--hf-cache', type=Path, default=Path.home() / '.cache/huggingface')
    parser.add_argument('--kernel-cache', type=Path, default=Path.home() / '.cache/tensorfold-qwen38')
    args = parser.parse_args()
    p, cache, kernels = profile(), args.hf_cache.resolve(), args.kernel_cache.resolve()
    if args.action == '_download-worker':
        from huggingface_hub import snapshot_download
        target = snapshot_download(p['repo'], revision=p['revision'], cache_dir=str(CONTAINER_HF / 'hub'),
                                   allow_patterns=list(p['files']), max_workers=1, token=False)
        check_snapshot(CONTAINER_HF, p)
        print(json.dumps({'download_complete': True, 'snapshot': target}), flush=True)
    elif args.action in ('_runtime-worker', '_check-worker'):
        result = runtime_check(p)
        if args.action == '_check-worker':
            target = check_snapshot(CONTAINER_HF, p)
            result.update({'snapshot': str(target), 'ngram_rows': table_header(target / 'ngram_embedding.safetensors'),
                           'capacity': capacity_report(target, p)})
        print(json.dumps(result, indent=2))
    elif args.action == 'prepare':
        prepare(p, cache)
    elif args.action == 'status':
        result = progress(cache, p)
        state = inspect_container(p['container'] + '-download')
        result['download_state'] = state['State'] if state else None
        print(json.dumps(result, indent=2))
    elif args.action in ('check', 'runtime'):
        run(check_command(p, cache, '_check-worker' if args.action == 'check' else '_runtime-worker'))
    elif args.action == 'command':
        print(shlex.join(launch_command(p, cache, kernels)))
    else:
        start(p, cache, kernels)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'Tinfield: {exc}', file=sys.stderr)
        sys.exit(1)
