"""Post-official 15-second pass: drain, then two PP4 microbatches per worker."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path


def request(url, body, deadline):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=max(.01, deadline-time.monotonic())) as response:
        return response.read().decode()


def summarize(directory):
    import numpy as np
    seen = set(); samples = []; boundaries = {}
    for worker in range(3):
        for rank, layers in enumerate((range(24),range(24,48),range(48,72),range(72,94))):
            uuid = None
            for sample in range(2):
                with np.load(directory/f'prefill-{worker}'/f'pp-{rank}-sample-{sample}.npz') as data:
                    meta = json.loads(str(data['metadata'])); ids = data['ids']
                    assert meta['schema'] == 'prefill-routing-pp4/v1' and meta['worker'] == f'prefill-{worker}' and meta['pp_rank'] == rank and meta['sample'] == sample
                    assert meta['global_layer_ids'] == list(layers) and meta['experts'] == 128 and meta['topk'] == 8
                    assert ids.dtype == np.int32 and ids.shape == (len(layers),meta['token_count'],8) and 0 <= ids.min() <= ids.max() < 128
                    assert 1 <= len(meta['request_ids']) == len(meta['request_lengths']) == len(meta['request_prefix_lengths']) <= 2 and sum(meta['request_lengths']) == meta['token_count']
                    assert all(r.startswith('r20-routing-') for r in meta['request_ids']) and not any(meta['request_prefix_lengths'])
                    assert uuid is None or uuid == meta['gpu_uuid']
                    boundary = (meta['request_ids'],meta['request_lengths'],meta['request_prefix_lengths'])
                    assert boundaries.setdefault((worker,sample),boundary) == boundary, 'PP request boundaries differ'
                    uuid = meta['gpu_uuid']
                    samples.append(meta)
            assert uuid not in seen
            seen.add(uuid)
    assert len(seen) == 12
    return samples


def run(root, wall_deadline):
    started = time.monotonic(); deadline = started+max(0,wall_deadline-time.time())
    directory = root/'results/prefill-routing-capture'
    directory.mkdir(parents=True,exist_ok=True)
    status = {'schema':'prefill-routing-pp4-pass/v1', 'complete':False}
    def save():
        status['elapsed_seconds'] = time.monotonic()-started
        (directory/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    try:
        p = json.loads((root/'results/prefill-fa3-prepack/selection.json').read_text())
        assert p['adopted'] is True and p['selected'] == 'P', 'P required'
        endpoints = [line.split() for line in (root/'source/worker-endpoints.tsv').read_text().splitlines()]
        # flush_cache waits for is_fully_idle, including PP microbatches, grammar,
        # bootstrap, transfer and in-flight queues, on every scheduler rank.
        # This is after official-aiperf returned and score.json was preserved.
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            drained = list(pool.map(lambda e:request(f'http://{e[1]}:{e[2]}/flush_cache?timeout=3',{},deadline),endpoints))
        assert len(drained) == 4 and all('Cache flushed.' in r for r in drained), 'outstanding requests'
        # Tokenizer flush returns its first DP result. Check every DP snapshot
        # as well, so an idle Decode rank cannot hide an unfinished peer.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            snapshots = list(pool.map(lambda e:json.loads(request(f'http://{e[1]}:{e[2]}/v1/loads?include=all',None,deadline)),endpoints))
        for endpoint,snapshot in zip(endpoints,snapshots,strict=True):
            assert snapshot['loads'], 'missing load snapshots'
            keys = ('prefill_bootstrap_queue_reqs','prefill_inflight_queue_reqs') if endpoint[0].startswith('prefill-') else ('decode_prealloc_queue_reqs','decode_transfer_queue_reqs','decode_retracted_queue_reqs')
            for load in snapshot['loads']:
                assert load['num_running_reqs'] == load['num_waiting_reqs'] == 0, 'outstanding DP requests'
                assert all(load['disaggregation'][key] == 0 for key in keys), 'outstanding transfers'
        status['drained'] = True
        decode = next(e for e in endpoints if e[0] == 'decode-0')
        rng = random.Random(201101)
        inputs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            jobs = []
            for e in endpoints:
                if not e[0].startswith('prefill-'): continue
                worker = int(e[0].split('-')[1])
                ids = [[rng.randrange(100,10000) for _ in range(3994)] for _ in range(4)]
                body = {'input_ids':ids, 'rid':[f'r20-routing-{root.name}-{worker}-{i}' for i in range(4)], 'sampling_params':{'temperature':0,'max_new_tokens':1,'ignore_eos':True}, 'stream':False, 'bootstrap_host':[e[1]]*4, 'bootstrap_port':[9000+worker]*4, 'bootstrap_room':[time.time_ns()%((1<<50)-1)+worker*4+i for i in range(4)]}
                inputs.append({'worker':e[0], 'payload':body})
                for target in (e,decode):
                    jobs.append(pool.submit(request,f'http://{target[1]}:{target[2]}/generate',body,deadline))
            (directory/'inputs.json').write_text(json.dumps(inputs)+'\n')
            for job in jobs: json.loads(job.result())
        # No filesystem polling: each prefill response follows its captured forward.
        samples = summarize(directory)
        assert {m['gpu_uuid'] for m in samples} == {r['uuid'] for r in p['gpus']}, 'P/capture UUID set'
        status.update(complete=True,samples=samples,input_sha256=hashlib.sha256((directory/'inputs.json').read_bytes()).hexdigest())
    except Exception as exc:  # noqa: BLE001 - retain official score, diagnostics are best effort
        status['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--deadline',type=float,required=True)
    a = p.parse_args()
    run(a.root,a.deadline)
