"""Measure segmented GPU KV writes with the deployment's NIXL UCX backend."""
import itertools
import time


def destinations(source_tp, destination_tp, rank):
    if rank >= source_tp:
        return []
    if destination_tp >= source_tp:
        replicas = destination_tp // source_tp
        return [8 + rank * replicas + offset for offset in range(replicas)]
    replicas = source_tp // destination_tp
    return [8 + rank // replicas] if rank % replicas == 0 else []


def measure(torch, dist, rank, local_rank):
    from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

    agent = nixl_agent(f'calibration-{rank}', nixl_agent_config(backends=[], num_threads=1, sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT))
    agent.create_backend('UCX', {'num_threads': '1'})
    # 94 layers x K/V, one KV head per TP rank, BF16, head_dim=128.
    # Gaps prevent descriptor merging from turning this into one contiguous copy.
    stride = 4200 * 256 + 4096
    buffer = torch.zeros((188, stride), dtype=torch.uint8, device='cuda')
    registered = agent.register_memory([(buffer.data_ptr(), buffer.numel(), local_rank, '')], 'VRAM')
    peers = [None] * 16
    dist.all_gather_object(peers, {'metadata': agent.get_agent_metadata(), 'ptr': buffer.data_ptr(), 'device': local_rank, 'name': f'calibration-{rank}'})
    if rank < 8:
        for peer in peers[8:]:
            agent.add_remote_agent(peer['metadata'])
    measurements = []
    for senders, receivers in itertools.product((4, 8), repeat=2):
        for tokens in (128, 1024, 4000, 4200):
            length = tokens * 256
            samples = []
            for iteration in range(8):
                buffer.fill_(rank + 1 if rank < senders else 0)
                torch.cuda.synchronize()
                dist.barrier()
                elapsed_ms = 0.0
                peers_to_write = destinations(senders, receivers, rank)
                if peers_to_write:
                    local = agent.get_xfer_descs([(buffer.data_ptr() + i * stride, length, local_rank) for i in range(188)], 'VRAM')
                    handles = []
                    started = time.perf_counter()
                    for destination in peers_to_write:
                        peer = peers[destination]
                        remote = agent.get_xfer_descs([(peer['ptr'] + i * stride, length, peer['device']) for i in range(188)], 'VRAM')
                        handle = agent.initialize_xfer('WRITE', local, remote, peer['name'])
                        handles.append(handle)
                        if agent.transfer(handle) == 'ERR':
                            raise RuntimeError('NIXL posting failed')
                    pending = list(handles)
                    while pending:
                        states = [(h, agent.check_xfer_state(h)) for h in pending]
                        if any(state == 'ERR' for _, state in states):
                            raise RuntimeError('NIXL completion failed')
                        pending = [h for h, state in states if state != 'DONE']
                        if time.perf_counter() - started > 60:
                            raise TimeoutError('NIXL write exceeded 60 seconds')
                        if pending:
                            time.sleep(0)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    for handle in handles:
                        agent.release_xfer_handle(handle)
                duration = torch.tensor(elapsed_ms, dtype=torch.float64, device='cuda')
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                if 8 <= rank < 8 + receivers:
                    expected = (rank - 8) * senders // receivers + 1
                    if not torch.all(buffer[:, :length] == expected).item():
                        raise ValueError('NIXL destination bytes do not match source')
                dist.barrier()
                if iteration >= 3:
                    samples.append(duration.item())
            measurements.append({'source_tp': senders, 'destination_tp': receivers, 'prompt_tokens': tokens, 'descriptors_per_destination': 188, 'bytes_per_destination': 188 * length, 'physical_bytes': receivers * 188 * length, 'unique_kv_bytes': 4 * 188 * length, 'latency_ms_samples': samples, 'warmup_iterations': 3, 'integrity_verified': True, 'timing_boundary': 'remote descriptor creation, initialize_xfer, post, completion; registration and local descriptors excluded; slowest sender'})
    dist.barrier()
    agent.deregister_memory(registered)
    return measurements
