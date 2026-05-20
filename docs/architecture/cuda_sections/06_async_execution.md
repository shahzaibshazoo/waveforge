# Section 6: Asynchronous Execution and Streams

## 6.1 CUDA Streams Fundamentals

A CUDA stream is an in-order queue of GPU operations. Operations within one stream execute sequentially; operations across different streams may execute concurrently.

```python
# PyTorch stream API
compute_stream = torch.cuda.Stream(priority=-1)  # High priority
io_stream = torch.cuda.Stream(priority=0)        # Low priority

with torch.cuda.stream(compute_stream):
    result = kernel_A(data)  # Runs on compute_stream

with torch.cuda.stream(io_stream):
    buffer.copy_(result, non_blocking=True)  # Concurrent with compute
```

**Default stream behavior:** All PyTorch operations without explicit stream assignment go to stream 0. Stream 0 implicitly synchronizes with all other streams — **avoid using stream 0 in performance-critical code.**

## 6.2 Stream Architecture for FDTD

| Stream | Priority | Purpose | Operations |
|--------|----------|---------|-----------|
| compute | HIGH (-1) | Field updates (critical path) | H-update, E-update, PML |
| source | HIGH (-1) | Source injection | Sparse field writes at source cells |
| detect | LOW (0) | Detector recording | Field gather + DFT accumulation |
| io | LOW (0) | Checkpoint/export | GPU→pinned host async DMA |
| viz | LOW (0) | Live monitoring | Decimated field copy for display |

### Why Source Gets High Priority

Source injection modifies field values that the update kernel will read. If source injection is delayed (low priority), the update kernel may read stale values. High priority ensures source executes before or concurrently with independent update regions.

## 6.3 Synchronization with Events

Events are lightweight markers on streams. One stream can wait for another stream's event without blocking the CPU.

```python
class StreamManager:
    def __init__(self):
        self.compute = torch.cuda.Stream(priority=-1)
        self.source = torch.cuda.Stream(priority=-1)
        self.detect = torch.cuda.Stream(priority=0)
        self.io = torch.cuda.Stream(priority=0)
        
        self.h_done = torch.cuda.Event()
        self.e_done = torch.cuda.Event()
        self.src_done = torch.cuda.Event()
    
    def timestep(self, fields, sources, detectors, t):
        # Source injection (can start immediately)
        with torch.cuda.stream(self.source):
            sources.inject(fields.E, t)
        self.source.record_event(self.src_done)
        
        # H-update (waits for source to finish modifying E)
        with torch.cuda.stream(self.compute):
            self.compute.wait_event(self.src_done)
            update_H(fields)
        self.compute.record_event(self.h_done)
        
        # E-update (sequential after H on same stream)
        with torch.cuda.stream(self.compute):
            update_E(fields)
            apply_pml(fields)
        self.compute.record_event(self.e_done)
        
        # Detector (waits for E-update, runs concurrently with next step's source)
        with torch.cuda.stream(self.detect):
            self.detect.wait_event(self.e_done)
            detectors.record(fields, t)
```

### Synchronization Costs

| Operation | Latency | When to Use |
|-----------|---------|-------------|
| `event.record(stream)` | ~0.5 μs | Every stream transition |
| `stream.wait_event(event)` | ~1 μs | Cross-stream dependency |
| `torch.cuda.synchronize()` | 3-10 μs | **NEVER in hot loop** |
| `event.synchronize()` | 3-10 μs | Only for CPU-side result access |

## 6.4 Overlap Timeline

### Single Timestep (512³ grid, A100)

```
Time:  0μs      50μs     100μs    200μs    300μs    360μs    400μs
       │         │         │         │         │         │         │
Compute: [═══════ H_update (180μs) ═══════][═══════ E_update+PML (180μs) ═════]
Source:  [src(5μs)]                         [src(5μs)]
Detect:                                                  [DFT(30μs)]
I/O:                                                          [checkpoint(100μs)─►
       │         │         │         │         │         │         │
```

**Critical path: 360 μs** (H_update + E_update, sequential on compute stream)

**Overlapped (free):**
- Source injection: 5 μs (hidden behind H_update)
- Detector DFT: 30 μs (runs after E_update, overlaps with next step's source)
- Checkpoint I/O: 100 μs (fully async, no impact on compute)

### Without Streams (Naive Sequential)

```
H_update(180) → src(5) → E_update(180) → src(5) → PML(included) → detect(30) → checkpoint(100)
Total: 500 μs/step
```

**Speedup from streams: 500/360 = 1.39×** (39% improvement)

## 6.5 CUDA Graphs

### Concept

CUDA Graphs record a sequence of kernel launches into a static executable graph. Replaying the graph incurs near-zero CPU overhead — the GPU executes the entire sequence without CPU interaction.

### FDTD as CUDA Graph

FDTD is ideal for graph capture because every timestep is IDENTICAL:
- Same kernels
- Same tensor addresses (pre-allocated, never reallocated)
- Same launch configurations
- No dynamic control flow

```python
# Capture phase (once at initialization)
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    stream.wait_stream(torch.cuda.current_stream())
    
    # Warm up
    for _ in range(3):
        one_timestep(fields, t=0)
    
    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        one_timestep(fields, t=0)  # t is read from a tensor, not Python int

# Replay phase (every timestep)
for t in range(N_steps):
    t_tensor.fill_(t)  # Update time parameter in-place
    graph.replay()      # Single CPU call, ~3 μs
```

### Graph Benefits

| Metric | Without Graph | With Graph | Improvement |
|--------|--------------|-----------|-------------|
| CPU overhead per step | 50-100 μs | 3 μs | 17-33× |
| Kernel launch overhead | 5 μs × 10 kernels = 50 μs | 0 (pre-recorded) | Eliminated |
| GPU idle time (waiting for CPU) | 10-50 μs | 0 | Eliminated |
| Effective step time (512³) | 410 μs | 363 μs | 13% faster |

### Graph Limitations

- All tensor addresses must be static (no `torch.empty` in the loop)
- No Python control flow during replay (if/else based on field values)
- No dynamic shapes (grid size fixed after capture)
- Cannot capture operations that call back to CPU (print, assert, logging)

All satisfied by steady-state FDTD time-stepping.

## 6.6 Async Memory Operations

### GPU → Host (Checkpoint)

```python
# Pinned host buffer (allocated once at init)
host_buffer = torch.empty(field_shape, pin_memory=True)

# Async copy (non-blocking)
with torch.cuda.stream(io_stream):
    host_buffer.copy_(gpu_fields, non_blocking=True)

# Later (on CPU thread, after io_stream completes):
io_stream.synchronize()  # Wait only on I/O stream, not compute
save_to_disk(host_buffer)
```

### Double Buffering

```python
buffers = [torch.empty(shape, pin_memory=True) for _ in range(2)]
write_thread = None

def checkpoint_async(fields, step):
    buf_idx = step % 2
    
    # Wait for previous disk write to finish (if any)
    if write_thread and write_thread.is_alive():
        write_thread.join()
    
    # Async GPU → pinned host
    with torch.cuda.stream(io_stream):
        buffers[buf_idx].copy_(fields, non_blocking=True)
    io_stream.synchronize()
    
    # Background disk write
    write_thread = threading.Thread(target=save_hdf5, args=(buffers[buf_idx], step))
    write_thread.start()
```

## 6.7 Performance Summary

| Optimization | Time Saved per Step (512³) | Implementation Complexity |
|-------------|--------------------------|--------------------------|
| Multi-stream overlap | 140 μs (39%) | Medium (event management) |
| CUDA Graph replay | 47 μs (13%) | Low (capture/replay pattern) |
| Async I/O (non-blocking) | Eliminates checkpoint stall | Low |
| Priority streams | 5-10 μs (scheduling) | Trivial |
| **Combined** | **~190 μs (53%)** | **Medium** |

**Final optimized step time: ~360 μs** for 512³ on A100 (2,778 steps/second).
