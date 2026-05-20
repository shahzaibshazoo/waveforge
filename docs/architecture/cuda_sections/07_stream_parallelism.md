# Section 7: Advanced Stream Parallelism and Pipelining

## 7.1 Concurrent Kernel Execution

A single GPU can execute multiple kernels simultaneously if:
1. They are on different CUDA streams
2. Combined SM usage doesn't exceed total SMs
3. Neither kernel individually saturates all SMs

### FDTD Concurrency Analysis

```
E-field update (512³): 262,144 blocks × 512 threads
A100 SMs: 108, each runs 4 blocks = 432 active blocks
Waves: 262,144 / 432 = 607 waves → SATURATES GPU

Source injection: 1,000 threads (sparse)
→ 2 blocks → uses 2/108 SMs = 1.8%
```

**Reality:** The E-field kernel saturates all SMs. A concurrent source injection kernel can only execute during the "tail effect" — the last few waves when some SMs finish early and become idle.

**Practical concurrency:** Between the large field-update kernels. Small kernels (source, detect) overlap with the startup/tail of large kernels.

## 7.2 Pipeline Parallelism for Multi-Simulation (MIMO)

MIMO imaging requires N_TX independent FDTD simulations (same geometry, different sources). If a single simulation doesn't fill VRAM, multiple can run concurrently.

### VRAM Budget for Concurrent Simulations

```
Single sim (256³ FP32): ~900 MB (fields + materials + PML)
A100 80 GB VRAM: floor(80 GB / 0.9 GB) = 88 concurrent sims (memory-limited)
But SM bandwidth: 88 sims × 0.9 GB working set = 79 GB active → L2 cache thrashing

Practical limit: 4-8 concurrent sims (balance memory + cache efficiency)
```

### Stream-Per-Simulation Pattern

```python
class MultiSimRunner:
    def __init__(self, n_concurrent, grid):
        self.sims = [SimState(grid, device='cuda') for _ in range(n_concurrent)]
        self.streams = [torch.cuda.Stream() for _ in range(n_concurrent)]
    
    def run_batch(self, tx_configs, n_steps):
        for step in range(n_steps):
            for i, (sim, stream) in enumerate(zip(self.sims, self.streams)):
                with torch.cuda.stream(stream):
                    sim.inject_source(tx_configs[i], step)
                    sim.update_H()
                    sim.update_E()
                    sim.apply_pml()
                    sim.record_detectors(step)
        
        # Synchronize all
        torch.cuda.synchronize()
        return [sim.get_results() for sim in self.sims]
```

### Throughput Gain (Small Grids)

| Grid | Single Sim SM Usage | Concurrent Sims | Throughput Multiplier |
|------|-------------------|-----------------|---------------------|
| 64³ | 8% | 8 | 5.2× |
| 128³ | 15% | 6 | 4.1× |
| 256³ | 60% | 2 | 1.6× |
| 512³ | 100% | 1 | 1.0× (saturated) |

Concurrency helps most for small grids that under-utilize the GPU individually.

## 7.3 Batched Simulation (Single Kernel, Multiple Grids)

Alternative to multi-stream: stack simulations along a batch dimension.

```python
# Batched field tensors: all sims in one allocation
# Shape: (N_batch, Nx, Ny, Nz) per component
Ex_batch = torch.zeros(N_batch, Nx, Ny, Nz, device='cuda')
Ey_batch = torch.zeros(N_batch, Nx, Ny, Nz, device='cuda')
# ... 

def update_H_batched(E_batch, H_batch, coeffs):
    """Single kernel updates all simulations simultaneously."""
    # Curl operates on last 3 dims; batch dim adds more parallel work
    dEz_dy = E_batch.Ez[:, :, 1:, :] - E_batch.Ez[:, :, :-1, :]
    dEy_dz = E_batch.Ey[:, :, :, 1:] - E_batch.Ey[:, :, :, :-1]
    H_batch.Hx[:, :, :-1, :-1] -= Db * (dEz_dy[:, :, :, :-1] - dEy_dz[:, :, :-1, :])
```

**Advantage:** Single kernel launch for all sims. Batch dimension adds threads without increasing per-cell memory traffic. Better SM utilization for small grids.

**Disadvantage:** All sims must have identical geometry/grid (only source differs for MIMO — this is satisfied).

### Batched Performance Model

```
Single sim (128³): 2.1M cells → 4096 blocks → 4096/432 = 9 waves (SM under-utilized)
Batched ×8: 16.8M cells → 32768 blocks → 32768/432 = 76 waves (better utilization)

Throughput: batch=8 achieves ~6× throughput of sequential (not 8× due to cache pressure)
```

## 7.4 Priority Streams

CUDA stream priority determines warp scheduling preference when multiple streams have ready work.

```python
# High priority: warps from this stream get scheduled first
compute = torch.cuda.Stream(priority=-1)  # -1 = highest

# Low priority: warps yield to high-priority streams
io = torch.cuda.Stream(priority=0)  # 0 = default (lowest)
```

**Effect on FDTD:**
- Compute stream (field updates): always high priority → maximum SM allocation
- I/O stream (checkpoint copy): low priority → only uses SMs when compute has no ready warps
- Result: checkpoint DMA runs in "cracks" between compute waves, near-zero impact on step time

### Priority Levels

```
CUDA defines: cudaStreamPriorityRange(&lowest, &highest)
Typically: lowest=0, highest=-1 (only 2 levels on most GPUs)
Some GPUs support -5 to 0 (6 levels) but benefit is marginal beyond 2.
```

## 7.5 Pipelined Adjoint Computation

The adjoint method requires:
1. Forward pass → save checkpoints
2. Backward (adjoint) pass → recompute from checkpoints + run adjoint simultaneously

### Two-Stream Adjoint Pipeline

```python
recompute_stream = torch.cuda.Stream(priority=-1)
adjoint_stream = torch.cuda.Stream(priority=-1)
recompute_done = torch.cuda.Event()

def adjoint_pass(checkpoints, n_steps, checkpoint_interval):
    for segment in reversed(range(n_segments)):
        # Stream A: recompute forward states from checkpoint
        with torch.cuda.stream(recompute_stream):
            states = recompute_segment(checkpoints[segment], checkpoint_interval)
        recompute_stream.record_event(recompute_done)
        
        # Stream B: run adjoint using previously recomputed states
        with torch.cuda.stream(adjoint_stream):
            adjoint_stream.wait_event(recompute_done)
            for state in reversed(states):
                adjoint_step(state)
                accumulate_gradient(state)
```

**Overlap opportunity:** While adjoint processes segment N, recompute can prepare segment N-1. With 2 streams: recompute and adjoint of DIFFERENT segments overlap → ~30% speedup over sequential.

## 7.6 Synchronization Best Practices

### DO

```python
# Fine-grained event sync (cheap, precise)
event = torch.cuda.Event()
stream_a.record_event(event)
stream_b.wait_event(event)
```

### DON'T

```python
# Device-wide sync (expensive, unnecessary)
torch.cuda.synchronize()  # Blocks CPU until ALL GPU work completes

# Stream sync when event suffices
stream.synchronize()  # Blocks CPU; use event-based cross-stream sync instead
```

### When CPU Sync Is Required

- Reading a GPU tensor value on CPU (e.g., checking for NaN)
- Printing/logging field statistics
- End of simulation (before extracting results)
- Checkpoint disk write (need data in host buffer)

**Rule:** At most 1 synchronization per N_checkpoint_interval steps (typically every 1000 steps).

## 7.7 Complete Stream Configuration

```python
class FDTDStreamConfig:
    def __init__(self):
        self.compute = torch.cuda.Stream(priority=-1)
        self.source = torch.cuda.Stream(priority=-1)
        self.detect = torch.cuda.Stream(priority=0)
        self.io = torch.cuda.Stream(priority=0)
        self.viz = torch.cuda.Stream(priority=0)
        
        self.events = {
            'src_done': torch.cuda.Event(enable_timing=False),
            'h_done': torch.cuda.Event(enable_timing=False),
            'e_done': torch.cuda.Event(enable_timing=False),
        }
        
        self.graph = None  # Populated after capture
    
    def capture_graph(self, step_fn):
        """Capture steady-state timestep as CUDA graph."""
        with torch.cuda.stream(self.compute):
            for _ in range(3):  # Warm up
                step_fn()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.compute):
                step_fn()
    
    def replay_step(self):
        """Execute one timestep with near-zero CPU overhead."""
        self.graph.replay()
```
