# Section 7: Data Pipeline and I/O Architecture

## 7.1 Simulation Data Flow

### End-to-End Pipeline

```
INPUT                          RUNTIME                         OUTPUT
─────                          ───────                         ──────
Geometry (CSG/STL/SDF)         Source waveform gen (GPU)       Detector time series
       │                              │                              │
       ▼                              ▼                              ▼
Material mapping (CPU)  ──►    Field updates (GPU)    ──►    DFT results (GPU→CPU)
       │                              │                              │
       ▼                              ▼                              ▼
Grid voxelization (CPU) ──►    Detector accumulation  ──►    Imaging recon (GPU)
       │                         (GPU)                               │
       ▼                              │                              ▼
GPU tensor upload       ──►    [checkpoint async]     ──►    HDF5/Zarr export
(one-time)                     (GPU→pinned→disk)             (async)
```

**Key Invariant:** Zero CPU↔GPU transfers during the time-stepping loop. All runtime tensors are GPU-resident. Checkpoints use async DMA and do not stall the compute pipeline.

---

## 7.2 Geometry Pipeline

### Input Formats

| Format | Use Case | Processing |
|--------|----------|------------|
| CSG primitives | Analytical shapes (sphere, box, cylinder) | Direct voxelization |
| STL mesh | CAD imports | Ray-casting voxelization |
| SDF field | Smooth boundaries | Subpixel averaging at interfaces |
| Image stack (DICOM/PNG) | Medical CT, MRI data | Direct mapping to epsilon |
| NumPy array | Programmatic geometry | Direct assignment |

### Voxelization Pipeline

```
1. CSG tree → evaluate SDF at each grid cell center
2. SDF → material index: material_idx[i,j,k] = argmin(SDF_m(x_i, y_j, z_k)) for m in materials
3. Subpixel averaging at interfaces:
   eps_eff[i,j,k] = Σ_m (volume_fraction_m × eps_m)   [for cells crossing boundaries]
4. Upload material_idx (int16) and eps_eff (float32) to GPU
```

**Tensor shapes after voxelization:**
- `material_index`: `(Nx, Ny, Nz)` int16
- `epsilon_xx, epsilon_yy, epsilon_zz`: `(Nx, Ny, Nz)` float32
- `sigma_xx, sigma_yy, sigma_zz`: `(Nx, Ny, Nz)` float32
- `mu_xx, mu_yy, mu_zz`: `(Nx, Ny, Nz)` float32 (usually uniform → scalar)

### Subpixel Averaging

At material interfaces, staircasing artifacts degrade accuracy. Subpixel smoothing:

```
For cell (i,j,k) containing interface:
  eps_eff = (1/V_cell) ∫∫∫_cell eps(x,y,z) dV
  ≈ Σ_{sub} eps(x_sub) / N_sub    (Monte Carlo or tensor-product quadrature)
```

GPU-accelerated: launch kernel over interface cells (identified by SDF sign change between neighbors), compute weighted average using 8-27 sub-samples.

---

## 7.3 Checkpoint and Restart

### Async Checkpoint Pipeline

```
GPU field tensors ──► Pinned host buffer ──► Background thread ──► HDF5 on disk
     (no stall)         (async DMA)            (Python thread)       (chunked write)
```

### Implementation

```python
class CheckpointManager:
    def __init__(self, path, interval_steps=1000):
        self.buf_A = torch.empty(field_shape, pin_memory=True)  # Double buffer
        self.buf_B = torch.empty(field_shape, pin_memory=True)
        self.stream = torch.cuda.Stream()
        self.writer_thread = threading.Thread(target=self._disk_writer, daemon=True)

    def save_async(self, fields, step):
        buf = self.buf_A if step % 2 == 0 else self.buf_B
        with torch.cuda.stream(self.stream):
            buf.copy_(fields, non_blocking=True)  # GPU→pinned (DMA)
        self.stream.synchronize()  # Wait for DMA, not compute stream
        self.write_queue.put((buf, step))  # Background thread writes to disk
```

### Checkpoint Format (HDF5)

```
checkpoint_step_005000.h5
├── fields/
│   ├── Ex  [dataset: (Nx,Ny,Nz) float32, chunked (64,64,64), lz4 compressed]
│   ├── Ey  [...]
│   ├── Ez  [...]
│   ├── Hx  [...]
│   ├── Hy  [...]
│   └── Hz  [...]
├── pml/
│   ├── psi_Exy [...]
│   └── ... (12 psi tensors)
├── metadata/
│   ├── step (int)
│   ├── time (float64)
│   ├── grid_shape (3,)
│   ├── dx, dy, dz (float64)
│   └── config (JSON string)
```

### Restart Protocol

```python
def restart_from_checkpoint(path):
    with h5py.File(path, 'r') as f:
        fields = {name: torch.from_numpy(f[f'fields/{name}'][:]).cuda()
                  for name in ['Ex','Ey','Ez','Hx','Hy','Hz']}
        step = f['metadata/step'][()]
    engine.load_state(fields, step)
    engine.run(until=T_final)  # Resumes from checkpoint step
```

---

## 7.4 Detector Data Pipeline

### Time-Domain Detectors

```
Recording:    field[probe_idx] → ring_buffer[probe_idx, t % buf_size]
Flush:        ring_buffer (GPU) → pinned_host (DMA) → output_array (CPU)
Flush trigger: buffer full OR simulation end
```

**Tensor shapes:**
- Point probes: `(N_probes, buf_size)` float32 — ring buffer on GPU
- Surface probes: `(N_surface_cells, buf_size)` float32 — larger, flush more frequently

### Frequency-Domain (DFT) Detectors

Running DFT avoids storing full time series:

```python
# On GPU, every timestep:
for m in range(N_freqs):
    phase = 2 * pi * freqs[m] * t * dt
    dft_real[m, :] += field[monitor_cells] * cos(phase)
    dft_imag[m, :] += field[monitor_cells] * sin(phase)
```

**Tensor shape:** `(N_freqs, N_monitor_cells)` complex64 (stored as 2× float32)

At simulation end: `S_param[freq] = dft_complex / N_steps` (normalized)

### Near-to-Far Field Transform

```
Surface currents: J_s[surface_cells, N_freqs], M_s[surface_cells, N_freqs]
Far-field: E_ff(theta, phi, f) = ∫∫ (J_s × r̂ + M_s) × Green's × e^{jkr} dS
```

Computed as GPU matrix-vector product: `E_ff = G @ J_s` where G is the Green's function matrix `(N_angles, N_surface_cells)` complex64.

---

## 7.5 Imaging Reconstruction Pipeline

### Multi-Simulation Orchestration

```
for tx_idx in range(N_tx):           # Or batched across GPUs
    configure_source(tx_positions[tx_idx], waveform)
    sim.run(until=T_max)
    raw_data[tx_idx, :, :] = detectors.extract()  # (N_rx, N_t)
    sim.reset_fields()               # Zero fields, keep geometry
```

**Batching strategy:** If VRAM allows, run M simulations in parallel (M independent grids on same GPU) or distribute across GPUs (1 TX per GPU).

### Reconstruction Kernels

**Delay-and-Sum (DAS) Backprojection:**

```
image[x,y,z] = Σ_{tx} Σ_{rx} signal[tx, rx, τ(tx,rx,x,y,z)]
where τ = (|pos_tx - r| + |pos_rx - r|) / c
```

GPU implementation: one thread per image voxel, loops over TX/RX pairs, interpolates signal at computed delay.

**Tensor shapes:**
- `raw_signals`: `(N_tx, N_rx, N_t)` float32 — input
- `delays`: `(N_tx, N_rx, Nx_img, Ny_img, Nz_img)` float32 — precomputed or computed on-the-fly
- `image`: `(Nx_img, Ny_img, Nz_img)` float32 — output

### Pipeline Timing (32 TX, 32 RX, 256³ image, 4096 time samples)

| Stage | GPU Time | Notes |
|-------|----------|-------|
| Forward simulations (32×) | 32 × 5s = 160s | Parallelizable across GPUs |
| Delay computation | 0.5s | Precomputed geometry |
| Backprojection | 2.1s | Memory-bound, all on GPU |
| Post-processing (filter) | 0.1s | |
| **Total (4 GPUs)** | **~42s** | 4× speedup on forward sims |

---

## 7.6 Streaming and Real-Time Monitoring

### Field Snapshot Streaming

```python
class LiveStreamer:
    def __init__(self, decimation=100, slice_axis='z', slice_idx=None):
        self.zmq_pub = zmq.Context().socket(zmq.PUB)
        self.zmq_pub.bind("tcp://*:5555")

    def on_step(self, fields, step):
        if step % self.decimation == 0:
            slice_data = fields.Ex[:, :, self.slice_idx].cpu().numpy()
            self.zmq_pub.send_pyobj({'step': step, 'field': slice_data})
```

### Decimation Strategy

- **Spatial:** Send every Nth cell (2× decimation = 8× data reduction in 3D)
- **Temporal:** Send every Mth step (M=100 typical for 10k+ step sims)
- **Component:** Send only requested field component (6× reduction)
- **Combined:** 100× temporal × 8× spatial = 800× reduction → 8 KB/frame for 512³ grid

### Output Format Summary

| Consumer | Format | Transport |
|----------|--------|-----------|
| Post-processing scripts | HDF5/Zarr | Filesystem |
| Jupyter notebooks | NumPy arrays (in-memory) | Direct return |
| Live visualization | Decimated slices | ZMQ PUB/SUB |
| Web dashboard | PNG/JPEG frames | WebSocket |
| ML pipelines | PyTorch tensors | Direct (same GPU) |
