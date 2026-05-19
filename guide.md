# GPU-EM-FDTD FRAMEWORK — MASTER GUIDE.md
# SINGLE SOURCE OF TRUTH FOR ALL AGENTS

You are working on a GPU-native electromagnetic FDTD simulation framework.

This system is designed to replace CPU-bound EM solvers (like Meep) with a GPU-first architecture optimized for:
- FDTD simulation
- MIMO imaging
- inverse scattering
- differentiable physics (future phase)

This file is the ONLY authoritative execution guide.

If any instruction elsewhere conflicts with this file → this file wins.

====================================================
🧠 SYSTEM GOAL
====================================================

Build a modular, GPU-accelerated electromagnetic simulation engine with:

- PyTorch CUDA backend
- Yee-grid FDTD solver
- scalable architecture for imaging (MIMO/SAR)
- future support for inverse problems and AI reconstruction

====================================================
🚨 EXECUTION RULES (CRITICAL)
====================================================

1. ONE PHASE AT A TIME
2. ONE MODULE AT A TIME
3. NO skipping steps
4. NO mixing implementation phases
5. NO redesigning architecture mid-way
6. ALWAYS validate physics correctness before finalizing code

====================================================
🧠 AGENT SYSTEM (MANDATORY)
====================================================

Every module must be processed using subagents:

### Physics Agent
- Validates Maxwell equations
- Ensures Yee grid correctness
- Checks numerical stability (CFL condition)

### GPU Agent
- Ensures CUDA tensor correctness
- Verifies memory layout and coalescing
- Eliminates CPU bottlenecks

### Implementation Agent
- Writes production-grade PyTorch code
- Follows module structure strictly

### Verification Agent
- Detects logical errors
- Validates boundary conditions
- Ensures field updates completeness

### Performance Agent
- Estimates VRAM usage
- Checks computational complexity
- Flags bottlenecks early

====================================================
📦 FINAL PROJECT STRUCTURE
====================================================

gpu-em-engine/
│
├── src/
│   ├── core/
│   │   ├── grid.py
│   │   ├── fields.py
│   │   ├── sources.py
│   │   ├── boundaries.py
│   │   ├── fdtd2d.py
│   │   └── materials.py
│   │
│   ├── visualization/
│   ├── utils/
│
├── examples/
├── tests/
├── benchmarks/
└── docs/

====================================================
🧭 PHASE EXECUTION PLAN (STRICT ORDER)
====================================================

----------------------------------------------------
PHASE 0 — ARCHITECTURE LOCK
----------------------------------------------------
Goal:
- finalize system design
- confirm tensor layout
- confirm simulation loop
- confirm module boundaries

Output:
- architecture validation only
- NO CODE

----------------------------------------------------
PHASE 1 — GRID SYSTEM
----------------------------------------------------
File: src/core/grid.py

Responsibilities:
- spatial discretization (dx, dy)
- CFL condition enforcement
- dt computation
- coordinate system
- CUDA-aware design

STOP after completion.

----------------------------------------------------
PHASE 2 — FIELD SYSTEM
----------------------------------------------------
File: src/core/fields.py

Responsibilities:
- Ex, Ey, Hz tensors
- GPU memory initialization
- Yee grid compatibility

STOP.

----------------------------------------------------
PHASE 3 — SOURCES
----------------------------------------------------
File: src/core/sources.py

Responsibilities:
- Gaussian pulse
- sinusoidal source
- Ricker wavelet
- injection logic

STOP.

----------------------------------------------------
PHASE 4 — BOUNDARIES
----------------------------------------------------
File: src/core/boundaries.py

Responsibilities:
- absorbing boundaries
- Mur boundary
- PML scaffold

STOP.

----------------------------------------------------
PHASE 5 — FDTD CORE ENGINE (CRITICAL)
----------------------------------------------------
File: src/core/fdtd2d.py

Responsibilities:
- Maxwell equations implementation
- Yee grid update loop
- full time-stepping solver
- CUDA tensor execution

This is the MOST IMPORTANT module.

STOP.

----------------------------------------------------
PHASE 6 — VISUALIZATION
----------------------------------------------------
File: src/visualization/plot2d.py

Responsibilities:
- Hz field visualization
- CPU transfer ONLY for plotting

STOP.

----------------------------------------------------
PHASE 7 — EXAMPLE SIMULATION
----------------------------------------------------
File: examples/basic_2d_wave.py

Responsibilities:
- run full simulation pipeline
- initialize grid + fields
- execute solver loop
- visualize output

STOP.

----------------------------------------------------
PHASE 8 — TESTING
----------------------------------------------------
File: tests/test_fdtd2d.py

Responsibilities:
- stability checks
- correctness validation
- propagation sanity tests

STOP.

----------------------------------------------------
PHASE 9 — BENCHMARKING
----------------------------------------------------
File: benchmarks/benchmark_gpu_vs_cpu.py

Responsibilities:
- GPU vs CPU timing
- performance scaling
- memory profiling

STOP.

====================================================
⚙️ GPU ENGINEERING RULES
====================================================

- ALL tensors MUST be torch.cuda tensors
- NO Python loops inside solver
- MUST use vectorized operations
- MUST respect CFL condition
- MUST minimize CPU-GPU transfers
- MUST preserve Yee grid structure

====================================================
🧪 VERIFICATION REQUIREMENTS
====================================================

Before finalizing any module:

Verification Agent MUST confirm:
- Maxwell equation correctness
- field update completeness
- boundary correctness
- GPU placement correctness
- no numerical instability risk

If any check fails → regenerate module.

====================================================
🚀 STRATEGIC OBJECTIVE
====================================================

This system is being built to eventually support:

- MIMO imaging acceleration
- SAR reconstruction
- inverse scattering problems
- differentiable EM physics
- AI-based reconstruction pipelines

====================================================
START CONDITION
====================================================

Begin ONLY with PHASE 0.

Wait for explicit instruction before moving forward.
