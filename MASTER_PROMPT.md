You are not acting as a casual coding assistant.

You are acting as:
- Senior Computational Electromagnetics Researcher
- CUDA Performance Engineer
- PyTorch Kernel Architect
- Scientific Computing Framework Designer
- Technical Documentation Lead
- Open Source Infrastructure Architect

Your task is to design and architect a COMPLETE next-generation GPU-native electromagnetic simulation framework intended to become a modern alternative to Meep.

This framework must:
- Be GPU-first
- Python-first
- Optimized for FDTD
- Optimized for MIMO imaging
- Optimized for inverse scattering
- Optimized for differentiable physics
- Support future AI-assisted EM reconstruction
- Support large-scale CUDA acceleration
- Eventually exceed Meep performance for imaging applications

You MUST think deeply before generating anything.

DO NOT produce shallow toy examples.

I want a PROFESSIONAL SYSTEM DESIGN DOCUMENTATION similar to what a real startup or advanced research lab would create before implementation.

The output must be EXTREMELY detailed and TECHNICAL.

====================================================
PROJECT OVERVIEW
====================================================

We are building:

A GPU-native differentiable electromagnetic simulation engine using FDTD methods.

Primary goals:
- Massive GPU acceleration
- Multi-GPU scalability
- MIMO imaging optimization
- Microwave imaging
- SAR reconstruction
- Through-wall imaging
- Near-field imaging
- Differentiable simulation
- AI integration
- Inverse scattering
- Scientific reproducibility
- Modern Python API

Core inspiration:
- Meep
- openEMS
- gprMax

But our architecture must be MODERN and GPU-CENTRIC.

====================================================
IMPORTANT DESIGN PHILOSOPHY
====================================================

The framework MUST:
- Prioritize GPU execution
- Avoid CPU bottlenecks
- Use tensor operations
- Be modular
- Be scalable
- Be research-friendly
- Be startup-grade
- Be open-source quality
- Be production-ready
- Be future-proof for AI integration

====================================================
TECH STACK REQUIREMENTS
====================================================

Primary language:
- Python

GPU backend:
- PyTorch CUDA
- CuPy where beneficial
- Triton kernels later
- Optional custom CUDA kernels

Scientific backend:
- NumPy
- SciPy

Visualization:
- PyVista
- Matplotlib

Future support:
- JAX backend possibility
- ROCm possibility
- Vulkan compute possibility

====================================================
YOUR RESPONSIBILITIES
====================================================

You MUST generate FULL TECHNICAL DOCUMENTATION including:

1. SYSTEM ARCHITECTURE
2. CORE PHYSICS ENGINE
3. GPU EXECUTION MODEL
4. MEMORY MANAGEMENT STRATEGY
5. CUDA OPTIMIZATION STRATEGY
6. DOMAIN DECOMPOSITION
7. MULTI-GPU STRATEGY
8. FDTD IMPLEMENTATION DETAILS
9. PML IMPLEMENTATION
10. DIFFERENTIABLE PHYSICS DESIGN
11. ADJOINT METHOD ARCHITECTURE
12. MIMO IMAGING PIPELINE
13. SAR IMAGING PIPELINE
14. INVERSE SCATTERING PIPELINE
15. AI INTEGRATION ARCHITECTURE
16. API DESIGN
17. FILE STRUCTURE
18. TESTING FRAMEWORK
19. BENCHMARKING FRAMEWORK
20. PERFORMANCE PROFILING STRATEGY
21. FUTURE RESEARCH DIRECTIONS
22. STARTUP/COMMERCIALIZATION POTENTIAL
23. OPEN SOURCE STRATEGY
24. ROADMAP
25. RISK ANALYSIS
26. COMPARISON AGAINST MEEP
27. COMPUTATIONAL COMPLEXITY ANALYSIS
28. GPU MEMORY ESTIMATION
29. PARALLELIZATION STRATEGY
30. FUTURE CUDA KERNEL PLANS

====================================================
VERY IMPORTANT
====================================================

You MUST produce:
- Real equations
- Mathematical derivations
- CUDA considerations
- Tensor memory layouts
- Data flow diagrams
- Module interaction diagrams
- Directory structures
- Pseudocode
- API examples
- GPU optimization concepts
- Computational complexity analysis
- Numerical stability analysis
- CFL condition analysis
- Precision considerations
- Mixed precision strategies
- Memory bandwidth analysis
- GPU occupancy considerations

====================================================
FDTD REQUIREMENTS
====================================================

You MUST explain:
- Yee grid implementation
- Ex Ey Ez storage
- Hx Hy Hz storage
- Staggered grid
- Time stepping
- Curl operators
- Boundary handling
- PML implementation
- Numerical dispersion
- Stability conditions
- GPU tensor update flow

====================================================
GPU REQUIREMENTS
====================================================

You MUST deeply explain:
- Why GPUs are ideal for FDTD
- Tensorized field updates
- CUDA thread mapping
- Shared memory opportunities
- Coalesced memory access
- Kernel fusion
- Asynchronous execution
- Stream parallelism
- Multi-GPU synchronization
- Memory bottlenecks
- VRAM optimization
- Sparse updates
- Mixed precision

====================================================
MIMO IMAGING REQUIREMENTS
====================================================

You MUST deeply explain:
- Multiple transmitter handling
- Receiver array handling
- FMCW processing
- Synthetic aperture processing
- Backprojection
- Delay-and-sum
- Beamforming
- Time reversal
- Reconstruction acceleration
- Batched simulations on GPU

====================================================
DIFFERENTIABLE PHYSICS REQUIREMENTS
====================================================

You MUST deeply explain:
- Autograd integration
- Differentiable material tensors
- Gradient propagation
- Adjoint methods
- Physics-informed optimization
- Inverse reconstruction
- Neural parameter estimation
- Optimization loops
- Learned EM reconstruction

====================================================
OUTPUT REQUIREMENTS
====================================================

You MUST generate:

1. FULL MASTER DOCUMENTATION
2. COMPLETE SYSTEM DESIGN
3. IMPLEMENTATION ROADMAP
4. DETAILED DEVELOPMENT PHASES
5. DETAILED MODULE-BY-MODULE EXPLANATION
6. FUTURE RESEARCH IDEAS
7. POTENTIAL PUBLICATION DIRECTIONS
8. STARTUP VISION
9. SCALABILITY PLAN
10. LONG-TERM ECOSYSTEM PLAN

====================================================
CRITICAL INSTRUCTION
====================================================

Do NOT give generic advice.

Do NOT give motivational speech.

Do NOT summarize superficially.

Do NOT generate small toy architecture.

Think like:
- NVIDIA CUDA architect
- Meep core developer
- PyTorch systems engineer
- Scientific computing researcher
- Electromagnetic imaging expert
- HPC systems architect

The documentation should feel like:
- A DARPA research proposal
- A startup technical architecture document
- A PhD-level systems design paper
- A next-generation computational physics framework specification

====================================================
FINAL REQUIREMENT
====================================================

Generate the documentation in professionally organized sections with:
- Titles
- Subsections
- Equations
- Technical depth
- Implementation details
- Engineering tradeoffs
- Research-grade explanations

The result should become the MASTER BLUEPRINT for building the framework.
