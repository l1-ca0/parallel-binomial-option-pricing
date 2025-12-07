# Parallel Binomial Option Pricing Engine

**High-Performance American Option Pricing using Parallel Algorithms on CPU, GPU, and Distributed Clusters.**

📄 **[Read the Full Project Report (PDF)](docs/Parallel_BOPM_Report.pdf)**

This project implements the Binomial Options Pricing Model (BOPM) using a suite of parallel architectures: Serial, OpenMP (Multi-core), CUDA (GPU), and MPI (Distributed Cluster). It features a highly optimized Hybrid Adaptive Pipeline on the GPU that achieves 384x speedup over the serial baseline and 9.35x speedup over optimized 8-core OpenMP execution.

*All experiments were conducted on the CMU GHC machines, featuring an **8-core Intel CPU** and a single **NVIDIA GeForce RTX 2080 GPU**.*

---

## Key Features

*   **Diverse Parallel Backends:**
    *   Serial CPU: Optimized O(N) space baseline.
    *   OpenMP: Multi-core CPU implementation with Static & Dynamic Scheduling.
    *   CUDA: Suite of 10 distinct GPU kernels exploring the Latency-Throughput trade-off.
    *   MPI: Distributed Memory Master-Worker pattern for massive batch processing.
*   **Hybrid Adaptive GPU Pipeline:** Dynamically switches execution strategies based on problem size (N):
    *   Large N: Shared Memory Tiling + Thread Coarsening (Max Throughput).
    *   Medium N: Warp Per Block Kernel (Low Latency).
    *   Small N: CPU Fallback (Zero Launch Overhead).
*   **Advanced Synchronization:** Explores Global Barriers, Warp Shuffles, and Implicit Warp Synchronization.
*   **Cluster Scalability:** MPI implementation demonstrates 95% efficiency strong scaling on 8 ranks.

---

## Performance Summary

| Architecture | Strategy | Time (N=10k) | Time (N=100k) | Time (N=1M) | Speedup vs Serial (N=100k) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Serial CPU** | Baseline | 1,380 ms | 137,801 ms | | 1.0x |
| **OpenMP CPU** | 8-Thread Multi-core CPU | 33.5 ms | 3,353 ms |  | 41.1x |
| **CUDA GPU** | 1. Wavefront (Baseline) | 28.93 ms | 551.44 ms | 42,425 ms | 249.9x |
| | 2. Tiled (Global Mem) | 10.15 ms | 428.51 ms | 37,877 ms | 321.6x |
| | 3. **Shared Mem Tiling** | 26.72 ms | 405.99 ms | **28,332 ms** | **339.4x** |
| | 4. Warp Shuffle Tiling | 28.34 ms | 440.37 ms | 30,082 ms | 312.9x |
| | 5. Time Parallel | 24.71 ms | 465.39 ms | 36,146 ms | 296.1x |
| | 6. Cooperative Multi Warp | 58.24 ms | 633.17 ms | 50,216 ms | 217.6x |
| | 7. Persistent Barrier | 47.41 ms | 522.64 ms | 29,958 ms | 263.7x |
| | 8. Independent Multi Warp | 30.11 ms | 525.31 ms | 37,477 ms | 262.3x |
| | 9. **Warp Per Block** | 8.44 ms | **352.01 ms** | 32,189 ms | **391.5x** |
| | 10. **Hybrid CPU-GPU** | **8.14 ms** | 358.59 ms | 28,579 ms | **384.8x** |

**MPI for CPU/GPU Cluster**

Below results use Serial CPU as backend.
| Ranks | Time (1000 Options) | Speedup | Efficiency |
| :--- | :--- | :--- | :--- |
| 1 | 13.84 s | 1.00x | 100% |
| 2 | 6.98 s | 1.98x | 99.0% |
| 4 | 3.65 s | 3.80x | 94.9% |
| 8 | 1.82 s | 7.60x | 95.0% |

---

## Build Instructions

**Prerequisites:**
*   C++17 Compiler 
*   NVIDIA CUDA Toolkit (nvcc)
*   MPI Library (OpenMPI)
*   OpenMP

**Compile:**
```bash
make
```
This produces all executables in the `bin/` directory.

---

## Usage

### 1. Single Option Pricing
Run the individual pricing engines. Note that the financial parameters (`S0=100, K=100, T=1, r=0.05, sigma=0.2`) are used for consistent benchmarking.

*   **Serial:**
    ```bash
    ./bin/serial_binomial <N>
    ```

*   **OpenMP:**
    ```bash
    ./bin/openmp_binomial <N> [num_threads]
    ```

*   **CUDA (Hybrid):**
    ```bash
    ./bin/cuda_binomial <N> [options]
    
    # Options:
    #   --filter <name>       Run only kernels matching name 
    #   --thresh-cpu <val>    Override Hybrid CPU-GPU threshold
    ```

### 2. Benchmarking
Run the detailed benchmark suites to generate performance data:

*   **Deep CUDA Analysis (All 10 Kernels):**
    ```bash
    ./bin/benchmark_cuda
    ```
    *Output:* detailed CSV results comparing all kernel strategies across range of N.

*   **OpenMP Scaling Analysis:**
    ```bash
    ./bin/benchmark_openmp
    ```

### 3. MPI Batch Processing
To run the distributed batch pricer:

```bash
# Usage: ./bin/mpi_batch [num_options] [steps] [method]
# Example: 5000 options, 2000 steps, using OpenMP backend
mpirun -np 8 ./bin/mpi_batch 5000 2000 openmp
```

---

## Visualization
The core binomial option pricing algorithm visualised (Forward Lattice Construction -> Backward Induction):

![Binomial Option Pricing Animation](binomial_pricing.gif)
