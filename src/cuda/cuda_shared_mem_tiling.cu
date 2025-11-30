#include "cuda_kernels.cuh"
#include "cuda_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/**
 * @file cuda_shared_mem_tiling.cu
 * @brief Tiled Implementation with Thread Coarsening 
 *
 * Optimization Strategy: Thread Coarsening (Register/ILP Optimization)
 *
 * This version enhances the standard tiled approach by assigning multiple
 * nodes to a single thread (Thread Coarsening), allowing for better
 * register usage and instruction-level parallelism.
 *
 * In the standard Tiled approach,
    each thread processes 1 node.*In this Improved approach,
    each thread processes VT nodes(e.g., VT = 4)
            .**Benefits : *1. Reduced Block Overhead
    : We need fewer blocks to cover the same N.*
        -Standard
    : Blocks = N / TILE_SIZE * -Improved : Blocks = N / (TILE_SIZE * VT) *
                                                    2. Reduced Halo Overhead
    : Each block has a fixed "halo"(STEPS)that is
      *
      wasted.* -By doing more work per block,
      the ratio of(Useful Work / Halo Overhead) * improves.*
          3. Instruction Level Parallelism(ILP)
    : The compiler can unroll the inner * loop over VT,
      hiding latency.*/

#define TILE_SIZE 256 // Threads per block
#define VT 4          // Values per thread (Coarsening factor)
#define STEPS 16      // Time steps per kernel

__device__ double device_callPayoff_improved(double S, double K) {
  return fmax(S - K, 0.0);
}

__device__ double device_putPayoff_improved(double S, double K) {
  return fmax(K - S, 0.0);
}

/**
 * @brief Thread Coarsened Kernel
 *
 * Each block computes (TILE_SIZE * VT) output values.
 * To do this for `STEPS` steps, it needs to load (TILE_SIZE * VT + STEPS) input
 * values.
 */
__global__ void improvedKernel(const double *V_in, double *V_out,
                               const double *u_pow, const double *d_pow,
                               int t_start, int steps, double S0, double K,
                               bool isCall, double discount, double p) {

  // Shared memory size: Enough for all threads' data + halo
  // Size = TILE_SIZE * VT + STEPS
  extern __shared__ double shared_mem[];
  double *V_s = shared_mem;

  int tid = threadIdx.x;
  // int block_output_size = TILE_SIZE * VT; // Unused
  // The actual output size shrinks by `steps` (halo effect)
  // But we define the block's responsibility based on the *output* at t_end.
  // Let's stick to the logic: Block covers a fixed range of OUTPUT at t_end.
  // Range size = block_output_size - steps?
  // No, let's say the block is responsible for computing `block_output_size`
  // valid values *if possible*. But due to halo, we usually define the block by
  // the INPUT tile and accept the output shrinks.
  // Let's use the "Overlapping Blocks" strategy from cuda_tiled.cu but adapted.

  // Strategy:
  // We want to produce `OUTPUT_WIDTH` valid values at `t_end`.
  // `OUTPUT_WIDTH` = (TILE_SIZE * VT) - steps.
  // Block `b` is responsible for output indices [b * OUTPUT_WIDTH, (b+1) *
  // OUTPUT_WIDTH - 1]. To do this, it must load `OUTPUT_WIDTH + steps` values
  // from `t_start`. `OUTPUT_WIDTH + steps` = `TILE_SIZE * VT`.

  int output_width = (TILE_SIZE * VT) - steps;
  int block_start_idx = blockIdx.x * output_width;

  // 1. Cooperative Load into Shared Memory
  // We need to load `TILE_SIZE * VT` elements.
  // Each thread loads `VT` elements?
  // Yes, straightforward mapping.
  int thread_start_idx = block_start_idx + tid * VT;

  // Load VT elements per thread
  // double r_val[VT]; // Unused

#pragma unroll
  for (int i = 0; i < VT; ++i) {
    int global_idx = thread_start_idx + i;
    if (global_idx <= t_start) {
      V_s[tid * VT + i] = V_in[global_idx];
    } else {
      V_s[tid * VT + i] = 0.0;
    }
  }
  __syncthreads();

  // 2. Compute `steps` iterations
  // We operate on Shared Memory directly for simplicity (easier to handle
  // neighbor dependency). True register tiling would require shuffling.
  // Without double buffering, we have a race between threads.
  // Let's assume we allocated 2x shared memory in the kernel launch.
  // V_s_in = &shared_mem[0]
  // V_s_out = &shared_mem[TILE_SIZE * VT]

  // Pointers for double buffering
  double *V_s_in = &shared_mem[0];
  double *V_s_out = &shared_mem[(TILE_SIZE * VT)];

  for (int k = 0; k < steps; ++k) {
    int current_t = t_start - k;

    // Each thread computes its VT elements
#pragma unroll
    for (int i = 0; i < VT; ++i) {
      int local_idx = tid * VT + i;
      // Check if this index is valid for computation
      // The valid range shrinks by 1 every step.
      // Initial valid: [0, TILE_SIZE*VT - 1]
      // Step k valid: [0, TILE_SIZE*VT - 1 - k]
      if (local_idx < (TILE_SIZE * VT) - 1 - k) {
        int global_idx = block_start_idx + local_idx;
        if (global_idx <= current_t - 1) {
          double S =
              S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];
          double V_hold = discount * (p * V_s_in[local_idx + 1] +
                                      (1.0 - p) * V_s_in[local_idx]);
          double V_exercise = isCall ? device_callPayoff_improved(S, K)
                                     : device_putPayoff_improved(S, K);
          V_s_out[local_idx] = fmax(V_hold, V_exercise);
        }
      }
    }
    __syncthreads();

    // Swap pointers for next iteration
    double *temp = V_s_in;
    V_s_in = V_s_out;
    V_s_out = temp;
  }

  // 3. Write back
  // Result is in V_s_in (because we swapped after writing to V_s_out)
  double *V_final = V_s_in;
  int final_t = t_start - steps;

  // Only write valid output range
#pragma unroll
  for (int i = 0; i < VT; ++i) {
    int local_idx = tid * VT + i;
    if (local_idx < output_width) {
      int global_idx = block_start_idx + local_idx;
      if (global_idx <= final_t) {
        V_out[global_idx] = V_final[local_idx];
      }
    }
  }
}

// Helper for terminal values (reused)
__global__ void initTerminalValuesKernelImproved(double *V, const double *u_pow,
                                                 const double *d_pow, double S0,
                                                 double K, int N, bool isCall) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= N) {
    double S = S0 * u_pow[i] * d_pow[N - i];
    V[i] = isCall ? device_callPayoff_improved(S, K)
                  : device_putPayoff_improved(S, K);
  }
}

double priceAmericanOptionCUDASharedMemTiling(const OptionParams &opt) {
  BinomialParams params = computeBinomialParams(opt);

  // Precompute powers
  std::vector<double> h_u_pow(opt.N + 1);
  std::vector<double> h_d_pow(opt.N + 1);
  h_u_pow[0] = 1.0;
  h_d_pow[0] = 1.0;
  for (int i = 1; i <= opt.N; ++i) {
    h_u_pow[i] = h_u_pow[i - 1] * params.u;
    h_d_pow[i] = h_d_pow[i - 1] * params.d;
  }

  double *d_V_in, *d_V_out, *d_u_pow, *d_d_pow;
  CUDA_CHECK(cudaMalloc(&d_V_in, (opt.N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_V_out, (opt.N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_u_pow, (opt.N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_d_pow, (opt.N + 1) * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_u_pow, h_u_pow.data(), (opt.N + 1) * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_d_pow, h_d_pow.data(), (opt.N + 1) * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Init
  int threadsPerBlock = 256;
  int blocks = (opt.N + 1 + threadsPerBlock - 1) / threadsPerBlock;
  initTerminalValuesKernelImproved<<<blocks, threadsPerBlock>>>(
      d_V_in, d_u_pow, d_d_pow, opt.S0, opt.K, opt.N, opt.isCall);
  CUDA_CHECK(cudaGetLastError());

  // Loop
  int current_t = opt.N;
  int steps_per_kernel = STEPS;
  int tile_size = TILE_SIZE; // 256
  int vt = VT;               // 4
  // int output_width = (tile_size * vt) - steps_per_kernel; // Unused

  while (current_t > 0) {
    int steps = std::min(steps_per_kernel, current_t);

    // Recalculate output width if steps < steps_per_kernel (last iter)
    int current_output_width = (tile_size * vt) - steps;
    int num_blocks =
        (current_t + current_output_width - 1) / current_output_width;

    // Double buffered shared mem
    size_t shared_mem_size = 2 * (tile_size * vt) * sizeof(double);

    improvedKernel<<<num_blocks, tile_size, shared_mem_size>>>(
        d_V_in, d_V_out, d_u_pow, d_d_pow, current_t, steps, opt.S0, opt.K,
        opt.isCall, params.discount, params.p);
    CUDA_CHECK(cudaGetLastError());

    std::swap(d_V_in, d_V_out);
    current_t -= steps;
  }

  double result;
  CUDA_CHECK(
      cudaMemcpy(&result, &d_V_in[0], sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_V_in));
  CUDA_CHECK(cudaFree(d_V_out));
  CUDA_CHECK(cudaFree(d_u_pow));
  CUDA_CHECK(cudaFree(d_d_pow));

  return result;
}
