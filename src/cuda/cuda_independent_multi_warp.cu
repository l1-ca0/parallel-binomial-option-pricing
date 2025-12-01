#include "cuda_kernels.cuh"
#include "cuda_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/**
 * @file cuda_independent_multi_warp.cu
 * @brief Tiled Implementation (Multi-Warp Independent)
 *
 * Optimization Strategy:
 * 1. Multi-Warp: Block Size = 256 (8 Warps).
 * 2. Independent Warps: Each warp acts as a "virtual block".
 * 3. No Synchronization: No __syncthreads(), only implicit warp sync.
 * 4. No Shared Memory: All communication via __shfl_down_sync.
 * 5. Register Tiling: VT=4.
 *
 * Optimization Strategy:
 * 1. Multi-Warp: Block Size = 256 (8 Warps).
 * 2. Independent Warps: Each warp acts as a "virtual block".
 * 3. No Synchronization: No __syncthreads(), only implicit warp sync.
 * 4. No Shared Memory: All communication via __shfl_down_sync.
 * 5. Register Tiling: VT=4.
 */

#define TILE_SIZE 32 // Warp Size (Virtual Tile Size)
#define VT 4
#define WARPS_PER_BLOCK 8
#define BLOCK_DIM (TILE_SIZE * WARPS_PER_BLOCK) // 256

__device__ double device_callPayoff_v5(double S, double K) {
  return fmax(S - K, 0.0);
}

__device__ double device_putPayoff_v5(double S, double K) {
  return fmax(K - S, 0.0);
}

__global__ void __launch_bounds__(256)
    tiledKernelV5(const double *__restrict__ V_in, double *__restrict__ V_out,
                  const double *__restrict__ u_pow,
                  const double *__restrict__ d_pow, int t_start, int steps,
                  double S0, double K, bool isCall, double discount, double p) {

  // No shared memory needed!

  int tid = threadIdx.x; // 0 to 255
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  // Calculate Virtual Block ID
  // Each warp is a "virtual block"
  int virtual_block_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;

  int output_width = (TILE_SIZE * VT) - steps;
  int block_start_idx = virtual_block_id * output_width;
  int thread_start_idx = block_start_idx + lane_id * VT;

  // Registers for the thread's tile
  double r_val[VT];

  // 1. Load from Global Memory
#pragma unroll
  for (int i = 0; i < VT; ++i) {
    int global_idx = thread_start_idx + i;
    if (global_idx <= t_start) {
      r_val[i] = V_in[global_idx];
    } else {
      r_val[i] = 0.0;
    }
  }

  for (int k = 0; k < steps; ++k) {
    int current_t = t_start - k;

    // Get neighbor value via shuffle
    // We need the first value of the next thread (lane_id + 1)
    double my_first_val = r_val[0];
    unsigned mask = 0xffffffff;
    double neighbor_val = __shfl_down_sync(mask, my_first_val, 1);

    // For the last lane (31), neighbor_val is undefined (or 0).
    // This is correct for the boundary of the virtual tile.

    double r_next[VT];

    // Compute internal updates
#pragma unroll
    for (int i = 0; i < VT - 1; ++i) {
      double val_curr = r_val[i];
      double val_next = r_val[i + 1];

      int global_idx = thread_start_idx + i;
      if (global_idx <= current_t - 1) {
        double S = S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];
        double V_hold = discount * (p * val_next + (1.0 - p) * val_curr);
        double V_exercise =
            isCall ? device_callPayoff_v5(S, K) : device_putPayoff_v5(S, K);
        r_next[i] = fmax(V_hold, V_exercise);
      } else {
        r_next[i] = 0.0;
      }
    }

    // Compute boundary update (using neighbor_val)
    {
      int i = VT - 1;
      double val_curr = r_val[i];
      double val_next = neighbor_val;

      int global_idx = thread_start_idx + i;
      if (global_idx <= current_t - 1) {
        double S = S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];
        double V_hold = discount * (p * val_next + (1.0 - p) * val_curr);
        double V_exercise =
            isCall ? device_callPayoff_v5(S, K) : device_putPayoff_v5(S, K);
        r_next[i] = fmax(V_hold, V_exercise);
      } else {
        r_next[i] = 0.0;
      }
    }

    // Update registers
#pragma unroll
    for (int i = 0; i < VT; ++i) {
      r_val[i] = r_next[i];
    }

    // Implicit sync within warp for shuffle
  }

  // Write back
  int final_t = t_start - steps;
#pragma unroll
  for (int i = 0; i < VT; ++i) {
    int local_idx = lane_id * VT + i;
    if (local_idx < output_width) {
      int global_idx = block_start_idx + local_idx;
      if (global_idx <= final_t) {
        V_out[global_idx] = r_val[i];
      }
    }
  }
}

__global__ void initTerminalValuesKernelV5(double *V, const double *u_pow,
                                           const double *d_pow, double S0,
                                           double K, int N, bool isCall) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= N) {
    double S = S0 * u_pow[i] * d_pow[N - i];
    V[i] = isCall ? device_callPayoff_v5(S, K) : device_putPayoff_v5(S, K);
  }
}

double priceAmericanOptionCUDAIndependentMultiWarp(const OptionParams &opt) {
  BinomialParams params = computeBinomialParams(opt);

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

  int threadsPerBlock = 256;
  int blocks = (opt.N + 1 + threadsPerBlock - 1) / threadsPerBlock;
  initTerminalValuesKernelV5<<<blocks, threadsPerBlock>>>(
      d_V_in, d_u_pow, d_d_pow, opt.S0, opt.K, opt.N, opt.isCall);
  CUDA_CHECK(cudaGetLastError());

  int current_t = opt.N;
  int steps_per_kernel = 16;

  int tile_size = 32; // Virtual Tile Size (Warp Size)

  while (current_t > 0) {
    int steps = std::min(steps_per_kernel, current_t);

    int current_output_width = (tile_size * VT) - steps;

    // Total number of virtual blocks needed
    int num_virtual_blocks =
        (current_t + current_output_width - 1) / current_output_width;

    // Total number of physical blocks
    // Each physical block handles 'warps_per_block' virtual blocks
    int num_physical_blocks =
        (num_virtual_blocks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    tiledKernelV5<<<num_physical_blocks, threadsPerBlock>>>(
        d_V_in, d_V_out, d_u_pow, d_d_pow, current_t, steps, opt.S0, opt.K,
        opt.isCall, params.discount, params.p);
    CUDA_CHECK(cudaGetLastError());

    std::swap(d_V_in, d_V_out);
    current_t -= steps;
  }

  double h_price;
  CUDA_CHECK(
      cudaMemcpy(&h_price, d_V_in, sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_V_in));
  CUDA_CHECK(cudaFree(d_V_out));
  CUDA_CHECK(cudaFree(d_u_pow));
  CUDA_CHECK(cudaFree(d_d_pow));

  return h_price;
}
