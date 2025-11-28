#include "cuda_kernels.cuh"
#include "cuda_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/**
 * @file cuda_cooperative_multi_warp.cu
 * @brief Cooperative Multi-Warp Implementation
 *
 * Strategy:
 * - All warps in a block cooperate to process a single large contiguous tile.
 * - Uses shared memory for inter-warp communication and synchronization.
 *
 * Tile size = 256 * 8 = 2048 elements
 * Each thread holds 8 elements in registers.
 * Warp 0 handles elements 0 - 255,
 * Warp 1 handles 256 - 511, etc.
 */

#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE_TP 256
#define VT_TP 8

          __device__ __forceinline__ double payoff_mw(double S, double K,
                                                      bool isCall) {
  return isCall ? fmax(S - K, 0.0) : fmax(K - S, 0.0);
}

/**
 * @brief Warp shuffle helper for double precision
 */
__device__ __forceinline__ double shfl_down_double_mw(double var,
                                                      unsigned int delta) {
  int lo = __double2loint(var);
  int hi = __double2hiint(var);
  lo = __shfl_down_sync(0xffffffff, lo, delta);
  hi = __shfl_down_sync(0xffffffff, hi, delta);
  return __hiloint2double(hi, lo);
}

__global__ void __launch_bounds__(256)
    multiWarpKernel(const double *__restrict__ V_in, double *__restrict__ V_out,
                    const double *__restrict__ u_pow,
                    const double *__restrict__ d_pow, int t_start, int steps,
                    double S0, double K, bool isCall, double discount,
                    double p) {

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warps = 8;

  // Tile layout: 256 threads * 8 values = 2048 elements per block
  // Thread tid owns elements [tid*VT, tid*VT + VT)
  const int tile_width = BLOCK_SIZE_TP * VT_TP; // 2048
  const int output_width = tile_width - steps;
  const int block_start_idx = blockIdx.x * output_width;
  const int thread_start_idx = block_start_idx + tid * VT_TP;

  // Shared memory for inter-warp halo exchange
  // Each warp publishes its lane 0's r_val[0] (the first element of the warp's
  // region)
  __shared__ double warp_first_elem[8];

  // Registers for this thread's values
  double r_val[VT_TP];

// Load from global memory
#pragma unroll
  for (int i = 0; i < VT_TP; ++i) {
    int global_idx = thread_start_idx + i;
    r_val[i] = (global_idx <= t_start) ? V_in[global_idx] : 0.0;
  }

  // Process time steps
  for (int k = 0; k < steps; ++k) {
    int current_t = t_start - k;

    // Step 1: Each warp's lane 0 publishes its first value for the PREVIOUS
    // warp to read Actually, we need lane 0 to publish for lane 31 of the
    // previous warp So warp W's lane 0 publishes, and warp W-1's lane 31 reads
    // it
    if (lane_id == 0) {
      warp_first_elem[warp_id] = r_val[0];
    }
    __syncthreads();

    // Step 2: Get the neighbor value needed for r_val[VT-1]
    // For lane L (L < 31): neighbor comes from lane L+1's r_val[0] via shuffle
    // For lane 31: neighbor comes from next warp's lane 0's r_val[0] via shared
    // memory

    double my_last_neighbor;

    // Get lane L+1's first register value via shuffle
    double next_lane_first = shfl_down_double_mw(r_val[0], 1);

    if (lane_id == 31) {
      // Lane 31 needs value from next warp
      if (warp_id < num_warps - 1) {
        my_last_neighbor = warp_first_elem[warp_id + 1];
      } else {
        my_last_neighbor = 0.0; // Last warp, last lane - boundary
      }
    } else {
      my_last_neighbor = next_lane_first;
    }

    // Step 3: Compute new values
    double r_next[VT_TP];

// Elements 0 to VT-2: use r_val[i+1]
#pragma unroll
    for (int i = 0; i < VT_TP - 1; ++i) {
      int global_idx = thread_start_idx + i;
      if (global_idx <= current_t - 1) {
        double S = S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];
        double V_hold = discount * (p * r_val[i + 1] + (1.0 - p) * r_val[i]);
        double V_exercise = payoff_mw(S, K, isCall);
        r_next[i] = fmax(V_hold, V_exercise);
      } else {
        r_next[i] = 0.0;
      }
    }

    // Element VT-1: use my_last_neighbor
    {
      int i = VT_TP - 1;
      int global_idx = thread_start_idx + i;
      if (global_idx <= current_t - 1) {
        double S = S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];
        double V_hold =
            discount * (p * my_last_neighbor + (1.0 - p) * r_val[i]);
        double V_exercise = payoff_mw(S, K, isCall);
        r_next[i] = fmax(V_hold, V_exercise);
      } else {
        r_next[i] = 0.0;
      }
    }

// Update registers
#pragma unroll
    for (int i = 0; i < VT_TP; ++i) {
      r_val[i] = r_next[i];
    }

    __syncthreads(); // Ensure all warps complete before next iteration
  }

  // Write results back to global memory
  const int final_t = t_start - steps;
#pragma unroll
  for (int i = 0; i < VT_TP; ++i) {
    int local_idx = tid * VT_TP + i;
    if (local_idx < output_width) {
      int global_idx = block_start_idx + local_idx;
      if (global_idx <= final_t) {
        V_out[global_idx] = r_val[i];
      }
    }
  }
}

__global__ void initTerminalKernelMW(double *V, const double *u_pow,
                                     const double *d_pow, double S0, double K,
                                     int N, bool isCall) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= N) {
    double S = S0 * u_pow[i] * d_pow[N - i];
    V[i] = payoff_mw(S, K, isCall);
  }
}

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", __FILE__, __LINE__, \
              err, cudaGetErrorString(err));                                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

/**
 * @brief Host function using multi-warp kernel
 */
double priceAmericanOptionCUDACooperativeMultiWarp(double S0, double K,
                                                   double r, double sigma,
                                                   double T, int N,
                                                   bool isCall) {
  double dt = T / N;
  double u = exp(sigma * sqrt(dt));
  double d = 1.0 / u;
  double p = (exp(r * dt) - d) / (u - d);
  double discount = exp(-r * dt);

  std::vector<double> h_u_pow(N + 1);
  std::vector<double> h_d_pow(N + 1);
  h_u_pow[0] = 1.0;
  h_d_pow[0] = 1.0;
  for (int i = 1; i <= N; ++i) {
    h_u_pow[i] = h_u_pow[i - 1] * u;
    h_d_pow[i] = h_d_pow[i - 1] * d;
  }

  double *d_V_in, *d_V_out, *d_u_pow, *d_d_pow;
  CUDA_CHECK(cudaMalloc(&d_V_in, (N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_V_out, (N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_u_pow, (N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_d_pow, (N + 1) * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_u_pow, h_u_pow.data(), (N + 1) * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_d_pow, h_d_pow.data(), (N + 1) * sizeof(double),
                        cudaMemcpyHostToDevice));

  int blocks = (N + 256) / 256;
  initTerminalKernelMW<<<blocks, 256>>>(d_V_in, d_u_pow, d_d_pow, S0, K, N,
                                        isCall);
  CUDA_CHECK(cudaGetLastError());

  int current_t = N;
  const int tile_width = BLOCK_SIZE_TP * VT_TP; // 2048
  const int steps_per_kernel = 32;

  while (current_t > 0) {
    int steps = std::min(steps_per_kernel, current_t);

    int output_width = tile_width - steps;
    int num_blocks = (current_t + output_width - 1) / output_width;

    multiWarpKernel<<<num_blocks, 256>>>(d_V_in, d_V_out, d_u_pow, d_d_pow,
                                         current_t, steps, S0, K, isCall,
                                         discount, p);
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
