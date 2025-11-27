#include "cuda_kernels.cuh"
#include "cuda_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/**
 * @file cuda_time_parallel.cu
 * @brief Time-Axis Parallelization Implementation
 *
 * This implementation parallelizes across the time axis, allowing for
 * concurrent computation of multiple time steps.
 */

// Configuration
#define BLOCK_SIZE_TP 256
#define VT_TP 4
#define STEPS_L1 256 // Steps per kernel launch

__device__ __forceinline__ double payoff_tp(double S, double K, bool isCall) {
  return isCall ? fmax(S - K, 0.0) : fmax(K - S, 0.0);
}

/**
 * @brief Two-level temporal tiling kernel
 *
 * This kernel processes STEPS_L1 time steps per launch.
 * Uses shared memory double-buffering for the tile.
 */
__global__ void __launch_bounds__(256)
    tiledKernelTimeParallel(const double *__restrict__ V_in,
                            double *__restrict__ V_out,
                            const double *__restrict__ u_pow,
                            const double *__restrict__ d_pow, int t_start,
                            int total_steps, double S0, double K, bool isCall,
                            double discount, double p) {

  // Shared memory for double buffering
  extern __shared__ double smem[];

  const int tid = threadIdx.x;
  const int tile_width = BLOCK_SIZE_TP * VT_TP; // 2048 elements

  // Block output coverage
  const int output_width = tile_width - total_steps;
  const int block_start_idx = blockIdx.x * output_width;

  // Double buffering pointers
  double *V_s_in = &smem[0];
  double *V_s_out = &smem[tile_width];

// Load initial values into shared memory (coalesced)
#pragma unroll
  for (int i = 0; i < VT_TP; ++i) {
    int local_idx = tid * VT_TP + i;
    int global_idx = block_start_idx + local_idx;
    if (global_idx <= t_start) {
      V_s_in[local_idx] = V_in[global_idx];
    } else {
      V_s_in[local_idx] = 0.0;
    }
  }
  __syncthreads();

  // Process all time steps
  int current_t = t_start;

  for (int k = 0; k < total_steps; ++k) {
// Each thread computes VT_TP values
#pragma unroll
    for (int i = 0; i < VT_TP; ++i) {
      int local_idx = tid * VT_TP + i;
      int global_idx = block_start_idx + local_idx;

      // Check bounds: need local_idx and local_idx+1 to be valid in shared mem
      // After k steps, the valid range shrinks by k on the right
      if (local_idx < tile_width - 1 - k && global_idx <= current_t - 1) {
        double S = S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];
        double V_hold = discount * (p * V_s_in[local_idx + 1] +
                                    (1.0 - p) * V_s_in[local_idx]);
        double V_exercise = payoff_tp(S, K, isCall);
        V_s_out[local_idx] = fmax(V_hold, V_exercise);
      }
    }
    __syncthreads();

    // Swap buffers
    double *temp = V_s_in;
    V_s_in = V_s_out;
    V_s_out = temp;

    current_t--;
  }

  // Write back results
  const int final_t = t_start - total_steps;
#pragma unroll
  for (int i = 0; i < VT_TP; ++i) {
    int local_idx = tid * VT_TP + i;
    if (local_idx < output_width) {
      int global_idx = block_start_idx + local_idx;
      if (global_idx <= final_t) {
        V_out[global_idx] = V_s_in[local_idx];
      }
    }
  }
}

__global__ void initTerminalKernelTP(double *V, const double *u_pow,
                                     const double *d_pow, double S0, double K,
                                     int N, bool isCall) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= N) {
    double S = S0 * u_pow[i] * d_pow[N - i];
    V[i] = payoff_tp(S, K, isCall);
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
 * @brief Host function using two-level tiled kernel
 */
double priceAmericanOptionTimeParallel(double S0, double K, double r,
                                       double sigma, double T, int N,
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
  initTerminalKernelTP<<<blocks, 256>>>(d_V_in, d_u_pow, d_d_pow, S0, K, N,
                                        isCall);
  CUDA_CHECK(cudaGetLastError());

  int current_t = N;
  const int tile_width = BLOCK_SIZE_TP * VT_TP; // 2048

  while (current_t > 0) {
    int steps = std::min(STEPS_L1, current_t);

    int output_width = tile_width - steps;
    int num_blocks = (current_t + output_width - 1) / output_width;
    size_t smem_size = 2 * tile_width * sizeof(double); // Double buffer

    tiledKernelTimeParallel<<<num_blocks, BLOCK_SIZE_TP, smem_size>>>(
        d_V_in, d_V_out, d_u_pow, d_d_pow, current_t, steps, S0, K, isCall,
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
