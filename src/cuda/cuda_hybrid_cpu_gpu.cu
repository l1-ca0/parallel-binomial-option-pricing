#include "cuda_kernels.cuh"
#include "cuda_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/**
 * @file cuda_hybrid_cpu_gpu.cu
 * @brief Hybrid Implementation CPU-GPU (Adaptive GPU + CPU Fallback)
 *
 * Strategy:
 * - Use Tiled Shared Mem for Large N.
 * - Use Tiled Warp-Per-Block for Medium N.
 * - Switch to CPU when N is small.
 */

// Default Thresholds
int g_cpu_threshold_cpugpu = 500;
int g_large_n_threshold_cpugpu = 80000;

void setHybridCPUGPUThresholds(int cpu_thresh, int large_n_thresh) {
  g_cpu_threshold_cpugpu = cpu_thresh;
  g_large_n_threshold_cpugpu = large_n_thresh;
}

#define LARGE_N_THRESHOLD g_large_n_threshold_cpugpu
#define CPU_THRESHOLD g_cpu_threshold_cpugpu

// =================================================================================
// KERNEL 1: Large N (Tiled Shared Mem)
// =================================================================================

#define TILE_SIZE_V2 256
#define VT_V2 4
#define STEPS_V2 16

__device__ double device_callPayoff_hybrid_cpugpu(double S, double K) {
  return fmax(S - K, 0.0);
}

__device__ double device_putPayoff_hybrid_cpugpu(double S, double K) {
  return fmax(K - S, 0.0);
}

__global__ void hybridCPUGPU_LargeN_Kernel(const double *V_in, double *V_out,
                                           const double *u_pow,
                                           const double *d_pow, int t_start,
                                           int steps, double S0, double K,
                                           bool isCall, double discount,
                                           double p) {
  extern __shared__ double shared_mem[];
  double *V_s = shared_mem;

  int tid = threadIdx.x;
  int output_width = (TILE_SIZE_V2 * VT_V2) - steps;
  int block_start_idx = blockIdx.x * output_width;
  int thread_start_idx = block_start_idx + tid * VT_V2;

  // Load
#pragma unroll
  for (int i = 0; i < VT_V2; ++i) {
    int global_idx = thread_start_idx + i;
    if (global_idx <= t_start) {
      V_s[tid * VT_V2 + i] = V_in[global_idx];
    } else {
      V_s[tid * VT_V2 + i] = 0.0;
    }
  }
  __syncthreads();

  double *V_s_in = &shared_mem[0];
  double *V_s_out = &shared_mem[(TILE_SIZE_V2 * VT_V2)];

  for (int k = 0; k < steps; ++k) {
    int current_t = t_start - k;
#pragma unroll
    for (int i = 0; i < VT_V2; ++i) {
      int local_idx = tid * VT_V2 + i;
      if (local_idx < (TILE_SIZE_V2 * VT_V2) - 1 - k) {
        int global_idx = block_start_idx + local_idx;
        if (global_idx <= current_t - 1) {
          double S =
              S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];
          double V_hold = discount * (p * V_s_in[local_idx + 1] +
                                      (1.0 - p) * V_s_in[local_idx]);
          double V_exercise = isCall ? device_callPayoff_hybrid_cpugpu(S, K)
                                     : device_putPayoff_hybrid_cpugpu(S, K);
          V_s_out[local_idx] = fmax(V_hold, V_exercise);
        }
      }
    }
    __syncthreads();
    double *temp = V_s_in;
    V_s_in = V_s_out;
    V_s_out = temp;
  }

  // Write back
  double *V_final = V_s_in;
  int final_t = t_start - steps;
#pragma unroll
  for (int i = 0; i < VT_V2; ++i) {
    int local_idx = tid * VT_V2 + i;
    if (local_idx < output_width) {
      int global_idx = block_start_idx + local_idx;
      if (global_idx <= final_t) {
        V_out[global_idx] = V_final[local_idx];
      }
    }
  }
}

// =================================================================================
// KERNEL 2: Small N (Tiled Warp-Per-Block)
// =================================================================================

#define TILE_SIZE_V4 32
#define VT_V4 4

__global__ void __launch_bounds__(32)
    hybridCPUGPU_SmallN_Kernel(const double *__restrict__ V_in,
                               double *__restrict__ V_out,
                               const double *__restrict__ u_pow,
                               const double *__restrict__ d_pow, int t_start,
                               int steps, double S0, double K, bool isCall,
                               double discount, double p) {
  int tid = threadIdx.x;
  int output_width = (TILE_SIZE_V4 * VT_V4) - steps;
  int block_start_idx = blockIdx.x * output_width;
  int thread_start_idx = block_start_idx + tid * VT_V4;

  double r_val[VT_V4];

#pragma unroll
  for (int i = 0; i < VT_V4; ++i) {
    int global_idx = thread_start_idx + i;
    if (global_idx <= t_start) {
      r_val[i] = V_in[global_idx];
    } else {
      r_val[i] = 0.0;
    }
  }

  for (int k = 0; k < steps; ++k) {
    int current_t = t_start - k;
    double my_first_val = r_val[0];
    unsigned mask = 0xffffffff;
    double neighbor_val = __shfl_down_sync(mask, my_first_val, 1);

    double r_next[VT_V4];
#pragma unroll
    for (int i = 0; i < VT_V4 - 1; ++i) {
      double val_curr = r_val[i];
      double val_next = r_val[i + 1];
      int global_idx = thread_start_idx + i;
      if (global_idx <= current_t - 1) {
        double S = S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];
        double V_hold = discount * (p * val_next + (1.0 - p) * val_curr);
        double V_exercise = isCall ? device_callPayoff_hybrid_cpugpu(S, K)
                                   : device_putPayoff_hybrid_cpugpu(S, K);
        r_next[i] = fmax(V_hold, V_exercise);
      } else {
        r_next[i] = 0.0;
      }
    }
    {
      int i = VT_V4 - 1;
      double val_curr = r_val[i];
      double val_next = neighbor_val;
      int global_idx = thread_start_idx + i;
      if (global_idx <= current_t - 1) {
        double S = S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];
        double V_hold = discount * (p * val_next + (1.0 - p) * val_curr);
        double V_exercise = isCall ? device_callPayoff_hybrid_cpugpu(S, K)
                                   : device_putPayoff_hybrid_cpugpu(S, K);
        r_next[i] = fmax(V_hold, V_exercise);
      } else {
        r_next[i] = 0.0;
      }
    }
#pragma unroll
    for (int i = 0; i < VT_V4; ++i) {
      r_val[i] = r_next[i];
    }
  }

  int final_t = t_start - steps;
#pragma unroll
  for (int i = 0; i < VT_V4; ++i) {
    int local_idx = tid * VT_V4 + i;
    if (local_idx < output_width) {
      int global_idx = block_start_idx + local_idx;
      if (global_idx <= final_t) {
        V_out[global_idx] = r_val[i];
      }
    }
  }
}

__global__ void initTerminalValuesKernelHybridCPUGPU(double *V,
                                                     const double *u_pow,
                                                     const double *d_pow,
                                                     double S0, double K, int N,
                                                     bool isCall) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= N) {
    double S = S0 * u_pow[i] * d_pow[N - i];
    V[i] = isCall ? device_callPayoff_hybrid_cpugpu(S, K)
                  : device_putPayoff_hybrid_cpugpu(S, K);
  }
}

// =================================================================================
// HOST FUNCTION
// =================================================================================

double priceAmericanOptionCUDAHybridCPUGPU(const OptionParams &opt) {
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
  initTerminalValuesKernelHybridCPUGPU<<<blocks, threadsPerBlock>>>(
      d_V_in, d_u_pow, d_d_pow, opt.S0, opt.K, opt.N, opt.isCall);
  CUDA_CHECK(cudaGetLastError());

  int current_t = opt.N;

  // Phase 1: Large N (Shared Mem Tiling)
  while (current_t > LARGE_N_THRESHOLD) {
    int steps = std::min(STEPS_V2, current_t - LARGE_N_THRESHOLD);
    if (steps <= 0)
      break;

    int current_output_width = (TILE_SIZE_V2 * VT_V2) - steps;
    int num_blocks =
        (current_t + current_output_width - 1) / current_output_width;
    size_t shared_mem_size = 2 * (TILE_SIZE_V2 * VT_V2) * sizeof(double);

    hybridCPUGPU_LargeN_Kernel<<<num_blocks, TILE_SIZE_V2, shared_mem_size>>>(
        d_V_in, d_V_out, d_u_pow, d_d_pow, current_t, steps, opt.S0, opt.K,
        opt.isCall, params.discount, params.p);
    CUDA_CHECK(cudaGetLastError());

    std::swap(d_V_in, d_V_out);
    current_t -= steps;
  }

  // Phase 2: Medium N (Warp Per Block)
  while (current_t > CPU_THRESHOLD) {
    int steps = std::min(16, current_t - CPU_THRESHOLD);
    if (steps <= 0)
      break;

    int current_output_width = (TILE_SIZE_V4 * VT_V4) - steps;
    int num_blocks =
        (current_t + current_output_width - 1) / current_output_width;

    hybridCPUGPU_SmallN_Kernel<<<num_blocks, TILE_SIZE_V4>>>(
        d_V_in, d_V_out, d_u_pow, d_d_pow, current_t, steps, opt.S0, opt.K,
        opt.isCall, params.discount, params.p);
    CUDA_CHECK(cudaGetLastError());

    std::swap(d_V_in, d_V_out);
    current_t -= steps;
  }

  // Phase 3: Transfer to CPU
  std::vector<double> h_V(current_t + 1);
  CUDA_CHECK(cudaMemcpy(h_V.data(), d_V_in, (current_t + 1) * sizeof(double),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_V_in));
  CUDA_CHECK(cudaFree(d_V_out));
  CUDA_CHECK(cudaFree(d_u_pow));
  CUDA_CHECK(cudaFree(d_d_pow));

  // Phase 4: CPU Serial
  for (int t = current_t - 1; t >= 0; --t) {
    for (int i = 0; i <= t; ++i) {
      double S = opt.S0 * h_u_pow[i] * h_d_pow[t - i];
      double V_hold =
          params.discount * (params.p * h_V[i + 1] + (1.0 - params.p) * h_V[i]);
      double V_exercise;
      if (opt.isCall) {
        V_exercise = std::max(S - opt.K, 0.0);
      } else {
        V_exercise = std::max(opt.K - S, 0.0);
      }
      h_V[i] = std::max(V_hold, V_exercise);
    }
  }

  return h_V[0];
}
