#include "cuda_kernels.cuh"
#include "cuda_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/**
 * @file cuda_warp_shuffle_tiling.cu
 * @brief Tiled Implementation V3 (Warp Shuffle + Register Tiling)
 *
 * Optimization Strategy:
 * 1. Thread Coarsening (VT=4): Each thread computes 4 values.
 * 2. Register Tiling: Data is kept in registers, minimizing shared mem access.
 * 3. Warp Shuffle: Intra-warp communication uses __shfl_down_sync.
 * 4. Reduced Shared Memory: Only used for inter-warp communication (halo).
 */

#define TILE_SIZE 256
#define VT 4
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (TILE_SIZE / WARP_SIZE)

__device__ double device_callPayoff_v3(double S, double K) {
  return fmax(S - K, 0.0);
}

__device__ double device_putPayoff_v3(double S, double K) {
  return fmax(K - S, 0.0);
}

__global__ void tiledKernelV3(const double *__restrict__ V_in,
                              double *__restrict__ V_out,
                              const double *__restrict__ u_pow,
                              const double *__restrict__ d_pow, int t_start,
                              int steps, double S0, double K, bool isCall,
                              double discount, double p) {

  // Shared memory only needed for inter-warp halo
  // Each warp stores its "first" value (lane 0's first value) here
  // so the previous warp's last thread (lane 31) can read it.
  __shared__ double warp_halo[WARPS_PER_BLOCK];

  int tid = threadIdx.x;
  int lane_id = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

  int output_width = (TILE_SIZE * VT) - steps;
  int block_start_idx = blockIdx.x * output_width;
  int thread_start_idx = block_start_idx + tid * VT;

  // Registers for the thread's tile
  double r_val[VT];

  // 1. Load from Global Memory directly to Registers
  // We need to handle boundary conditions carefully.
  // We load VT elements.
#pragma unroll
  for (int i = 0; i < VT; ++i) {
    int global_idx = thread_start_idx + i;
    if (global_idx <= t_start) {
      r_val[i] = V_in[global_idx];
    } else {
      r_val[i] = 0.0;
    }
  }

  // We also need the "next" value for the computation.
  // r_val[i] depends on r_val[i] and r_val[i+1].
  // Inside the thread, r_val[i] needs r_val[i+1].
  // r_val[VT-1] needs r_val[0] of the NEXT thread.

  for (int k = 0; k < steps; ++k) {
    int current_t = t_start - k;

    // Identify the value needed from the neighbor (next thread)
    // This is the first value of the next thread.
    // We can get this via shuffle from the next lane.
    // For the last lane, we need it from shared memory.

    double neighbor_val;
    double my_first_val = r_val[0];

    // 1. Publish my first value to shared memory if I am Lane 0
    if (lane_id == 0) {
      warp_halo[warp_id] = my_first_val;
    }
    __syncthreads(); // Wait for all Lane 0s to write

    // 2. Get neighbor value
    // Shuffle down: get value from lane_id + 1
    // The 'mask' should be all active threads. 0xffffffff is usually fine for
    // full blocks.
    unsigned mask = 0xffffffff;
    neighbor_val = __shfl_down_sync(mask, my_first_val, 1);

    // 3. If I am Lane 31, I need to read from the next warp's halo
    if (lane_id == WARP_SIZE - 1) {
      if (warp_id < WARPS_PER_BLOCK - 1) {
        neighbor_val = warp_halo[warp_id + 1];
      } else {
        // Last warp, last thread.
        // We might need to load from global if we are tiling across blocks?
        // In this implementation (Wavefront/Tiled), we assume the block covers
        // the necessary range plus halo. But wait, standard Tiled
        // implementation loads a larger tile to compute a smaller output. The
        // "halo" is internal to the block. The rightmost thread of the block
        // doesn't have a valid neighbor in the block. But that's fine, because
        // its computed value is likely outside the "valid output" cone or we
        // handled it by loading 0s. Actually, for the rightmost boundary, we
        // assume 0 or boundary condition. In the standard tiled logic, we just
        // don't compute the invalid region.
        neighbor_val = 0.0; // Default
      }
    }

    // 4. Compute
    // We compute in-place or use a temporary?
    // r_val[i] = f(r_val[i], r_val[i+1])
    // We need to be careful about dependencies.
    // r_val[0] needs r_val[1]. r_val[1] needs r_val[2].
    // If we update r_val[1], r_val[0] will use the NEW value.
    // So we need a temporary buffer or iterate carefully.
    // Since it's a small array, we can copy to temp registers.

    double r_next[VT];

#pragma unroll
    for (int i = 0; i < VT - 1; ++i) {
      // Internal dependencies
      double val_curr = r_val[i];
      double val_next = r_val[i + 1];

      int global_idx = thread_start_idx + i;
      // Check bounds? The logic in TiledV2 checked bounds.
      // Optimization: Compute blindly, mask out later?
      // Let's stick to correctness checks.

      if (global_idx <= current_t - 1) {
        double S = S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];
        double V_hold = discount * (p * val_next + (1.0 - p) * val_curr);
        double V_exercise =
            isCall ? device_callPayoff_v3(S, K) : device_putPayoff_v3(S, K);
        r_next[i] = fmax(V_hold, V_exercise);
      } else {
        r_next[i] = 0.0; // Or keep old value? Doesn't matter for invalid zone.
      }
    }

    // Last element (VT-1) needs neighbor_val
    {
      int i = VT - 1;
      double val_curr = r_val[i];
      double val_next = neighbor_val;

      int global_idx = thread_start_idx + i;
      if (global_idx <= current_t - 1) {
        double S = S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];
        double V_hold = discount * (p * val_next + (1.0 - p) * val_curr);
        double V_exercise =
            isCall ? device_callPayoff_v3(S, K) : device_putPayoff_v3(S, K);
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

    __syncthreads(); // Sync before next iteration (for shared mem safety)
  }

  // Write back
  int final_t = t_start - steps;
#pragma unroll
  for (int i = 0; i < VT; ++i) {
    int local_idx = tid * VT + i;
    // Check if within output width
    if (local_idx < output_width) {
      int global_idx = block_start_idx + local_idx;
      if (global_idx <= final_t) {
        V_out[global_idx] = r_val[i];
      }
    }
  }
}

__global__ void initTerminalValuesKernelV3(double *V, const double *u_pow,
                                           const double *d_pow, double S0,
                                           double K, int N, bool isCall) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= N) {
    double S = S0 * u_pow[i] * d_pow[N - i];
    V[i] = isCall ? device_callPayoff_v3(S, K) : device_putPayoff_v3(S, K);
  }
}

double priceAmericanOptionCUDAWarpShuffleTiling(const OptionParams &opt) {
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
  initTerminalValuesKernelV3<<<blocks, threadsPerBlock>>>(
      d_V_in, d_u_pow, d_d_pow, opt.S0, opt.K, opt.N, opt.isCall);
  CUDA_CHECK(cudaGetLastError());

  int current_t = opt.N;
  int steps_per_kernel = 16; // Adjust as needed
  int tile_size = TILE_SIZE;
  int vt = VT;

  while (current_t > 0) {
    int steps = std::min(steps_per_kernel, current_t);

    int current_output_width = (tile_size * vt) - steps;
    int num_blocks =
        (current_t + current_output_width - 1) / current_output_width;

    // No shared mem size needed for kernel call (static allocation)
    tiledKernelV3<<<num_blocks, tile_size>>>(
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
