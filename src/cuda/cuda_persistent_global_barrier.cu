#include "cuda_kernels.cuh"
#include "cuda_utils.cuh"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/**
 * @file cuda_persistent_global_barrier.cu
 * @brief Persistent Kernel Implementation with Global Barrier
 *
 * Optimization Strategy: Persistent Kernel
 *
 * This implementation launches a persistent kernel that stays on the GPU
 * for the entire duration of the time-stepping loop, synchronizing via
 * a global barrier (Sense-Reversing Barrier) to avoid kernel launch overhead.
 *
 *
 * Problem:
 * - Standard Wavefront/Tiled approaches launch a new kernel for every `STEPS`
 * (e.g., 16) time steps.
 * - For N=10,000, this means ~625 kernel launches.
 * - Kernel launch overhead (~5-10us) accumulates and becomes significant (30%+
 * of runtime).
 *
 * Solution:
 * - Launch a single "Persistent" kernel that stays alive for the entire
 * duration.
 * - Use a Global Software Barrier to synchronize all blocks between time steps.
 * - This eliminates 99% of kernel launch overhead.
 */

#define TILE_SIZE 256
#define VT 4     // Reuse Thread Coarsening (VT=4)
#define STEPS 16 // Steps between global barriers

__device__ double device_callPayoff_persistent(double S, double K) {
  return fmax(S - K, 0.0);
}

__device__ double device_putPayoff_persistent(double S, double K) {
  return fmax(K - S, 0.0);
}

/**
 * @brief Global Software Barrier
 *
 * Synchronizes all blocks in the grid.
 * Only thread 0 of each block participates in the global atomic operations.
 *
 * @param count Pointer to global counter (initialized to 0)
 * @param sense Pointer to global sense variable (initialized to 0)
 * @param num_blocks Total number of blocks in the grid
 */
__device__ void global_sync(int *count, volatile int *sense, int num_blocks) {
  __syncthreads(); // Ensure all threads in block reached this point

  if (threadIdx.x == 0) {
    bool my_sense = *sense == 0 ? false : true; // Read current sense
    // We want to wait until sense flips.
    // If we are the last block, we flip it.

    if (atomicAdd(count, 1) == num_blocks - 1) {
      *count = 0;         // Reset counter
      *sense = !my_sense; // Flip sense
    } else {
      // Spin wait until sense changes
      while ((*sense == 0 ? false : true) == my_sense) {
        // Busy wait
      }
    }
  }

  __syncthreads(); // Wait for thread 0 to signal completion
}

/**
 * @brief Persistent Kernel
 *
 * Loops from t_start down to t_end inside the kernel.
 */
__global__ void persistentKernel(double *V_in, double *V_out,
                                 const double *u_pow, const double *d_pow,
                                 int t_max, int t_min, double S0, double K,
                                 bool isCall, double discount, double p,
                                 int *barrier_count,
                                 volatile int *barrier_sense, int num_blocks) {

  // Shared memory for Tiled+Coarsened logic
  extern __shared__ double shared_mem[];

  // Pointers for double buffering in shared mem
  // Size needed: 2 * (TILE_SIZE * VT)
  // Pointers for double buffering
  double *V_s_in = &shared_mem[0];
  double *V_s_out = &shared_mem[(TILE_SIZE * VT)];

  int tid = threadIdx.x;
  int output_width = (TILE_SIZE * VT) - STEPS;
  // int block_start_idx = blockIdx.x * output_width; // Removed, using
  // Grid-Stride Loop

  // Main Time Loop
  double *curr_global_in = V_in;
  double *curr_global_out = V_out;

  for (int t = t_max; t > t_min; t -= STEPS) {
    int steps =
        (STEPS < (t - t_min)) ? STEPS : (t - t_min); // Steps for this iteration
    int current_t = t;

    // 1. Cooperative Load
    // Grid-Stride Loop over spatial domain
    for (int block_start_idx = blockIdx.x * output_width;
         block_start_idx <= current_t;
         block_start_idx += gridDim.x * output_width) {

      bool is_active_block = block_start_idx <= current_t;

      if (is_active_block) {
        int thread_start_idx = block_start_idx + tid * VT;

#pragma unroll
        for (int i = 0; i < VT; ++i) {
          int global_idx = thread_start_idx + i;
          if (global_idx <= current_t) {
            V_s_in[tid * VT + i] = curr_global_in[global_idx];
          } else {
            V_s_in[tid * VT + i] = 0.0;
          }
        }
      }
      __syncthreads();

      // 2. Compute (Only active blocks)
      if (is_active_block) {
        for (int k = 0; k < steps; ++k) {
          int step_t = current_t - k;

#pragma unroll
          for (int i = 0; i < VT; ++i) {
            int local_idx = tid * VT + i;
            if (local_idx < (TILE_SIZE * VT) - 1 - k) {
              int global_idx = block_start_idx + local_idx;
              if (global_idx <= step_t - 1) {
                double S =
                    S0 * u_pow[global_idx] * d_pow[(step_t - 1) - global_idx];
                double V_hold = discount * (p * V_s_in[local_idx + 1] +
                                            (1.0 - p) * V_s_in[local_idx]);
                double V_exercise = isCall ? device_callPayoff_persistent(S, K)
                                           : device_putPayoff_persistent(S, K);
                V_s_out[local_idx] = fmax(V_hold, V_exercise);
              }
            }
          }
          __syncthreads();

          // Swap shared pointers
          double *temp = V_s_in;
          V_s_in = V_s_out;
          V_s_out = temp;
        }
      }

      // 3. Write Back (Only active blocks)
      if (is_active_block) {
        // Result is in V_s_in (because we swapped after writing to V_s_out)
        double *V_final = V_s_in;
        int final_t = current_t - steps;

#pragma unroll
        for (int i = 0; i < VT; ++i) {
          int local_idx = tid * VT + i;
          if (local_idx < output_width) {
            int global_idx = block_start_idx + local_idx;
            if (global_idx <= final_t) {
              curr_global_out[global_idx] = V_final[local_idx];
            }
          }
        }
      }
      __syncthreads(); // Sync before next tile in grid-stride loop (reuse
                       // shared mem)
    } // End Grid-Stride Loop

    // 4. Global Barrier (ALL BLOCKS MUST PARTICIPATE)
    global_sync(barrier_count, barrier_sense, num_blocks);

    // 5. Swap Global Pointers for next iteration
    double *temp = curr_global_in;
    curr_global_in = curr_global_out;
    curr_global_out = temp;
  }
}

// Helper for terminal values
__global__ void initTerminalValuesKernelPersistent(double *V,
                                                   const double *u_pow,
                                                   const double *d_pow,
                                                   double S0, double K, int N,
                                                   bool isCall) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= N) {
    double S = S0 * u_pow[i] * d_pow[N - i];
    V[i] = isCall ? device_callPayoff_persistent(S, K)
                  : device_putPayoff_persistent(S, K);
  }
}

double priceAmericanOptionCUDAPersistentGlobalBarrier(const OptionParams &opt) {
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

  // Barrier variables
  int *d_barrier_count;
  int *d_barrier_sense;
  CUDA_CHECK(cudaMalloc(&d_barrier_count, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_barrier_sense, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_barrier_count, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_barrier_sense, 0, sizeof(int)));

  // Init
  int threadsPerBlock = 256;
  int blocks = (opt.N + 1 + threadsPerBlock - 1) / threadsPerBlock;
  initTerminalValuesKernelPersistent<<<blocks, threadsPerBlock>>>(
      d_V_in, d_u_pow, d_d_pow, opt.S0, opt.K, opt.N, opt.isCall);
  CUDA_CHECK(cudaGetLastError());

  // Persistent Launch
  // We use a fixed number of blocks to ensure they all fit on the GPU.
  // If we launch more than the GPU can hold, the global barrier will deadlock.
  int num_sms;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  int num_blocks =
      num_sms * 4; // Heuristic: 4 blocks per SM (adjust based on occupancy)

  int tile_size = TILE_SIZE;
  int vt = VT;

  // int num_blocks = (opt.N + output_width - 1) / output_width; // OLD:
  // Dependent on N

  // Shared mem: 2 buffers
  size_t shared_mem_size = 2 * (tile_size * vt) * sizeof(double);

  // We run down to t=0 (or close to it).
  // Let's run fully to 0 for simplicity, or switch to CPU if needed.
  // For persistent, let's go all the way to 0 (or STEPS).

  persistentKernel<<<num_blocks, tile_size, shared_mem_size>>>(
      d_V_in, d_V_out, d_u_pow, d_d_pow, opt.N, 0, opt.S0, opt.K, opt.isCall,
      params.discount, params.p, d_barrier_count, d_barrier_sense, num_blocks);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize()); // Wait for persistent kernel

  // Result is in d_V_in or d_V_out depending on iterations.
  // Total iterations = ceil(N / STEPS).
  // If iterations is even, result in d_V_in (because we started with in,
  // swapped). Wait, loop:
  // 1. Read In, Write Out.
  // 2. Swap (In=Out, Out=In).
  // If 1 iter: Read In, Write Out. Swap. In points to result.
  // So result is always in `curr_global_in` at end of loop?
  // Let's check logic:
  // Start: curr=V_in.
  // Loop: Write to curr_out. Swap (curr=curr_out).
  // End: curr points to valid data.
  // Yes. BUT `curr_global_in` is a local variable in kernel.
  // Host doesn't know which one has result.
  // Host knows total iterations.
  int total_steps = (opt.N + STEPS - 1) / STEPS;
  double *d_result_ptr = (total_steps % 2 == 0) ? d_V_in : d_V_out;
  // Wait, if 1 step: Write to Out. Swap -> curr=Out.
  // So result is in Out.
  // If 2 steps:
  // 1. In->Out. Swap (curr=Out).
  // 2. Out->In. Swap (curr=In).
  // Result in In.
  // So if even steps, result in In. If odd, result in Out.

  double result;
  CUDA_CHECK(cudaMemcpy(&result, &d_result_ptr[0], sizeof(double),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_V_in));
  CUDA_CHECK(cudaFree(d_V_out));
  CUDA_CHECK(cudaFree(d_u_pow));
  CUDA_CHECK(cudaFree(d_d_pow));
  CUDA_CHECK(cudaFree(d_barrier_count));
  CUDA_CHECK(cudaFree(d_barrier_sense));

  return result;
}