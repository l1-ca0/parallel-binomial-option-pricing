#include "cuda_tiled.cuh"
#include "cuda_utils.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/**
 * @file cuda_tiled.cu
 * @brief Optimized Tiled Implementation
 *
 * =================================================================================================
 * ALGORITHM DETAILS
 * =================================================================================================
 * This implementation uses a "Tiled" or "Block-Iterative" approach to improve
 * memory bandwidth efficiency.
 *
 * Problem:
 * - The Naive Wavefront approach reads 2 doubles and writes 1 double from
 * Global Memory for EVERY node update.
 * - This makes the algorithm memory-bound.
 *
 * Solution (Tiling):
 * - We load a chunk of data (Tile) from Global Memory into Shared Memory (L1
 * Cache).
 * - We perform multiple time steps (STEPS_PER_KERNEL) entirely within Shared
 * Memory.
 * - We write the valid results back to Global Memory.
 *
 * Dependency Handling:
 * - To compute V(t, i), we need V(t+1, i) and V(t+1, i+1).
 * - To compute 'k' steps, the dependency cone widens.
 * - If we load a tile of size TILE_SIZE, after 1 step, we have TILE_SIZE-1
 * valid values.
 * - After 'k' steps, we have TILE_SIZE-k valid values.
 * - This "Halo" effect means we load more data than we output, but the ratio of
 *   (Compute Ops / Global Memory Access) is significantly improved.
 *
 * =================================================================================================
 * MEMORY USAGE & DATA STRUCTURES
 * =================================================================================================
 * Global Memory:
 * - V_in, V_out: Double buffered arrays for full tree state.
 * - u_pow, d_pow: Precomputed powers.
 *
 * Shared Memory:
 * - V_s_in  (size TILE_SIZE): Input buffer for the current step within the
 * block.
 * - V_s_out (size TILE_SIZE): Output buffer for the current step within the
 * block.
 * - We use Double Buffering in Shared Memory to avoid race conditions between
 * threads in the same block.
 *
 * Parallelization:
 * - Grid of Blocks covers the array at time t.
 * - Each Block loads TILE_SIZE elements.
 * - Each Block advances STEPS_PER_KERNEL steps.
 * - Each Block writes back (TILE_SIZE - STEPS_PER_KERNEL) elements.
 *
 * Complexity:
 * - Bandwidth Reduction: ~Factor of STEPS_PER_KERNEL (ignoring halo overhead).
 * - Performance: Significantly faster than Wavefront for large N.
 * =================================================================================================
 */

// Error handling macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__,          \
              __LINE__, err, cudaGetErrorString(err));                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Device helper for payoff
__device__ double device_callPayoff_tiled(double S, double K) {
  return fmax(S - K, 0.0);
}

__device__ double device_putPayoff_tiled(double S, double K) {
  return fmax(K - S, 0.0);
}

// Constants for tiling
#define TILE_SIZE 256 // Size of the tile in shared memory (number of threads)
#define STEPS_PER_KERNEL                                                       \
  16 // Number of time steps to compute per kernel launch (or per block
     // iteration)

/**
 * @brief Tiled kernel for multiple time steps.
 *
 * Implementation Strategy:
 * This kernel employs a block-iterative tiling approach to process multiple
 * time steps within a single kernel launch, reducing global memory bandwidth
 * usage.
 *
 * Dependency Management:
 * The binomial lattice structure implies that computing V[i] at time t requires
 * V[i] and V[i+1] at time t+1. Consequently, the dependency cone widens with
 * each time step. To compute a valid block of values after `k` steps, the
 * initial data load must include a halo region.
 *
 * Overlapping Blocks Strategy:
 * - The kernel computes values at `t_end = t_start - STEPS`.
 * - Blocks are launched to cover the domain at `t_end`.
 * - Each block computes a segment of size `OUTPUT_TILE_SIZE`.
 * - To satisfy dependencies, each block loads `OUTPUT_TILE_SIZE + STEPS`
 * elements from `t_start`.
 * - `TILE_SIZE` (threads) is set to `OUTPUT_TILE_SIZE + STEPS`.
 *
 * Example Configuration:
 * - STEPS = 16
 * - TILE_SIZE = 256
 * - OUTPUT_TILE_SIZE = 240
 * - Block `b` computes indices `[b*240, (b+1)*240 - 1]` at `t_end`.
 * - Block `b` loads indices `[b*240, b*240 + 255]` from `t_start`.
 */
__global__ void tiledKernel(const double *V_in, double *V_out,
                            const double *u_pow, const double *d_pow,
                            int t_start, int steps, double S0, double K,
                            bool isCall, double discount, double p) {

  // Allocate shared memory for double buffering.
  // Double buffering is required to prevent race conditions where a thread
  // updates a value that is still needed by a neighbor in the current step.

  extern __shared__ double shared_mem[];
  double *V_s_in = shared_mem;
  double *V_s_out = shared_mem + TILE_SIZE;

  int tid = threadIdx.x;
  int output_tile_size = TILE_SIZE - steps;

  // Global index of the first element this block is responsible for OUTPUTTING
  int block_output_start = blockIdx.x * output_tile_size;

  // Global index this thread loads
  int global_idx = block_output_start + tid;

  // 1. Load data from global memory (at time t_start)
  // Boundary check
  if (global_idx <= t_start) {
    V_s_in[tid] = V_in[global_idx];
  } else {
    V_s_in[tid] = 0.0; // Padding
  }
  __syncthreads();

  // 2. Perform `steps` iterations in shared memory
  for (int k = 0; k < steps; ++k) {
    // Time step being computed from.
    int current_t = t_start - k;

    // The valid data range in shared memory decreases by 1 element per step
    // due to the dependency cone.
    // Step 0: [0, TILE_SIZE-1]
    // Step k: [0, TILE_SIZE - 1 - k]

    // Only threads within the valid range for THIS step need to work
    if (tid < TILE_SIZE - 1 - k) {
      // Verify that the node index is within the valid range for the current
      // time step.
      if (global_idx <= current_t - 1) {
        // Calculate stock price S at node (current_t - 1, global_idx) for early
        // exercise check.

        double S = S0 * u_pow[global_idx] * d_pow[(current_t - 1) - global_idx];

        double V_hold =
            discount * (p * V_s_in[tid + 1] + (1.0 - p) * V_s_in[tid]);

        double V_exercise;
        if (isCall) {
          V_exercise = device_callPayoff_tiled(S, K);
        } else {
          V_exercise = device_putPayoff_tiled(S, K);
        }

        V_s_out[tid] = fmax(V_hold, V_exercise);
      }
    }
    __syncthreads();

    // Swap shared memory pointers
    double *temp = V_s_in;
    V_s_in = V_s_out;
    V_s_out = temp;
  }

  // 3. Write result back to global memory (at time t_start - steps)
  // Only threads in the valid OUTPUT range write back
  if (tid < output_tile_size) {
    int final_t = t_start - steps;
    if (global_idx <= final_t) {
      V_out[global_idx] = V_s_in[tid]; // Result is in V_s_in after swap
    }
  }
}

// Helper to initialize terminal values (same as wavefront, reused or copied)
__global__ void initTerminalValuesKernelTiled(double *V, const double *u_pow,
                                              const double *d_pow, double S0,
                                              double K, int N, bool isCall) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= N) {
    double S = S0 * u_pow[i] * d_pow[N - i];
    if (isCall) {
      V[i] = device_callPayoff_tiled(S, K);
    } else {
      V[i] = device_putPayoff_tiled(S, K);
    }
  }
}

double priceAmericanOptionCUDATiled(const OptionParams &opt) {
  BinomialParams params = computeBinomialParams(opt);

  // Precompute powers on host
  std::vector<double> h_u_pow(opt.N + 1);
  std::vector<double> h_d_pow(opt.N + 1);
  h_u_pow[0] = 1.0;
  h_d_pow[0] = 1.0;
  for (int i = 1; i <= opt.N; ++i) {
    h_u_pow[i] = h_u_pow[i - 1] * params.u;
    h_d_pow[i] = h_d_pow[i - 1] * params.d;
  }

  // Allocate device memory
  double *d_V_in, *d_V_out;
  double *d_u_pow, *d_d_pow;

  CUDA_CHECK(cudaMalloc(&d_V_in, (opt.N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_V_out, (opt.N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_u_pow, (opt.N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_d_pow, (opt.N + 1) * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_u_pow, h_u_pow.data(), (opt.N + 1) * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_d_pow, h_d_pow.data(), (opt.N + 1) * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Initialize terminal values
  int threadsPerBlock = 256;
  int blocks = (opt.N + 1 + threadsPerBlock - 1) / threadsPerBlock;
  initTerminalValuesKernelTiled<<<blocks, threadsPerBlock>>>(
      d_V_in, d_u_pow, d_d_pow, opt.S0, opt.K, opt.N, opt.isCall);
  CUDA_CHECK(cudaGetLastError());

  // Main loop
  // We process `STEPS_PER_KERNEL` steps at a time
  int current_t = opt.N;
  int steps_per_kernel = STEPS_PER_KERNEL;
  int tile_size = TILE_SIZE;
  int output_tile_size = tile_size - steps_per_kernel; // 240

  while (current_t > 0) {
    int steps = std::min(steps_per_kernel, current_t);

    // Adjust output tile size based on the number of steps in this iteration.
    // If steps < STEPS_PER_KERNEL (e.g., last iteration), the output tile size
    // increases effectively, as fewer halo elements are consumed.

    int current_output_tile_size = tile_size - steps;
    int num_blocks =
        (current_t + current_output_tile_size - 1) / current_output_tile_size;

    // Shared memory needed: 2 * TILE_SIZE * sizeof(double)
    size_t shared_mem_size = 2 * tile_size * sizeof(double);

    tiledKernel<<<num_blocks, tile_size, shared_mem_size>>>(
        d_V_in, d_V_out, d_u_pow, d_d_pow, current_t, steps, opt.S0, opt.K,
        opt.isCall, params.discount, params.p);
    CUDA_CHECK(cudaGetLastError());

    // Swap pointers
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
