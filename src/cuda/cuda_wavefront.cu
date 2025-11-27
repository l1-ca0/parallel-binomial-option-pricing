#include "cuda_utils.cuh" 
#include "cuda_kernels.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

/**
 * @file cuda_wavefront.cu
 * @brief Naive Wavefront Implementation
 *
 * =================================================================================================
 * ALGORITHM DETAILS
 * =================================================================================================
 * This implementation uses a "Wavefront" or level-by-level parallelization
 * strategy for the Binomial Option Pricing model.
 *
 * Mathematical Model:
 * - The binomial tree is traversed backwards from t = N down to t = 0.
 * - At each time step t, there are t+1 nodes (indices i = 0 to t).
 * - The value V(t, i) depends on V(t+1, i) and V(t+1, i+1).
 * - V(t, i) = max( Payoff(S(t, i)), Discount * (p * V(t+1, i+1) + (1-p) *
 * V(t+1, i)) )
 *
 * Parallelization Approach:
 * - We parallelize the loop over 'i' at each time step 't'.
 * - Each thread computes one node V(t, i).
 * - A kernel is launched for EACH time step t (from N-1 down to 0).
 * - This requires N kernel launches, which incurs significant overhead for
 * large N.
 *
 * =================================================================================================
 * MEMORY USAGE & DATA STRUCTURES
 * =================================================================================================
 * Global Memory:
 * - V_in  (size N+1): Stores option values at time t+1 (Input to kernel).
 * - V_out (size N+1): Stores option values at time t   (Output from kernel).
 * - u_pow (size N+1): Precomputed powers of u (u^0, u^1, ... u^N).
 * - d_pow (size N+1): Precomputed powers of d (d^0, d^1, ... d^N).
 *
 * Shared Memory:
 * - Not used in this naive implementation. All data access goes to Global
 * Memory.
 *
 * Data Flow:
 * - We use Double Buffering (Ping-Pong) to avoid race conditions.
 * - In step t, we read from V_in (holding t+1 data) and write to V_out (holding
 * t data).
 * - After the kernel, we swap pointers (V_in <-> V_out) on the host.
 *
 * Complexity:
 * - Time: O(N^2 / P) where P is number of processors (idealized).
 * - Space: O(N) global memory.
 * - Bandwidth: High. We read 2 doubles and write 1 double per node per step.
 *   Total global memory transactions: ~3 * (N^2/2) * sizeof(double).
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
__device__ double device_callPayoff(double S, double K) {
  return fmax(S - K, 0.0);
}

__device__ double device_putPayoff(double S, double K) {
  return fmax(K - S, 0.0);
}

/**
 * @brief Kernel to initialize terminal values at t = N
 */
__global__ void initTerminalValuesKernel(double *V, const double *u_pow,
                                         const double *d_pow, double S0,
                                         double K, int N, bool isCall) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= N) {
    double S = S0 * u_pow[i] * d_pow[N - i];
    if (isCall) {
      V[i] = device_callPayoff(S, K);
    } else {
      V[i] = device_putPayoff(S, K);
    }
  }
}

/**
 * @brief Wavefront kernel for one time step t
 *
 * Computes V[i] at time t using V[i] and V[i+1] from time t+1.
 * Note: We can overwrite V[i] in place because V[i] at time t depends on V[i]
 * and V[i+1] at time t+1. If we process from i = 0 to t, we read V[i] (old) and
 * V[i+1] (old) and write V[i] (new). This is safe because V[i+1] is not
 * modified by thread i, and thread i+1 (which modifies V[i+1]) doesn't need the
 * old V[i]. Wait, thread i needs V[i] and V[i+1]. Thread i-1 needs V[i-1] and
 * V[i]. If we overwrite V[i], thread i-1 might read the NEW V[i] instead of the
 * OLD V[i] if there's a race.
 *
 * ACTUALLY: In the serial version:
 *   V[i] = ... V[i+1] ... V[i]
 * We iterate i from 0 to t.
 * When we compute V[0], we use V[0] and V[1].
 * When we compute V[1], we use V[1] and V[2].
 *
 * If we parallelize this:
 * Thread i reads V[i] and V[i+1], writes V[i].
 * Thread i-1 reads V[i-1] and V[i], writes V[i-1].
 *
 * There is a RACE CONDITION if we do this in-place with a single array if we
 * are not careful? No, because thread i writes to V[i]. Thread i-1 reads V[i].
 * If thread i writes V[i] BEFORE thread i-1 reads V[i], then thread i-1 gets
 * the wrong value (new instead of old).
 *
 * SO WE NEED DOUBLE BUFFERING or separate input/output arrays for the wavefront
 * approach to be safe in parallel. Or we can use `__syncthreads()` if we were
 * in a block, but this is global memory.
 *
 * CORRECTION: The "Wavefront" approach usually implies we might need two
 * buffers (ping-pong) or be very careful. Let's use Double Buffering (V_in and
 * V_out) to be safe and correct.
 */
__global__ void wavefrontStepKernel(const double *V_in, double *V_out,
                                    const double *u_pow, const double *d_pow,
                                    int t, double S0, double K, bool isCall,
                                    double discount, double p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i <= t) {
    // Calculate S at node (t, i)
    double S = S0 * u_pow[i] * d_pow[t - i];

    // Continuation value
    // V_in has values from time t+1
    double V_hold = discount * (p * V_in[i + 1] + (1.0 - p) * V_in[i]);

    // Exercise value
    double V_exercise;
    if (isCall) {
      V_exercise = device_callPayoff(S, K);
    } else {
      V_exercise = device_putPayoff(S, K);
    }

    // American option max
    V_out[i] = fmax(V_hold, V_exercise);
  }
}

double priceAmericanOptionCUDAWavefront(const OptionParams &opt) {
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

  // We need two buffers for V to avoid race conditions
  CUDA_CHECK(cudaMalloc(&d_V_in, (opt.N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_V_out, (opt.N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_u_pow, (opt.N + 1) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_d_pow, (opt.N + 1) * sizeof(double)));

  // Copy powers to device
  CUDA_CHECK(cudaMemcpy(d_u_pow, h_u_pow.data(), (opt.N + 1) * sizeof(double),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_d_pow, h_d_pow.data(), (opt.N + 1) * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Initialize terminal values (at t=N) into d_V_in
  // We treat d_V_in as the "current" valid values (initially at t=N)
  int threadsPerBlock = 256;
  int blocks = (opt.N + 1 + threadsPerBlock - 1) / threadsPerBlock;

  initTerminalValuesKernel<<<blocks, threadsPerBlock>>>(
      d_V_in, d_u_pow, d_d_pow, opt.S0, opt.K, opt.N, opt.isCall);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Loop from t = N-1 down to 0
  for (int t = opt.N - 1; t >= 0; --t) {
    blocks = (t + 1 + threadsPerBlock - 1) / threadsPerBlock;

    // Compute t from t+1 (d_V_in) and store in d_V_out
    wavefrontStepKernel<<<blocks, threadsPerBlock>>>(
        d_V_in, d_V_out, d_u_pow, d_d_pow, t, opt.S0, opt.K, opt.isCall,
        params.discount, params.p);
    CUDA_CHECK(cudaGetLastError());

    // Swap pointers: d_V_out becomes the input for the next step
    std::swap(d_V_in, d_V_out);
  }

  // Result is now in d_V_in (because we swapped after the last step t=0)
  // Wait, if t=0, we computed into d_V_out, then swapped, so d_V_in has the
  // result. Correct.

  double result;
  CUDA_CHECK(
      cudaMemcpy(&result, &d_V_in[0], sizeof(double), cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_V_in));
  CUDA_CHECK(cudaFree(d_V_out));
  CUDA_CHECK(cudaFree(d_u_pow));
  CUDA_CHECK(cudaFree(d_d_pow));

  return result;
}
