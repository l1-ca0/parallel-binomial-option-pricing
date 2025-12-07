#include "openmp_binomial.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

/**
 * OpenMP parallel American option pricer
 *
 * Parallelization strategy:
 * - Parallelize the inner loop at each time step using #pragma omp parallel for
 * - Implicit barrier at the end of each parallel region ensures correctness
 * - Each thread computes a subset of nodes at time step t
 *
 * Performance considerations:
 * - N barriers (one per time step) can add significant overhead
 * - Load imbalance: at t=0, only 1 node; at t=N, N+1 nodes
 * - Memory bandwidth: multiple threads reading/writing shared arrays
 * - Cache effects: each thread's access pattern affects others
 *
 * Scheduling choice:
 * - Static scheduling used for predictable, uniform workload
 * - Each thread gets a contiguous chunk, which is good for cache locality
 * - For this regular computation pattern, static performs better than dynamic
 */

double priceAmericanOptionOpenMP(const OptionParams &opt, int num_threads) {
  // Set number of threads if specified
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  // Compute binomial parameters (includes validation)
  BinomialParams params = computeBinomialParams(opt);

  // Allocate space for option values at current time step
  // We need double buffering to avoid race conditions in the parallel loop
  std::vector<double> V_in(opt.N + 1);
  std::vector<double> V_out(opt.N + 1);

  // Precompute powers of u and d for efficiency
  std::vector<double> u_pow(opt.N + 1); // u^i for i=0..N
  std::vector<double> d_pow(opt.N + 1); // d^i for i=0..N

  u_pow[0] = 1.0;
  d_pow[0] = 1.0;
  for (int i = 1; i <= opt.N; ++i) {
    u_pow[i] = u_pow[i - 1] * params.u;
    d_pow[i] = d_pow[i - 1] * params.d;
  }

// Step 1: Initialize terminal values at expiration (t = N)
// Initialize into V_in (which represents values at time t+1)

// Start parallel region ONCE to avoid repeated thread creation overhead
#pragma omp parallel shared(V_in, V_out, u_pow, d_pow, opt, params)
  {
// Initialize terminal values
#pragma omp for schedule(static)
    for (int i = 0; i <= opt.N; ++i) {
      // Stock price: S0 * u^i * d^(N-i)
      double S = opt.S0 * u_pow[i] * d_pow[opt.N - i];

      // Terminal value = intrinsic value
      if (opt.isCall) {
        V_in[i] = callPayoff(S, opt.K);
      } else {
        V_in[i] = putPayoff(S, opt.K);
      }
    }

    // Step 2: Backward induction from t = N-1 down to t = 0
    for (int t = opt.N - 1; t >= 0; --t) {
// Parallelize across the t+1 nodes at this time step
// Read from V_in (time t+1), write to V_out (time t)
#pragma omp for schedule(static)
      for (int i = 0; i <= t; ++i) {
        // Stock price at node (t, i)
        double S = opt.S0 * u_pow[i] * d_pow[t - i];

        // Continuation value (hold the option)
        // Use V_in which holds values from time t+1
        double V_hold = params.discount *
                        (params.p * V_in[i + 1] + (1.0 - params.p) * V_in[i]);

        // Exercise value (exercise immediately)
        double V_exercise;
        if (opt.isCall) {
          V_exercise = callPayoff(S, opt.K);
        } else {
          V_exercise = putPayoff(S, opt.K);
        }

        // American option: max of hold vs exercise
        V_out[i] = std::max(V_hold, V_exercise);
      }

// Swap buffers: V_out becomes V_in for the next iteration (t-1)
// Only one thread should perform the swap to avoid data races.
// The implicit barrier at the end of 'single' ensures all threads see
// the new pointers before the next iteration.
#pragma omp single
      {
        std::swap(V_in, V_out);
      }
    }
  }

  // After the last swap (at t=0), the result is in V_in[0]
  // (Because we computed into V_out, then swapped, so V_in has the result)
  return V_in[0];
}

/**
 * OpenMP parallel European option pricer
 * Similar to American but without early exercise check
 */
double priceEuropeanOptionOpenMP(const OptionParams &opt, int num_threads) {
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  BinomialParams params = computeBinomialParams(opt);
  std::vector<double> V(opt.N + 1);

  // Precompute powers
  std::vector<double> u_pow(opt.N + 1);
  std::vector<double> d_pow(opt.N + 1);
  u_pow[0] = 1.0;
  d_pow[0] = 1.0;
  for (int i = 1; i <= opt.N; ++i) {
    u_pow[i] = u_pow[i - 1] * params.u;
    d_pow[i] = d_pow[i - 1] * params.d;
  }

// Initialize terminal values
#pragma omp parallel for schedule(static) shared(V, u_pow, d_pow, opt, params)
  for (int i = 0; i <= opt.N; ++i) {
    double S = opt.S0 * u_pow[i] * d_pow[opt.N - i];
    if (opt.isCall) {
      V[i] = callPayoff(S, opt.K);
    } else {
      V[i] = putPayoff(S, opt.K);
    }
  }

  // Backward induction - European (no early exercise)
  for (int t = opt.N - 1; t >= 0; --t) {
#pragma omp parallel for schedule(static) shared(V, params, t)
    for (int i = 0; i <= t; ++i) {
      // European: only continuation value (no max with exercise)
      V[i] = params.discount * (params.p * V[i + 1] + (1.0 - params.p) * V[i]);
    }
  }

  return V[0];
}

/**
 * Get the actual number of threads OpenMP is using
 */
int getOMPThreadCount() {
  int num_threads = 0;
#pragma omp parallel
  {
#pragma omp single
    num_threads = omp_get_num_threads();
  }
  return num_threads;
}
/**
 * OpenMP parallel American option pricer using Dynamic Scheduling
 * Used for benchmarking purposes to compare against Static scheduling.
 */
double priceAmericanOptionOpenMPDynamic(const OptionParams &opt, int num_threads) {
  // Set number of threads if specified
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  // Compute binomial parameters (includes validation)
  BinomialParams params = computeBinomialParams(opt);

  // Allocate space for option values at current time step
  std::vector<double> V_in(opt.N + 1);
  std::vector<double> V_out(opt.N + 1);

  // Precompute powers of u and d for efficiency
  std::vector<double> u_pow(opt.N + 1); // u^i for i=0..N
  std::vector<double> d_pow(opt.N + 1); // d^i for i=0..N
  
  u_pow[0] = 1.0;
  d_pow[0] = 1.0;
  for (int i = 1; i <= opt.N; ++i) {
    u_pow[i] = u_pow[i - 1] * params.u;
    d_pow[i] = d_pow[i - 1] * params.d;
  }

  // Hoisted parallel region
  #pragma omp parallel shared(V_in, V_out, u_pow, d_pow, opt, params)
  {
    // Step 1: Initialize terminal values (Dynamic Scheduling)
    #pragma omp for schedule(dynamic, 1024)
    for (int i = 0; i <= opt.N; ++i) {
      double S = opt.S0 * u_pow[i] * d_pow[opt.N - i];
      if (opt.isCall) {
        V_in[i] = callPayoff(S, opt.K);
      } else {
        V_in[i] = putPayoff(S, opt.K);
      }
    }

    // Step 2: Backward induction
    for (int t = opt.N - 1; t >= 0; --t) {
      // Dynamic Scheduling for time stepping
      #pragma omp for schedule(dynamic, 1024)
      for (int i = 0; i <= t; ++i) {
        double S = opt.S0 * u_pow[i] * d_pow[t - i];

        double V_hold = params.discount *
                        (params.p * V_in[i + 1] + (1.0 - params.p) * V_in[i]);

        double V_exercise;
        if (opt.isCall) {
          V_exercise = callPayoff(S, opt.K);
        } else {
          V_exercise = putPayoff(S, opt.K);
        }

        V_out[i] = std::max(V_hold, V_exercise);
      }

      // Swap buffers (Single thread)
      #pragma omp single
      { std::swap(V_in, V_out); }
    }
  }

  return V_in[0];
}
