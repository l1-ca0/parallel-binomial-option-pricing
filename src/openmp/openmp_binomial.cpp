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
 */
double priceAmericanOptionOpenMP(const OptionParams &opt, int num_threads) {
  // Set number of threads if specified
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }

  // Compute binomial parameters
  BinomialParams params = computeBinomialParams(opt);

  // Allocate space for option values at current time step
  std::vector<double> V(opt.N + 1);

  // Precompute powers of u and d for efficiency
  // This avoids redundant pow() calls in the parallel loop
  std::vector<double> u_pow(opt.N + 1); // u^i for i=0..N
  std::vector<double> d_pow(opt.N + 1); // d^i for i=0..N

  u_pow[0] = 1.0;
  d_pow[0] = 1.0;
  for (int i = 1; i <= opt.N; ++i) {
    u_pow[i] = u_pow[i - 1] * params.u;
    d_pow[i] = d_pow[i - 1] * params.d;
  }

// Step 1: Initialize terminal values at expiration (t = N)
// This can be parallelized since all computations are independent
#pragma omp parallel for schedule(static)
  for (int i = 0; i <= opt.N; ++i) {
    // Stock price: S0 * u^i * d^(N-i)
    double S = opt.S0 * u_pow[i] * d_pow[opt.N - i];

    // Terminal value = intrinsic value
    if (opt.isCall) {
      V[i] = callPayoff(S, opt.K);
    } else {
      V[i] = putPayoff(S, opt.K);
    }
  }
  // Implicit barrier here ensures all terminal values are computed

  // Step 2: Backward induction from t = N-1 down to t = 0
  for (int t = opt.N - 1; t >= 0; --t) {
// Parallelize across the t+1 nodes at this time step
// Note: As t decreases, parallelism declines (load imbalance)
#pragma omp parallel for schedule(static)
    for (int i = 0; i <= t; ++i) {
      // Stock price at node (t, i)
      double S = opt.S0 * u_pow[i] * d_pow[t - i];

      // Continuation value (hold the option)
      double V_hold =
          params.discount * (params.p * V[i + 1] + (1.0 - params.p) * V[i]);

      // Exercise value (exercise immediately)
      double V_exercise;
      if (opt.isCall) {
        V_exercise = callPayoff(S, opt.K);
      } else {
        V_exercise = putPayoff(S, opt.K);
      }

      // American option: max of hold vs exercise
      V[i] = std::max(V_hold, V_exercise);
    }
    // Implicit barrier at end of parallel region
    // Critical: ensures all V[i] at time t are computed before moving to t-1
  }

  return V[0];
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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