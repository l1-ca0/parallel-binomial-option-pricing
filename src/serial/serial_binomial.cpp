#include "serial_binomial.h"
#include <algorithm>
#include <cmath>
#include <vector>

/**
 * Serial American option pricer using binomial tree
 *
 * Algorithm:
 * 1. Forward pass: Build price lattice (implicit - we calculate prices on the
 * fly)
 * 2. Backward pass: Calculate option values via backward induction
 *
 * Memory optimization: Use only O(N) space by keeping only current time step
 * ("wavefront" approach - only store values at current time step)
 */
double priceAmericanOptionSerial(const OptionParams &opt) {
  // Compute binomial parameters
  BinomialParams params = computeBinomialParams(opt);

  // Allocate space for option values at current time step
  // At time t, we have t+1 nodes (indexed 0 to t)
  // We only need to store the current wavefront
  std::vector<double> V(opt.N + 1); // Values at time step t

  // Step 1: Initialize terminal values at expiration (t = N)
  // At expiration, option value = intrinsic value
  for (int i = 0; i <= opt.N; ++i) {
    // Calculate stock price at node (N, i)
    // After N time steps: i up-moves and (N-i) down-moves
    double S = opt.S0 * std::pow(params.u, i) * std::pow(params.d, opt.N - i);

    // Terminal value = intrinsic value (exercise immediately)
    if (opt.isCall) {
      V[i] = callPayoff(S, opt.K);
    } else {
      V[i] = putPayoff(S, opt.K);
    }
  }

  // Step 2: Backward induction from t = N-1 down to t = 0
  for (int t = opt.N - 1; t >= 0; --t) {
    // At time t, we have t+1 nodes (i = 0, 1, ..., t)
    for (int i = 0; i <= t; ++i) {
      // Calculate stock price at node (t, i)
      double S = opt.S0 * std::pow(params.u, i) * std::pow(params.d, t - i);

      // Calculate continuation value (hold the option)
      // V_hold = discounted expected value of the two child nodes
      double V_hold =
          params.discount * (params.p * V[i + 1] + (1.0 - params.p) * V[i]);

      // Calculate exercise value (exercise immediately)
      double V_exercise;
      if (opt.isCall) {
        V_exercise = callPayoff(S, opt.K);
      } else {
        V_exercise = putPayoff(S, opt.K);
      }

      // American option: take maximum of hold vs exercise
      // This is the key difference from European options
      V[i] = std::max(V_hold, V_exercise);
    }

    // Note: V now contains values at time t (for next iteration)
    // The array naturally overwrites the previous time step
  }

  // After backward induction, V[0] contains the option value at t=0
  return V[0];
}

/**
 * Serial European option pricer (for validation)
 * European options can only be exercised at expiration, so no early exercise
 * check
 */
double priceEuropeanOptionSerial(const OptionParams &opt) {
  BinomialParams params = computeBinomialParams(opt);
  std::vector<double> V(opt.N + 1);

  // Initialize terminal values
  for (int i = 0; i <= opt.N; ++i) {
    double S = opt.S0 * std::pow(params.u, i) * std::pow(params.d, opt.N - i);
    if (opt.isCall) {
      V[i] = callPayoff(S, opt.K);
    } else {
      V[i] = putPayoff(S, opt.K);
    }
  }

  // Backward induction - no early exercise check for European
  for (int t = opt.N - 1; t >= 0; --t) {
    for (int i = 0; i <= t; ++i) {
      // European option: only continuation value (no max with exercise)
      V[i] = params.discount * (params.p * V[i + 1] + (1.0 - params.p) * V[i]);
    }
  }

  return V[0];
}