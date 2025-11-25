#include "option.h"
#include <cmath>
#include <stdexcept> // For exception types
#include <string>    // For std::to_string

BinomialParams computeBinomialParams(const OptionParams &opt) {
  // Validate input parameters
  if (opt.N <= 0) {
    throw std::invalid_argument("N (number of time steps) must be positive");
  }
  if (opt.T <= 0.0) {
    throw std::invalid_argument("T (time to maturity) must be positive");
  }
  if (opt.sigma <= 0.0) {
    throw std::invalid_argument("sigma (volatility) must be positive");
  }
  if (opt.S0 <= 0.0) {
    throw std::invalid_argument("S0 (initial stock price) must be positive");
  }
  if (opt.K <= 0.0) {
    throw std::invalid_argument("K (strike price) must be positive");
  }

  BinomialParams params;
  params.dt = opt.T / opt.N;
  params.u = std::exp(opt.sigma * std::sqrt(params.dt));
  params.d = 1.0 / params.u;

  // Check for numerical issues
  double denominator = params.u - params.d;
  if (std::abs(denominator) < 1e-10) {
    throw std::runtime_error(
        "Invalid binomial parameters: u ≈ d (numerical instability)");
  }

  params.p = (std::exp(opt.r * params.dt) - params.d) / denominator;

  // Validate risk-neutral probability
  // If p is outside [0,1], it indicates arbitrage or invalid parameters
  if (params.p < -1e-10 || params.p > 1.0 + 1e-10) {
    throw std::runtime_error(
        "Invalid risk-neutral probability: p = " + std::to_string(params.p) +
        " (must be in [0,1]). This indicates arbitrage or invalid parameters. "
        "Check that r, sigma, and T are reasonable values.");
  }

  // Clamp to [0,1] in case of small numerical errors
  if (params.p < 0.0)
    params.p = 0.0;
  if (params.p > 1.0)
    params.p = 1.0;

  params.discount = std::exp(-opt.r * params.dt);

  return params;
}

double callPayoff(double S, double K) { return std::max(S - K, 0.0); }

double putPayoff(double S, double K) { return std::max(K - S, 0.0); }