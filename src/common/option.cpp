#include "option.h"
#include <cmath>

BinomialParams computeBinomialParams(const OptionParams &opt) {
  BinomialParams params;
  params.dt = opt.T / opt.N;
  params.u = std::exp(opt.sigma * std::sqrt(params.dt));
  params.d = 1.0 / params.u;
  params.p = (std::exp(opt.r * params.dt) - params.d) / (params.u - params.d);
  params.discount = std::exp(-opt.r * params.dt);
  return params;
}

double callPayoff(double S, double K) { return std::fmax(S - K, 0.0); }

double putPayoff(double S, double K) { return std::fmax(K - S, 0.0); }