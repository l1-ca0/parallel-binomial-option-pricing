#ifndef OPTION_H
#define OPTION_H

struct OptionParams {
  double S0;    // Initial stock price
  double K;     // Strike price
  double r;     // Risk-free rate
  double sigma; // Volatility
  double T;     // Time to maturity
  int N;        // Number of time steps
  bool isCall;  // true for call, false for put
};

// Risk-neutral probabilities
struct BinomialParams {
  double u;        // Up factor
  double d;        // Down factor
  double p;        // Risk-neutral probability
  double dt;       // Time step size
  double discount; // Discount factor per step
};

// Calculate binomial parameters from option params
BinomialParams computeBinomialParams(const OptionParams &opt);

// Payoff functions
double callPayoff(double S, double K);
double putPayoff(double S, double K);

#endif // OPTION_H