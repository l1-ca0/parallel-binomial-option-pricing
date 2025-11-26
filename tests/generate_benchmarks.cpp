#include "../src/common/option.h"
#include "../src/serial/serial_binomial.h"
#include <iomanip>
#include <iostream>
#include <vector>

struct TestCase {
  double S0;
  double K;
  double r;
  double sigma;
  double T;
  int N_ref; // Reference N for generation
};

int main() {
  std::vector<TestCase> cases = {
      {100.0, 100.0, 0.05, 0.20, 1.0, 20000},    // Case 1
      {90.0, 100.0, 0.05, 0.20, 1.0, 20000},     // Case 2
      {110.0, 100.0, 0.05, 0.20, 1.0, 20000},    // Case 3
      {100.0, 100.0, 0.05, 0.40, 1.0, 20000},    // Case 4
      {100.0, 100.0, 0.05, 0.10, 1.0, 20000},    // Case 5
      {100.0, 100.0, 0.05, 0.20, 0.25, 20000},   // Case 6
      {100.0, 100.0, 0.05, 0.20, 3.0, 20000},    // Case 7
      {100.0, 100.0, 0.00, 0.20, 1.0, 20000},    // Case 8
      {100.0, 100.0, 0.10, 0.20, 1.0, 20000},    // Case 9
      {80.0, 100.0, 0.05, 0.20, 1.0, 20000},     // Case 10
      {120.0, 100.0, 0.05, 0.20, 1.0, 20000},    // Case 11
      {100.0, 100.0, 0.05, 0.60, 1.0, 20000},    // Case 13
      {100.0, 100.0, 0.05, 0.20, 0.0833, 20000}, // Case 14
      {100.0, 100.0, 0.05, 0.20, 5.0, 20000}     // Case 15
  };

  std::cout << std::fixed << std::setprecision(4);

  for (const auto &test : cases) {
    OptionParams opt;
    opt.S0 = test.S0;
    opt.K = test.K;
    opt.r = test.r;
    opt.sigma = test.sigma;
    opt.T = test.T;
    opt.N = test.N_ref;
    opt.isCall = false;

    double price = priceAmericanOptionSerial(opt);

    std::cout << test.S0 << ", " << test.K << ", " << test.r << ", "
              << test.sigma << ", " << test.T << ", " << price << std::endl;
  }

  return 0;
}
