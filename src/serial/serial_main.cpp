#include "../common/timer.h"
#include "../common/validator.h"
#include "serial_binomial.h"
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
  // Default parameters
  OptionParams opt;
  opt.S0 = 100.0;
  opt.K = 100.0;
  opt.r = 0.05;
  opt.sigma = 0.2;
  opt.T = 1.0;
  opt.N = 1000;
  opt.isCall = false;

  if (argc > 1) {
    opt.N = std::atoi(argv[1]);
  }

  std::cout << "=== Serial Binomial Option Pricer ===" << std::endl;
  Validator::printOptionDetails(opt);

  // Price American option
  Timer timer;
  timer.start();
  double american_price = priceAmericanOptionSerial(opt);
  timer.stop();

  std::cout << "Results:" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  American Option Price: " << american_price << std::endl;
  timer.print("  Execution Time");

  long long total_nodes = static_cast<long long>(opt.N + 1) * (opt.N + 2) / 2;
  std::cout << "  Nodes Computed:        " << total_nodes << std::endl;
  std::cout << "  Throughput:            " << std::setprecision(2)
            << (static_cast<double>(total_nodes) / timer.elapsed_ms())
            << " nodes/ms" << std::endl;
  std::cout << std::endl;

  // Compare with European
  timer.start();
  double european_price = priceEuropeanOptionSerial(opt);
  timer.stop();
  std::cout << "  European Option Price: " << std::setprecision(6)
            << european_price << std::endl;
  timer.print("  European Time");
  std::cout << "  Early Exercise Value:  " << (american_price - european_price)
            << std::endl;
  std::cout << std::endl;

  // Validation
  Validator::runValidationSuite(american_price, opt, "Serial");
  Validator::validateAmericanVsEuropean(american_price, european_price, opt);

  return 0;
}