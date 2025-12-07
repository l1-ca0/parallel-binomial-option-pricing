#include "../src/common/option.h"
#include "../src/common/timer.h"
#include "../src/serial/serial_binomial.h"
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void run_benchmark(int N, const std::string &method_name,
                   std::function<double(const OptionParams &)> pricer) {
  OptionParams opt;
  opt.S0 = 100.0;
  opt.K = 100.0;
  opt.r = 0.05;
  opt.sigma = 0.2;
  opt.T = 10.0;
  opt.N = N;
  opt.isCall = false;

  // Warmup
  pricer(opt);

  // Measure
  Timer timer;
  timer.start();
  double price = pricer(opt);
  timer.stop();

  std::cout << std::left << std::setw(35) << method_name << std::setw(10) << N
            << std::setw(15) << std::fixed << std::setprecision(6)
            << timer.elapsed_sec() * 1000.0 << price << std::endl;
}

int main() {
  std::vector<int> N_bench = {1000, 5000, 10000, 50000}; 

  std::cout
      << "===================================================================="
      << std::endl;
  std::cout << "Serial Binomial Option Pricing Benchmark" << std::endl;
  std::cout
      << "===================================================================="
      << std::endl;
  std::cout << std::left << std::setw(35) << "Method" << std::setw(10) << "N"
            << std::setw(15) << "Time (ms)" << std::setw(15) << "Price"
            << std::endl;
  std::cout
      << "--------------------------------------------------------------------"
      << std::endl;

  for (int N : N_bench) {
    run_benchmark(N, "Serial", [](const OptionParams &opt) {
      return priceAmericanOptionSerial(opt);
    });
  }

  return 0;
}
