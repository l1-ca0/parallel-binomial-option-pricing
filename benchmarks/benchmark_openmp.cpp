#include "../src/common/option.h"
#include "../src/common/timer.h"
#include "../src/openmp/openmp_binomial.h"
#include "../src/serial/serial_binomial.h"
#include <functional>
#include <iomanip>
#include <iostream>
#include <omp.h>
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
  std::vector<int> N_bench = {1000, 5000, 10000, 50000, 100000};

  std::cout
      << "===================================================================="
      << std::endl;
  std::cout << "OpenMP Binomial Option Pricing Benchmark" << std::endl;
  std::cout << "Threads: " << omp_get_max_threads() << std::endl;
  std::cout
      << "===================================================================="
      << std::endl;
  std::cout << std::left << std::setw(35) << "Method" << std::setw(10) << "N"
            << std::setw(15) << "Time (ms)" << std::setw(15) << "Price"
            << std::endl;
  std::cout
      << "--------------------------------------------------------------------"
      << std::endl;

  std::vector<int> thread_counts = {1, 2, 4, 8};
  int max_threads = omp_get_max_threads();

  for (int N : N_bench) {
    // Run Serial Baseline first
    run_benchmark(N, "Serial", [](const OptionParams &opt) {
      return priceAmericanOptionSerial(opt);
    });

    for (int t : thread_counts) {
      if (t > max_threads)
        continue; // Skip if requesting more threads than available hardware

      omp_set_num_threads(t);
      std::string method_name = "OpenMP (" + std::to_string(t) + " threads)";

      run_benchmark(N, method_name, [](const OptionParams &opt) {
        return priceAmericanOptionOpenMP(opt);
      });
    }
  }

  return 0;
}
