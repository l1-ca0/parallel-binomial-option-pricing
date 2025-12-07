#include "../src/common/option.h"
#include "../src/common/timer.h"
#include "../src/openmp/openmp_binomial.h"
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

// Helper to run benchmark and output to stream
void run_benchmark(int N, const std::string &method_name,
                   std::function<double(const OptionParams &)> pricer,
                   std::ostream &out) {
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

  out << std::left << std::setw(40) << method_name << std::setw(10) << N
      << std::setw(15) << std::fixed << std::setprecision(6)
      << timer.elapsed_sec() * 1000.0 << price << std::endl;
}

int main() {
  // Create directory if it doesn't exist
  system("mkdir -p analysis_openmp_results_dynamic");
  std::string filename =
      "analysis_openmp_results_dynamic/openmp_dynamic_results.txt";

  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open output file: " << filename << std::endl;
    return 1;
  }

  std::cout << "Running OpenMP Dynamic Benchmark. Output saved to " << filename
            << std::endl;

  std::vector<int> N_bench = {1000, 5000, 10000, 50000, 100000};
  std::vector<int> thread_counts = {1, 2, 4, 8};
  int max_threads = omp_get_max_threads();

  // Header
  ofs << "===================================================================="
      << std::endl;
  ofs << "OpenMP Benchmark: Dynamic Scheduling" << std::endl;
  ofs << "Threads Available: " << max_threads << std::endl;
  ofs << "===================================================================="
      << std::endl;
  ofs << std::left << std::setw(40) << "Method" << std::setw(10) << "N"
      << std::setw(15) << "Time (ms)" << std::setw(15) << "Price" << std::endl;
  ofs << "--------------------------------------------------------------------"
      << std::endl;

  for (int N : N_bench) {
    for (int t : thread_counts) {
      if (t > max_threads)
        continue;

      omp_set_num_threads(t);

      // ONLY Dynamic Scheduling
      run_benchmark(
          N, "OpenMP (Dynamic, " + std::to_string(t) + " threads)",
          [t](const OptionParams &p) {
            return priceAmericanOptionOpenMPDynamic(p, t);
          },
          ofs);
    }
  }

  std::cout << "Benchmark complete." << std::endl;
  return 0;
}
