#include "../common/timer.h"
#include "../common/validator.h"
#include "../serial/serial_binomial.h"
#include "openmp_binomial.h"
#include <iomanip>
#include <iostream>
#include <omp.h>

/**
 * Main driver for OpenMP binomial option pricer
 *
 * Usage: ./openmp_binomial [N] [num_threads]
 * Example: ./openmp_binomial 10000 16
 *
 * Or set threads via environment variable:
 * OMP_NUM_THREADS=16 ./openmp_binomial 10000
 */

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

  int num_threads = 0; // 0 = use default (OMP_NUM_THREADS or system default)

  // Parse command line arguments
  if (argc > 1) {
    try {
      int n = std::stoi(argv[1]);
      if (n <= 0) {
        std::cerr << "Error: N must be positive" << std::endl;
        return 1;
      }
      opt.N = n;
    } catch (const std::exception &e) {
      std::cerr << "Error: Invalid N value: " << argv[1] << std::endl;
      std::cerr << "Usage: " << argv[0] << " [N] [num_threads]" << std::endl;
      return 1;
    }
  }

  if (argc > 2) {
    try {
      int nt = std::stoi(argv[2]);
      if (nt <= 0) {
        std::cerr << "Error: num_threads must be positive" << std::endl;
        return 1;
      }
      num_threads = nt;
    } catch (const std::exception &e) {
      std::cerr << "Error: Invalid num_threads value: " << argv[2] << std::endl;
      return 1;
    }
  }

  std::cout << "=== OpenMP Binomial Option Pricer ===" << std::endl;
  Validator::printOptionDetails(opt);

  // Print thread configuration
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }
  int actual_threads = getOMPThreadCount();
  std::cout << "OpenMP Configuration:" << std::endl;
  std::cout << "  Requested threads:     "
            << (num_threads > 0 ? std::to_string(num_threads) : "default")
            << std::endl;
  std::cout << "  Actual threads:        " << actual_threads << std::endl;
  std::cout << "  Max threads available: " << omp_get_max_threads()
            << std::endl;
  std::cout << std::endl;

  // Price American option with OpenMP
  Timer timer;
  timer.start();
  double american_price_omp = priceAmericanOptionOpenMP(opt, num_threads);
  timer.stop();
  double omp_time = timer.elapsed_ms();

  std::cout << "Results (OpenMP):" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  American Option Price: " << american_price_omp << std::endl;
  std::cout << "  Execution Time:        " << std::setprecision(3) << omp_time
            << " ms" << std::endl;

  long long total_nodes = static_cast<long long>(opt.N + 1) * (opt.N + 2) / 2;
  std::cout << "  Nodes Computed:        " << total_nodes << std::endl;
  std::cout << "  Throughput:            " << std::setprecision(2)
            << (static_cast<double>(total_nodes) / omp_time) << " nodes/ms"
            << std::endl;
  std::cout << std::endl;

  // Compare with European
  timer.start();
  double european_price_omp = priceEuropeanOptionOpenMP(opt, num_threads);
  timer.stop();
  std::cout << "  European Option Price: " << std::setprecision(6)
            << european_price_omp << std::endl;
  std::cout << "  European Time:         " << std::setprecision(3)
            << timer.elapsed_ms() << " ms" << std::endl;
  std::cout << "  Early Exercise Value:  "
            << (american_price_omp - european_price_omp) << std::endl;
  std::cout << std::endl;

  // Validation
  Validator::runValidationSuite(american_price_omp, opt, "OpenMP");
  Validator::validateAmericanVsEuropean(american_price_omp, european_price_omp,
                                        opt);

  // Compare with serial implementation for correctness
  std::cout << "=== Correctness Check vs Serial ===" << std::endl;
  timer.start();
  double american_price_serial = priceAmericanOptionSerial(opt);
  timer.stop();
  double serial_time = timer.elapsed_ms();

  std::cout << "Serial Results:" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  Price:          " << american_price_serial << std::endl;
  std::cout << "  Time:           " << std::setprecision(3) << serial_time
            << " ms" << std::endl;
  std::cout << std::endl;

  Validator::compareImplementations(american_price_serial, "Serial",
                                    american_price_omp, "OpenMP");

  // Speedup analysis
  if (omp_time < 1e-6) {
    std::cout << "  Speedup:                N/A (time too small)" << std::endl;
  } else {
    double speedup = serial_time / omp_time;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Speedup:                " << speedup << "x" << std::endl;
    std::cout << "  Parallel Efficiency:    "
              << (speedup / actual_threads * 100.0) << "%" << std::endl;
  }
  std::cout << "=== Performance Analysis ===" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  // Only print aggregated speedup/efficiency if omp_time was large enough
  if (omp_time >= 1e-6) {
    double speedup = serial_time / omp_time;
    std::cout << "  Speedup:                " << speedup << "x" << std::endl;
    std::cout << "  Parallel Efficiency:    "
              << (speedup / actual_threads * 100.0) << "%" << std::endl;
  } else {
    std::cout << "  Speedup:                N/A" << std::endl;
    std::cout << "  Parallel Efficiency:    N/A" << std::endl;
  }
  std::cout << "  Time per barrier:       " << std::setprecision(3)
            << (omp_time / opt.N) << " ms" << std::endl;

  // Estimate overhead
  double ideal_time = serial_time / actual_threads;
  double overhead = omp_time - ideal_time;
  std::cout << "  Ideal parallel time:    " << ideal_time << " ms" << std::endl;
  std::cout << "  Estimated overhead:     " << overhead << " ms ("
            << std::setprecision(1) << (overhead / omp_time * 100.0) << "%)"
            << std::endl;
  std::cout << std::endl;

  return 0;
}