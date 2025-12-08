#include "../common/timer.h"
#include "../common/validator.h"
#include "cuda_kernels.cuh"
#include <iomanip>
#include <iostream>

/**
 * @file cuda_main.cu
 * @brief Main Driver for CUDA Implementations
 *
 * This file orchestrates the execution and comparison of various CUDA
 * strategies for parallel binomial option pricing:
 *
 *
 * It runs each implementation, measures execution time, and reports results.
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

  std::string filter = "";
  int thresh_cpu = -1;
  int thresh_large_n = -1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--filter" && i + 1 < argc) {
      filter = argv[++i];
    } else if (arg == "--thresh-cpu" && i + 1 < argc) {
      thresh_cpu = std::stoi(argv[++i]);
    } else if (arg == "--thresh-large-n" && i + 1 < argc) {
      thresh_large_n = std::stoi(argv[++i]);
    } else if (arg.find("--") == 0) {
      // Ignore other flags
    } else {
      try {
        int n = std::stoi(arg);
        if (n > 0)
          opt.N = n;
      } catch (...) {
      }
    }
  }

  // Apply thresholds if specified
  if (thresh_cpu != -1 || thresh_large_n != -1) {
    int cpu = (thresh_cpu != -1) ? thresh_cpu : 2000; // Default fallback
    int large = (thresh_large_n != -1) ? thresh_large_n : 171000;

    std::cout << "Setting Thresholds: CPU=" << cpu << ", LargeN=" << large
              << std::endl;
    setHybridCPUGPUThresholds(cpu, large);
    setHybridOpenMPThresholds(cpu, large);
    setHybridGPUThresholds(large);
  }

  std::cout << "=== CUDA Binomial Option Pricer (N=" << opt.N
            << ") ===" << std::endl;
  if (!filter.empty()) {
    std::cout << "Filter: " << filter << std::endl;
  }
  Validator::printOptionDetails(opt);
  std::cout << std::endl;

  auto run_kernel = [&](const char *name, auto func) {
    if (!filter.empty() &&
        std::string(name).find(filter) == std::string::npos) {
      return std::make_pair(0.0, 0.0);
    }
    std::cout << "Running " << name << "..." << std::endl;
    Timer timer;
    timer.start();
    double price = func();
    timer.stop();
    std::cout << "  Price: " << std::fixed << std::setprecision(6) << price
              << std::endl;
    timer.print("  Time");
    std::cout << std::endl;
    return std::make_pair(price, timer.elapsed_ms());
  };

  // 1. Wavefront (Baseline)
  auto res_wavefront = run_kernel(
      "Wavefront", [&]() { return priceAmericanOptionCUDAWavefront(opt); });

  // 2. Tiled Variants
  run_kernel("Tiled (Original)",
             [&]() { return priceAmericanOptionCUDATiled(opt); });
  run_kernel("Tiled V2 (Shared Mem)",
             [&]() { return priceAmericanOptionCUDASharedMemTiling(opt); });
  run_kernel("Tiled V3 (Warp Shuffle)",
             [&]() { return priceAmericanOptionCUDAWarpShuffleTiling(opt); });
  run_kernel("Tiled V4 (Warp Per Block)",
             [&]() { return priceAmericanOptionCUDAWarpPerBlock(opt); });
  run_kernel("Tiled V5 (Indep Multi Warp)", [&]() {
    return priceAmericanOptionCUDAIndependentMultiWarp(opt);
  });

  // 3. Persistent Variants
  run_kernel("Persistent (Global Barrier)", [&]() {
    return priceAmericanOptionCUDAPersistentGlobalBarrier(opt);
  });

  // 4. Hybrid Variants
  run_kernel("Hybrid GPU-Only",
             [&]() { return priceAmericanOptionCUDAHybridGPU(opt); });
  run_kernel("Hybrid CPU-GPU",
             [&]() { return priceAmericanOptionCUDAHybridCPUGPU(opt); });
  run_kernel("Hybrid OpenMP",
             [&]() { return priceAmericanOptionCUDAOpenMPHybrid(opt); });

  // 5. Others
  run_kernel("Time Parallel", [&]() {
    return priceAmericanOptionTimeParallel(opt.S0, opt.K, opt.r, opt.sigma,
                                           opt.T, opt.N, opt.isCall);
  });
  run_kernel("Cooperative Multi Warp", [&]() {
    return priceAmericanOptionCUDACooperativeMultiWarp(
        opt.S0, opt.K, opt.r, opt.sigma, opt.T, opt.N, opt.isCall);
  });

  std::cout << "=== Done ===" << std::endl;

  return 0;
}
