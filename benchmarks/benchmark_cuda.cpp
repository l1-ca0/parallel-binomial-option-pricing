#include "../src/common/option.h"
#include "../src/common/timer.h"
#include "../src/cuda/cuda_kernels.cuh"
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Helper to run benchmark for a specific N and method
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

int main(int argc, char **argv) {
  std::vector<int> N_values = {1000,   5000,   10000,  50000,
                               100000, 500000, 1000000};

  std::cout
      << "===================================================================="
      << std::endl;
  std::cout << "CUDA Binomial Option Pricing Benchmark" << std::endl;
  std::cout
      << "===================================================================="
      << std::endl;
  std::cout << std::left << std::setw(35) << "Method" << std::setw(10) << "N"
            << std::setw(15) << "Time (ms)" << std::setw(15) << "Price"
            << std::endl;
  std::cout
      << "--------------------------------------------------------------------"
      << std::endl;

  for (int N : N_values) {
    run_benchmark(N, "Wavefront", [](const OptionParams &opt) {
      return priceAmericanOptionCUDAWavefront(opt);
    });

    run_benchmark(N, "Tiled", [](const OptionParams &opt) {
      return priceAmericanOptionCUDATiled(opt);
    });

    // Tiled Shared Mem Tiling
    run_benchmark(N, "CUDA Shared Mem Tiling", [](const OptionParams &opt) {
      return priceAmericanOptionCUDASharedMemTiling(opt);
    });

    // Tiled Warp Shuffle Tiling
    run_benchmark(N, "CUDA Warp Shuffle Tiling", [](const OptionParams &opt) {
      return priceAmericanOptionCUDAWarpShuffleTiling(opt);
    });

    // Tiled Warp Per Block
    run_benchmark(N, "CUDA Warp Per Block", [](const OptionParams &opt) {
      return priceAmericanOptionCUDAWarpPerBlock(opt);
    });

    // Tiled Independent Multi Warp
    run_benchmark(N, "CUDA Independent Multi Warp",
                  [](const OptionParams &opt) {
                    return priceAmericanOptionCUDAIndependentMultiWarp(opt);
                  });


    // Persistent (Global Barrier)
    run_benchmark(N, "CUDA Persistent Global Barrier",
                  [](const OptionParams &opt) {
                    return priceAmericanOptionCUDAPersistentGlobalBarrier(opt);
                  });

    // Hybrid GPU
    run_benchmark(N, "CUDA Hybrid GPU", [](const OptionParams &opt) {
      return priceAmericanOptionCUDAHybridGPU(opt);
    });

    // Hybrid CPU-GPU
    run_benchmark(N, "CUDA Hybrid CPU-GPU", [](const OptionParams &opt) {
      return priceAmericanOptionCUDAHybridCPUGPU(opt);
    });

    // Hybrid OpenMP
    run_benchmark(N, "CUDA Hybrid OpenMP", [](const OptionParams &opt) {
      return priceAmericanOptionCUDAOpenMPHybrid(opt);
    });

    // Time Parallel
    run_benchmark(N, "CUDA Time Parallel", [](const OptionParams &opt) {
      return priceAmericanOptionTimeParallel(opt.S0, opt.K, opt.r, opt.sigma,
                                             opt.T, opt.N, opt.isCall);
    });

    // Cooperative Multi Warp
    run_benchmark(
        N, "CUDA Cooperative Multi Warp", [](const OptionParams &opt) {
          return priceAmericanOptionCUDACooperativeMultiWarp(
              opt.S0, opt.K, opt.r, opt.sigma, opt.T, opt.N, opt.isCall);
        });

    std::cout << "-------------------------------------------------------------"
                 "-------"
              << std::endl;
  }

  return 0;
}
