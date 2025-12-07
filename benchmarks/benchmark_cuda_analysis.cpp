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
  std::string target_method = "ALL";
  int target_N = -1;

  // Simple argument parsing
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--method" && i + 1 < argc) {
      target_method = argv[++i];
    } else if (arg == "--n" && i + 1 < argc) {
      target_N = std::stoi(argv[++i]);
    }
  }

  // Filter N values if specific N requested
  if (target_N != -1) {
    N_values.clear();
    N_values.push_back(target_N);
  }

  std::cout
      << "===================================================================="
      << std::endl;
  std::cout << "CUDA Binomial Option Pricing Benchmark (Analysis Mode)"
            << std::endl;
  if (target_method != "ALL")
    std::cout << "Target Method: " << target_method << std::endl;
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
    // Helper macro to run specific or all
    auto should_run = [&](const std::string &name) {
      return target_method == "ALL" || target_method == name;
    };

    if (should_run("Wavefront"))
      run_benchmark(N, "Wavefront", [](const OptionParams &opt) {
        return priceAmericanOptionCUDAWavefront(opt);
      });

    if (should_run("Tiled"))
      run_benchmark(N, "Tiled", [](const OptionParams &opt) {
        return priceAmericanOptionCUDATiled(opt);
      });

    // Tiled (Shared Mem Tiling)
    if (should_run("CUDA Shared Mem Tiling"))
      run_benchmark(N, "CUDA Shared Mem Tiling", [](const OptionParams &opt) {
        return priceAmericanOptionCUDASharedMemTiling(opt);
      });

    // Tiled (Warp Shuffle Tiling)
    if (should_run("CUDA Warp Shuffle Tiling"))
      run_benchmark(N, "CUDA Warp Shuffle Tiling", [](const OptionParams &opt) {
        return priceAmericanOptionCUDAWarpShuffleTiling(opt);
      });

    // Tiled (Warp Per Block)
    if (should_run("CUDA Warp Per Block"))
      run_benchmark(N, "CUDA Warp Per Block", [](const OptionParams &opt) {
        return priceAmericanOptionCUDAWarpPerBlock(opt);
      });

    // Tiled (Independent Multi Warp)
    if (should_run("CUDA Independent Multi Warp"))
      run_benchmark(N, "CUDA Independent Multi Warp",
                    [](const OptionParams &opt) {
                      return priceAmericanOptionCUDAIndependentMultiWarp(opt);
                    });

    // Persistent (Global Barrier)
    if (should_run("CUDA Persistent Global Barrier"))
      run_benchmark(
          N, "CUDA Persistent Global Barrier", [](const OptionParams &opt) {
            return priceAmericanOptionCUDAPersistentGlobalBarrier(opt);
          });

    // Hybrid GPU
    if (should_run("CUDA Hybrid GPU"))
      run_benchmark(N, "CUDA Hybrid GPU", [](const OptionParams &opt) {
        return priceAmericanOptionCUDAHybridGPU(opt);
      });

    // Hybrid CPU-GPU
    if (should_run("CUDA Hybrid CPU-GPU"))
      run_benchmark(N, "CUDA Hybrid CPU-GPU", [](const OptionParams &opt) {
        return priceAmericanOptionCUDAHybridCPUGPU(opt);
      });

    // Hybrid OpenMP
    if (should_run("CUDA Hybrid OpenMP"))
      run_benchmark(N, "CUDA Hybrid OpenMP", [](const OptionParams &opt) {
        return priceAmericanOptionCUDAOpenMPHybrid(opt);
      });

    // Time Parallel
    if (should_run("CUDA Time Parallel"))
      run_benchmark(N, "CUDA Time Parallel", [](const OptionParams &opt) {
        return priceAmericanOptionTimeParallel(opt.S0, opt.K, opt.r, opt.sigma,
                                               opt.T, opt.N, opt.isCall);
      });

    // Cooperative Multi Warp
    if (should_run("CUDA Cooperative Multi Warp"))
      run_benchmark(
          N, "CUDA Cooperative Multi Warp", [](const OptionParams &opt) {
            return priceAmericanOptionCUDACooperativeMultiWarp(
                opt.S0, opt.K, opt.r, opt.sigma, opt.T, opt.N, opt.isCall);
          });

    if (target_method == "ALL")
      std::cout
          << "-------------------------------------------------------------"
             "-------"
          << std::endl;
  }

  return 0;
}
