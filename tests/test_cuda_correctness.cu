#include "../src/common/option.h"
#include "../src/cuda/cuda_kernels.cuh"
#include "../src/serial/serial_binomial.h" // For European price comparison
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Test result tracking
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;

// Helper macros
#define ASSERT(condition, message)                                             \
  if (!(condition)) {                                                          \
    throw std::runtime_error(message);                                         \
  }

#define ASSERT_NEAR(actual, expected, tolerance, message)                      \
  if (std::abs((actual) - (expected)) > (tolerance)) {                         \
    throw std::runtime_error(                                                  \
        std::string(message) + " (got " + std::to_string(actual) +             \
        ", expected " + std::to_string(expected) +                             \
        ", diff = " + std::to_string(std::abs((actual) - (expected))) + ")");  \
  }

// Standard test option
OptionParams createTestOption() {
  OptionParams opt;
  opt.S0 = 100.0;
  opt.K = 100.0;
  opt.r = 0.05;
  opt.sigma = 0.2;
  opt.T = 1.0;
  opt.N = 1000;
  opt.isCall = false;
  return opt;
}

// Function pointer type for pricing functions
using PricingFunc = std::function<double(const OptionParams &)>;

// Test Suite Class
class CUDATestSuite {
  std::string name;
  PricingFunc pricer;

public:
  CUDATestSuite(const std::string &n, PricingFunc p) : name(n), pricer(p) {}

  void run() {
    std::cout << "=== Running Tests for " << name << " ===" << std::endl;
    run_test("put_price_positive", [this]() { test_put_price_positive(); });
    run_test("put_price_below_strike",
             [this]() { test_put_price_below_strike(); });
    run_test("put_price_above_intrinsic",
             [this]() { test_put_price_above_intrinsic(); });
    run_test("american_put_ge_european",
             [this]() { test_american_put_ge_european(); });
    run_test("convergence_with_N", [this]() { test_convergence_with_N(); });
    run_test("zero_volatility", [this]() { test_zero_volatility(); });
    std::cout << std::endl;
  }

private:
  void run_test(const std::string &test_name, std::function<void()> test_func) {
    tests_run++;
    std::cout << "  Test: " << test_name << " ... ";
    try {
      test_func();
      tests_passed++;
      std::cout << "PASS" << std::endl;
    } catch (const std::exception &e) {
      tests_failed++;
      std::cout << "FAIL: " << e.what() << std::endl;
    }
  }

  void test_put_price_positive() {
    OptionParams opt = createTestOption();
    double price = pricer(opt);
    ASSERT(price >= 0, "Put price must be non-negative");
  }

  void test_put_price_below_strike() {
    OptionParams opt = createTestOption();
    double price = pricer(opt);
    ASSERT(price <= opt.K, "American put price cannot exceed strike price");
  }

  void test_put_price_above_intrinsic() {
    OptionParams opt = createTestOption();
    double price = pricer(opt);
    double intrinsic = std::max(opt.K - opt.S0, 0.0);
    ASSERT(price >= intrinsic - 1e-10,
           "Put price must be at least intrinsic value");
  }

  void test_american_put_ge_european() {
    OptionParams opt = createTestOption();
    double american = pricer(opt);
    double european =
        priceEuropeanOptionSerial(opt); // Use Serial for European baseline
    ASSERT(american >= european - 1e-10,
           "American put must be >= European put");
  }

  void test_convergence_with_N() {
    OptionParams opt = createTestOption();
    opt.N = 100;
    double price_100 = pricer(opt);
    opt.N = 1000;
    double price_1000 = pricer(opt);
    opt.N = 2000;
    double price_2000 = pricer(opt);

    double diff1 = std::abs(price_1000 - price_100);
    double diff2 = std::abs(price_2000 - price_1000);

    // Loose check for convergence
    ASSERT(diff2 < diff1 || diff2 < 0.05, "Price should converge or be stable");
  }

  void test_zero_volatility() {
    OptionParams opt = createTestOption();
    opt.sigma = 1e-5; // Small sigma
    opt.r = 0.0;      // Set r=0 to ensure p is valid
    opt.S0 = 90.0;
    opt.K = 100.0;
    opt.N = 100;

    double price = pricer(opt);
    double expected = opt.K - opt.S0;
    ASSERT_NEAR(price, expected, 0.1, "Zero vol should equal intrinsic");
  }
};

int main() {
  std::cout << "=== CUDA Correctness Tests ===" << std::endl;

  // Test Wavefront
  CUDATestSuite wavefront("Wavefront", [](const OptionParams &opt) {
    return priceAmericanOptionCUDAWavefront(opt);
  });
  wavefront.run();

  // Test Tiled
  CUDATestSuite tiled("Tiled", [](const OptionParams &opt) {
    return priceAmericanOptionCUDATiled(opt);
  });
  tiled.run();

    
  // Tiled (Shared Mem Tiling)
  CUDATestSuite tiled_shared_mem(
      "SharedMemTiling", [](const OptionParams &opt) {
        return priceAmericanOptionCUDASharedMemTiling(opt);
      });
  tiled_shared_mem.run();

  // Tiled (Warp Shuffle Tiling)
  // Tiled (Warp Shuffle Tiling)
  CUDATestSuite suite_warp_shuffle(
      "WarpShuffleTiling", [](const OptionParams &opt) {
        return priceAmericanOptionCUDAWarpShuffleTiling(opt);
      });
  suite_warp_shuffle.run();

  // Tiled (Warp Per Block)
  CUDATestSuite tiled_warp_per_block(
      "WarpPerBlock", [](const OptionParams &opt) {
        return priceAmericanOptionCUDAWarpPerBlock(opt);
      });
  tiled_warp_per_block.run();

  // Tiled (Independent Multi Warp)
  CUDATestSuite tiled_independent_multi_warp(
      "IndependentMultiWarp", [](const OptionParams &opt) {
        return priceAmericanOptionCUDAIndependentMultiWarp(opt);
      });
  tiled_independent_multi_warp.run();

  // Test Persistent (Global Barrier)
  CUDATestSuite persistent_global_barrier(
      "PersistentGlobalBarrier", [](const OptionParams &opt) {
        return priceAmericanOptionCUDAPersistentGlobalBarrier(opt);
      });
  persistent_global_barrier.run();

  // Test HybridGPU
  CUDATestSuite hybrid_gpu("HybridGPU", [](const OptionParams &opt) {
    return priceAmericanOptionCUDAHybridGPU(opt);
  });
  hybrid_gpu.run();

  // Test TimeParallel
  CUDATestSuite time_parallel("TimeParallel", [](const OptionParams &opt) {
    return priceAmericanOptionTimeParallel(opt.S0, opt.K, opt.r, opt.sigma,
                                           opt.T, opt.N, opt.isCall);
  });
  time_parallel.run();

  // Test Cooperative Multi Warp
  CUDATestSuite cooperative_multi_warp(
      "CooperativeMultiWarp", [](const OptionParams &opt) {
        return priceAmericanOptionCUDACooperativeMultiWarp(
            opt.S0, opt.K, opt.r, opt.sigma, opt.T, opt.N, opt.isCall);
      });
  cooperative_multi_warp.run();

  std::cout << "=== Summary ===" << std::endl;
  std::cout << "Total:  " << tests_run << std::endl;
  std::cout << "Passed: " << tests_passed << std::endl;
  std::cout << "Failed: " << tests_failed << std::endl;

  return (tests_failed == 0) ? 0 : 1;
}
