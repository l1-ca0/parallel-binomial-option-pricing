#include "../src/common/option.h"
#include "../src/common/timer.h"
#include "../src/cuda/cuda_kernels.cuh"
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct TestCase {
  double S0;
  double K;
  double r;
  double sigma;
  double T;
  int N;
  double expected_price;
};

bool parseTestCase(const std::string &line, TestCase &test) {
  if (line.empty() || line[0] == '#')
    return false;
  std::istringstream iss(line);
  char comma;
  if (!(iss >> test.S0 >> comma >> test.K >> comma >> test.r >> comma >>
        test.sigma >> comma >> test.T >> comma >> test.N >> comma >>
        test.expected_price)) {
    return false;
  }
  return true;
}

// Determine tolerance based on N (relative error)
double getTolerance(int N) {
  if (N >= 100000)
    return 0.00005; // 0.005%
  if (N >= 50000)
    return 0.0001; // 0.01%
  if (N >= 10000)
    return 0.0002; // 0.02%
  if (N >= 5000)
    return 0.0005; // 0.05%
  if (N >= 1000)
    return 0.001; // 0.1%
  if (N >= 100)
    return 0.01; // 1%
  return 0.05;   // 5% for very coarse grids
}

void run_benchmark(const std::string &name,
                   std::function<double(const OptionParams &)> pricer,
                   const std::vector<TestCase> &tests, bool verbose) {

  std::cout << "Running " << name << " Benchmarks..." << std::endl;
  int passed = 0;
  Timer timer;
  double total_time = 0;

  for (const auto &test : tests) {
    OptionParams opt;
    opt.S0 = test.S0;
    opt.K = test.K;
    opt.r = test.r;
    opt.sigma = test.sigma;
    opt.T = test.T;
    opt.N = test.N;
    opt.isCall = false;

    timer.start();
    double price = pricer(opt);
    timer.stop();
    total_time += timer.elapsed_ms();

    double error = std::abs(price - test.expected_price);
    double rel_error = error / std::max(1.0, std::abs(test.expected_price));

    // Tolerance based on N
    double tol = getTolerance(test.N);

    bool pass = rel_error <= tol;
    if (pass)
      passed++;

    if (verbose || !pass) {
      std::cout << "  N=" << test.N << " Expected=" << test.expected_price
                << " Got=" << price << " Error=" << (rel_error * 100) << "% "
                << (pass ? "PASS" : "FAIL") << " Time=" << timer.elapsed_ms()
                << "ms" << std::endl;
    }
  }
  std::cout << "  Passed: " << passed << "/" << tests.size()
            << " Total Time: " << total_time << "ms" << std::endl;
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  std::string filename = "tests/test_data/american_put_benchmarks.txt";
  bool verbose = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-v" || arg == "--verbose")
      verbose = true;
    else
      filename = arg;
  }

  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open " << filename << std::endl;
    return 1;
  }

  std::vector<TestCase> tests;
  std::string line;
  while (std::getline(file, line)) {
    TestCase test;
    if (parseTestCase(line, test))
      tests.push_back(test);
  }

  std::cout << "Loaded " << tests.size() << " test cases." << std::endl
            << std::endl;
  run_benchmark(
    "Wavefront",
    [](const OptionParams &o) {
      return priceAmericanOptionCUDAWavefront(o);
    },
    tests, verbose);

  run_benchmark(
    "Tiled",
    [](const OptionParams &o) {
      return priceAmericanOptionCUDATiled(o);
    },
    tests, verbose);

  run_benchmark(
    "Tiled (Warp Shuffle)",
    [](const OptionParams &o) {
      return priceAmericanOptionCUDAWarpShuffleTiling(o);
    },
    tests, verbose);

  run_benchmark(
    "Time Parallel",
    [](const OptionParams &o) {
      return priceAmericanOptionTimeParallel(o.S0, o.K, o.r, o.sigma, o.T,
                                            o.N, o.isCall);
    },
    tests, verbose);

  run_benchmark(
    "Cooperative Multi Warp",
    [](const OptionParams &o) {
      return priceAmericanOptionCUDACooperativeMultiWarp(
        o.S0, o.K, o.r, o.sigma, o.T, o.N, o.isCall);
    },
    tests, verbose);

  run_benchmark(
    "Persistent (Global Barrier)",
    [](const OptionParams &o) {
      return priceAmericanOptionCUDAPersistentGlobalBarrier(o);
    },
    tests, verbose);

  return 0;
}
