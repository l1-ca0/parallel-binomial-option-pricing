#include "../src/common/option.h"
#include "../src/common/timer.h"
#include "../src/openmp/openmp_binomial.h"
#include "../src/serial/serial_binomial.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @file test_openmp_benchmarks.cpp
 * @brief Validate OpenMP implementation against known benchmark values
 *
 * Reads test cases from american_put_benchmarks.txt and validates
 * that OpenMP implementation produce correct results.
 * Includes performance timing for large N values.
 */

struct TestCase {
  double S0;
  double K;
  double r;
  double sigma;
  double T;
  int N;
  double expected_price;
};

// Parse a single test case from a line
bool parseTestCase(const std::string &line, TestCase &test) {
  // Skip empty lines and comments
  if (line.empty() || line[0] == '#') {
    return false;
  }

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

int main(int argc, char **argv) {
  std::string filename = "tests/test_data/american_put_benchmarks.txt";
  bool verbose = false;
  bool skip_slow = false;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-v" || arg == "--verbose") {
      verbose = true;
    } else if (arg == "--skip-slow") {
      skip_slow = true;
    } else {
      filename = arg;
    }
  }

  std::cout << "=== Benchmark Validation Test ===" << std::endl;
  std::cout << "Reading test cases from: " << filename << std::endl;
  if (skip_slow) {
    std::cout
        << "Skipping tests with N > 10000 (use without --skip-slow to run all)"
        << std::endl;
  }
  std::cout << std::endl;

  // Open test data file
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "ERROR: Cannot open file: " << filename << std::endl;
    std::cerr << "Make sure the file exists in tests/test_data/" << std::endl;
    return 1;
  }

  std::vector<TestCase> tests;
  std::string line;
  int line_number = 0;

  // Read all test cases
  while (std::getline(file, line)) {
    line_number++;
    TestCase test;
    if (parseTestCase(line, test)) {
      // Skip slow tests if requested
      if (skip_slow && test.N > 10000) {
        continue;
      }
      tests.push_back(test);
    }
  }
  file.close();

  std::cout << "Loaded " << tests.size() << " test cases" << std::endl;
  std::cout << std::endl;

  if (tests.empty()) {
    std::cerr << "ERROR: No valid test cases found" << std::endl;
    return 1;
  }

  // Run tests
  int passed_serial = 0;
  int passed_openmp = 0;
  int total = 0;

  Timer timer;

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Running validation tests..." << std::endl;
  std::cout << "---------------------------------------------------------------"
               "----------"
            << std::endl;

  for (size_t i = 0; i < tests.size(); ++i) {
    const TestCase &test = tests[i];
    total++;

    // Create option parameters
    OptionParams opt;
    opt.S0 = test.S0;
    opt.K = test.K;
    opt.r = test.r;
    opt.sigma = test.sigma;
    opt.T = test.T;
    opt.N = test.N;
    opt.isCall = false; // All test cases are puts

    std::cout << "Test " << std::setw(3) << (i + 1) << ": ";
    std::cout << "S0=" << std::setw(5) << test.S0 << ", K=" << std::setw(5)
              << test.K << ", r=" << std::setprecision(2) << test.r
              << ", σ=" << test.sigma << ", T=" << test.T
              << ", N=" << std::setw(6) << test.N << std::fixed
              << std::setprecision(4);

    // Determine tolerance
    double tolerance = getTolerance(test.N);

    // Declare timing variables outside try blocks for later use
    double serial_time = 0.0;
    double openmp_time = 0.0;

    // Test serial implementation
    try {
      timer.start();
      double serial_price = priceAmericanOptionSerial(opt);
      timer.stop();
      serial_time = timer.elapsed_ms();

      double serial_error = std::abs(serial_price - test.expected_price);
      double serial_rel_error = serial_error / test.expected_price;

      bool serial_pass = serial_rel_error <= tolerance;

      if (verbose || !serial_pass) {
        std::cout << std::endl;
        std::cout << "  Serial:  " << std::setprecision(6) << serial_price
                  << " (expected: " << test.expected_price
                  << ", error: " << std::setprecision(4)
                  << (serial_rel_error * 100.0) << "%, "
                  << "time: " << std::setprecision(2) << serial_time << " ms) ";
      }

      if (serial_pass) {
        if (verbose)
          std::cout << "PASS";
        passed_serial++;
      } else {
        if (!verbose)
          std::cout << std::endl << "  Serial:  ";
        std::cout << "FAIL (tolerance: " << (tolerance * 100.0) << "%)";
      }
      if (verbose)
        std::cout << std::endl;
    } catch (const std::exception &e) {
      std::cout << std::endl << "  Serial:  ERROR - " << e.what() << std::endl;
    }

    // Test OpenMP implementation
    try {
      timer.start();
      double openmp_price =
          priceAmericanOptionOpenMP(opt, 0); // Use default threads
      timer.stop();
      openmp_time = timer.elapsed_ms();

      double openmp_error = std::abs(openmp_price - test.expected_price);
      double openmp_rel_error = openmp_error / test.expected_price;

      bool openmp_pass = openmp_rel_error <= tolerance;

      if (verbose || !openmp_pass) {
        std::cout << "  OpenMP:  " << std::setprecision(6) << openmp_price
                  << " (expected: " << test.expected_price
                  << ", error: " << std::setprecision(4)
                  << (openmp_rel_error * 100.0) << "%, "
                  << "time: " << std::setprecision(2) << openmp_time << " ms) ";
      }

      if (openmp_pass) {
        if (verbose)
          std::cout << "PASS";
        passed_openmp++;
      } else {
        if (!verbose)
          std::cout << std::endl << "  OpenMP:  ";
        std::cout << "FAIL (tolerance: " << (tolerance * 100.0) << "%)";
      }
      if (verbose)
        std::cout << std::endl;

      // Show speedup for large N
      if (verbose && serial_time > 0) {
        double speedup = serial_time / openmp_time;
        std::cout << "  Speedup: " << std::setprecision(2) << speedup << "x"
                  << std::endl;
      }
    } catch (const std::exception &e) {
      std::cout << std::endl << "  OpenMP:  ERROR - " << e.what() << std::endl;
    }

    // Print compact summary for non-verbose mode
    if (!verbose) {
      std::cout << std::endl;
    } else {
      std::cout << std::endl;
    }
  }

  // Summary
  std::cout << "---------------------------------------------------------------"
               "----------"
            << std::endl;
  std::cout << "=== Test Summary ===" << std::endl;
  std::cout << "Total test cases:     " << total << std::endl;
  std::cout << "Serial passed:        " << passed_serial << " / " << total
            << std::endl;

  std::cout << "OpenMP passed:        " << passed_openmp << " / " << total
            << std::endl;
  std::cout << std::endl;

  if (passed_serial == total && passed_openmp == total) {
    std::cout << "ALL TESTS PASSED!" << std::endl;
    return 0;
  } else {
    std::cout << "SOME TESTS FAILED" << std::endl;
    std::cout << "Run with -v flag for detailed output" << std::endl;
    return 1;
  }
}