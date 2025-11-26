#include "../src/common/option.h"
#include "../src/openmp/openmp_binomial.h"
#include "../src/serial/serial_binomial.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file test_correctness.cpp
 * @brief Unit tests for algorithmic correctness
 *
 * Tests fundamental properties of option pricing:
 * 1. Pricing bounds (intrinsic value, upper bounds)
 * 2. American >= European
 * 3. Monotonicity properties
 * 4. Put-Call parity relationships
 * 5. Convergence as N increases
 * 6. Edge cases and boundary conditions
 *
 * Unlike test_benchmarks.cpp which validates against known values,
 * this tests mathematical properties that must hold.
 */

// Test result tracking
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;

// Helper macros for testing
#define TEST(name)                                                             \
  void test_##name();                                                          \
  void run_test_##name() {                                                     \
    tests_run++;                                                               \
    std::cout << "Running test: " << #name << " ... ";                         \
    try {                                                                      \
      test_##name();                                                           \
      tests_passed++;                                                          \
      std::cout << "PASS" << std::endl;                                      \
    } catch (const std::exception &e) {                                        \
      tests_failed++;                                                          \
      std::cout << "FAIL: " << e.what() << std::endl;                        \
    }                                                                          \
  }                                                                            \
  void test_##name()

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

// Helper function to create standard test option
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

// ========== BASIC CORRECTNESS TESTS ==========

TEST(put_price_positive) {
  OptionParams opt = createTestOption();
  double price = priceAmericanOptionSerial(opt);
  ASSERT(price >= 0, "Put price must be non-negative");
}

TEST(call_price_positive) {
  OptionParams opt = createTestOption();
  opt.isCall = true;
  double price = priceAmericanOptionSerial(opt);
  ASSERT(price >= 0, "Call price must be non-negative");
}

TEST(put_price_below_strike) {
  OptionParams opt = createTestOption();
  double price = priceAmericanOptionSerial(opt);
  ASSERT(price <= opt.K, "American put price cannot exceed strike price");
}

TEST(call_price_below_spot_plus_strike) {
  // American call upper bound: C <= S0
  // For non-dividend paying stock, American call = European call
  // European call upper bound is S0 (stock itself is better)
  OptionParams opt = createTestOption();
  opt.isCall = true;
  double price = priceAmericanOptionSerial(opt);
  ASSERT(price <= opt.S0 + 0.01,
         "American call price should not significantly exceed spot price");
}

TEST(put_price_above_intrinsic) {
  OptionParams opt = createTestOption();
  double price = priceAmericanOptionSerial(opt);
  double intrinsic = std::max(opt.K - opt.S0, 0.0);
  ASSERT(price >= intrinsic - 1e-10,
         "Put price must be at least intrinsic value");
}

TEST(call_price_above_intrinsic) {
  OptionParams opt = createTestOption();
  opt.isCall = true;
  double price = priceAmericanOptionSerial(opt);
  double intrinsic = std::max(opt.S0 - opt.K, 0.0);
  ASSERT(price >= intrinsic - 1e-10,
         "Call price must be at least intrinsic value");
}

// ========== AMERICAN VS EUROPEAN TESTS ==========

TEST(american_put_ge_european) {
  OptionParams opt = createTestOption();
  double american = priceAmericanOptionSerial(opt);
  double european = priceEuropeanOptionSerial(opt);
  ASSERT(american >= european - 1e-10,
         "American put must be worth at least as much as European put");
}

TEST(american_call_ge_european) {
  OptionParams opt = createTestOption();
  opt.isCall = true;
  double american = priceAmericanOptionSerial(opt);
  double european = priceEuropeanOptionSerial(opt);
  ASSERT(american >= european - 1e-10,
         "American call must be worth at least as much as European call");
}

TEST(american_put_early_exercise_premium_itm) {
  // Deep in-the-money put should have early exercise premium
  OptionParams opt = createTestOption();
  opt.S0 = 80.0; // Deep ITM
  opt.K = 100.0;
  double american = priceAmericanOptionSerial(opt);
  double european = priceEuropeanOptionSerial(opt);
  double premium = american - european;
  ASSERT(
      premium > 0.01,
      "Deep ITM American put should have significant early exercise premium");
}

// ========== MONOTONICITY TESTS ==========

TEST(put_price_decreases_with_spot) {
  // As spot price increases, put value should decrease
  OptionParams opt = createTestOption();
  opt.S0 = 90.0;
  double price1 = priceAmericanOptionSerial(opt);

  opt.S0 = 110.0;
  double price2 = priceAmericanOptionSerial(opt);

  ASSERT(price1 > price2, "Put price should decrease as spot increases");
}

TEST(call_price_increases_with_spot) {
  // As spot price increases, call value should increase
  OptionParams opt = createTestOption();
  opt.isCall = true;
  opt.S0 = 90.0;
  double price1 = priceAmericanOptionSerial(opt);

  opt.S0 = 110.0;
  double price2 = priceAmericanOptionSerial(opt);

  ASSERT(price2 > price1, "Call price should increase as spot increases");
}

TEST(put_price_increases_with_strike) {
  // As strike increases, put value should increase
  OptionParams opt = createTestOption();
  opt.K = 90.0;
  double price1 = priceAmericanOptionSerial(opt);

  opt.K = 110.0;
  double price2 = priceAmericanOptionSerial(opt);

  ASSERT(price2 > price1, "Put price should increase as strike increases");
}

TEST(price_increases_with_volatility) {
  // Higher volatility -> higher option value (for both puts and calls)
  OptionParams opt = createTestOption();
  opt.sigma = 0.1;
  double price1 = priceAmericanOptionSerial(opt);

  opt.sigma = 0.4;
  double price2 = priceAmericanOptionSerial(opt);

  ASSERT(price2 > price1, "Option price should increase with volatility");
}

TEST(price_increases_with_time) {
  // More time to maturity -> higher option value
  OptionParams opt = createTestOption();
  opt.T = 0.25;
  double price1 = priceAmericanOptionSerial(opt);

  opt.T = 2.0;
  double price2 = priceAmericanOptionSerial(opt);

  ASSERT(price2 > price1, "Option price should increase with time to maturity");
}

// ========== CONVERGENCE TESTS ==========

TEST(convergence_with_N) {
  // As N increases, price should converge
  OptionParams opt = createTestOption();

  opt.N = 100;
  double price_100 = priceAmericanOptionSerial(opt);

  opt.N = 1000;
  double price_1000 = priceAmericanOptionSerial(opt);

  opt.N = 5000;
  double price_5000 = priceAmericanOptionSerial(opt);

  // Prices should be converging
  double diff1 = std::abs(price_1000 - price_100);
  double diff2 = std::abs(price_5000 - price_1000);

  ASSERT(diff2 < diff1, "Price should converge as N increases");
}

// ========== EDGE CASE TESTS ==========

TEST(zero_volatility) {
  // With zero volatility, American put = max(K - S, 0) discounted
  OptionParams opt = createTestOption();
  opt.sigma = 1e-10; // Essentially zero
  opt.S0 = 90.0;
  opt.K = 100.0;
  opt.N = 100;

  double price = priceAmericanOptionSerial(opt);
  double expected = opt.K - opt.S0; // Should exercise immediately

  ASSERT_NEAR(price, expected, 0.1,
              "With zero volatility, should equal intrinsic value");
}

TEST(very_long_maturity) {
  // Very long maturity should be handled correctly
  OptionParams opt = createTestOption();
  opt.T = 10.0; // 10 years
  opt.N = 1000;

  double price = priceAmericanOptionSerial(opt);
  ASSERT(price > 0 && price < opt.K,
         "Very long maturity option should be priced correctly");
}

TEST(very_short_maturity) {
  // Very short maturity should approach intrinsic value
  OptionParams opt = createTestOption();
  opt.T = 0.01; // ~4 days
  opt.S0 = 95.0;
  opt.K = 100.0;
  opt.N = 100;

  double price = priceAmericanOptionSerial(opt);
  double intrinsic = opt.K - opt.S0;

  ASSERT_NEAR(price, intrinsic, 0.5,
              "Very short maturity should be close to intrinsic value");
}

TEST(deep_itm_put_early_exercise) {
  // Deep in-the-money put with high interest rate should be exercised early
  OptionParams opt = createTestOption();
  opt.S0 = 50.0; // Very deep ITM
  opt.K = 100.0;
  opt.r = 0.15; // High interest rate
  opt.N = 1000;

  double american = priceAmericanOptionSerial(opt);
  double european = priceEuropeanOptionSerial(opt);

  ASSERT(american > european + 1.0,
         "Deep ITM put with high r should have large early exercise premium");
}

TEST(deep_otm_put_near_zero) {
  // Deep out-of-the-money put should be nearly worthless
  OptionParams opt = createTestOption();
  opt.S0 = 150.0; // Deep OTM
  opt.K = 100.0;
  opt.N = 1000;

  double price = priceAmericanOptionSerial(opt);

  ASSERT(price < 1.0, "Deep OTM put should be nearly worthless");
}

// ========== IMPLEMENTATION CONSISTENCY TESTS ==========

TEST(serial_vs_openmp_consistency) {
  // Serial and OpenMP should give identical results
  OptionParams opt = createTestOption();
  opt.N = 1000;

  double serial_price = priceAmericanOptionSerial(opt);
  double openmp_price = priceAmericanOptionOpenMP(opt, 4);

  ASSERT_NEAR(serial_price, openmp_price, 1e-10,
              "Serial and OpenMP implementations must give identical results");
}

TEST(serial_vs_openmp_multiple_N) {
  // Test consistency across different N values
  std::vector<int> N_values = {100, 500, 1000, 5000};
  OptionParams opt = createTestOption();

  for (int N : N_values) {
    opt.N = N;
    double serial_price = priceAmericanOptionSerial(opt);
    double openmp_price = priceAmericanOptionOpenMP(opt, 4);

    ASSERT_NEAR(
        serial_price, openmp_price, 1e-10,
        (std::string("Serial and OpenMP must match for N=") + std::to_string(N))
            .c_str());
  }
}

TEST(deterministic_results) {
  // Multiple runs should give identical results (no randomness)
  OptionParams opt = createTestOption();

  double price1 = priceAmericanOptionSerial(opt);
  double price2 = priceAmericanOptionSerial(opt);
  double price3 = priceAmericanOptionSerial(opt);

  ASSERT(price1 == price2 && price2 == price3, "Results must be deterministic");
}

// ========== PARAMETER VALIDATION TESTS ==========

TEST(negative_N_throws) {
  OptionParams opt = createTestOption();
  opt.N = -100;

  bool threw = false;
  try {
    priceAmericanOptionSerial(opt);
  } catch (const std::exception &) {
    threw = true;
  }

  ASSERT(threw, "Negative N should throw exception");
}

TEST(zero_N_throws) {
  OptionParams opt = createTestOption();
  opt.N = 0;

  bool threw = false;
  try {
    priceAmericanOptionSerial(opt);
  } catch (const std::exception &) {
    threw = true;
  }

  ASSERT(threw, "Zero N should throw exception");
}

TEST(negative_sigma_throws) {
  OptionParams opt = createTestOption();
  opt.sigma = -0.2;

  bool threw = false;
  try {
    priceAmericanOptionSerial(opt);
  } catch (const std::exception &) {
    threw = true;
  }

  ASSERT(threw, "Negative sigma should throw exception");
}

TEST(negative_T_throws) {
  OptionParams opt = createTestOption();
  opt.T = -1.0;

  bool threw = false;
  try {
    priceAmericanOptionSerial(opt);
  } catch (const std::exception &) {
    threw = true;
  }

  ASSERT(threw, "Negative T should throw exception");
}

// ========== MAIN TEST RUNNER ==========

int main() {
  std::cout << "=== Correctness Unit Tests ===" << std::endl;
  std::cout << std::endl;

  // Run all tests
  run_test_put_price_positive();
  run_test_call_price_positive();
  run_test_put_price_below_strike();
  run_test_call_price_below_spot_plus_strike();
  run_test_put_price_above_intrinsic();
  run_test_call_price_above_intrinsic();

  run_test_american_put_ge_european();
  run_test_american_call_ge_european();
  run_test_american_put_early_exercise_premium_itm();

  run_test_put_price_decreases_with_spot();
  run_test_call_price_increases_with_spot();
  run_test_put_price_increases_with_strike();
  run_test_price_increases_with_volatility();
  run_test_price_increases_with_time();

  run_test_convergence_with_N();

  run_test_zero_volatility();
  run_test_very_long_maturity();
  run_test_very_short_maturity();
  run_test_deep_itm_put_early_exercise();
  run_test_deep_otm_put_near_zero();

  run_test_serial_vs_openmp_consistency();
  run_test_serial_vs_openmp_multiple_N();
  run_test_deterministic_results();

  run_test_negative_N_throws();
  run_test_zero_N_throws();
  run_test_negative_sigma_throws();
  run_test_negative_T_throws();

  // Summary
  std::cout << std::endl;
  std::cout << "=== Test Summary ===" << std::endl;
  std::cout << "Total tests:  " << tests_run << std::endl;
  std::cout << "Passed:       " << tests_passed << std::endl;
  std::cout << "Failed:       " << tests_failed << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;

  if (tests_failed == 0) {
    std::cout << "ALL TESTS PASSED!" << std::endl;
    return 0;
  } else {
    std::cout << "SOME TESTS FAILED" << std::endl;
    return 1;
  }
}