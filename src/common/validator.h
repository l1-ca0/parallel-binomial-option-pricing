#ifndef VALIDATOR_H
#define VALIDATOR_H

#include "option.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

/**
 * @file validator.h
 * @brief Validation utilities for option pricing correctness
 *
 * Provides functions to validate option prices against known bounds,
 * compare implementations, and check for numerical correctness.
 */

class Validator {
public:
  /**
   * Compare two option prices with tolerance
   * @param price1 First price
   * @param price2 Second price
   * @param rel_tol Relative tolerance (default 1e-6 = 0.0001%)
   * @param abs_tol Absolute tolerance (default 1e-9)
   * @return true if prices are within tolerance
   */
  static bool comparePrices(double price1, double price2, double rel_tol = 1e-6,
                            double abs_tol = 1e-9) {
    if (std::isnan(price1) || std::isnan(price2)) {
      return false;
    }
    if (std::isinf(price1) || std::isinf(price2)) {
      return false;
    }

    double abs_diff = std::abs(price1 - price2);
    double rel_diff = abs_diff / std::max(std::abs(price1), std::abs(price2));

    return abs_diff <= abs_tol || rel_diff <= rel_tol;
  }

  /**
   * Validate American call option bounds
   * An American call should satisfy:
   * 1. Price >= intrinsic value: max(S - K, 0)
   * 2. Price <= spot price: S
   * 3. Price >= European call price
   *
   * @param price Option price to validate
   * @param opt Option parameters
   * @return true if all bounds are satisfied
   */
  static bool validateAmericanCall(double price, const OptionParams &opt) {
    if (!opt.isCall)
      return false;

    double intrinsic = std::max(opt.S0 - opt.K, 0.0);
    double upper_bound = opt.S0;

    bool valid = true;

    // Check lower bound: price >= intrinsic value
    if (price < intrinsic - 1e-10) {
      std::cerr << "ERROR: Call price (" << price << ") below intrinsic value ("
                << intrinsic << ")" << std::endl;
      valid = false;
    }

    // Check upper bound: price <= spot price
    if (price > upper_bound + 1e-10) {
      std::cerr << "ERROR: Call price (" << price << ") above spot price ("
                << upper_bound << ")" << std::endl;
      valid = false;
    }

    return valid;
  }

  /**
   * Validate American put option bounds
   * An American put should satisfy:
   * 1. Price >= intrinsic value: max(K - S, 0)
   * 2. Price <= strike price: K
   * 3. Price >= European put price
   *
   * @param price Option price to validate
   * @param opt Option parameters
   * @return true if all bounds are satisfied
   */
  static bool validateAmericanPut(double price, const OptionParams &opt) {
    if (opt.isCall)
      return false;

    double intrinsic = std::max(opt.K - opt.S0, 0.0);
    double upper_bound = opt.K;

    bool valid = true;

    // Check lower bound: price >= intrinsic value
    if (price < intrinsic - 1e-10) {
      std::cerr << "ERROR: Put price (" << price << ") below intrinsic value ("
                << intrinsic << ")" << std::endl;
      valid = false;
    }

    // Check upper bound: price <= strike
    if (price > upper_bound + 1e-10) {
      std::cerr << "ERROR: Put price (" << price << ") above strike price ("
                << upper_bound << ")" << std::endl;
      valid = false;
    }

    return valid;
  }

  /**
   * Validate American option price (call or put)
   * @param price Option price to validate
   * @param opt Option parameters
   * @return true if bounds are satisfied
   */
  static bool validateAmericanOption(double price, const OptionParams &opt) {
    if (opt.isCall) {
      return validateAmericanCall(price, opt);
    } else {
      return validateAmericanPut(price, opt);
    }
  }

  /**
   * Validate that American price >= European price
   * @param american_price American option price
   * @param european_price European option price
   * @param opt Option parameters (for display)
   * @return true if American >= European (within tolerance)
   */
  static bool validateAmericanVsEuropean(double american_price,
                                         double european_price,
                                         const OptionParams &opt) {
    if (american_price < european_price - 1e-10) {
      std::cerr << "ERROR: American price (" << american_price
                << ") less than European price (" << european_price << ")"
                << std::endl;
      return false;
    }
    return true;
  }

  /**
   * Run comprehensive validation suite
   * @param price Option price to validate
   * @param opt Option parameters
   * @param implementation_name Name of implementation (for reporting)
   * @return true if all validations pass
   */
  static bool runValidationSuite(double price, const OptionParams &opt,
                                 const std::string &implementation_name = "") {
    std::cout << "=== Validation Suite";
    if (!implementation_name.empty()) {
      std::cout << " for " << implementation_name;
    }
    std::cout << " ===" << std::endl;

    bool all_passed = true;

    // Check for NaN or Inf
    std::cout << "  NaN/Inf check:            ";
    if (std::isnan(price) || std::isinf(price)) {
      std::cout << "FAIL (price is " << price << ")" << std::endl;
      all_passed = false;
    } else {
      std::cout << "PASS" << std::endl;
    }

    // Check positivity
    std::cout << "  Positivity check:         ";
    if (price < 0.0) {
      std::cout << "FAIL (price = " << price << ")" << std::endl;
      all_passed = false;
    } else {
      std::cout << "PASS" << std::endl;
    }

    // Check bounds
    std::cout << "  Bounds check:             ";
    if (validateAmericanOption(price, opt)) {
      std::cout << "PASS" << std::endl;
    } else {
      std::cout << "FAIL" << std::endl;
      all_passed = false;
    }

    std::cout << std::endl;
    return all_passed;
  }

  /**
   * Compare two implementations
   * @param price1 Price from first implementation
   * @param name1 Name of first implementation
   * @param price2 Price from second implementation
   * @param name2 Name of second implementation
   * @param rel_tol Relative tolerance
   */
  static void compareImplementations(double price1, const std::string &name1,
                                     double price2, const std::string &name2,
                                     double rel_tol = 1e-6) {
    std::cout << "=== Implementation Comparison ===" << std::endl;
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "  " << name1 << ": " << price1 << std::endl;
    std::cout << "  " << name2 << ": " << price2 << std::endl;

    double abs_diff = std::abs(price1 - price2);
    double rel_diff = abs_diff / std::max(std::abs(price1), std::abs(price2));

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Absolute difference: " << abs_diff << std::endl;
    std::cout << "  Relative difference: " << rel_diff << " ("
              << (rel_diff * 100.0) << "%)" << std::endl;

    std::cout << "  Match (tol=" << rel_tol << "): "
              << (comparePrices(price1, price2, rel_tol) ? "PASS" : "FAIL")
              << std::endl;
    std::cout << std::endl;
  }

  /**
   * Print detailed option info (for debugging)
   */
  static void printOptionDetails(const OptionParams &opt) {
    std::cout << "=== Option Details ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Type:              " << (opt.isCall ? "Call" : "Put")
              << std::endl;
    std::cout << "  Style:             American" << std::endl;
    std::cout << "  Spot price (S0):   " << opt.S0 << std::endl;
    std::cout << "  Strike (K):        " << opt.K << std::endl;
    std::cout << "  Risk-free rate:    " << opt.r << std::endl;
    std::cout << "  Volatility:        " << opt.sigma << std::endl;
    std::cout << "  Time to maturity:  " << opt.T << " years" << std::endl;
    std::cout << "  Time steps (N):    " << opt.N << std::endl;
    std::cout << "  Moneyness:         ";
    if (opt.isCall) {
      double moneyness = opt.S0 / opt.K;
      if (moneyness > 1.05)
        std::cout << "In-the-money";
      else if (moneyness < 0.95)
        std::cout << "Out-of-the-money";
      else
        std::cout << "At-the-money";
    } else {
      double moneyness = opt.K / opt.S0;
      if (moneyness > 1.05)
        std::cout << "In-the-money";
      else if (moneyness < 0.95)
        std::cout << "Out-of-the-money";
      else
        std::cout << "At-the-money";
    }
    std::cout << std::endl;

    // Calculate intrinsic value
    double intrinsic = opt.isCall ? std::max(opt.S0 - opt.K, 0.0)
                                  : std::max(opt.K - opt.S0, 0.0);
    std::cout << "  Intrinsic value:   " << intrinsic << std::endl;
    std::cout << std::endl;
  }
};

#endif // VALIDATOR_H