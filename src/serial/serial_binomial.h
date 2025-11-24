#ifndef SERIAL_BINOMIAL_H
#define SERIAL_BINOMIAL_H

#include "../common/option.h"

/**
 * @file serial_binomial.h
 * @brief Serial implementation of binomial option pricing (baseline)
 *
 * This is the reference implementation for correctness validation.
 * Time complexity: O(N^2)
 * Space complexity: O(N) using wavefront approach
 */

/**
 * Price an American option using the Cox-Ross-Rubinstein binomial model
 * Serial implementation (baseline for comparison)
 *
 * @param opt Option parameters (S0, K, r, sigma, T, N, isCall)
 * @return Option price at t=0
 */
double priceAmericanOptionSerial(const OptionParams &opt);

/**
 * Price a European option using binomial model (for testing)
 * European options have analytical solutions, useful for validation
 *
 * @param opt Option parameters
 * @return Option price at t=0
 */
double priceEuropeanOptionSerial(const OptionParams &opt);

#endif // SERIAL_BINOMIAL_H