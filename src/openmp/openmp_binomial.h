#ifndef OPENMP_BINOMIAL_H
#define OPENMP_BINOMIAL_H

#include "../common/option.h"

/**
 * @file openmp_binomial.h
 * @brief OpenMP parallel implementation of binomial option pricing
 *
 * Parallelizes the inner loop at each time step using OpenMP.
 * Each time step requires a barrier synchronization.
 *
 * Key challenges:
 * - N barriers (one per time step) introduce synchronization overhead
 * - Load imbalance: parallelism declines from N+1 tasks to 1 task
 * - Memory bandwidth: all threads access shared arrays
 *
 */

/**
 * Price an American option using OpenMP parallelization
 * Parallelizes across nodes at each time step
 *
 * @param opt Option parameters
 * @param num_threads Number of OpenMP threads (0 = use default)
 * @return Option price at t=0
 */
double priceAmericanOptionOpenMP(const OptionParams &opt, int num_threads = 0);

/**
 * Price a European option using OpenMP (for validation)
 *
 * @param opt Option parameters
 * @param num_threads Number of OpenMP threads (0 = use default)
 * @return Option price at t=0
 */
double priceEuropeanOptionOpenMP(const OptionParams &opt, int num_threads = 0);

/**
 * Price an American option using OpenMP parallelization with Dynamic scheduling
 * Used for comparing scheduling strategies (Static vs Dynamic)
 *
 * @param opt Option parameters
 * @param num_threads Number of OpenMP threads
 * @return Option price at t=0
 */
double priceAmericanOptionOpenMPDynamic(const OptionParams &opt,
                                        int num_threads = 0);

/**
 * Get the number of OpenMP threads being used
 * @return Number of threads
 */
int getOMPThreadCount();

#endif // OPENMP_BINOMIAL_H