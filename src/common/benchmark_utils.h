#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include "option.h"
#include "timer.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

/**
 * @file benchmark_utils.h
 * @brief Utilities for performance benchmarking
 *
 * Provides functions for running multiple trials, computing statistics,
 * and exporting results to CSV format.
 */

struct BenchmarkResult {
  std::string implementation;
  int N;                        // Number of time steps
  double price;                 // Option price
  std::vector<double> times_ms; // Timing measurements (ms)
  double mean_time_ms;
  double std_dev_ms;
  double min_time_ms;
  double max_time_ms;
  double median_time_ms;
  long long total_nodes;
  double throughput; // Nodes per ms
};

class BenchmarkUtils {
public:
  /**
   * Compute statistics from timing measurements
   */
  static void computeStatistics(BenchmarkResult &result) {
    if (result.times_ms.empty())
      return;

    // Mean
    result.mean_time_ms =
        std::accumulate(result.times_ms.begin(), result.times_ms.end(), 0.0) /
        result.times_ms.size();

    // Standard deviation
    double sq_sum = 0.0;
    for (double t : result.times_ms) {
      sq_sum += (t - result.mean_time_ms) * (t - result.mean_time_ms);
    }
    result.std_dev_ms = std::sqrt(sq_sum / result.times_ms.size());

    // Min and max
    result.min_time_ms =
        *std::min_element(result.times_ms.begin(), result.times_ms.end());
    result.max_time_ms =
        *std::max_element(result.times_ms.begin(), result.times_ms.end());

    // Median
    std::vector<double> sorted = result.times_ms;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 0) {
      result.median_time_ms = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    } else {
      result.median_time_ms = sorted[n / 2];
    }

    // Throughput
    result.total_nodes =
        static_cast<long long>(result.N + 1) * (result.N + 2) / 2;
    result.throughput = result.total_nodes / result.mean_time_ms;
  }

  /**
   * Print benchmark result to console
   */
  static void printResult(const BenchmarkResult &result) {
    std::cout << "=== Benchmark Result: " << result.implementation
              << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  N (time steps):     " << result.N << std::endl;
    std::cout << "  Option price:       " << result.price << std::endl;
    std::cout << std::setprecision(3);
    std::cout << "  Mean time:          " << result.mean_time_ms << " ms"
              << std::endl;
    std::cout << "  Std deviation:      " << result.std_dev_ms << " ms"
              << std::endl;
    std::cout << "  Min time:           " << result.min_time_ms << " ms"
              << std::endl;
    std::cout << "  Max time:           " << result.max_time_ms << " ms"
              << std::endl;
    std::cout << "  Median time:        " << result.median_time_ms << " ms"
              << std::endl;
    std::cout << "  Total nodes:        " << result.total_nodes << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "  Throughput:         " << result.throughput << " nodes/ms"
              << std::endl;
    std::cout << std::endl;
  }

  /**
   * Export results to CSV file
   */
  static void exportToCSV(const std::vector<BenchmarkResult> &results,
                          const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "ERROR: Cannot open file " << filename << std::endl;
      return;
    }

    // Header
    file
        << "Implementation,N,Price,MeanTime_ms,StdDev_ms,MinTime_ms,MaxTime_ms,"
        << "MedianTime_ms,TotalNodes,Throughput_nodes_per_ms" << std::endl;

    // Data
    file << std::fixed << std::setprecision(10);
    for (const auto &result : results) {
      file << result.implementation << "," << result.N << "," << result.price
           << "," << result.mean_time_ms << "," << result.std_dev_ms << ","
           << result.min_time_ms << "," << result.max_time_ms << ","
           << result.median_time_ms << "," << result.total_nodes << ","
           << result.throughput << std::endl;
    }

    file.close();
    std::cout << "Results exported to " << filename << std::endl;
  }

  /**
   * Compute speedup relative to baseline
   */
  static double computeSpeedup(const BenchmarkResult &baseline,
                               const BenchmarkResult &optimized) {
    return baseline.mean_time_ms / optimized.mean_time_ms;
  }

  /**
   * Print speedup comparison
   */
  static void printSpeedup(const BenchmarkResult &baseline,
                           const BenchmarkResult &optimized) {
    double speedup = computeSpeedup(baseline, optimized);
    std::cout << "=== Speedup Analysis ===" << std::endl;
    std::cout << "  Baseline (" << baseline.implementation
              << "): " << baseline.mean_time_ms << " ms" << std::endl;
    std::cout << "  Optimized (" << optimized.implementation
              << "): " << optimized.mean_time_ms << " ms" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
    std::cout << std::endl;
  }
};

#endif // BENCHMARK_UTILS_H