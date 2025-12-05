/**
 * @file mpi_batch.cpp
 * @brief MPI Implementation for High-Throughput Batch Option Pricing
 *
 * This file implements the standalone benchmark executable.
 * It uses the shared logic defined in mpi_pricing.h/cpp.
 */

#include "../common/timer.h"
#include "mpi_pricing.h"
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Default Configuration
  int num_options = 1024;
  int steps = 1000; // Changed from 100000 to 1000
  std::string method = "serial";

  // Argument Parsing
  if (argc >= 2)
    num_options = std::stoi(argv[1]);
  if (argc >= 3)
    steps = std::stoi(argv[2]);
  if (argc >= 4)
    method = argv[3];

  // Validate Method
  if (!isValidMethod(method)) {
    if (rank == 0) {
      std::cerr << "Error: Unknown method '" << method << "'" << std::endl;
      std::cerr << "Supported methods: serial, openmp, cuda_hybrid, "
                   "cuda_hybrid_cpu_gpu"
                << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Print Configuration (Rank 0 only)
  if (rank == 0) {
    std::cout << "========================================" << std::endl;
    std::cout << "      MPI Option Pricing Benchmark      " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Processes: " << size << std::endl;
    std::cout << "  Options:   " << num_options << std::endl;
    std::cout << "  Steps (N): " << steps << std::endl;
    std::cout << "  Method:    " << method << std::endl;
    std::cout << "----------------------------------------" << std::endl;
  }

  // Generate Synthetic Workload (Rank 0 only)
  std::vector<OptionParams> options;
  if (rank == 0) {
    options.resize(num_options);
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> distS(80.0, 120.0);
    std::uniform_real_distribution<double> distK(90.0, 110.0);
    std::uniform_real_distribution<double> distVol(0.1, 0.5);
    std::uniform_real_distribution<double> distT(0.5, 2.0);

    for (int i = 0; i < num_options; ++i) {
      // Generate American Put options
      options[i] = {distS(rng), distK(rng), 0.05, distVol(rng),
                    distT(rng), steps,      false};
    }
    std::cout << "Rank 0: Generated " << num_options << " synthetic options."
              << std::endl;
  }

  // Synchronization before timing
  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  // Execute the MPI Pricing Logic
  std::vector<double> results = run_mpi_pricing(options, method, rank, size);

  // Synchronization after timing
  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();

  // Report Results
  if (rank == 0) {
    double total_time = end_time - start_time;
    double sum = 0.0;
    for (double p : results)
      sum += p;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Benchmark Complete." << std::endl;
    std::cout << "  Total Time:      " << std::fixed << std::setprecision(4)
              << total_time << " s" << std::endl;
    std::cout << "  Throughput:      " << std::fixed << std::setprecision(2)
              << num_options / total_time << " options/sec" << std::endl;
    std::cout << "  Total Price Sum: " << std::scientific << sum
              << " (Verification Check)" << std::endl;
    std::cout << "========================================" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
