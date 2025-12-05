/**
 * @file mpi_pricing.cpp
 * @brief Core MPI Implementation for High-Throughput Batch Option Pricing
 *
 * This file implements the core logic for distributing a large batch of option
 * pricing tasks across multiple MPI nodes. It uses a "Master-Worker" pattern
 * with static partitioning to achieve embarrassing parallelism.
 *
 * Key Features:
 * - **Hybrid Parallelism**: Combines MPI (inter-node) with OpenMP or CUDA
 * (intra-node).
 * - **Static Partitioning**: Workload is divided evenly among ranks based on
 * their ID.
 * - **Efficient Communication**: Uses `MPI_Bcast` for input distribution and
 * `MPI_Gatherv` for result collection.
 */

#include "mpi_pricing.h"
#include "../cuda/cuda_kernels.cuh"
#include "../openmp/openmp_binomial.h"
#include "../serial/serial_binomial.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <mpi.h>

/**
 * @brief Validates if the requested pricing method is supported.
 *
 * @param method The string identifier for the method (e.g., "serial",
 * "cuda_hybrid").
 * @return true if the method is valid, false otherwise.
 */
bool isValidMethod(const std::string &method) {
  return method == "serial" || method == "openmp" ||
         method == "cuda_hybrid_cpu_gpu";
}

/**
 * @brief Executes the MPI batch pricing workflow.
 *
 * This is the core function that coordinates the parallel execution. It
 * performs the following steps:
 * 1. **Broadcast Metadata**: Rank 0 sends the total number of options to all
 * other ranks.
 * 2. **Broadcast Data**: Rank 0 sends the actual option parameters to all
 * ranks.
 *    - Note: For extremely large datasets (>2GB), `MPI_Scatterv` would be
 * preferred, but `MPI_Bcast` is sufficient and simpler for typical batch sizes
 * in this context.
 * 3. **Partition Workload**: Each rank calculates its specific start and end
 * indices [start, end) to process. This ensures no overlap and full coverage.
 * 4. **Execute Pricing**: Each rank iterates through its assigned chunk and
 * calls the specified pricing kernel (Serial, OpenMP, or CUDA).
 * 5. **Gather Results**: All partial results are collected back to Rank 0 using
 * `MPI_Gatherv` to handle potentially uneven chunk sizes (if N is not divisible
 * by size).
 *
 * @param options Input vector of options. Must be populated on Rank 0. Ignored
 * on other ranks.
 * @param method  The pricing backend to use.
 * @param rank    Current MPI process rank.
 * @param size    Total number of MPI processes.
 * @return std::vector<double> The collected prices (valid only on Rank 0).
 */
std::vector<double> run_mpi_pricing(const std::vector<OptionParams> &options,
                                    const std::string &method, int rank,
                                    int size) {

  int num_options = options.size();

  // -------------------------------------------------------------------------
  // Step 1: Metadata Broadcast
  // -------------------------------------------------------------------------
  // Ensure all ranks know how many options are being processed.
  MPI_Bcast(&num_options, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // -------------------------------------------------------------------------
  // Step 2: Data Preparation & Broadcast
  // -------------------------------------------------------------------------
  // Workers allocate memory to receive the workload.
  std::vector<OptionParams> local_options;
  if (rank == 0) {
    local_options = options; // Master already has the data
  } else {
    local_options.resize(num_options); // Workers prepare buffer
  }

  // Broadcast the entire dataset. OptionParams is a POD struct, so it can be
  // sent as raw bytes.
  MPI_Bcast(local_options.data(), num_options * sizeof(OptionParams), MPI_BYTE,
            0, MPI_COMM_WORLD);

  // -------------------------------------------------------------------------
  // Step 3: Workload Partitioning
  // -------------------------------------------------------------------------
  // Calculate the range [start_idx, end_idx) for this rank.
  // We distribute the remainder (num_options % size) to the first few ranks to
  // ensure balance.
  int items_per_rank = num_options / size;
  int remainder = num_options % size;
  int start_idx = rank * items_per_rank + std::min(rank, remainder);
  int end_idx = start_idx + items_per_rank + (rank < remainder ? 1 : 0);
  int local_count = end_idx - start_idx;

  // -------------------------------------------------------------------------
  // Step 4: Parallel Computation
  // -------------------------------------------------------------------------
  std::vector<double> local_results;
  local_results.reserve(local_count);

  for (int i = start_idx; i < end_idx; ++i) {
    const auto &opt = local_options[i];
    double price = 0.0;

    // Dispatch to the requested pricing kernel
    if (method == "serial") {
      price = priceAmericanOptionSerial(opt);
    } else if (method == "openmp") {
      // Note: OpenMP handles intra-node parallelism here
      price = priceAmericanOptionOpenMP(opt);
    } else if (method == "cuda_hybrid_cpu_gpu") {
      // Adaptive CPU/GPU hybrid approach
      price = priceAmericanOptionCUDAHybridCPUGPU(opt);
    } else {
      if (rank == 0)
        std::cerr << "Error: Unknown method: " << method << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    local_results.push_back(price);
  }

  // -------------------------------------------------------------------------
  // Step 5: Result Gathering
  // -------------------------------------------------------------------------
  std::vector<double> global_results;
  if (rank == 0) {
    global_results.resize(num_options);
  }

  // Prepare receive counts and displacements for MPI_Gatherv
  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      int i_start = i * items_per_rank + std::min(i, remainder);
      int i_end = i_start + items_per_rank + (i < remainder ? 1 : 0);
      recvcounts[i] = i_end - i_start;
      displs[i] = i_start;
    }
  }

  // Gather all partial results into the global vector on Rank 0
  MPI_Gatherv(local_results.data(), local_count, MPI_DOUBLE,
              global_results.data(), recvcounts.data(), displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return global_results;
}
