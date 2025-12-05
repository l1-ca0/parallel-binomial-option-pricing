#ifndef MPI_PRICING_H
#define MPI_PRICING_H

#include "../common/option.h"
#include <string>
#include <vector>

/**
 * @brief Validates if the requested pricing method is supported.
 *
 * @param method The string identifier for the method.
 * @return true if valid, false otherwise.
 */
bool isValidMethod(const std::string &method);

/**
 * @brief Runs the MPI pricing batch for a given set of options.
 *
 * This function handles the distribution of work (broadcasting options),
 * parallel computation using the specified method, and gathering of results.
 *
 * @param options Vector of options to price. On Rank 0, this must contain the
 * full workload.
 * @param method  The pricing method to use.
 * @param rank    The MPI rank of the calling process.
 * @param size    The total number of MPI processes.
 * @return std::vector<double> Vector of calculated prices. Only significant on
 * Rank 0.
 */
std::vector<double> run_mpi_pricing(const std::vector<OptionParams> &options,
                                    const std::string &method, int rank,
                                    int size);

#endif // MPI_PRICING_H
