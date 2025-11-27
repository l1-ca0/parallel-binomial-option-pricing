#ifndef CUDA_TILED_CUH
#define CUDA_TILED_CUH

#include "../common/option.h"

/**
 * @brief Computes the American option price using the Tiled (Shared Memory)
 * parallelization strategy on GPU.
 *
 * Strategy:
 * - Uses shared memory to cache a tile of the option values.
 * - Performs multiple time steps (e.g., 16 or 32) entirely within shared
 * memory.
 * - Reduces global memory bandwidth significantly.
 *
 * @param opt Option parameters
 * @return double Option price at t=0
 */
double priceAmericanOptionCUDATiled(const OptionParams &opt);

#endif // CUDA_TILED_CUH
