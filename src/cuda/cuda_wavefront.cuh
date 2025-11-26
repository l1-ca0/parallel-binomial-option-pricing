#ifndef CUDA_WAVEFRONT_CUH
#define CUDA_WAVEFRONT_CUH

#include "../common/option.h"

/**
 * @brief Computes the American option price using the Wavefront parallelization strategy on GPU.
 * 
 * Strategy:
 * - Launch one kernel per time step (from t = N-1 down to 0).
 * - Each kernel computes the values for the current time step's wavefront.
 * - Uses global memory for storing the option values array V.
 * - This is the "naive" GPU implementation.
 * 
 * @param opt Option parameters
 * @return double Option price at t=0
 */
double priceAmericanOptionCUDAWavefront(const OptionParams &opt);

#endif // CUDA_WAVEFRONT_CUH
