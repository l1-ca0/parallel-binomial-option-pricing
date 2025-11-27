#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "../common/option.h"

// Wavefront Implementation
double priceAmericanOptionCUDAWavefront(const OptionParams &opt);

// Tiled Implementations
double priceAmericanOptionCUDATiled(const OptionParams &opt);

// Time Parallel Implementation
double priceAmericanOptionTimeParallel(double S0, double K, double r,
                                       double sigma, double T, int N,
                                       bool isCall);

#endif // CUDA_KERNELS_CUH