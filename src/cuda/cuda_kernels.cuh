#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "../common/option.h"

// Wavefront Implementation
double priceAmericanOptionCUDAWavefront(const OptionParams &opt);

// Tiled Implementations
double priceAmericanOptionCUDATiled(const OptionParams &opt);

#endif // CUDA_KERNELS_CUH