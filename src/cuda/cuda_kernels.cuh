#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "../common/option.h"

// Wavefront Implementation
double priceAmericanOptionCUDAWavefront(const OptionParams &opt);

// Tiled Implementations
double priceAmericanOptionCUDATiled(const OptionParams &opt);
double priceAmericanOptionCUDASharedMemTiling(const OptionParams &opt);
double priceAmericanOptionCUDAWarpShuffleTiling(const OptionParams &opt);
double priceAmericanOptionCUDAWarpPerBlock(const OptionParams &opt);
double priceAmericanOptionCUDAIndependentMultiWarp(const OptionParams &opt);


// Persistent Implementations
double priceAmericanOptionCUDAPersistentGlobalBarrier(const OptionParams &opt);

// Time Parallel Implementation
double priceAmericanOptionTimeParallel(double S0, double K, double r,
                                       double sigma, double T, int N,
                                       bool isCall);


// Multi-Warp Implementation
double priceAmericanOptionCUDACooperativeMultiWarp(double S0, double K,
                                                   double r, double sigma,
                                                   double T, int N,
                                                   bool isCall);

#endif // CUDA_KERNELS_CUH