#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

/**
 * @file cuda_kernels.cuh
 * @brief CUDA Kernel Declarations
 *
 * This header file contains declarations for all CUDA kernel host functions
 */

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

// Hybrid Implementations
double priceAmericanOptionCUDAHybridGPU(const OptionParams &opt);
double priceAmericanOptionCUDAHybridCPUGPU(const OptionParams &opt);
double priceAmericanOptionCUDAOpenMPHybrid(const OptionParams &opt);

// Threshold Configuration
void setHybridCPUGPUThresholds(int cpu_thresh, int large_n_thresh);
void setHybridOpenMPThresholds(int cpu_thresh, int large_n_thresh);
void setHybridGPUThresholds(int large_n_thresh);

#endif // CUDA_KERNELS_CUH
