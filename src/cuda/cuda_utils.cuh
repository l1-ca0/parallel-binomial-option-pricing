#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Error handling macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__,          \
              __LINE__, err, cudaGetErrorString(err));                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif // CUDA_UTILS_CUH
