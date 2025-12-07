CXX = g++
NVCC = nvcc
MPICXX = mpic++
CXXFLAGS = -O3 -std=c++17 -Wall
NVCCFLAGS = -O3 -std=c++17 -arch=sm_70 -ccbin g++-11 -Xcompiler -fopenmp
# OpenMP flags (Standard Linux)
OMPFLAGS = -fopenmp

BIN_DIR = bin
COMMON = src/common/option.cpp

# Automatically find all CUDA source files, excluding the main driver
CUDA_ALL_SRCS = $(wildcard src/cuda/*.cu)
CUDA_MAIN = src/cuda/cuda_main.cu
CUDA_SRCS = $(filter-out $(CUDA_MAIN), $(CUDA_ALL_SRCS))

# Define targets
SERIAL_TARGETS = $(BIN_DIR)/serial_binomial
OPENMP_TARGETS = $(BIN_DIR)/openmp_binomial $(BIN_DIR)/test_openmp_correctness $(BIN_DIR)/test_openmp_benchmarks $(BIN_DIR)/benchmark_openmp
CUDA_TARGETS = $(BIN_DIR)/cuda_binomial $(BIN_DIR)/test_cuda_correctness $(BIN_DIR)/test_cuda_benchmarks $(BIN_DIR)/benchmark_cuda
MPI_TARGETS = $(BIN_DIR)/mpi_batch

# Build all targets by default
all: dirs $(SERIAL_TARGETS) $(OPENMP_TARGETS) $(CUDA_TARGETS) $(MPI_TARGETS)

# Create output directory
dirs:
	mkdir -p $(BIN_DIR) obj

# Serial implementation
$(BIN_DIR)/serial_binomial: src/serial/serial_main.cpp src/serial/serial_binomial.cpp $(COMMON) | dirs
	$(CXX) $(CXXFLAGS) $^ -o $@

# CUDA Object generation
CUDA_OBJS = $(patsubst src/cuda/%.cu, obj/%.o, $(CUDA_SRCS))

obj/%.o: src/cuda/%.cu | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# MPI Shared Logic
obj/mpi_pricing.o: src/mpi/mpi_pricing.cpp src/mpi/mpi_pricing.h | dirs
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -c src/mpi/mpi_pricing.cpp -o $@

# MPI Targets
obj/mpi_batch.o: src/mpi/mpi_batch.cpp src/mpi/mpi_pricing.h | dirs
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -c src/mpi/mpi_batch.cpp -o $@

$(BIN_DIR)/test_mpi_benchmarks: tests/test_mpi_benchmarks.cpp obj/mpi_pricing.o $(COMMON) src/serial/serial_binomial.cpp src/openmp/openmp_binomial.cpp $(CUDA_OBJS)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -L/usr/local/cuda/lib64 -lcudart $^ -o $@

$(BIN_DIR)/mpi_batch: obj/mpi_batch.o obj/mpi_pricing.o $(COMMON) src/serial/serial_binomial.cpp src/openmp/openmp_binomial.cpp obj/cuda_hybrid_cpu_gpu.o
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -I/usr/include/mpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lmpi_cxx $^ -o $@

# CUDA Tests
$(BIN_DIR)/test_cuda_correctness: tests/test_cuda_correctness.cu $(CUDA_SRCS) $(COMMON) src/serial/serial_binomial.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BIN_DIR)/test_cuda_benchmarks: tests/test_cuda_benchmarks.cu $(CUDA_SRCS) $(COMMON)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# OpenMP parallel implementation
$(BIN_DIR)/openmp_binomial: src/openmp/openmp_main.cpp src/openmp/openmp_binomial.cpp src/serial/serial_binomial.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

# Unit tests (Serial & OpenMP)
$(BIN_DIR)/test_openmp_correctness: tests/test_openmp_correctness.cpp $(COMMON) src/serial/serial_binomial.cpp src/openmp/openmp_binomial.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

$(BIN_DIR)/test_openmp_benchmarks: tests/test_openmp_benchmarks.cpp $(COMMON) src/serial/serial_binomial.cpp src/openmp/openmp_binomial.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

$(BIN_DIR)/benchmark_openmp: benchmarks/benchmark_openmp.cpp src/openmp/openmp_binomial.cpp src/serial/serial_binomial.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

$(BIN_DIR)/benchmark_openmp_dynamic: benchmarks/benchmark_openmp_dynamic.cpp src/openmp/openmp_binomial.cpp src/serial/serial_binomial.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

# CUDA implementation
$(BIN_DIR)/cuda_binomial: $(CUDA_MAIN) $(CUDA_SRCS) $(COMMON)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# CUDA Benchmark
$(BIN_DIR)/benchmark_cuda: benchmarks/benchmark_cuda.cpp $(CUDA_SRCS) $(COMMON)
	$(NVCC) $(NVCCFLAGS) -x cu $^ -o $@

# Run all tests
test: all
	@echo "===================="
	@echo "Running OpenMP correctness tests..."
	@echo "===================="
	./$(BIN_DIR)/test_openmp_correctness
	@echo ""
	@echo "===================="
	@echo "Running OpenMP benchmark tests..."
	@echo "===================="
	./$(BIN_DIR)/test_openmp_benchmarks --skip-slow
	@echo ""
	@echo "===================="
	@echo "Running CUDA correctness tests..."
	@echo "===================="
	./$(BIN_DIR)/test_cuda_correctness
	@echo ""
	@echo "===================="
	@echo "Running CUDA benchmark tests..."
	@echo "===================="
	./$(BIN_DIR)/test_cuda_benchmarks --skip-slow

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR) obj

.PHONY: all dirs clean test