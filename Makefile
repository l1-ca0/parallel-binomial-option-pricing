CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++17 -Wall
NVCCFLAGS = -O3 -std=c++17 -arch=sm_70
OMPFLAGS = -fopenmp

BIN_DIR = bin
COMMON = src/common/option.cpp

# Build all targets by default
all: dirs serial openmp test_benchmarks test_correctness

# Create output directory
dirs:
	mkdir -p $(BIN_DIR)

# Serial implementation
serial: src/serial/serial_main.cpp src/serial/serial_binomial.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $^ -o $(BIN_DIR)/serial_binomial

# OpenMP parallel implementation
openmp: src/openmp/openmp_main.cpp src/openmp/openmp_binomial.cpp src/serial/serial_binomial.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $(BIN_DIR)/openmp_binomial

# Benchmark validation tests (numerical accuracy)
test_benchmarks: tests/test_benchmarks.cpp src/serial/serial_binomial.cpp src/openmp/openmp_binomial.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $(BIN_DIR)/test_benchmarks

# Correctness unit tests (mathematical properties)
test_correctness: tests/test_correctness.cpp src/serial/serial_binomial.cpp src/openmp/openmp_binomial.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $(BIN_DIR)/test_correctness

# Run all tests
test: test_benchmarks test_correctness
	@echo "===================="
	@echo "Running correctness unit tests..."
	@echo "===================="
	./$(BIN_DIR)/test_correctness
	@echo ""
	@echo "===================="
	@echo "Running benchmark validation tests..."
	@echo "===================="
	./$(BIN_DIR)/test_benchmarks --skip-slow

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR)

.PHONY: all dirs clean test