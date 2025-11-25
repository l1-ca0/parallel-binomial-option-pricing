CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++17 -Wall
NVCCFLAGS = -O3 -std=c++17 -arch=sm_70
OMPFLAGS = -fopenmp

BIN_DIR = bin
COMMON = src/common/option.cpp

all: dirs serial openmp

dirs:
	mkdir -p $(BIN_DIR)

serial: src/serial/serial_main.cpp src/serial/serial_binomial.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $^ -o $(BIN_DIR)/serial_binomial

openmp: src/openmp/openmp_main.cpp src/openmp/openmp_binomial.cpp src/serial/serial_binomial.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $(BIN_DIR)/openmp_binomial

clean:
	rm -rf $(BIN_DIR)

.PHONY: all dirs clean