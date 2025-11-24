CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall
OMPFLAGS = -fopenmp

BIN_DIR = bin
COMMON = src/common/option.cpp

all: dirs serial

dirs:
	mkdir -p $(BIN_DIR)

serial: src/serial/serial_main.cpp src/serial/serial_binomial.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $^ -o $(BIN_DIR)/serial_binomial

clean:
	rm -rf $(BIN_DIR)

.PHONY: all dirs clean