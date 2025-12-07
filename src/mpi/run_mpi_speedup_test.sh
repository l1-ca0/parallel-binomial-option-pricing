#!/bin/bash

# Usage: ./run_speedup_test.sh [METHOD] [NUM_OPTIONS] [PROCS]
# Example: ./run_speedup_test.sh serial 1000 8

METHOD=${1:-"cuda_hybrid_cpu_gpu"}
NUM_OPTIONS=${2:-200}
PROCS=${3:-8}
STEPS=100000

# Compile first
echo "Building executables..."
make bin/mpi_batch > /dev/null 2>&1

if [ ! -f bin/mpi_batch ]; then
    echo "Error: bin/mpi_batch not found. Compilation failed?"
    exit 1
fi

echo "=========================================================="
echo "       MPI Speedup Comparison: Sequential vs Batch"
echo "=========================================================="
echo "Method:      $METHOD"
echo "Options:     $NUM_OPTIONS"
echo "Steps:       $STEPS"
echo "MPI Ranks:   $PROCS"
echo "----------------------------------------------------------"

# 1. Sequential Baseline
# We use 'mpirun -np 1' to run the benchmark on a single process.
# This forces the code to loop through all 1000 options sequentially.
# This is functionally equivalent to "running the individual pricer 1000 times"
# but without the heavy overhead of starting a new process for every option.
echo "[1] Running Individual Pricer $NUM_OPTIONS times (Sequential)..."
OUT_1=$(mpirun -np 1 ./bin/mpi_batch $NUM_OPTIONS $STEPS $METHOD)
TIME_1=$(echo "$OUT_1" | grep "Total Time" | awk '{print $3}')

if [ -z "$TIME_1" ]; then
    echo "Error running baseline. Output:"
    echo "$OUT_1"
    exit 1
fi

echo "    Time: $TIME_1 seconds"

# 2. Parallel Batch
# We use 'mpirun -np $PROCS' to distribute the 1000 options across N ranks.
echo ""
echo "[2] Running Batch of $NUM_OPTIONS options with MPI (Parallel)..."
OUT_N=$(mpirun -np $PROCS ./bin/mpi_batch $NUM_OPTIONS $STEPS $METHOD)
TIME_N=$(echo "$OUT_N" | grep "Total Time" | awk '{print $3}')

if [ -z "$TIME_N" ]; then
    echo "Error running parallel. Output:"
    echo "$OUT_N"
    exit 1
fi

echo "    Time: $TIME_N seconds"

# 3. Calculate Speedup
SPEEDUP=$(awk "BEGIN {print $TIME_1 / $TIME_N}")
EFFICIENCY=$(awk "BEGIN {print ($SPEEDUP / $PROCS) * 100}")

echo "----------------------------------------------------------"
echo "Speedup:       ${SPEEDUP}x"
echo "Efficiency:    ${EFFICIENCY}%"
echo "=========================================================="

