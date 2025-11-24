#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>
#include <string>

/**
 * @file timer.h
 * @brief High-resolution timer utility for performance measurement
 *
 * Provides microsecond-precision timing for benchmarking parallel
 * implementations.
 */

class Timer {
private:
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  bool is_running;
  std::string name;

public:
  /**
   * Constructor
   * @param timer_name Optional name for the timer (for logging)
   */
  explicit Timer(const std::string &timer_name = "")
      : is_running(false), name(timer_name) {}

  /**
   * Start the timer
   */
  void start() {
    start_time = std::chrono::high_resolution_clock::now();
    is_running = true;
  }

  /**
   * Stop the timer
   */
  void stop() {
    end_time = std::chrono::high_resolution_clock::now();
    is_running = false;
  }

  /**
   * Get elapsed time in milliseconds
   * @return Elapsed time in ms (with microsecond precision)
   */
  double elapsed_ms() const {
    auto end =
        is_running ? std::chrono::high_resolution_clock::now() : end_time;
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_time);
    return duration.count() / 1000.0;
  }

  /**
   * Get elapsed time in seconds
   * @return Elapsed time in seconds
   */
  double elapsed_sec() const { return elapsed_ms() / 1000.0; }

  /**
   * Get elapsed time in microseconds
   * @return Elapsed time in microseconds
   */
  long long elapsed_us() const {
    auto end =
        is_running ? std::chrono::high_resolution_clock::now() : end_time;
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start_time);
    return duration.count();
  }

  /**
   * Print elapsed time with optional message
   * @param message Custom message to print
   */
  void print(const std::string &message = "") const {
    std::string msg = message.empty() ? name : message;
    if (!msg.empty()) {
      std::cout << msg << ": ";
    }
    std::cout << elapsed_ms() << " ms" << std::endl;
  }

  /**
   * Reset the timer (stops if running)
   */
  void reset() { is_running = false; }
};

/**
 * RAII-style scoped timer that automatically prints on destruction
 * Useful for timing code blocks
 */
class ScopedTimer {
private:
  Timer timer;
  std::string message;
  bool print_on_destruct;

public:
  explicit ScopedTimer(const std::string &msg, bool print = true)
      : message(msg), print_on_destruct(print) {
    timer.start();
  }

  ~ScopedTimer() {
    if (print_on_destruct) {
      timer.stop();
      timer.print(message);
    }
  }

  double elapsed_ms() const { return timer.elapsed_ms(); }
};

#endif // TIMER_H