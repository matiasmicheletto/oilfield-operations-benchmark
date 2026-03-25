#pragma once
#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <unistd.h>
#include <sstream>
#include <limits.h>
#include <iomanip>
#include <string>
#include <random>
#include <type_traits>
#include <filesystem>
#include <stdexcept>
#include <sys/resource.h>

#if defined(_WIN32)
    #include <windows.h>
#elif defined(__APPLE__)
    #include <mach-o/dyld.h>
    #include <limits.h>
#elif defined(__linux__)
    #include <unistd.h>
    #include <limits.h>
#endif

// Same as __DBL_MAX__ from <cfloat> but compatible with C++17
#define DBL_MAX std::numeric_limits<double>::max()
#define DBL_MIN std::numeric_limits<double>::lowest()

/**
 * 
 * @brief Utility functions and specification constants
 * 
 */

namespace utils { // Utility functions

// Get directory of the executable (to load the manual file if not specified)
std::filesystem::path getBinaryDir();
std::string getBinaryDirStr();

// Generate a simple UUID (not RFC4122 compliant, just for unique IDs)
std::string generate_uuid();
std::string generate_uuid_short();

// Print help message from file
inline constexpr const char defaultMessage[] = "";
void printHelp(const char* file, const char* message = defaultMessage); 

std::string currentDateTime();

long getPeakMemoryUsageKB();

long long getElapsedMs(const std::chrono::high_resolution_clock::time_point& start_time);

// Random number generator
static std::random_device rd;
static std::mt19937 gen(rd());

struct NullBuffer : std::streambuf {
    int overflow(int c) override { return c; }
};

// Debug output stream (disabled by default)
// Usage: dbg << "Debug info: " << value << std::endl;
inline NullBuffer null_buffer;
inline std::ostream null_stream(&null_buffer);
inline std::ostream& dbg = null_stream;

// ANSI color codes for terminal output
constexpr const char* red   = "\033[31m";
constexpr const char* green = "\033[32m";
constexpr const char* reset = "\033[0m";

inline void throw_runtime_error(const std::string& message) { 
    dbg << red << message << reset << "\n"; 
    std::cerr << red << message << reset << std::endl; 
    throw std::runtime_error(message); 
};

inline bool areEqual(double a, double b) { return std::fabs(a - b) < 1e-9; }

double randNormal(double mean, double stddev);
double clamp(double value, double minVal, double maxVal);

} // namespace utils

#endif // UTILS_HPP