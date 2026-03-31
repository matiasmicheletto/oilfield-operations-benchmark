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

// ---------------------------------------------------------------------------
// Routing helpers
// ---------------------------------------------------------------------------

// Nearest-Neighbour TSP route starting and ending at the depot (index 0).
//
//   well_indices  – dist_matrix row/col indices of the wells to visit
//                   (each index matches the corresponding well.id)
//   dist          – full (n+1)×(n+1) travel-time matrix
//
// Returns the ordered visit sequence (first and last element are 0 = depot)
// and the accumulated travel time.
std::pair<std::vector<int>, double>
nearest_neighbor_route(const std::vector<int>& well_indices,
                       const std::vector<std::vector<double>>& dist);

// 2-opt local-search improvement applied in-place to a single crew route.
//   route – [0, w1, ..., wk, 0] (depot at both ends, not modified as endpoints)
//   dist  – full (n+1)×(n+1) travel-time matrix
// Iterates until no improving 2-opt swap is found.
void two_opt(std::vector<int>& route,
             const std::vector<std::vector<double>>& dist);

// Partition well_ids into `crews` subsets (round-robin by distance from depot),
// run nearest-neighbour TSP + 2-opt per crew, and return the per-crew routes
// along with the total combined distance.
//
//   well_ids  – well IDs to route (each doubles as dist_matrix index)
//   dist      – full (n+1)×(n+1) travel-time matrix
//   crews     – number of crews (>= 1)
//
// Returns {crew_routes, total_distance}.
// Crews with no assigned wells are represented as an empty vector.
// When crews == 1 the result is equivalent to nearest_neighbor_route + two_opt.
std::pair<std::vector<std::vector<int>>, double>
multi_crew_route(const std::vector<int>& well_ids,
                 const std::vector<std::vector<double>>& dist,
                 int crews);

} // namespace utils

#endif // UTILS_HPP