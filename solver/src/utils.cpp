#include "../include/utils.h"

namespace utils {

std::filesystem::path getBinaryDir(){
#if defined(_WIN32)

    wchar_t buffer[MAX_PATH];
    DWORD len = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    if (len == 0 || len == MAX_PATH)
        throw_runtime_error("GetModuleFileNameW failed");

    return std::filesystem::path(buffer).parent_path();

#elif defined(__APPLE__)

    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size); // get required size

    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0)
        throw_runtime_error("_NSGetExecutablePath failed");

    return std::filesystem::canonical(buffer).parent_path();

#elif defined(__linux__)

    char buffer[PATH_MAX];
    ssize_t len = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len == -1)
        throw_runtime_error("Cannot resolve /proc/self/exe");

    buffer[len] = '\0';
    return std::filesystem::path(buffer).parent_path();

#else
    #error "Unsupported platform"
#endif
}

std::string getBinaryDirStr() {
#ifdef _WIN32
    char result[MAX_PATH];
    GetModuleFileName(NULL, result, MAX_PATH);
    std::string path(result);
    return path.substr(0, path.find_last_of("\\/"));
#else
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count == -1) throw_runtime_error("Cannot resolve /proc/self/exe");    
    std::string path(result, count);
    return path.substr(0, path.find_last_of('/'));
#endif
}

std::string generate_uuid() {
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    int i;
    ss << std::hex;
    for (i = 0; i < 8; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 4; i++) {
        ss << dis(gen);
    }
    ss << "-4";
    for (i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    ss << dis2(gen);
    for (i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 12; i++) {
        ss << dis(gen);
    }
    return ss.str();
}

std::string generate_uuid_short() {
    std::string full_uuid = generate_uuid();
    return full_uuid.substr(0, 8); // Return first 8 characters
}


void printHelp(const char* file, const char* message) { // Open readme file with manual and print on terminal   
    std::cerr << std::endl << message << std::endl << std::endl;
    std::ifstream manualFile(file);
    if (manualFile.is_open()) {
        std::string line;
        while (getline(manualFile, line)) {
            std::cout << line << std::endl;
        }
        manualFile.close();
        exit(1);
    } else { // try to load from executable dir
        std::string execDir = getBinaryDirStr();
        std::string fullPath = execDir + "/" + file;
        std::ifstream defaultManualFile(fullPath);
        if (defaultManualFile.is_open()) {
            std::string line;
            while (getline(defaultManualFile, line)) {
                std::cout << line << std::endl;
            }
            defaultManualFile.close();
            exit(1);
        }
    }

    throw_runtime_error("Unable to open manual file: " + std::string(file));
    exit(1);
}

std::string currentDateTime() {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

long getPeakMemoryUsageKB() {
#if defined(__linux__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw_runtime_error("getrusage failed");
    }
    return usage.ru_maxrss; // Already in KB on Linux
#elif defined(__APPLE__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw_runtime_error("getrusage failed");
    }
    return usage.ru_maxrss / 1024; // Convert from bytes to KB on macOS
#else
    throw_runtime_error("Peak memory usage not supported on this platform");
    return -1;
#endif
}

long long getElapsedMs(const std::chrono::high_resolution_clock::time_point& start_time) {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
}

double randNormal(double mean, double stddev) {
    // Thread-local random number generator
    std::normal_distribution<double> distribution(mean, stddev);
    return distribution(gen);
}

double clamp(double value, double minVal, double maxVal) {
    if (value < minVal) return minVal;
    if (value > maxVal) return maxVal;
    return value;
}

// ---------------------------------------------------------------------------
// Routing helpers
// ---------------------------------------------------------------------------

std::pair<std::vector<int>, double>
nearest_neighbor_route(const std::vector<int>& well_indices,
                       const std::vector<std::vector<double>>& dist) {
    if (well_indices.empty()) return {{0, 0}, 0.0};

    const int n = static_cast<int>(well_indices.size());
    std::vector<bool> visited(n, false);
    std::vector<int>  route;
    route.reserve(n + 2);
    route.push_back(0); // depart from depot

    int    current   = 0;
    double total     = 0.0;

    for (int step = 0; step < n; ++step) {
        double best_d   = DBL_MAX;
        int    best_pos = -1;

        for (int j = 0; j < n; ++j) {
            if (!visited[j] && dist[current][well_indices[j]] < best_d) {
                best_d   = dist[current][well_indices[j]];
                best_pos = j;
            }
        }

        visited[best_pos] = true;
        total   += best_d;
        current  = well_indices[best_pos];
        route.push_back(current);
    }

    // Return to depot
    total += dist[current][0];
    route.push_back(0);

    return {route, total};
}

} // namespace utils
