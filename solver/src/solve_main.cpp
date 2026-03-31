#define MANUAL "assets/solve_manual.txt"

#include <iostream>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <getopt.h>

#include "../include/utils.h"
#include "../include/loader.h"
#include "../include/solver.h"

// ---------------------------------------------------------------------------
// CLI options
// ---------------------------------------------------------------------------
static struct option long_options[] = {
    {"help",      no_argument,       0, 'h'},
    {"version",   no_argument,       0, 'v'},
    {"dbg",       no_argument,       0, 'D'},
    {"config",    required_argument, 0, 'c'},
    {"set",       required_argument, 0, 's'},
    {"output",    required_argument, 0, 'o'},
    {"param",     required_argument, 0, 'p'},
    {"battery",   required_argument, 0, 'b'},
    {"dist",      required_argument, 0, 'd'},
    {0,           0,                 0,  0 }
};

int main(int argc, char **argv) {

    // Explicit CLI overrides (std::optional = "not provided by user")
    std::string                cfg_filename;
    std::vector<std::string>   cfg_overrides;
    std::optional<std::string> cli_param_file;
    std::optional<std::string> cli_battery_file;
    std::optional<std::string> cli_dist_file;
    std::optional<std::string> cli_output_file;
    bool                       cli_debug = false;

    int opt, option_index = 0;
    while ((opt = getopt_long(argc, argv, "vhDc:s:o:p:b:d:",
                              long_options, &option_index)) != -1) {
        switch (opt) {
            case 'h': utils::printHelp(MANUAL); return 0;
            case 'v': std::cout << "Oilfield Greedy Solver v0.1.0\n"; return 0;
            case 'D': cli_debug = true; break;
            case 'c': cfg_filename = optarg; break;
            case 's': cfg_overrides.emplace_back(optarg); break;
            case 'o': cli_output_file = optarg; break;
            case 'p': cli_param_file   = optarg; break;
            case 'b': cli_battery_file = optarg; break;
            case 'd': cli_dist_file    = optarg; break;
            case '?': return 1;
        }
    }

    // ------------------------------------------------------------------
    // Build configuration: defaults → yaml file → --set → explicit CLI flags
    // ------------------------------------------------------------------
    loader::SolverConfig cfg;

    if (!cfg_filename.empty()) {
        try {
            loader::load_yaml_config(cfg, cfg_filename);
            std::cout << "[config] Loaded: " << cfg_filename << "\n";
        } catch (const std::exception& e) {
            std::cerr << utils::red << e.what() << "\n" << utils::reset;
            return 1;
        }
    }

    for (const auto& kv : cfg_overrides)
        loader::apply_override(cfg, kv);

    // Explicit CLI flags always win over the config file
    if (cli_param_file)   cfg.param_file   = *cli_param_file;
    if (cli_battery_file) cfg.battery_file = *cli_battery_file;
    if (cli_dist_file)    cfg.dist_file    = *cli_dist_file;
    if (cli_output_file)  cfg.output_file  = *cli_output_file;
    if (cli_debug)        cfg.debug        = true;

    if (cfg.debug)
        utils::dbg.rdbuf(std::cout.rdbuf());

    // ------------------------------------------------------------------
    // Validate required inputs
    // ------------------------------------------------------------------
    if (cfg.param_file.empty() || cfg.battery_file.empty() || cfg.dist_file.empty()) {
        std::cerr << utils::red
                  << "[main] Error: param, battery and dist files are required.\n"
                  << "       Provide them via -p/-b/-d flags, --set input.*, "
                     "or the -c config file.\n"
                  << utils::reset;
        return 1;
    }

    // ------------------------------------------------------------------
    // Load instance
    // ------------------------------------------------------------------
    Instance inst;
    try {
        inst = loader::load(cfg.param_file, cfg.battery_file, cfg.dist_file);
    } catch (const std::exception& e) {
        std::cerr << utils::red << "[main] Load error: " << e.what()
                  << "\n" << utils::reset;
        return 1;
    }

    std::cout << "[main] Loaded " << inst.wells.size() << " wells, "
              << inst.batteries.size() << " batteries, "
              << inst.dist_matrix.size() << "x"
              << (inst.dist_matrix.empty() ? 0 : inst.dist_matrix[0].size())
              << " distance matrix.\n";
    std::cout << "[main] max_wells=" << cfg.max_wells
              << "  tolerance=" << cfg.tolerance
              << "  crews=" << cfg.crews
              << "  max_quantity=" << cfg.max_quantity
              << "  max_cost=" << cfg.max_cost
              << "  max_loss=" << cfg.max_loss << "\n";

    // ------------------------------------------------------------------
    // Solve
    // ------------------------------------------------------------------
    Solution sol;
    if (!solver::solve(inst, sol, cfg)) {
        std::cerr << utils::red << "[main] No feasible solution found.\n"
                  << utils::reset;
        return 2;
    }

    // ------------------------------------------------------------------
    // Report
    // ------------------------------------------------------------------

    // Build a well-id → Well* lookup for the report
    std::unordered_map<int, const Well*> well_by_id;
    for (const Well& w : inst.wells) well_by_id[w.id] = &w;

    // Helper to compute per-crew distance from a route vector
    auto route_dist = [&](const std::vector<int>& r) {
        double d = 0.0;
        for (int k = 0; k + 1 < static_cast<int>(r.size()); ++k)
            d += inst.dist_matrix[r[k]][r[k + 1]];
        return d;
    };

    // Helper to build a printable route string "0 → w1 → w2 → 0"
    auto route_str = [](const std::vector<int>& r) {
        std::string s;
        for (int k = 0; k < static_cast<int>(r.size()); ++k) {
            if (k) s += " → ";
            s += std::to_string(r[k]);
        }
        return s;
    };

    auto print_solution = [&](std::ostream& os) {
        os << "\n=== Solution ===\n";

        os << "Selected wells (" << sol.selected_ids.size() << "):";
        for (int id : sol.selected_ids) os << " " << id;
        os << "\n";

        os << "Per-well regimes:\n";
        for (int id : sol.selected_ids) {
            const Well* w = well_by_id.count(id) ? well_by_id.at(id) : nullptr;
            if (!w) continue;
            os << "  Well " << id
               << ": current_regime=" << w->current_regime
               << " -> new_regime=";
            if (id < static_cast<int>(sol.new_regimes.size()))
                os << sol.new_regimes[id];
            else
                os << "N/A";
            os << "\n";
        }

        const int n_crews = static_cast<int>(sol.crew_routes.size());
        os << "Per-crew routes (" << cfg.crews << " crews):\n";
        for (int c = 0; c < n_crews; ++c) {
            const auto& cr = sol.crew_routes[c];
            if (cr.empty()) {
                os << "  Crew " << (c + 1) << ": (no wells assigned)\n";
            } else {
                os << "  Crew " << (c + 1) << ": " << route_str(cr)
                   << "   (distance: " << route_dist(cr) << ")\n";
            }
        }

        os << "Total distance: " << sol.total_distance << "\n";
        os << "Total cost: "     << sol.total_cost     << "\n";
        os << "Total loss (actual): " << sol.total_loss << "\n";
    };

    print_solution(std::cout);

    if (!cfg.output_file.empty()) {
        std::ofstream out(cfg.output_file);
        if (out) {
            print_solution(out);
            std::cout << "[main] Solution written to " << cfg.output_file << "\n";
        } else {
            std::cerr << utils::red
                      << "[main] Warning: cannot open output file: "
                      << cfg.output_file << "\n" << utils::reset;
        }
    }

    return 0;
}

