#pragma once
#ifndef MODELS_H
#define MODELS_H

#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Core data structures – mirrors the columns produced by WellGenerator
// and the .dat files written by the Python pipeline:
//   parameters_*.dat  →  ID  G  N  r  R  P  C  B
//   batteries_*.dat   →  ID  Gpt
//   distance_*.dat    →  (n+1)×(n+1) matrix, row/col 0 = depot
// ---------------------------------------------------------------------------

struct Well {
    int    id;              // unique 1-indexed well ID
    double gross_prod;      // G – max gross production capacity at 100 % regime
                            //     (file column G is production at current_regime;
                            //      the loader normalises it: gross_prod = G * 100 / r)
    double net_prod;        // N – max net production at 100 % regime
                            //     (normalised from file column N the same way as gross_prod)
    double current_regime;  // r – current production regime (0-100)
    double risk;            // R – risk factor   (0-100)
    double priority;        // P – priority level (0-100)
    double cost;            // C – intervention cost
    int    battery_id;      // B – parent battery ID
};

struct Battery {
    int    id;
    double target_gross;    // Gpt – target gross production for this battery
	double max_loss;
	double max_cost;
};

struct Instance {
    std::string name;
    std::vector<Well>                 wells;
    std::vector<Battery>              batteries;
    // (n+1) × (n+1) travel-time matrix.
    // Index 0   = operations-centre depot.
    // Index i   = well whose id == i  (1-indexed, matching the parameters file).
    std::vector<std::vector<double>>  dist_matrix;
};

enum class PRINT_FORMAT { TXT, ROUTES };

struct Solution {
    std::vector<int>    selected_ids;   // Well IDs chosen for intervention
    double              total_distance;
    bool                feasible;

    // new_regimes[i] = assigned production regime (0-100) for well with id==i.
    // Indexed by well ID (1-based); index 0 is unused.
    // Only meaningful for IDs in selected_ids; all others are 0.
    std::vector<double> new_regimes;

    // One route per crew: each route is [0, w1, w2, ..., wk, 0] (depot at both ends).
    // Empty crews are represented as an empty vector (not {0,0}).
    std::vector<std::vector<int>> crew_routes;

    double total_cost;  // sum C[i] over selected wells
    double total_loss;  // sum (G[i]-N[i]) * newregime[i] / 100 over selected wells

    // Legacy single-route view (union of all crew routes, excluding depot repetitions).
    // Kept for backward compatibility; prefer crew_routes for new code.
    std::vector<int> route;

    void print(std::ostream& os, const Instance& inst, int n_crews,
               PRINT_FORMAT fmt = PRINT_FORMAT::TXT) const {
        if (fmt == PRINT_FORMAT::ROUTES) {
            // One line per crew: crew_id well_id well_id ...
            // Depot (0) is omitted so the caller gets plain well sequences.
            const int nc = static_cast<int>(crew_routes.size());
            for (int c = 0; c < nc; ++c) {
                const auto& cr = crew_routes[c];
                os << (c + 1);
                for (int node : cr)
                    if (node != 0) os << " " << node;
                os << "\n";
            }
            return;
        }

        // PRINT_FORMAT::TXT (default)
        std::unordered_map<int, const Well*> well_by_id;
        for (const Well& w : inst.wells) well_by_id[w.id] = &w;

        auto route_dist = [&](const std::vector<int>& r) {
            double d = 0.0;
            for (int k = 0; k + 1 < static_cast<int>(r.size()); ++k)
                d += inst.dist_matrix[r[k]][r[k + 1]];
            return d;
        };

        auto route_str = [](const std::vector<int>& r) {
            std::string s;
            for (int k = 0; k < static_cast<int>(r.size()); ++k) {
                if (k) s += " → ";
                s += std::to_string(r[k]);
            }
            return s;
        };

        os << "\n=== Solution ===\n";

        os << "Selected wells (" << selected_ids.size() << "):";
        for (int id : selected_ids) os << " " << id;
        os << "\n";

        os << "Per-well regimes:\n";
        for (int id : selected_ids) {
            const Well* w = well_by_id.count(id) ? well_by_id.at(id) : nullptr;
            if (!w) continue;
            os << "  Well " << id
               << ": current_regime=" << w->current_regime
               << " -> new_regime=";
            if (id < static_cast<int>(new_regimes.size()))
                os << new_regimes[id];
            else
                os << "N/A";
            os << "\n";
        }

        const int nc = static_cast<int>(crew_routes.size());
        os << "Per-crew routes (" << n_crews << " crews):\n";
        for (int c = 0; c < nc; ++c) {
            const auto& cr = crew_routes[c];
            if (cr.empty()) {
                os << "  Crew " << (c + 1) << ": (no wells assigned)\n";
            } else {
                os << "  Crew " << (c + 1) << ": " << route_str(cr)
                   << "   (distance: " << route_dist(cr) << ")\n";
            }
        }

        os << "Total distance: " << total_distance << "\n";
        os << "Total cost: "     << total_cost     << "\n";
        os << "Total loss (actual): " << total_loss << "\n";
    }
};

#endif // MODELS_H
