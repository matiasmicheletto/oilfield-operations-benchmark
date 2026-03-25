#include "../include/solver.h"
#include "../include/utils.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace solver {

// ---------------------------------------------------------------------------
// Regime-adjustment feasibility check (per battery, per candidate window).
//
// Given K visited wells and all wells in the battery:
//   - Non-visited wells are FIXED at current regime r[i]:
//       contribution = G[i] * r[i] / 100
//   - Visited wells have adjustable regime in [0, 100]:
//       contribution in [0, G[i]]
//
// Total achievable range:
//   min_prod = base_unvisited  (visited set to 0 %)
//   max_prod = base_unvisited + sum(G[i] for visited)  (visited at 100 %)
//
// Feasible if Gpt lies inside [min_prod, max_prod] (with tolerance band).
//
// Edge-case: Gpt > sum(G) for the battery is possible due to noise in data
// generation (noise_std = 0.1).  In that situation the effective target is
// clamped to max_prod before the tolerance check.
// ---------------------------------------------------------------------------
static bool production_feasible(
        const std::vector<size_t>& window,
        const std::vector<size_t>& battery_all,
        const std::vector<Well>&   wells,
        double                     target_gross,
        double                     tolerance) {

    std::unordered_set<size_t> sel_set(window.begin(), window.end());

    // Fixed production from non-visited wells
    double base = 0.0;
    for (size_t idx : battery_all)
        if (!sel_set.count(idx))
            base += wells[idx].gross_prod * wells[idx].current_regime / 100.0;

    // Maximum additional production from visited wells (regime = 100 %)
    double max_from_sel = 0.0;
    for (size_t idx : window)
        max_from_sel += wells[idx].gross_prod;

    const double max_prod = base + max_from_sel;

    // Clamp over-capacity targets to the physically achievable maximum
    const double eff_target = std::min(target_gross, max_prod);

    return (eff_target >= base * (1.0 - tolerance)) &&
           (eff_target <= max_prod * (1.0 + tolerance));
}

// ---------------------------------------------------------------------------
// Core solver: greedy cyclic-window selection + nearest-neighbour routing
// ---------------------------------------------------------------------------
bool solve(const Instance& inst, Solution& sol,
           int max_wells, double tolerance) {
    sol.feasible       = false;
    sol.total_distance = std::numeric_limits<double>::max();
    sol.selected_ids.clear();
    sol.route.clear();

    // Group well indices by battery ID
    std::unordered_map<int, std::vector<size_t>> bat_map;
    for (size_t i = 0; i < inst.wells.size(); ++i)
        bat_map[inst.wells[i].battery_id].push_back(i);

    std::vector<int> all_selected_dist_idx;

    // ------------------------------------------------------------------
    // Process each battery independently
    // ------------------------------------------------------------------
    for (const Battery& bat : inst.batteries) {

        auto it = bat_map.find(bat.id);
        if (it == bat_map.end()) {
            std::cerr << utils::red << "[solver] Warning: battery " << bat.id
                      << " has no wells – skipped.\n" << utils::reset;
            continue;
        }

        const std::vector<size_t>& battery_all = it->second;

        // Warn if target exceeds physical maximum production
        double sum_G_bat = 0.0;
        for (size_t idx : battery_all) sum_G_bat += inst.wells[idx].gross_prod;
        if (bat.target_gross > sum_G_bat * (1.0 + tolerance))
            std::cout << utils::red
                      << "[solver] Warning: battery " << bat.id
                      << " Gpt=" << bat.target_gross
                      << " exceeds max possible=" << sum_G_bat
                      << " – target will be clamped.\n" << utils::reset;

        // 1. Sort this battery's wells: priority DESC, then cost ASC
        std::vector<size_t> order = battery_all;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const Well& wa = inst.wells[a];
            const Well& wb = inst.wells[b];
            if (!utils::areEqual(wa.priority, wb.priority))
                return wa.priority > wb.priority;
            return wa.cost < wb.cost;
        });

        const int K = std::min(max_wells, static_cast<int>(order.size()));

        // 2. Cyclic-window search
        bool found = false;
        for (size_t start = 0; start < order.size(); ++start) {

            std::vector<size_t> window;
            window.reserve(K);
            for (int i = 0; i < K; ++i)
                window.push_back(order[(start + i) % order.size()]);

            // 3. Regime-adjustment feasibility check
            if (production_feasible(window, battery_all, inst.wells,
                                    bat.target_gross, tolerance)) {

                for (size_t idx : window) {
                    all_selected_dist_idx.push_back(inst.wells[idx].id);
                    sol.selected_ids.push_back(inst.wells[idx].id);
                }
                found = true;

                // Log achievable range vs. Gpt
                std::unordered_set<size_t> sel_set(window.begin(), window.end());
                double base = 0.0;
                for (size_t idx : battery_all)
                    if (!sel_set.count(idx))
                        base += inst.wells[idx].gross_prod
                              * inst.wells[idx].current_regime / 100.0;
                double max_sel = 0.0;
                for (size_t idx : window) max_sel += inst.wells[idx].gross_prod;

                std::cout << "[solver] Battery " << bat.id
                          << ": window start=" << start
                          << "  achievable=[" << base << ", " << (base + max_sel) << "]"
                          << "  Gpt=" << bat.target_gross << "\n";
                break;
            }
        }

        if (!found) {
            std::cout << utils::red
                      << "[solver] Battery " << bat.id
                      << ": no feasible selection (K=" << K
                      << ", tol=" << (tolerance * 100.0) << "%).\n"
                      << utils::reset;
            return false;
        }
    }

    // ------------------------------------------------------------------
    // 4. Nearest-Neighbour TSP for all selected wells (single crew).
    //    Multi-crew extension: partition all_selected_dist_idx into `crews`
    //    subsets (e.g. by proximity) and route each independently (TODO).
    // ------------------------------------------------------------------
    auto [route, dist] = utils::nearest_neighbor_route(
                             all_selected_dist_idx, inst.dist_matrix);

    sol.route          = route;
    sol.total_distance = dist;
    sol.feasible       = true;

    return true;
}

} // namespace solver
