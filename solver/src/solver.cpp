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
// Per-battery production-feasibility check.
//
// Given the indices (into inst.wells) of the wells currently selected for
// this battery, and all well indices belonging to the battery:
//
//   base     = sum G[i]*r[i]/100  for non-selected wells  (fixed at current regime)
//   max_add  = sum G[i]           for selected wells       (adjustable 0–100 %)
//
// The achievable production range is [base, base + max_add].
// The battery target Gpt is clamped to max_prod to handle noisy over-capacity
// targets, then checked against the range with the given tolerance.
//
// Returns true iff Gpt can be hit by some valid regime assignment.
// ---------------------------------------------------------------------------
static bool production_feasible(
        const std::vector<size_t>& selected,   // well indices in this battery that are selected
        const std::vector<size_t>& battery_all, // all well indices in this battery
        const std::vector<Well>&   wells,
        double                     target_gross,
        double                     tolerance)
{
    std::unordered_set<size_t> sel_set(selected.begin(), selected.end());

    double base = 0.0;
    for (size_t idx : battery_all)
        if (!sel_set.count(idx))
            base += wells[idx].gross_prod * wells[idx].current_regime / 100.0;

    double max_add = 0.0;
    for (size_t idx : selected)
        max_add += wells[idx].gross_prod;

    const double max_prod  = base + max_add;
    const double eff_target = std::min(target_gross, max_prod);

    return (eff_target >= base       * (1.0 - tolerance)) &&
           (eff_target <= max_prod   * (1.0 + tolerance));
}

// ---------------------------------------------------------------------------
// Core solver: globally-aware greedy selection + nearest-neighbour routing.
// ---------------------------------------------------------------------------
bool solve(const Instance& inst, Solution& sol,
           const loader::SolverConfig& cfg)
{
    sol.feasible       = false;
    sol.total_distance = std::numeric_limits<double>::max();
    sol.selected_ids.clear();
    sol.route.clear();

    const double tolerance   = cfg.tolerance;
    const int    max_qty     = cfg.max_quantity;
    const double max_cost    = cfg.max_cost;
    const double max_loss    = cfg.max_loss;

    // ------------------------------------------------------------------
    // Build lookup structures
    // ------------------------------------------------------------------
    std::unordered_map<int, const Battery*> bat_by_id;
    for (const Battery& b : inst.batteries)
        bat_by_id[b.id] = &b;

    // battery_id → indices into inst.wells
    std::unordered_map<int, std::vector<size_t>> bat_map;
    for (size_t i = 0; i < inst.wells.size(); ++i)
        bat_map[inst.wells[i].battery_id].push_back(i);

    // Warn about batteries whose Gpt exceeds their physical maximum
    for (const Battery& bat : inst.batteries) {
        const auto& ball = bat_map[bat.id];
        double sum_G = 0.0;
        for (size_t idx : ball) sum_G += inst.wells[idx].gross_prod;
        if (bat.target_gross > sum_G * (1.0 + tolerance))
            std::cout << utils::red
                      << "[solver] Warning: battery " << bat.id
                      << " Gpt=" << bat.target_gross
                      << " exceeds physical max=" << sum_G
                      << " – target will be clamped in feasibility check.\n"
                      << utils::reset;
    }

    // ------------------------------------------------------------------
    // Global sort: priority DESC, cost ASC, id ASC (stable tie-break)
    // ------------------------------------------------------------------
    std::vector<size_t> global_order(inst.wells.size());
    std::iota(global_order.begin(), global_order.end(), 0);
    std::sort(global_order.begin(), global_order.end(),
              [&](size_t a, size_t b) {
                  const Well& wa = inst.wells[a];
                  const Well& wb = inst.wells[b];
                  if (!utils::areEqual(wa.priority, wb.priority))
                      return wa.priority > wb.priority;
                  if (!utils::areEqual(wa.cost, wb.cost))
                      return wa.cost < wb.cost;
                  return wa.id < wb.id;
              });

    // ------------------------------------------------------------------
    // Global tracking state
    // ------------------------------------------------------------------
    // battery_id → indices of currently selected wells in that battery
    std::unordered_map<int, std::vector<size_t>> bat_selected;
    for (const Battery& b : inst.batteries)
        bat_selected[b.id] = {};

    int    sel_count    = 0;
    double sel_cost     = 0.0;
    // Conservative (worst-case) loss estimate: assume newregime = 100 for every
    // selected well, so loss_i = (G[i] - N[i]) * 100 / 100 = G[i] - N[i].
    double sel_loss_est = 0.0;

    // ------------------------------------------------------------------
    // Greedy selection pass
    // ------------------------------------------------------------------
    for (size_t idx : global_order) {
        const Well& w = inst.wells[idx];

        // --- global hard constraints ---
        if (sel_count >= max_qty)                                  continue;
        if (sel_cost + w.cost > max_cost)                          continue;
        const double loss_w = std::max(0.0, w.gross_prod - w.net_prod);
        if (sel_loss_est + loss_w > max_loss)                      continue;

        // --- per-battery feasibility check (tentative add) ---
        const auto& bat_all = bat_map.at(w.battery_id);
        const Battery& bat  = *bat_by_id.at(w.battery_id);

        auto& bsel = bat_selected.at(w.battery_id);
        bsel.push_back(idx);   // tentatively select

        if (!production_feasible(bsel, bat_all, inst.wells,
                                 bat.target_gross, tolerance)) {
            bsel.pop_back();   // revert
            continue;
        }

        // --- accept well ---
        sel_count++;
        sel_cost     += w.cost;
        sel_loss_est += loss_w;
        sol.selected_ids.push_back(w.id);
    }

    // ------------------------------------------------------------------
    // Verify every battery has a feasible selection
    // ------------------------------------------------------------------
    for (const Battery& bat : inst.batteries) {
        const auto& bat_all = bat_map.at(bat.id);
        const auto& bsel    = bat_selected.at(bat.id);

        if (!production_feasible(bsel, bat_all, inst.wells,
                                 bat.target_gross, tolerance)) {
            std::cout << utils::red
                      << "[solver] Battery " << bat.id
                      << ": no feasible selection found (Gpt=" << bat.target_gross
                      << ", tol=" << (tolerance * 100.0) << "%).\n"
                      << utils::reset;
            return false;
        }

        // Compute and log the achievable range for diagnostics
        std::unordered_set<size_t> sel_set(bsel.begin(), bsel.end());
        double base = 0.0;
        for (size_t i : bat_all)
            if (!sel_set.count(i))
                base += inst.wells[i].gross_prod * inst.wells[i].current_regime / 100.0;
        double max_add = 0.0;
        for (size_t i : bsel) max_add += inst.wells[i].gross_prod;

        std::cout << "[solver] Battery " << bat.id
                  << ": selected=" << bsel.size()
                  << "  achievable=[" << base << ", " << (base + max_add) << "]"
                  << "  Gpt=" << bat.target_gross << "\n";
    }

    std::cout << "[solver] Selection summary:"
              << "  wells=" << sel_count
              << "  cost=" << sel_cost
              << "  loss_est=" << sel_loss_est << "\n";

    // ------------------------------------------------------------------
    // Nearest-Neighbour routing (single crew).
    // Multi-crew partitioning is deferred to Step 3.
    // ------------------------------------------------------------------
    auto [route, dist] = utils::nearest_neighbor_route(
                             sol.selected_ids, inst.dist_matrix);

    sol.route          = route;
    sol.total_distance = dist;
    sol.feasible       = true;

    return true;
}

} // namespace solver

