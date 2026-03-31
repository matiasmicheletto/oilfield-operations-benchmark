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
    sol.total_cost     = 0.0;
    sol.total_loss     = 0.0;
    sol.selected_ids.clear();
    sol.new_regimes.clear();
    sol.crew_routes.clear();
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

    // well.id → index into inst.wells (used throughout to avoid O(n) scans)
    std::unordered_map<int, size_t> well_idx_by_id;
    well_idx_by_id.reserve(inst.wells.size());
    for (size_t i = 0; i < inst.wells.size(); ++i)
        well_idx_by_id[inst.wells[i].id] = i;

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
    // Diagnostic log: print achievable range per battery.
    // Feasibility was already guaranteed by the selection loop — every
    // accepted well passed production_feasible() at the moment of acceptance.
    // The lightweight range check below is a diagnostic-only guard.
    // ------------------------------------------------------------------
    for (const Battery& bat : inst.batteries) {
        const auto& bat_all = bat_map.at(bat.id);
        const auto& bsel    = bat_selected.at(bat.id);

        std::unordered_set<size_t> sel_set(bsel.begin(), bsel.end());
        double base = 0.0;
        for (size_t i : bat_all)
            if (!sel_set.count(i))
                base += inst.wells[i].gross_prod * inst.wells[i].current_regime / 100.0;
        double max_add = 0.0;
        for (size_t i : bsel) max_add += inst.wells[i].gross_prod;

        // Diagnostic guard: Gpt must lie within [base, base+max_add] ± tolerance
        const double eff_target = std::min(bat.target_gross, base + max_add);
        if (eff_target < base * (1.0 - tolerance) ||
            eff_target > (base + max_add) * (1.0 + tolerance)) {
            std::cout << utils::red
                      << "[solver] Battery " << bat.id
                      << ": INTERNAL ERROR – final selection is infeasible (Gpt="
                      << bat.target_gross << ", tol=" << (tolerance * 100.0) << "%).\n"
                      << utils::reset;
            return false;
        }

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
    // Step 2: Regime assignment — proportional scaling per battery.
    //
    // For each battery k and its selected wells S_k:
    //   target_contribution = Gpt[k] - base_unselected
    //   Where base_unselected = sum G[i]*r[i]/100 for non-selected wells.
    //
    //   Proportional scale: newregime[i] = r[i] * scale
    //     where scale = target_contribution / sum_{i in S_k} G[i]*r[i]/100
    //
    //   Wells that would exceed 100 are capped; any remaining shortfall is
    //   redistributed among uncapped wells in a single pass.
    // ------------------------------------------------------------------

    // new_regimes indexed by well.id (1-based); allocate with zeros
    const int max_well_id = static_cast<int>(inst.wells.size());  // IDs are 1..n
    sol.new_regimes.assign(max_well_id + 1, 0.0);

    for (const Battery& bat : inst.batteries) {
        const auto& bat_all = bat_map.at(bat.id);
        const auto& bsel    = bat_selected.at(bat.id);

        if (bsel.empty()) continue;

        // Base production from non-selected wells (fixed regimes)
        std::unordered_set<size_t> sel_set(bsel.begin(), bsel.end());
        double base = 0.0;
        for (size_t i : bat_all)
            if (!sel_set.count(i))
                base += inst.wells[i].gross_prod * inst.wells[i].current_regime / 100.0;

        // Effective target for the selected wells
        double sum_G_sel = 0.0;
        for (size_t i : bsel) sum_G_sel += inst.wells[i].gross_prod;
        const double max_prod   = base + sum_G_sel;
        const double eff_target = std::min(bat.target_gross, max_prod);
        const double need       = eff_target - base; // contribution required from selected wells

        // Current contribution at existing regimes
        double cur_contribution = 0.0;
        for (size_t i : bsel)
            cur_contribution += inst.wells[i].gross_prod
                              * inst.wells[i].current_regime / 100.0;

        // When all selected wells already operate at regime 0, proportional
        // scaling is undefined (0 * scale == 0 regardless of scale).  Instead,
        // assign the required production uniformly across selected wells by
        // capacity, then clamp.  This avoids the 100x overscale that would
        // occur if the fallback value were fed into the base_regime * scale path.
        if (cur_contribution <= 1e-9) {
            // newregime[i] = (need / sum_G_sel) * 100, clamped to [0, 100]
            const double uniform_regime = (sum_G_sel > 1e-9)
                                          ? utils::clamp(need / sum_G_sel * 100.0, 0.0, 100.0)
                                          : 0.0;
            for (size_t i : bsel)
                sol.new_regimes[inst.wells[i].id] = uniform_regime;
            continue;  // skip the proportional-scale block below
        }

        const double scale = need / cur_contribution;

        // First pass: apply scale, collect capped wells and leftover
        std::vector<size_t> uncapped;
        double leftover  = 0.0;
        double g_uncapped = 0.0;

        for (size_t i : bsel) {
            const Well& w = inst.wells[i];
            double nr = w.current_regime * scale;

            if (nr > 100.0) {
                leftover += (nr - 100.0) * w.gross_prod / 100.0; // excess G contribution
                sol.new_regimes[w.id] = 100.0;
            } else {
                sol.new_regimes[w.id] = nr;
                uncapped.push_back(i);
                g_uncapped += w.gross_prod;
            }
        }

        // Second pass: redistribute leftover among uncapped wells
        if (leftover > 1e-9 && g_uncapped > 1e-9) {
            for (size_t i : uncapped) {
                const Well& w = inst.wells[i];
                double extra  = leftover * (w.gross_prod / g_uncapped);
                double nr     = sol.new_regimes[w.id]
                              + extra / w.gross_prod * 100.0;
                sol.new_regimes[w.id] = utils::clamp(nr, 0.0, 100.0);
            }
        }
    }

    // Compute total_cost and total_loss from actual (post-assignment) regimes.
    // Uses well_idx_by_id for O(1) lookup instead of a linear scan per well.
    sol.total_cost = 0.0;
    sol.total_loss = 0.0;
    for (int id : sol.selected_ids) {
        const Well& w  = inst.wells[well_idx_by_id.at(id)];
        sol.total_cost += w.cost;
        const double nr = sol.new_regimes[id];
        sol.total_loss += std::max(0.0, w.gross_prod - w.net_prod) * nr / 100.0;
    }

    // ------------------------------------------------------------------
    // Step 3: Multi-crew routing — round-robin partition by depot distance,
    //         NN TSP per crew, 2-opt post-processing per crew.
    // ------------------------------------------------------------------
    auto [crew_routes, total_dist] = utils::multi_crew_route(
                                         sol.selected_ids, inst.dist_matrix,
                                         cfg.crews);

    sol.crew_routes    = crew_routes;
    sol.total_distance = total_dist;

    // Build legacy flat route (crew routes concatenated, depot repetitions collapsed)
    sol.route.clear();
    sol.route.push_back(0);
    for (const auto& cr : crew_routes) {
        // cr is [0, w1, ..., wk, 0]; skip leading 0 and trailing 0
        for (int k = 1; k + 1 < static_cast<int>(cr.size()); ++k)
            sol.route.push_back(cr[k]);
    }
    sol.route.push_back(0);

    sol.feasible = true;
    return true;
}

} // namespace solver

