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
// Core solver: per-battery regime-adjustment heuristic.
//
// For each battery the wells are sorted by cfg.sort_method, then scanned in
// order.  Each well's regime is pushed toward 100 % (target > current) or
// toward 0 % (target < current) until the battery target is matched.  The
// last adjusted well receives the exact partial regime required; subsequent
// wells are left unchanged.  Global hard constraints are checked per well.
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

    const double tolerance = cfg.tolerance;
    const int    max_qty   = cfg.max_quantity;
    const double max_cost  = cfg.max_cost;
    const double max_loss  = cfg.max_loss;

    // ------------------------------------------------------------------
    // Build lookup structures
    // ------------------------------------------------------------------
    std::unordered_map<int, size_t> well_idx_by_id;
    well_idx_by_id.reserve(inst.wells.size());
    for (size_t i = 0; i < inst.wells.size(); ++i)
        well_idx_by_id[inst.wells[i].id] = i;

    std::unordered_map<int, std::vector<size_t>> bat_map;
    for (size_t i = 0; i < inst.wells.size(); ++i)
        bat_map[inst.wells[i].battery_id].push_back(i);

    // new_regimes indexed by well.id (1-based); 0 = not selected / not used
    const int max_well_id = static_cast<int>(inst.wells.size());
    sol.new_regimes.assign(max_well_id + 1, 0.0);

    // Track selected IDs in a set to distinguish new_regime==0 from "not selected"
    std::unordered_set<int> selected_set;

    int    sel_count    = 0;
    double sel_cost     = 0.0;
    double sel_loss_acc = 0.0;

    // ------------------------------------------------------------------
    // Per-battery regime adjustment
    // ------------------------------------------------------------------
    for (const Battery& bat : inst.batteries) {
        const auto& bat_all = bat_map.at(bat.id);

        // Maximum achievable production (all wells at 100 %)
        double max_prod = 0.0;
        for (size_t i : bat_all)
            max_prod += inst.wells[i].gross_prod;

        // Current production at existing regimes
        double current_prod = 0.0;
        for (size_t i : bat_all)
            current_prod += inst.wells[i].gross_prod * inst.wells[i].current_regime / 100.0;

        // Clamp target to physical maximum (same as original feasibility logic)
        const double eff_target = std::min(bat.target_gross, max_prod);

        const bool need_increase = current_prod < eff_target * (1.0 - tolerance);
        const bool need_decrease = current_prod > eff_target * (1.0 + tolerance);

        if (!need_increase && !need_decrease) {
            std::cout << "[solver] Battery " << bat.id
                      << ": target already met (current=" << current_prod
                      << ", Gpt=" << bat.target_gross << ").\n";
            continue;
        }

        // --------------------------------------------------------------
        // Sort well indices according to cfg.sort_method
        // --------------------------------------------------------------
        std::vector<size_t> order(bat_all.begin(), bat_all.end());

        if (cfg.sort_method == "priority_cost") {
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                const Well& wa = inst.wells[a];
                const Well& wb = inst.wells[b];
                const double ra = (wa.cost > 1e-9) ? wa.priority / wa.cost
                                                   : std::numeric_limits<double>::max();
                const double rb = (wb.cost > 1e-9) ? wb.priority / wb.cost
                                                   : std::numeric_limits<double>::max();
                return ra < rb;  // ascending ratio
            });
        } else if (cfg.sort_method == "loss") {
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                return (inst.wells[a].gross_prod - inst.wells[a].net_prod) <
                       (inst.wells[b].gross_prod - inst.wells[b].net_prod);  // ascending loss
            });
        } else if (cfg.sort_method == "route") {
            std::vector<int> ids;
            ids.reserve(bat_all.size());
            for (size_t i : bat_all)
                ids.push_back(inst.wells[i].id);
            auto [route, dummy] = utils::nearest_neighbor_route(ids, inst.dist_matrix);
            // route = [0, w1, ..., wk, 0] — extract inner sequence
            order.clear();
            for (int k = 1; k + 1 < static_cast<int>(route.size()); ++k)
                order.push_back(well_idx_by_id.at(route[k]));
        } else {
            std::cerr << utils::red
                      << "[solver] Unknown sort_method '" << cfg.sort_method
                      << "', using natural order.\n" << utils::reset;
        }

        // --------------------------------------------------------------
        // Scan sorted wells and adjust regimes until gap is closed
        // --------------------------------------------------------------
        double gap = need_increase ? (eff_target - current_prod)
                                   : (current_prod - eff_target);

        std::vector<int> selected_in_bat;

        for (size_t i : order) {
            if (gap <= 1e-9) break;  // target met

            const Well& w = inst.wells[i];

            // Compute potential new regime and how much of the gap it covers
            double new_regime;
            double gap_reduction;

            if (need_increase) {
                const double room = w.gross_prod * (100.0 - w.current_regime) / 100.0;
                if (room <= 1e-9) continue;  // already at 100 %, nothing to gain
                if (gap >= room) {
                    new_regime    = 100.0;
                    gap_reduction = room;
                } else {
                    new_regime    = w.current_regime + gap * 100.0 / w.gross_prod;
                    gap_reduction = gap;
                }
            } else {
                const double room = w.gross_prod * w.current_regime / 100.0;
                if (room <= 1e-9) continue;  // already at 0 %, nothing to cut
                if (gap >= room) {
                    new_regime    = 0.0;
                    gap_reduction = room;
                } else {
                    new_regime    = w.current_regime - gap * 100.0 / w.gross_prod;
                    gap_reduction = gap;
                }
            }

            // Check global hard constraints before accepting
            const double loss_w = std::max(0.0, w.gross_prod - w.net_prod) * new_regime / 100.0;
            if (sel_count + 1  > max_qty)              continue;
            if (sel_cost + w.cost > max_cost)           continue;
            if (sel_loss_acc + loss_w > max_loss)       continue;

            // Accept adjustment
            gap                  -= gap_reduction;
            sol.new_regimes[w.id] = new_regime;
            selected_in_bat.push_back(w.id);
            selected_set.insert(w.id);
            sel_count++;
            sel_cost     += w.cost;
            sel_loss_acc += loss_w;
        }

        for (int id : selected_in_bat)
            sol.selected_ids.push_back(id);

        // Compute achieved production for this battery (diagnostic + feasibility)
        double achieved = 0.0;
        for (size_t i : bat_all) {
            const Well& w = inst.wells[i];
            const double r = selected_set.count(w.id) ? sol.new_regimes[w.id]
                                                      : w.current_regime;
            achieved += w.gross_prod * r / 100.0;
        }

        const bool bat_ok = (achieved >= eff_target * (1.0 - tolerance)) &&
                            (achieved <= eff_target * (1.0 + tolerance));

        std::cout << "[solver] Battery " << bat.id
                  << ": adjusted=" << selected_in_bat.size()
                  << "  achieved=" << achieved
                  << "  Gpt=" << bat.target_gross
                  << (bat_ok ? "" : "  [INFEASIBLE]") << "\n";

        if (!bat_ok) {
            std::cout << utils::red
                      << "[solver] Battery " << bat.id
                      << ": target not met within tolerance (tol=" << (tolerance * 100.0) << "%).\n"
                      << utils::reset;
            return false;
        }
    }

    std::cout << "[solver] Selection summary:"
              << "  wells=" << sel_count
              << "  cost=" << sel_cost
              << "  loss=" << sel_loss_acc << "\n";

    // ------------------------------------------------------------------
    // Objective values from actual new regimes
    // ------------------------------------------------------------------
    sol.total_cost = 0.0;
    sol.total_loss = 0.0;
    for (int id : sol.selected_ids) {
        const Well& w  = inst.wells[well_idx_by_id.at(id)];
        sol.total_cost += w.cost;
        const double nr = sol.new_regimes[id];
        sol.total_loss += std::max(0.0, w.gross_prod - w.net_prod) * nr / 100.0;
    }

    // ------------------------------------------------------------------
    // Routing: multi-crew TSP over selected wells
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
