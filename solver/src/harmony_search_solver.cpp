#include "../include/harmony_search_solver.h"

#include "../include/greedy_solver.h"
#include "../include/utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

namespace solver {
namespace {

struct EvaluatedHarmony {
    std::vector<double> regimes; // indexed by well.id (1-based)

    bool feasible = false;
    double fitness = std::numeric_limits<double>::max();

    int selected_count = 0;
    double total_cost = 0.0;
    double total_loss = 0.0;
    double total_distance = std::numeric_limits<double>::max();

    std::vector<int> selected_ids;
    std::vector<std::vector<int>> crew_routes;
    std::vector<int> route;
};

inline double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

inline double clamp_regime(double x) {
    if (x < 0.0) return 0.0;
    if (x > 100.0) return 100.0;
    return x;
}

std::vector<size_t> battery_order(const Instance& inst,
                                  const std::vector<size_t>& bat_all,
                                  const std::string& sort_method,
                                  const std::unordered_map<int, size_t>& well_idx_by_id,
                                  const std::vector<std::vector<double>>& dist)
{
    std::vector<size_t> order(bat_all.begin(), bat_all.end());

    if (sort_method == "priority_cost") {
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return inst.wells[a].cost < inst.wells[b].cost;
        });
    } else if (sort_method == "loss") {
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const double la = std::max(0.0, inst.wells[a].gross_prod - inst.wells[a].net_prod);
            const double lb = std::max(0.0, inst.wells[b].gross_prod - inst.wells[b].net_prod);
            return la < lb;
        });
    } else if (sort_method == "route") {
        std::vector<int> ids;
        ids.reserve(bat_all.size());
        for (size_t i : bat_all)
            ids.push_back(inst.wells[i].id);

        if (!ids.empty()) {
            auto nn = utils::nearest_neighbor_route(ids, dist);
            const auto& route = nn.first;
            order.clear();
            for (int k = 1; k + 1 < static_cast<int>(route.size()); ++k)
                order.push_back(well_idx_by_id.at(route[k]));
        }
    }

    return order;
}

void repair_per_battery_targets(std::vector<double>& regimes,
                                const Instance& inst,
                                const loader::SolverConfig& cfg,
                                const std::unordered_map<int, std::vector<size_t>>& bat_map,
                                const std::unordered_map<int, size_t>& well_idx_by_id)
{
    for (const Battery& bat : inst.batteries) {
        const auto map_it = bat_map.find(bat.id);
        if (map_it == bat_map.end()) continue;
        const auto& bat_all = map_it->second;

        double max_prod = 0.0;
        for (size_t wi : bat_all)
            max_prod += inst.wells[wi].gross_prod;

        const double eff_target = std::min(bat.target_gross, max_prod);

        double current_prod = 0.0;
        for (size_t wi : bat_all) {
            const Well& w = inst.wells[wi];
            current_prod += w.gross_prod * regimes[w.id] / 100.0;
        }

        const double lo = eff_target * (1.0 - cfg.tolerance);
        const double hi = eff_target * (1.0 + cfg.tolerance);
        if (current_prod >= lo && current_prod <= hi)
            continue;

        const bool need_increase = current_prod < lo;
        double gap = need_increase ? (lo - current_prod) : (current_prod - hi);

        std::vector<size_t> order = battery_order(inst, bat_all, cfg.sort_method,
                                                  well_idx_by_id, inst.dist_matrix);

        for (size_t wi : order) {
            if (gap <= 1e-9) break;
            const Well& w = inst.wells[wi];
            double& r = regimes[w.id];

            if (need_increase) {
                const double room = w.gross_prod * (100.0 - r) / 100.0;
                if (room <= 1e-9) continue;
                if (gap >= room) {
                    r = 100.0;
                    gap -= room;
                } else {
                    r += (gap / w.gross_prod) * 100.0;
                    gap = 0.0;
                }
            } else {
                const double room = w.gross_prod * r / 100.0;
                if (room <= 1e-9) continue;
                if (gap >= room) {
                    r = 0.0;
                    gap -= room;
                } else {
                    r -= (gap / w.gross_prod) * 100.0;
                    gap = 0.0;
                }
            }

            r = clamp_regime(r);
        }
    }
}

EvaluatedHarmony evaluate_harmony(const Instance& inst,
                                  const loader::SolverConfig& cfg,
                                  const std::unordered_map<int, std::vector<size_t>>& bat_map,
                                  const std::unordered_map<int, size_t>& well_idx_by_id,
                                  std::vector<double> candidate_regimes,
                                  double cost_scale,
                                  double loss_scale,
                                  double dist_scale,
                                  double target_scale)
{
    EvaluatedHarmony out;

    // Keep harmonies numerically stable and nudge them toward target feasibility.
    for (double& r : candidate_regimes)
        r = clamp_regime(r);
    repair_per_battery_targets(candidate_regimes, inst, cfg, bat_map, well_idx_by_id);

    out.regimes = std::move(candidate_regimes);

    std::vector<double> bat_cost(inst.batteries.size(), 0.0);
    std::vector<double> bat_loss(inst.batteries.size(), 0.0);

    std::unordered_map<int, size_t> bat_pos;
    bat_pos.reserve(inst.batteries.size());
    for (size_t b = 0; b < inst.batteries.size(); ++b)
        bat_pos[inst.batteries[b].id] = b;

    out.selected_ids.clear();
    out.selected_ids.reserve(inst.wells.size());

    for (const Well& w : inst.wells) {
        const double r = out.regimes[w.id];
        const size_t bp = bat_pos.at(w.battery_id);
        const double loss_w = std::max(0.0, w.gross_prod - w.net_prod) * r / 100.0;
        out.total_loss += loss_w;
        bat_loss[bp] += loss_w;

        if (std::fabs(r - w.current_regime) <= 1e-9)
            continue;

        out.selected_ids.push_back(w.id);
        out.total_cost += w.cost;
        bat_cost[bp] += w.cost;
    }

    out.selected_count = static_cast<int>(out.selected_ids.size());

    auto route_pair = utils::multi_crew_route(out.selected_ids, inst.dist_matrix, cfg.crews);
    out.crew_routes = std::move(route_pair.first);
    out.total_distance = route_pair.second;

    out.route.clear();
    out.route.push_back(0);
    for (const auto& cr : out.crew_routes)
        for (int k = 1; k + 1 < static_cast<int>(cr.size()); ++k)
            out.route.push_back(cr[k]);
    out.route.push_back(0);

    // ---- Violation terms ----
    double target_violation = 0.0;
    for (const Battery& bat : inst.batteries) {
        const auto map_it = bat_map.find(bat.id);
        if (map_it == bat_map.end()) continue;
        const auto& bat_all = map_it->second;

        double max_prod = 0.0;
        for (size_t wi : bat_all)
            max_prod += inst.wells[wi].gross_prod;

        const double eff_target = std::min(bat.target_gross, max_prod);
        const double lo = eff_target * (1.0 - cfg.tolerance);
        const double hi = eff_target * (1.0 + cfg.tolerance);

        double achieved = 0.0;
        for (size_t wi : bat_all) {
            const Well& w = inst.wells[wi];
            achieved += w.gross_prod * out.regimes[w.id] / 100.0;
        }

        if (achieved < lo) target_violation += (lo - achieved);
        if (achieved > hi) target_violation += (achieved - hi);

        const size_t bp = bat_pos.at(bat.id);
        if (bat_cost[bp] > bat.max_cost)
            target_violation += (bat_cost[bp] - bat.max_cost);
        if (bat_loss[bp] > bat.max_loss)
            target_violation += (bat_loss[bp] - bat.max_loss);
    }

    double qty_violation = std::max(0.0, static_cast<double>(out.selected_count - cfg.max_quantity));
    double cost_violation = std::max(0.0, out.total_cost - cfg.max_cost);
    double loss_violation = std::max(0.0, out.total_loss - cfg.max_loss);

    const bool targets_ok = target_violation <= 1e-9;
    const bool caps_ok = (qty_violation <= 1e-9) && (cost_violation <= 1e-9) && (loss_violation <= 1e-9);
    out.feasible = targets_ok && caps_ok;

    const double norm_cost = out.total_cost / std::max(1.0, cost_scale);
    const double norm_loss = out.total_loss / std::max(1.0, loss_scale);
    const double norm_dist = out.total_distance / std::max(1.0, dist_scale);
    const double norm_qty  = static_cast<double>(out.selected_count) / std::max(1.0, static_cast<double>(inst.wells.size()));

    const double base_obj =
          cfg.hs_weight_cost * norm_cost
        + cfg.hs_weight_loss * norm_loss
        + cfg.hs_weight_distance * norm_dist
        + cfg.hs_weight_quantity * norm_qty;

    const double penalty = cfg.hs_penalty * (
          target_violation / std::max(1.0, target_scale)
        + qty_violation / std::max(1.0, static_cast<double>(cfg.max_quantity))
        + cost_violation / std::max(1.0, cost_scale)
        + loss_violation / std::max(1.0, loss_scale));

    out.fitness = base_obj + penalty;
    return out;
}

} // namespace

bool HarmonySearchSolver::solve(const Instance& inst,
                                Solution& sol,
                                const loader::SolverConfig& cfg)
{
    sol.feasible = false;
    sol.total_distance = std::numeric_limits<double>::max();
    sol.total_cost = 0.0;
    sol.total_loss = 0.0;
    sol.selected_ids.clear();
    sol.new_regimes.clear();
    sol.crew_routes.clear();
    sol.route.clear();

    if (inst.wells.empty())
        return false;

    const int hms = std::max(2, cfg.hs_hms);
    const int iterations = std::max(1, cfg.hs_iterations);
    const double hmcr = clamp01(cfg.hs_hmcr);
    const double par  = clamp01(cfg.hs_par);
    const double bw   = std::max(1e-6, cfg.hs_bw);
    const double change_prob = clamp01(cfg.hs_change_prob);

    std::unordered_map<int, size_t> well_idx_by_id;
    well_idx_by_id.reserve(inst.wells.size());
    for (size_t i = 0; i < inst.wells.size(); ++i)
        well_idx_by_id[inst.wells[i].id] = i;

    std::unordered_map<int, std::vector<size_t>> bat_map;
    for (size_t i = 0; i < inst.wells.size(); ++i)
        bat_map[inst.wells[i].battery_id].push_back(i);

    double cost_scale = 0.0;
    double loss_scale = 0.0;
    for (const Well& w : inst.wells) {
        cost_scale += w.cost;
        loss_scale += std::max(0.0, w.gross_prod - w.net_prod);
    }
    if (cfg.max_cost < std::numeric_limits<double>::max() / 4.0)
        cost_scale = std::max(cost_scale, cfg.max_cost);
    if (cfg.max_loss < std::numeric_limits<double>::max() / 4.0)
        loss_scale = std::max(loss_scale, cfg.max_loss);

    double target_scale = 0.0;
    for (const Battery& b : inst.batteries)
        target_scale += std::max(1.0, b.target_gross);

    std::vector<int> all_ids;
    all_ids.reserve(inst.wells.size());
    for (const Well& w : inst.wells)
        all_ids.push_back(w.id);
    const double dist_scale = std::max(1.0, utils::multi_crew_route(all_ids, inst.dist_matrix, cfg.crews).second);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_real_distribution<double> u100(0.0, 100.0);

    auto random_harmony = [&]() {
        std::vector<double> r(inst.wells.size() + 1, 0.0);
        for (const Well& w : inst.wells) {
            double v = w.current_regime;
            if (u01(rng) < change_prob) {
                // Blend global random exploration with local perturbations.
                if (u01(rng) < 0.5) {
                    v = u100(rng);
                } else {
                    const double delta = (u01(rng) * 2.0 - 1.0) * std::max(10.0, bw * 3.0);
                    v = w.current_regime + delta;
                }
            }
            r[w.id] = clamp_regime(v);
        }
        return r;
    };

    std::vector<EvaluatedHarmony> hm;
    hm.reserve(hms);

    if (cfg.hs_seed_with_greedy) {
        GreedySolver gs;
        Solution greedy_sol;
        if (gs.solve(inst, greedy_sol, cfg)) {
            std::vector<double> seed(inst.wells.size() + 1, 0.0);
            for (const Well& w : inst.wells)
                seed[w.id] = w.current_regime;
            for (int id : greedy_sol.selected_ids) {
                if (id > 0 && id < static_cast<int>(greedy_sol.new_regimes.size()))
                    seed[id] = greedy_sol.new_regimes[id];
            }
            hm.push_back(evaluate_harmony(inst, cfg, bat_map, well_idx_by_id,
                                          std::move(seed),
                                          cost_scale, loss_scale, dist_scale, target_scale));
        }
    }

    while (static_cast<int>(hm.size()) < hms) {
        hm.push_back(evaluate_harmony(inst, cfg, bat_map, well_idx_by_id,
                                      random_harmony(),
                                      cost_scale, loss_scale, dist_scale, target_scale));
    }

    auto best_it = std::min_element(hm.begin(), hm.end(),
                                    [](const EvaluatedHarmony& a, const EvaluatedHarmony& b) {
                                        return a.fitness < b.fitness;
                                    });
    EvaluatedHarmony best = *best_it;
    EvaluatedHarmony best_feasible;
    bool has_feasible = false;
    for (const auto& h : hm) {
        if (h.feasible && (!has_feasible || h.fitness < best_feasible.fitness)) {
            best_feasible = h;
            has_feasible = true;
        }
    }

    std::uniform_int_distribution<int> idx_hm(0, hms - 1);

    for (int it = 0; it < iterations; ++it) {
        std::vector<double> cand(inst.wells.size() + 1, 0.0);

        for (const Well& w : inst.wells) {
            double value;

            if (u01(rng) < hmcr) {
                const int src = idx_hm(rng);
                value = hm[src].regimes[w.id];
                if (u01(rng) < par) {
                    const double delta = (u01(rng) * 2.0 - 1.0) * bw;
                    value += delta;
                }
            } else {
                value = (u01(rng) < change_prob) ? u100(rng) : w.current_regime;
            }

            cand[w.id] = clamp_regime(value);
        }

        EvaluatedHarmony hnew = evaluate_harmony(inst, cfg, bat_map, well_idx_by_id,
                                                 std::move(cand),
                                                 cost_scale, loss_scale, dist_scale, target_scale);

        auto worst_it = std::max_element(hm.begin(), hm.end(),
                                         [](const EvaluatedHarmony& a, const EvaluatedHarmony& b) {
                                             return a.fitness < b.fitness;
                                         });
        if (hnew.fitness + 1e-12 < worst_it->fitness)
            *worst_it = std::move(hnew);

        auto cur_best_it = std::min_element(hm.begin(), hm.end(),
                                            [](const EvaluatedHarmony& a, const EvaluatedHarmony& b) {
                                                return a.fitness < b.fitness;
                                            });
        if (cur_best_it->fitness + 1e-12 < best.fitness)
            best = *cur_best_it;

        for (const auto& h : hm) {
            if (h.feasible && (!has_feasible || h.fitness + 1e-12 < best_feasible.fitness)) {
                best_feasible = h;
                has_feasible = true;
            }
        }
    }

    const EvaluatedHarmony& final_h = has_feasible ? best_feasible : best;
    if (!final_h.feasible)
        return false;

    sol.new_regimes.assign(inst.wells.size() + 1, 0.0);
    
    for (const Well& w : inst.wells){
        sol.new_regimes[w.id] = final_h.regimes[w.id];
        sol.total_loss += std::max(0.0, w.gross_prod - w.net_prod) * final_h.regimes[w.id] / 100.0;
    }

    sol.selected_ids = final_h.selected_ids;
    sol.total_cost = final_h.total_cost;
    sol.crew_routes = final_h.crew_routes;
    sol.total_distance = final_h.total_distance;
    sol.route = final_h.route;
    sol.feasible = true;

    return true;
}

} // namespace solver
