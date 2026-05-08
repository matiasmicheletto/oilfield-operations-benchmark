if (cfg.sort_method == "priority_cost") {
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const Well& wa = inst.wells[a];
        const Well& wb = inst.wells[b];

        const double ra = (wa.current_regime > 1e-9) ? wa.current_regime
                                        : std::numeric_limits<double>::max();
        const double rb = (wb.current_regime > 1e-9) ?  wb.current_regime
                                        : std::numeric_limits<double>::max();
        if (need_increase){	
            return ra < rb;  // ascending ratio
        }
        else{
            return rb < ra;
        }
        });
} else if (cfg.sort_method == "loss") {
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (need_increase){
            return (inst.wells[a].gross_prod - inst.wells[a].net_prod) <
                    (inst.wells[b].gross_prod - inst.wells[b].net_prod);  // ascending loss
        }
        else{
            return (inst.wells[a].gross_prod - inst.wells[a].net_prod) >
                    (inst.wells[b].gross_prod - inst.wells[b].net_prod);  // ascending loss
        }
    });
} else if (cfg.sort_method == "route") {
    std::vector<int> ids;
    ids.reserve(bat_all.size());
    for (size_t i : bat_all)
        ids.push_back(inst.wells[i].id);
    auto [route, dummy] = utils::nearest_neighbor_route(ids, inst.dist_matrix);
    order.clear();
    for (int k = 1; k + 1 < static_cast<int>(route.size()); ++k)
        order.push_back(well_idx_by_id.at(route[k]));
} else {
    std::cerr << utils::red
        << "[solver] Unknown sort_method '" << cfg.sort_method
        << "', using natural order.\n" << utils::reset;
}
