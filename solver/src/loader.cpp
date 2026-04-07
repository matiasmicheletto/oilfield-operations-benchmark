#include "../include/loader.h"
#include "../include/utils.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <yaml-cpp/yaml.h>

namespace loader {

// ---------------------------------------------------------------------------
// Config: YAML file loading
// ---------------------------------------------------------------------------
void load_yaml_config(SolverConfig& cfg, const std::string& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        utils::throw_runtime_error("[config] Cannot load '" + path + "': " + e.what());
    }

    if (root["input"]) {
        const auto& in = root["input"];
        if (in["param_file"]   && in["param_file"].IsScalar())
            cfg.param_file   = in["param_file"].as<std::string>();
        if (in["battery_file"] && in["battery_file"].IsScalar())
            cfg.battery_file = in["battery_file"].as<std::string>();
        if (in["dist_file"]    && in["dist_file"].IsScalar())
            cfg.dist_file    = in["dist_file"].as<std::string>();
    }

    if (root["solver"]) {
        const auto& s = root["solver"];
        if (s["max_wells"]) cfg.max_wells = s["max_wells"].as<int>();
        if (s["tolerance"]) cfg.tolerance = s["tolerance"].as<double>();
        if (s["crews"])     cfg.crews     = s["crews"].as<int>();
        // These three are optional; an absent or null node leaves the C++ default intact.
        if (s["max_quantity"] && s["max_quantity"].IsScalar() && s["max_quantity"].Scalar() != "")
            cfg.max_quantity = s["max_quantity"].as<int>();
        if (s["max_cost"]     && s["max_cost"].IsScalar()     && s["max_cost"].Scalar()     != "")
            cfg.max_cost     = s["max_cost"].as<double>();
        if (s["max_loss"]     && s["max_loss"].IsScalar()     && s["max_loss"].Scalar()     != "")
            cfg.max_loss     = s["max_loss"].as<double>();
        if (s["sort_method"]  && s["sort_method"].IsScalar())
            cfg.sort_method  = s["sort_method"].as<std::string>();
    }

    if (root["output"]) {
        const auto& out = root["output"];
        if (out["file"] && out["file"].IsScalar())
            cfg.output_file = out["file"].as<std::string>();
        if (out["debug"])
            cfg.debug = out["debug"].as<bool>();
    }
}

// ---------------------------------------------------------------------------
// Config: single --set key=value override
// ---------------------------------------------------------------------------
void apply_override(SolverConfig& cfg, const std::string& kv) {
    const auto eq = kv.find('=');
    if (eq == std::string::npos) {
        std::cerr << utils::red
                  << "[config] --set: expected key=value format, got: '" << kv << "'\n"
                  << utils::reset;
        return;
    }

    const std::string key = kv.substr(0, eq);
    const std::string val = kv.substr(eq + 1);

    try {
        if      (key == "input.param_file")    cfg.param_file   = val;
        else if (key == "input.battery_file")  cfg.battery_file = val;
        else if (key == "input.dist_file")     cfg.dist_file    = val;
        else if (key == "solver.max_wells")    cfg.max_wells    = std::stoi(val);
        else if (key == "solver.tolerance")    cfg.tolerance    = std::stod(val);
        else if (key == "solver.crews")        cfg.crews        = std::stoi(val);
        else if (key == "solver.max_quantity") cfg.max_quantity = std::stoi(val);
        else if (key == "solver.max_cost")     cfg.max_cost     = std::stod(val);
        else if (key == "solver.max_loss")     cfg.max_loss     = std::stod(val);
        else if (key == "solver.sort_method")  cfg.sort_method  = val;
        else if (key == "output.file")         cfg.output_file  = val;
        else if (key == "output.debug")        cfg.debug        = (val == "true" || val == "1");
        else
            std::cerr << utils::red
                      << "[config] --set: unknown key '" << key << "'\n"
                      << utils::reset;
    } catch (const std::exception& e) {
        std::cerr << utils::red
                  << "[config] --set: invalid value for '" << key << "': " << e.what() << "\n"
                  << utils::reset;
    }
}

Instance load(const std::string& param_file,
              const std::string& bat_file,
              const std::string& dist_file) {
    Instance inst;
    inst.name = param_file;

    // -----------------------------------------------------------------------
    // 1. parameters_*.dat
    //    Tab-separated, header: ID  G  N  r  R  P  C  B
    //    Columns (1-indexed):   1   2  3  4  5  6  7  8
    // -----------------------------------------------------------------------
    {
        std::ifstream f(param_file);
        if (!f) utils::throw_runtime_error("Cannot open parameter file: " + param_file);

        std::string line;
        std::getline(f, line); // skip header

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            Well w;
            ss >> w.id >> w.gross_prod >> w.net_prod >> w.current_regime
               >> w.risk >> w.priority >> w.cost >> w.battery_id;
            // Column G (gross_prod) and N (net_prod) store the production at the
            // current regime r.  Convert both to their maximum values at 100 %
            // regime so the rest of the solver can treat gross_prod as capacity.
            if (w.current_regime > 1e-9) {
                w.gross_prod = w.gross_prod * 100.0 / w.current_regime;
                w.net_prod   = w.net_prod   * 100.0 / w.current_regime;
            }
            inst.wells.push_back(w);
        }
    }

    // -----------------------------------------------------------------------
    // 2. batteries_*.dat
    //    Tab-separated, header: ID  Gpt
    // -----------------------------------------------------------------------
    {
        std::ifstream f(bat_file);
        if (!f) utils::throw_runtime_error("Cannot open battery file: " + bat_file);

        std::string line;
        std::getline(f, line); // skip header

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            Battery b;
            ss >> b.id >> b.target_gross;
            inst.batteries.push_back(b);
        }
    }

    // -----------------------------------------------------------------------
    // 3. distance_*.dat
    //    First line: "Distance Matrix" (text, skip)
    //    Remaining:  (n+1) rows of (n+1) tab-separated integers.
    //    Row 0 / col 0 = depot (operations centre).
    //    Row i / col i = well with id == i  (1-indexed).
    // -----------------------------------------------------------------------
    {
        std::ifstream f(dist_file);
        if (!f) utils::throw_runtime_error("Cannot open distance file: " + dist_file);

        std::string line;
        std::getline(f, line); // skip "Distance Matrix" header

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            std::vector<double> row;
            double val;
            while (ss >> val) row.push_back(val);
            if (!row.empty()) inst.dist_matrix.push_back(row);
        }
    }

    return inst;
}

} // namespace loader
