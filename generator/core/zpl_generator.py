class ZPLGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self, output_path, param_file, bat_file, dist_file):
        # Extract constants from config
        limits = self.config.get("limits", {})
        res = self.config.get("resources", {})
        
        # Zimpl content with dynamic file paths
        zpl_content = f"""# Auto-generated Zimpl model for {param_file}
# Sets and Data Files
set P := {{ read "{param_file}" as "<1n>" skip 1 }};
set B := {{ read "{bat_file}" as "<1n>" skip 1 }};
set V := {{0 .. card(P)}};

# Parameters
param G[P]   := read "{param_file}" as "<1n> 2n" skip 1;
param N[P]   := read "{param_file}" as "<1n> 3n" skip 1;
param R[P]   := read "{param_file}" as "<1n> 4n" skip 1;
param C[P]   := read "{param_file}" as "<1n> 7n" skip 1;
param Bat[P] := read "{param_file}" as "<1n> 8n" skip 1;

param D[V*V] := read "{dist_file}" as "n+" skip 1;
param Gpt[B] := read "{bat_file}" as "<1n> 2n" skip 1;

# Bounds from YAML
param maxloss := {limits.get('max_loss', 500)};
param maxcost := {limits.get('max_cost', 900)};
param maxquantity := {limits.get('max_quantity', 9)};
param crews := {res.get('crews', 1)};

# Tolerances
param gpt_inferior := 10; # Percentage
param gpt_superior := 10; # Percentage

# Variables
var newregime[P] >= 0 <= 100;
var z[P] binary;
var x[V*V] binary;
var y[P] >= 0 <= card(P);

# Objectives
var distance >= 0;
var loss >= 0;
var cost >= 0;

# Objective Function
minimize objfunc: distance;

# Constraints
subto lossbound: loss <= maxloss;
subto costbound: cost <= maxcost;
subto quantitybound: (sum <i> in P: z[i]) <= maxquantity;

# Constraints to link production to regimes and batteries
subto grossproduction_superior: forall <k> in B:
    sum <i> in P with i > 0 and Bat[i] == k: G[i] * newregime[i] / 100 <= Gpt[k] * (1 + gpt_superior / 100);

# Constraints to link production to regimes and batteries
subto grossproduction_inferior: forall <k> in B:
    sum <i> in P with i > 0 and Bat[i] == k: G[i] * newregime[i] / 100 >= Gpt[k] * (1 - gpt_inferior / 100);

# Constraints to link regimes to whether a well is operated or not
subto linkzwupper: forall <i> in P with i > 0:
    newregime[i] >= R[i] - 100 * z[i];
subto linkzwlower: forall <i> in P with i > 0:
    newregime[i] <= R[i] + 100 * z[i];

# Constraints to link routing variables to whether a well is operated or not
subto routeentry: forall <i> in P with i > 0:
    sum <j> in V: x[j,i] == z[i];
subto routeexit: forall <i> in P with i > 0:
    sum <j> in V: x[i,j] == z[i];

# Constraints to limit the number of routes that can be taken from and to the depot
subto departure: sum <i> in P with i > 0: x[0,i] <= crews;
subto arrival:   sum <i> in P with i > 0: x[i,0] <= crews;

# Flow conservation constraints for the routing
subto flow: forall <i> in P with i > 0:
    sum <j> in V: x[j,i] == sum <j> in V: x[i,j];

# Constraints to link routing variables to the order of visits
subto mtz: forall <i,j> in P*P with i != j and i > 0 and j > 0:
    y[i] + 1 <= y[j] + card(P) * (1 - x[i,j]);

# Definitions of the objectives
subto defdistance: distance == sum <i,j> in V*V: D[i,j] * x[i,j];
subto defloss: loss == sum <i> in P: (G[i] - N[i]) * newregime[i] / 100;
subto defcost: cost == sum <i> in P: C[i] * z[i];

# Constraints to prevent routes from a node to itself
subto nonsensex: forall <i> in V: x[i,i] == 0;
"""
        with open(output_path, "w") as f:
            f.write(zpl_content)