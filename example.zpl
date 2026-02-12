# Set of wells
set P := { read "instance_50_2_1.dat" as "<1n>" skip 1 };

# Parameters for each well
param G[P] := read "instance_50_2_1.dat" as "<1n> 2n" skip 1;  # Gross production
param N[P] := read "instance_50_2_1.dat" as "<1n> 3n" skip 1;  # Net production
param R[P] := read "instance_50_2_1.dat" as "<1n> 4n" skip 1;  # Current regime
param C[P] := read "instance_50_2_1.dat" as "<1n> 5n" skip 1;  # Cost
param Bat[P] := read "instance_50_2_1.dat" as "<1n> 6n" skip 1;  # Battery assignment

# Distances between wells
param D[P*P] := read "distances_50_2_1.dat" as "n+";

# Set of batteries
set B := { read "batteries_50_2_1.dat" as "<1n>" skip 1 };

# Desired gross volume
param Gpt[B] := read "baterias.dat" as "<1n> 2n" skip 1;

# Maximum loss, cost and quantity
param maxloss := 500;
param maxcost := 900;
param maxquantity := 9;

# Crews
param crews := 2;

# New regime for each well
var newregime[P] >= 0 <= 100;

# z[i] = regime of well i is changed
var z[P] binary;

# Route variables
var x[P*P] binary;

# MTZ variables
var y[P] >= 0 <= card(P);

# Objectives
var distance >= 0;
var loss >= 0;
var cost >= 0;
var quantity >= 0;

# Objective function
minimize objfunc: distance;

# Subject to maximum loss
subto lossbound: loss <= maxloss;

# Subject to maximum cost
subto costbound: cost <= maxcost;

# Subject to maximum quantity
subto quantitybound: quantity <= maxquantity;

# Total gross production target for each battery
subto grossproduction: forall <k> in B:
    sum <i> in P with i > 0 and Bat[i] == k: G[i] * newregime[i] / 100 == Gpt[k];

# Link z and w variables
subto linkzwupper: forall <i> in P with i > 0:
    newregime[i] >= R[i] - 100 * z[i];
subto linkzwlower: forall <i> in P with i > 0:
    newregime[i] <= R[i] + 100 * z[i];

# Visit all modified wells
subto routeentry: forall <i> in P with i > 0:
    sum <j> in P: x[j,i] == z[i];

# Reinforcement: Visit all modified wells
subto routeexit: forall <i> in P with i > 0:
    sum <j> in P: x[j,i] == z[i];

# Number of crews departing from depot
subto departure: sum <i> in P: x[0,i] == crews;

# Reinforcement: Number of crews arriving at depot
subto arrival: sum <i> in P: x[i,0] == crews;

# Flow conservation
subto flow: forall <i> in P with i > 0:
    sum <j> in P: x[j,i] == sum <j> in P: x[i,j];

# MTZ constraints
subto mtz: forall <i,j> in P*P with i != j and i > 0 and j > 0:
    y[i] + 1 <= y[j] + card(P) * (1 - x[i,j]);

# Definition of objectives
subto defdistance: distance == sum <i,j> in P*P: D[i,j] * x[i,j];
subto defloss: loss == sum <i> in P: (G[i] - N[i]) * newregime[i] / 100;
subto defcost: cost == sum <i> in P: C[i] * z[i];
subto defquantity: quantity == sum <i> in P: z[i];

# Nonsensical variables
subto nonsensex: forall <i> in P:
    x[i,i] == 0;