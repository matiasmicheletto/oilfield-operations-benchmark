"""
LP Generator
------------
lp_generator.py — Generate CPLEX LP-format files directly from instance data.

This avoids the need for Zimpl (which may not be installed) while producing
files that SCIP, CPLEX, and other LP-compatible solvers can read.

Variable naming mirrors the ZPL model:
    z_<i>           binary well-selection  (i in 1..n)
    x_<i>_<j>       binary routing arc     (i,j in 0..n, i≠j)
    newregime_<i>   continuous new regime  (i in 1..n)
    y_<i>           MTZ ordering variable  (i in 1..n)
    distance        continuous objective alias
    loss            continuous loss alias
    cost            continuous cost alias
"""

from pathlib import Path


def _fmt(v: float) -> str:
    """Format a float coefficient compactly for LP format."""
    if v == int(v):
        return str(int(v))
    return f"{v:.10g}"


class LPGenerator:
    def __init__(self, config):
        self.config = config

    def generate(
        self,
        output_path: Path,
        stem: str,
        wells: dict,
        bat_ids: list,
        bat_targets: list,
        distance_matrix,
    ) -> None:
        """Write an LP file for the given instance data.

        Parameters
        ----------
        output_path : Path  – destination .lp file
        stem        : str   – instance identifier (used only in comment)
        wells       : dict  – arrays G, N, R, C indexed 0..n-1
        bat_ids     : list  – battery id (1-based) for each well, length n
        bat_targets : list  – Gpt for each battery, length n_batteries
        distance_matrix : 2-D array-like of shape (n+1, n+1)
        """
        limits = self.config.get("limits", {})
        res    = self.config.get("resources", {})

        maxloss     = np.sum(well_data["G"])-np.sum(well_data["N"])         
        maxcost     = round(np.sum(well_data["C"]*.8))
        maxquantity = round(len(wells["G"])/2)

        crews       = res.get("crews",             1)
        gpt_sup     = 1.10   # +10 %
        gpt_inf     = 0.90   # -10 %

        n          = len(wells["G"])          # number of wells
        P          = list(range(1, n + 1))    # 1-based well indices
        V          = list(range(0, n + 1))    # depot=0 + wells
        n_bats     = len(bat_targets)
        B          = list(range(1, n_bats + 1))

        G   = [int(wells["G"][i])   for i in range(n)]
        N   = [int(wells["N"][i])   for i in range(n)]
        R   = [float(wells["R"][i]) for i in range(n)]
        C   = [int(wells["C"][i])   for i in range(n)]
        Bat = [int(bat_ids[i])      for i in range(n)]   # 1-based battery id
        Gpt = [float(bat_targets[k]) for k in range(n_bats)]

        D = [[int(round(distance_matrix[i][j])) for j in V] for i in V]

        lines = []
        w = lines.append

        w(f"\\Problem name: {stem}\n")
        w("Minimize")
        w(" obj: distance\n")
        w("Subject To")

        # ── lossbound ─────────────────────────────────────────────
        w(f" c_lossbound: loss <= {_fmt(maxloss)}")

        # ── costbound ─────────────────────────────────────────────
        w(f" c_costbound: cost <= {_fmt(maxcost)}")

        # ── quantitybound ─────────────────────────────────────────
        terms = " + ".join(f"z_{i}" for i in P)
        w(f" c_quantitybound: {terms} <= {maxquantity}")

        # ── battery production bounds ──────────────────────────────
        for k in B:
            members = [i for i in P if Bat[i - 1] == k]
            if not members:
                continue
            kgpt = Gpt[k - 1]
            # sum G[i]*regime[i]/100 <= Gpt[k]*(1+sup)
            coeff_terms = " + ".join(f"{_fmt(G[i-1] / 100.0)} newregime_{i}" for i in members)
            w(f" c_gptsup_{k}: {coeff_terms} <= {_fmt(kgpt * gpt_sup)}")
            # sum G[i]*regime[i]/100 >= Gpt[k]*(1-inf)
            w(f" c_gptinf_{k}: {coeff_terms} >= {_fmt(kgpt * gpt_inf)}")

        # ── linkzwupper / linkzwlower ──────────────────────────────
        for i in P:
            ri = R[i - 1]
            # newregime[i] >= R[i] - 100*z[i]  →  newregime_i + 100 z_i >= R[i]
            w(f" c_zwupper_{i}: newregime_{i} + {_fmt(100.0)} z_{i} >= {_fmt(ri)}")
            # newregime[i] <= R[i] + 100*z[i]  →  newregime_i - 100 z_i <= R[i]
            w(f" c_zwlower_{i}: newregime_{i} - {_fmt(100.0)} z_{i} <= {_fmt(ri)}")

        # ── routeentry / routeexit ─────────────────────────────────
        for i in P:
            entry = " + ".join(f"x_{j}_{i}" for j in V if j != i)
            w(f" c_routeentry_{i}: {entry} - z_{i} = 0")
            exit_ = " + ".join(f"x_{i}_{j}" for j in V if j != i)
            w(f" c_routeexit_{i}: {exit_} - z_{i} = 0")

        # ── departure / arrival (depot row 0) ─────────────────────
        dep = " + ".join(f"x_0_{i}" for i in P)
        w(f" c_departure: {dep} <= {crews}")
        arr = " + ".join(f"x_{i}_0" for i in P)
        w(f" c_arrival: {arr} <= {crews}")

        # ── MTZ subtour elimination ────────────────────────────────
        for i in P:
            for j in P:
                if i == j:
                    continue
                # y_i + 1 <= y_j + n*(1-x_{i,j})
                # → y_i - y_j + n*x_{i,j} <= n - 1
                w(f" c_mtz_{i}_{j}: y_{i} - y_{j} + {n} x_{i}_{j} <= {n - 1}")

        # ── defdistance ────────────────────────────────────────────
        dist_terms = []
        for i in V:
            for j in V:
                if i != j and D[i][j] != 0:
                    dist_terms.append(f"{D[i][j]} x_{i}_{j}")
        dist_lhs = " + ".join(dist_terms) if dist_terms else "0"
        w(f" c_defdistance: distance - {dist_lhs} = 0")

        # ── defloss ────────────────────────────────────────────────
        loss_terms = " + ".join(
            f"{_fmt((G[i-1] - N[i-1]) / 100.0)} newregime_{i}" for i in P
            if G[i-1] != N[i-1]
        )
        if loss_terms:
            w(f" c_defloss: loss - {loss_terms} = 0")
        else:
            w(" c_defloss: loss = 0")

        # ── defcost ────────────────────────────────────────────────
        cost_terms = " + ".join(f"{C[i-1]} z_{i}" for i in P if C[i-1] != 0)
        if cost_terms:
            w(f" c_defcost: cost - {cost_terms} = 0")
        else:
            w(" c_defcost: cost = 0")

        # ── nonsensex (x[i,i] = 0 — handled via Bounds) ───────────

        w("\nBounds")
        w(" distance >= 0")
        w(" loss >= 0")
        w(" cost >= 0")
        for i in P:
            w(f" 0 <= newregime_{i} <= 100")
        for i in P:
            w(f" 0 <= y_{i} <= {n}")

        w("\nBinary")
        z_vars  = "  " + " ".join(f"z_{i}"     for i in P)
        x_vars  = []
        for i in V:
            for j in V:
                if i != j:
                    x_vars.append(f"x_{i}_{j}")
        # write in chunks of 10
        w(z_vars)
        for start in range(0, len(x_vars), 10):
            w("  " + " ".join(x_vars[start:start + 10]))

        w("\nEnd\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
