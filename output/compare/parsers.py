"""
parsers.py — Solution file parsers for CPLEX (.sol XML), greedy (.txt), and SCIP (.txt).

Each parser returns a dict with keys:
    distance, cost, loss  : float | None
    selected_wells        : list[int]
    n_wells               : int
or None when the file cannot be parsed or contains no feasible solution.
"""

import re
import sys
from pathlib import Path
from xml.etree import ElementTree


def parse_cplex_sol(path: Path) -> dict | None:
    """Parse a CPLEX XML .sol file."""
    try:
        tree = ElementTree.parse(path)
    except ElementTree.ParseError as exc:
        print(f"  [warn] Cannot parse XML in '{path.name}': {exc}", file=sys.stderr)
        return None

    root = tree.getroot()

    header = root.find("header")
    status  = (header.attrib.get("solutionStatusString", "") if header is not None else "")
    obj_str = (header.attrib.get("objectiveValue",       "") if header is not None else "")

    if not obj_str:
        print(f"  [warn] '{path.name}' has no objectiveValue – skipping.", file=sys.stderr)
        return None

    variables = root.find("variables")
    if variables is None:
        print(f"  [warn] '{path.name}' has no <variables> section.", file=sys.stderr)
        return None

    result: dict = {
        "status":         status,
        "distance":       None,
        "cost":           None,
        "loss":           None,
        "selected_wells": [],
    }

    for var in variables.findall("variable"):
        name = var.attrib.get("name", "")
        raw  = var.attrib.get("value", "0")
        try:
            val = float(raw)
        except ValueError:
            continue

        if name == "distance":
            result["distance"] = val
        elif name == "cost":
            result["cost"] = val
        elif name == "loss":
            result["loss"] = val
        # Binary selection variable z[i] — encoded by Zimpl as "z$<i>$" in LP
        elif re.fullmatch(r"z\$\d+\$", name) and val > 0.5:
            result["selected_wells"].append(int(name.split("$")[1]))

    # Fall back to header objective when the distance variable is absent
    if result["distance"] is None:
        try:
            result["distance"] = float(obj_str)
        except ValueError:
            pass

    result["n_wells"] = len(result["selected_wells"])
    return result


def parse_greedy_txt(path: Path) -> dict | None:
    """Parse a plain-text solution file written by solve_main.cpp."""
    try:
        text = path.read_text()
    except OSError as exc:
        print(f"  [warn] Cannot read '{path.name}': {exc}", file=sys.stderr)
        return None

    result: dict = {
        "distance":       None,
        "cost":           None,
        "loss":           None,
        "selected_wells": [],
        "n_wells":        0,
    }

    m = re.search(r"Selected wells \((\d+)\):\s*([\d ]+)", text)
    if m:
        result["n_wells"]        = int(m.group(1))
        result["selected_wells"] = [int(x) for x in m.group(2).split()]

    m = re.search(r"Total distance:\s*([\d.eE+\-]+)", text)
    if m:
        result["distance"] = float(m.group(1))

    m = re.search(r"Total cost:\s*([\d.eE+\-]+)", text)
    if m:
        result["cost"] = float(m.group(1))

    m = re.search(r"Total loss \(actual\):\s*([\d.eE+\-]+)", text)
    if m:
        result["loss"] = float(m.group(1))

    if result["distance"] is None:
        print(f"  [warn] '{path.name}' missing 'Total distance' – skipping.", file=sys.stderr)
        return None

    return result


def parse_scip_txt(path: Path) -> dict | None:
    """Parse a plain-text solution file produced by SCIP."""
    try:
        text = path.read_text()
    except OSError as exc:
        print(f"  [warn] Cannot read '{path.name}': {exc}", file=sys.stderr)
        return None

    if "primal solution" not in text:
        print(f"  [warn] '{path.name}' has no primal solution section – skipping.",
              file=sys.stderr)
        return None

    result: dict = {
        "distance":       None,
        "cost":           None,
        "loss":           None,
        "selected_wells": [],
        "n_wells":        0,
    }

    # Variable lines: "varname   value   (obj:...)"
    var_line = re.compile(r"^(\S+)\s+([\d.eE+\-]+)\s+\(obj:", re.MULTILINE)
    for m in var_line.finditer(text):
        name = m.group(1)
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        if name == "distance":
            result["distance"] = val
        elif name == "cost":
            result["cost"] = val
        elif name == "loss":
            result["loss"] = val
        elif re.fullmatch(r"z#\d+", name) and val > 0.5:
            result["selected_wells"].append(int(name.split("#")[1]))

    # Fall back to the objective value line when the distance variable is absent
    if result["distance"] is None:
        m = re.search(r"objective value:\s+([\d.eE+\-]+)", text)
        if m:
            try:
                result["distance"] = float(m.group(1))
            except ValueError:
                pass

    if result["distance"] is None:
        print(f"  [warn] '{path.name}' missing 'distance' variable – skipping.", file=sys.stderr)
        return None

    result["n_wells"] = len(result["selected_wells"])
    return result
