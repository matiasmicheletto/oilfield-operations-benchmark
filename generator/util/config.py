import yaml
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run the instance generator.")
    parser.add_argument(
        "config_file",
        nargs="?",
        default="generator_config.yaml",
        help="Configuration file name or path (default: generator_config.yaml)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="Override a config value using dot notation, e.g. --set general.num_instances=10",
    )
    return parser.parse_args()


def resolve_config_path(config_file: str) -> Path:
    config_path = Path(config_file)
    if config_path.exists():
        return config_path

    script_dir_candidate = Path(__file__).resolve().parent / config_file
    if script_dir_candidate.exists():
        return script_dir_candidate

    return config_path


def load_config(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' not found.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """
    Apply dot-notation overrides to a nested config dict.

    Each override must be in the form 'key.subkey.leaf=value'.
    Values are auto-cast: booleans first, then int, then float, then str.

    Example:
        apply_overrides(config, ["general.num_instances=10"])
    """
    def _cast(raw: str):
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        return raw

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid --set value '{override}': expected format 'key.path=value'"
            )
        key_path, _, raw_value = override.partition("=")
        keys = key_path.strip().split(".")
        value = _cast(raw_value.strip())
        normalized_path = ".".join(keys)

        d = config
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]
        leaf_key = keys[-1]
        old_value = d.get(leaf_key, "<unset>")
        d[leaf_key] = value
        print(f"Config override applied: {normalized_path} = {value!r} (was {old_value!r})")

    return config

