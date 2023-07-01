import toml
from tabulate import tabulate


def convert_str_na_to_none(d):
    """
    Since toml does not support None, we use "NA" to represent None.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = convert_str_na_to_none(v)
    elif isinstance(d, list):
        d = [convert_str_na_to_none(v) for v in d]
    elif isinstance(d, tuple):
        d = tuple(convert_str_na_to_none(v) for v in d)
    else:
        if d == "NA":
            return None
        else:
            return d
    return d


def convert_none_to_str_na(d):
    """
    Since toml does not support None, we use "NA" to represent None.
    Otherwise the none-value key will be missing in the toml file.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = convert_none_to_str_na(v)
    elif isinstance(d, list):
        d = [convert_none_to_str_na(v) for v in d]
    elif isinstance(d, tuple):
        d = tuple(convert_none_to_str_na(v) for v in d)
    else:
        if d is None:
            return "NA"
        else:
            return d
    return d


def load_config(config_path):
    """Load from a toml config file and convert "NA" to None."""
    with open(config_path, "r") as f:
        config = toml.load(f)
    config = convert_str_na_to_none(config)
    return config


def save_config(config, config_path):
    """Convert None to "NA" and save to a toml config file."""
    config = convert_none_to_str_na(config)
    with open(config_path, "w") as f:
        toml.dump(config, f)


def post_parse_load_config(parsed_args):
    """
    Load from a toml config file

    If a config key is also a CLI argument, the CLI argument will override the config value.

    If is_force_config is set, the CLI argument will not override the config value.
    """
    if parsed_args.config is not None:
        if not parsed_args.config.endswith(".toml"):
            raise ValueError("Dataset config must be a Toml file.")
        config = load_config(parsed_args.config)

        cli_override_flag = False
        swapped = []
        for k, v in config.items():
            if k in parsed_args:
                cli_v = getattr(parsed_args, k)
                if cli_v is not None and not parsed_args.is_force_config:
                    swapped.append([k, cli_v, v, False])
                    cli_override_flag = True
                else:
                    setattr(parsed_args, k, v)
                    swapped.append([k, cli_v, v, True])
            else:
                print(f"Unknown config key name [{k}] against command line.")
        if cli_override_flag:
            print("[Config swaps]")
            print(
                tabulate(
                    swapped,
                    headers=["Name", "CLI", "Config", "Swapped?"],
                    tablefmt="orgtbl",
                )
            )
            print(
                "[Config swaps] Swapped False means the CLI value is used. CLI values have higher priority. You can force swap by setting adding flag --force-config."
            )
    return parsed_args
