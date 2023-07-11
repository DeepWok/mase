import toml
from tabulate import tabulate
from textwrap import wrap


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


def post_parse_load_config(args, defaults):
    """
    Load and merge arguments from a toml configuration file. If the configuration key
    matches the "dest" value of an existing CLI argument, we use precedence to determine
    which argument value to choose (i.e. default < configuration < manual overrides).
    These arguments are then visualised in a table. :)
    """
    if args.config and not args.config.endswith(".toml"):
        raise ValueError(f"expected .toml configuration file, got {args.config}")

    # Helper function to colour the output gray
    fmt_gray = lambda x: f"\033[38;5;8m{x}\033[0m"

    fields = ["Name", "Default", "Config. File", "Manual Override", "Effective"]
    table = []

    config = load_config(args.config) if args.config else None
    for k in list(vars(args).keys()):
        if k not in defaults or k == "config":
            continue

        default_gray = fmt_gray(defaults[k])
        if config and k in config.keys():
            # Only merge the values from the configuration if there are no manual
            # overrides (i.e. the argument value doesn't deviate from its default).
            v = config[k]
            if getattr(args, k) == defaults[k]:
                setattr(args, k, v)
                table.append([k, default_gray, v, "", v])
            else:
                table.append(
                    [k, default_gray, fmt_gray(v), getattr(args, k), getattr(args, k)]
                )
        else:
            if getattr(args, k) == defaults[k]:
                table.append([k, defaults[k], "", "", defaults[k]])
            else:
                table.append([k, default_gray, "", getattr(args, k), getattr(args, k)])

    if not config:
        fields.remove("Config. File")
        table = [
            [k, default, override, effective]
            for k, default, _, override, effective in table
        ]

    # NOTE: We need to replace NoneType with a string 'None' for text wrapping to work
    # properly via maxcolwidths.
    table = [["None" if item is None else item for item in row] for row in table]

    print(
        tabulate(
            table,
            headers=fields,
            colalign=["left"] + ["center"] * (len(fields) - 1),
            tablefmt="pretty",
            maxheadercolwidths=24,
            maxcolwidths=24,
            disable_numparse=True,
        )
    )
    return args
