import toml
from tabulate import tabulate


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = toml.load(f)
    return config


def post_parse_load_config(p):
    '''
    Load from a toml config file
    If a config key is also a CLI argument, the CLI argument will override the config value.
    If is_force_config is set, the CLI argument will not override the config value.
    '''
    if p.config is not None:
        if not p.config.endswith('.toml'):
            raise ValueError('Dataset config must be a Toml file.')
        config = load_config(p.config)

        cli_override_flag = False
        swapped = []
        for k, v in config.items():
            if k in p:
                cli_v = getattr(p, k)
                if cli_v is not None and not p.is_force_config:
                    swapped.append([k, cli_v, v, False])
                    cli_override_flag = True
                else:
                    setattr(p, k, v)
                    swapped.append([k, cli_v, v, True])
            else:
                print(f'Unknown config key name [{k}] against command line.')
        if cli_override_flag:
            print('[Config swaps]')
            print(tabulate(
                    swapped, 
                    headers=['Name', 'CLI', 'Config', 'Swapped?'],tablefmt='orgtbl'))
            print(
                '[Config swaps] Swapped False means the CLI value is used. CLI values have higher priority. You can force swap by setting adding flag --force-config.')
    return p
