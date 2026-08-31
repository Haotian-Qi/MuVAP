import yaml


def load_config(path, module):
    with open(path) as file:
        config = yaml.safe_load(file)

    required = {"checkpoint_dir", module}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    return config


def init_yaml_config(path):
    with open(path) as file:
        return yaml.safe_load(file)
