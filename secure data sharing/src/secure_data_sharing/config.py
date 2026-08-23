from pathlib import Path
import argparse
import yaml


def load_config(path):
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def add_config_argument(parser):
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'))
    return parser


def apply_overrides(config, overrides):
    for item in overrides:
        key, value = item.split('=', 1)
        cursor = config
        parts = key.split('.')
        for part in parts[:-1]:
            cursor = cursor[part]
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError:
            pass
        cursor[parts[-1]] = value
    return config
