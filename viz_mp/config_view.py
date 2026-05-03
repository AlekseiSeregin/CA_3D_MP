"""Load pickled ``Config`` from DB for role-based viz access."""

from configuration import Config
from configuration.config_utils_classes import update_class_from_dict


def apply_pickled_config(unpickled_dict):
    """
    Merge pickled static config into ``Config`` class and return a fresh instance
    with role lists ready for plotting.
    """
    update_class_from_dict(Config, unpickled_dict)
    return Config()
