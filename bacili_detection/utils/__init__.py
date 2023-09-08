from .misc import *

import importlib


def load_config(config_name, configs_module="bacili_detection.configs"):
    """
    Load a python file as a module and return it as a dict
    """
    from ml_collections import ConfigDict
    config_module = importlib.import_module(configs_module)
    config_dict = getattr(config_module, config_name)
    # if a pydantic config is passed, convert it to a dict
    if hasattr(config_dict, "dict"):
        config_dict = config_dict.dict()
    config_dict = ConfigDict(config_dict)
    return config_dict