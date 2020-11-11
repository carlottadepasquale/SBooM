#
# BootMCHawkes
#
# @authors : Carlotta De Pasquale

import yaml

def read_yaml_param(path_file):
    """
    reads parameters from yaml file
    input: yaml file path
    output: dictionary
    """
    with open(path_file) as f:
        dict_file_param = yaml.safe_load(f)
    return dict_file_param