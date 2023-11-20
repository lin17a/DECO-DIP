"""This module parses and filters parameters."""
from pprint import pprint
import sys
import copy
import yaml

def get_parameters(config_file):
    """load parameter file, fill missing parameters with default ones

    Args:
        config_file (str): path to config file

    Returns:
        dict: parameters for DECO-DIP
    """
    # load custom parameters
    try:
        with open(config_file, 'r') as read_file:  
            custom_params = yaml.safe_load(read_file)
    except OSError:
        print(f"config file {config_file} does not exist")
        sys.exit()
    except yaml.YAMLError as exc:
        print(exc)

    # load default parameters
    default_params_file = "./default_parameters.yaml"
    try:
        with open(default_params_file, "r") as read_file:
            default_params = yaml.safe_load(read_file)
    except OSError:
        print(f"default parameters file '{default_params_file}' does not exist")
        sys.exit()
    except yaml.YAMLError as exc:
        print(exc)

    if not "path" in custom_params['image']:
        print("There is no image path given in the config file.")
        sys.exit()

    combined_params = fill_with_default_parameters(custom_params, default_params)

    filtered_params = filter_unknown_parameters(combined_params, default_params)
    final_params = filter_unused_parameters(filtered_params)


    if final_params["save_and_log"]["verbosity"] > 0:
        pprint(final_params)

    return final_params

def fill_with_default_parameters(custom_params, default_params):
    """fill custom parameters with defaults where they are not specified

    Args:
        custom_params (dict): custom parameters
        default_params (dict): default parameters

    Returns:
        dict: combined paramters
    """
    for param_group in default_params:
        if not param_group in custom_params:
            custom_params[param_group] = default_params[param_group]
        for param in default_params[param_group]:
            if not param in custom_params[param_group]:
                custom_params[param_group][param] = default_params[param_group][param]

    return custom_params

def filter_unknown_parameters(params, default_params):
    """filter parameters that are not known

    Args:
        params (dict): parameters for the run
        default_params (dict): default parameters

    Returns:
        dict: filtered parameters
    """
    filtered_params = copy.deepcopy(params)

    # check for unknown params
    for param_group in params:
        if not param_group in default_params:
            print(f"parameter group {param_group} is not recognized and thus ignored.")
            filtered_params.pop(param_group)
            continue
        for param in params[param_group]:
            if not param in default_params[param_group]:
                if param_group == "image" and param == "path":
                    continue
                print(f"parameter {param} in {param_group} is not recognized and thus ignored.")
                filtered_params[param_group].pop(param)

    return filtered_params


def filter_unused_parameters(params):
    """filter parameters that are not needed dependent on other parameter values

    Args:
        params (dict): parameters for the run

    Returns:
        dict: filtered parameters
    """
    filtered_params = copy.deepcopy(params)

    # throw out params that are not needed
    if not params["time_series"]["is_series"]:
        filtered_params["time_series"].pop("learning_rate")
        filtered_params["time_series"].pop("num_iter")
        filtered_params["time_series"].pop("back_and_forth")

    if params["superresolution"]["superres_factor"] == 1:
        filtered_params["superresolution"].pop("downsample_kernel")

    if params["loss"]["loss_type_main"] != "psf" and params["loss"]["loss_type2"] != "psf":
        filtered_params["psf"] = None

    if not params["loss"]["loss_type2"]:
        filtered_params["loss"].pop("loss2_fact")
        filtered_params["loss"].pop("loss2_incr_fact")

    if not params["loss"]["regularizer"]:
        filtered_params["loss"].pop("regularizer_fact")
        filtered_params["loss"].pop("regularizer_incr_fact")

    if not params["save_and_log"]["tensorboard"]:
        filtered_params["save_and_log"].pop("tensorboard_logdir")

    return filtered_params
