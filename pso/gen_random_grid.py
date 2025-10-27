"""
Given a pypower case, automatically generate grid operating configurations such as start-up/shut-down costs, etc
This is convenient for generating large scale test cases
The config is saved to a yaml file
"""

import numpy as np
import pypower.api as pp

gen_config = {
    "pmin_to_pmax_ratio": 0.1,
    "ramp_up_to_pmax_ratio": 0.25,
    "ramp_down_to_pmax_ratio": 0.5,
    "ramp_start_up_to_pmax_ratio": 0.5,
    "ramp_shut_down_to_pmax_ratio": 0.5,
    "ramp_up_rd_to_pmax_ratio": 0.1,
    "ramp_up_down_rd_to_pmax_ratio": 0.2,
    "random_ratio": 0.1
}
gencost_config = {
    "start_up_to_first_ratio": 18,
    "shut_down_to_start_up_ratio": 5,
    "second_to_first_ratio": 0.0,
    "zero_to_first_ratio": 12,
    "storage_to_first_ratio": 10,
    "rd_up_to_first_ratio": 13,
    "rd_down_to_first_ratio": 5,
    "random_ratio": 0.1,
}
solar_config = {
    "number": [20],
    "integrated_capacity_ratio": [0.3],
    "curtail_to_first_ratio": [30],
    "random_ratio": 0.1
}
load_config = {
    "shed_to_first_ratio": [30],
    "random_ratio": 0.1
}

default_value = {
    "min_on_time": [3],
    "min_off_time": [2]
}




