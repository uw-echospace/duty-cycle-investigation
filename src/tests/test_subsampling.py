from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime as dt

import sys

sys.path.append("../src/tests")

import activity.activity_assembly as actvt
import activity.subsampling as ss
from core import SITE_NAMES, FREQ_GROUPS
import test_activity_metric_functions as test_activity

def test_if_subsampling_reduces_number_of_calls_by_expected_factor_in_mock_dataset():
    """
    Create a simulated dataset of calls with a constant IPI that produces N calls in 30-min period.

    Calls the subsampling method with the main cycle lengths and percent ons including the continuous scheme.
    Checks if the N calls are reduced exactly by the listening proportion of each duty-cycle scheme.
    """

    data_params = dict()
    cycle_lengths = [1800, 360]
    percent_ons = [0.1667]
    data_params["cycle_lengths"] = cycle_lengths
    data_params["percent_ons"] = percent_ons
    dc_tags = ss.get_list_of_dc_tags(data_params["cycle_lengths"], data_params["percent_ons"])
    data_params["dc_tags"] = dc_tags

    amplitude = 9.95
    mock_constant_ipis = np.array([amplitude]*10000, dtype='float')
    mock_subsampling_df = test_activity.create_initial_mock_data_from_ipis(mock_constant_ipis)
    for dc_tag in data_params['dc_tags']:
        mock_df_subsampled = ss.simulate_dutycycle_on_detections(mock_subsampling_df.copy(), dc_tag)
        cycle_length = int(dc_tag.split('of')[-1])
        time_on = int(dc_tag.split('of')[0])
        assert(len(mock_df_subsampled)*(cycle_length/time_on) == len(mock_subsampling_df))


