from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime as dt

import sys

sys.path.append("../src/tests")

import activity.subsampling as ss

def create_initial_mock_data_of_length(num_calls):
    """
    Created a simulated dataset of calls for a period of 30-min where the provided IPIs are used to separate each call.
    """

    mock_df = pd.DataFrame()
    recording_start = dt.datetime(2022, 6, 15, 1, 00, 0)
    call_duration = 0.01
    start_time = np.linspace(0, 1800, num_calls, endpoint=False)
    end_time = start_time + call_duration
    call_start_time = recording_start + pd.to_timedelta(start_time*1e9)
    call_end_time = recording_start + pd.to_timedelta((start_time+call_duration)*1e9)
    ref_time = pd.to_datetime(call_start_time).floor('30T')
    mock_df['ref_time'] = ref_time
    mock_df['call_start_time'] = call_start_time
    mock_df['call_end_time'] = call_end_time
    mock_df['start_time'] = start_time
    mock_df['end_time'] = end_time
    mock_df['low_freq'] = [20000]*len(mock_df)
    mock_df['high_freq'] = [30000]*len(mock_df)
    mock_df['freq_group'] = 'LF1'

    return mock_df

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
    
    num_calls=2520
    mock_subsampling_df = create_initial_mock_data_of_length(num_calls)
    assert(len(mock_subsampling_df)==num_calls)
    for dc_tag in data_params['dc_tags']:
        mock_df_subsampled = ss.simulate_dutycycle_on_detections(mock_subsampling_df.copy(), dc_tag)
        cycle_length = int(dc_tag.split('of')[-1])
        time_on = int(dc_tag.split('of')[0])
        assert(len(mock_df_subsampled)*(cycle_length/time_on) == len(mock_subsampling_df))


