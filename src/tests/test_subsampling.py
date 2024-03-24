from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime as dt

import sys

sys.path.append("../src/tests")

import activity.subsampling as ss


def create_initial_mock_data_from_ipis(mock_start):
    """
    Created a simulated dataset of calls for a period of 30-min where the provided IPIs are used to separate each call.
    """

    mock_df = pd.DataFrame()
    call_duration = 0.01
    mock_df['start_time'] = np.linspace(0, (60*24) - call_duration, 2520)
    mock_df['end_time'] = mock_df['start_time']+call_duration
    mock_df['low_freq'] = [20000]*len(mock_df)
    mock_df['high_freq'] = [30000]*len(mock_df)
    mock_df['freq_group'] = ['LF']*len(mock_df)
    mock_df['call_start_time'] = mock_start + pd.to_timedelta(60e9*mock_df['start_time'])
    mock_df['call_end_time'] = mock_start + pd.to_timedelta(60e9*mock_df['end_time'])
    mock_df['ref_time'] = mock_df['call_start_time']

    resampled_cycle_length_df = mock_df.resample(f'30T', on='ref_time', origin='start_day')
    mock_df['ref_time'] = pd.DatetimeIndex(resampled_cycle_length_df['ref_time'].transform(lambda x: x.name))
    return mock_df


def test_if_subsampling_reduces_number_of_calls_by_expected_factor_in_mock_dataset():
    """
    Create a simulated dataset of calls with a constant IPI that produces N calls in 30-min period.

    Calls the subsampling method with the main cycle lengths and percent ons including the continuous scheme.
    Checks if the N calls are reduced exactly by the listening proportion of each duty-cycle scheme.
    """
    avail = np.arange(0, 720, 6) + 6
    reset_24 = avail[np.where((24*60 % avail) == 0)[0]]

    data_params = dict()
    cycle_lengths = reset_24
    percent_ons = [1/6, 1/3, 1/2, 2/3]
    data_params["cycle_lengths"] = cycle_lengths
    data_params["percent_ons"] = percent_ons
    dc_tags = ss.get_list_of_dc_tags(data_params["cycle_lengths"], data_params["percent_ons"])
    data_params["dc_tags"] = dc_tags
    data_params['recording_start'] = '00:00'
    data_params['recording_end'] = '16:00'

    mock_start = dt.datetime(2022, 6, 15, 0, 0, 0)
    mock_df = create_initial_mock_data_from_ipis(mock_start)
    for dc_tag in data_params['dc_tags']:
        
        cycle_length = int(dc_tag.split('of')[-1])
        time_on = int(dc_tag.split('of')[0])
        time_on_in_secs = (60*time_on)

        mock_df_subsampled = ss.simulate_dutycycle_on_detections(mock_df.copy(), cycle_length, 
                                                              time_on_in_secs, data_params)
        assert np.isclose(len(mock_df_subsampled), len(mock_df)/(cycle_length/time_on), atol=3)