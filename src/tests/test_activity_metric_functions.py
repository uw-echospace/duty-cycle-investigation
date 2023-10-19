from pathlib import Path
import pandas as pd
import numpy as np

import datetime as dt
from scipy import stats, signal

import sys

sys.path.append("../src")
sys.path.append("../src/bout")

import clustering as clstr
import data_handling as dh
import subsampling as ss


def create_initial_mock_data_from_ipis(ipis):
    """
    Created a simulated dataset of calls for a period of 30-min where the provided IPIs are used to separate each call.
    """

    mock_df = pd.DataFrame(columns=['ref_time', 'call_start_time', 'call_end_time', 'start_time', 'end_time', 'low_freq', 'high_freq', 'freq_group'])
    recording_start = dt.datetime(2022, 6, 15, 1, 00, 0)
    recording_end = recording_start+dt.timedelta(minutes=30)
    call_duration = 0.01
    for i in range(0, len(ipis)):
        start_time = ipis[:i+1].cumsum(axis=0)[-1]+ i*call_duration
        call_start_time = recording_start + dt.timedelta(seconds=start_time)
        call_end_time = recording_start + dt.timedelta(seconds=start_time+call_duration)
        ref_time = pd.to_datetime(call_start_time).floor('30T')
        if (call_end_time <= recording_end):
            mock_df.loc[i] = [ref_time, call_start_time, call_end_time, start_time, start_time+call_duration, 20000, 30000, 'LF1']

    return mock_df

def test_activity_metrics_using_simulated_bout_dataset():
    """
    Create a simulated dataset of calls with 5 bouts of calls where within-bout calls are separated by IPI of 90ms.
    Bouts are separated by intervals of 5-min. 
    The BCI is set to be 150ms so we test the bout clustering functions to see if we get 5 bouts exactly.

    The number of detections should be equal to the number of IPIs (including the IPI required before the first call)
    The activity index should be equal to the number of bouts because each bout falls completely within just a single time block.
    The measured bout duration should be equal to the derived bout duration.
    Bout duration derived by calculating the duration of 10 calls and 9 IPIs per bout.
    """

    points = 50
    t = np.linspace(0, 1, points, endpoint=False)

    A = 150
    mock_square_ipis = A*(signal.square(2 * np.pi * 5 * t, 1/points) + 1)
    call_duration = 0.01
    desired_ipi = 0.09
    desired_duration_of_call_and_ipi = 0.1

    mock_square_ipis[mock_square_ipis==0] = desired_ipi

    mock_bout_df = create_initial_mock_data_from_ipis(mock_square_ipis)
    mock_bout_df = ss.simulate_dutycycle_on_detections(mock_bout_df, '1800of1800')
    bout_params = dict()
    bout_params['LF1_bci'] = 150
    batdetect2_predictions = clstr.classify_bouts_in_bd2_predictions_for_freqgroups(mock_bout_df, bout_params)
    bout_metrics = clstr.construct_bout_metrics_from_location_df_for_freqgroups(batdetect2_predictions)
    data_params = dict()
    data_params['resolution_in_min'] = '30'
    data_params["index_time_block_in_secs"] = '5'

    test_preds = batdetect2_predictions.copy()
    assert(dh.get_number_of_detections_per_interval(test_preds, data_params).item() == points)
    assert(dh.get_activity_index_per_interval(test_preds, data_params).item() == len(bout_metrics))
    assert(dh.get_bout_duration_per_interval(bout_metrics, data_params).item() == (10*call_duration + 9*desired_ipi)*len(bout_metrics))



