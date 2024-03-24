from pathlib import Path
import pandas as pd
import numpy as np

import datetime as dt
from scipy import stats, signal

import sys
sys.path.append("../src")
sys.path.append("../src/bout")

import bout as bt
import activity.activity_assembly as actvt
import activity.subsampling as ss
from cli import get_file_paths
from core import SITE_NAMES


def create_initial_mock_data_from_ipis(ipis):
    """
    Created a simulated dataset of calls for a period of 30-min where the provided IPIs are used to separate each call.
    """

    mock_df = pd.DataFrame(columns=['ref_time', 'call_start_time', 'call_end_time', 'start_time', 'end_time', 'low_freq', 'high_freq', 'freq_group', 'class'])
    recording_start = dt.datetime(2022, 6, 15, 1, 00, 0)
    recording_end = recording_start+dt.timedelta(minutes=30)
    call_duration = 0.01
    for i in range(0, len(ipis)):
        start_time = ipis[:i+1].cumsum(axis=0)[-1]+ i*call_duration
        call_start_time = recording_start + dt.timedelta(seconds=start_time)
        call_end_time = recording_start + dt.timedelta(seconds=start_time+call_duration)
        ref_time = pd.to_datetime(call_start_time).floor('30T')
        if (call_end_time <= recording_end):
            mock_df.loc[i] = [ref_time, call_start_time, call_end_time, start_time, start_time+call_duration, 20000, 30000, 'LF', 'simulation']

    return mock_df


def test_bout_metrics_using_simulated_bout_dataset():
    """
    Create a simulated dataset of calls with 5 bouts of calls where within-bout calls are separated by IPI of 90ms.
    Bouts are separated by intervals of 5-min. 
    The BCI is set to be 150ms so we test the bout clustering functions to see if we get 5 bouts exactly.

    The number of detections should be equal to the number of IPIs (including the IPI required before the first call)
    The activity index should be equal to the number of bouts because each bout falls completely within just a single time block.
    The measured bout duration should be equal to the derived bout duration.
    Bout duration derived by calculating the duration of 10 calls and 9 IPIs per bout.
    """

    bout_params = dict()
    bout_params['LF_bci'] = 150
    data_params = dict()
    data_params['resolution_in_min'] = '30'
    data_params["index_time_block_in_secs"] = '5'
    data_params['recording_start'] = '00:00'
    data_params['recording_end'] = '16:00'

    data_params['cur_dc_tag'] = '6of6'
    data_params['cycle_length'] = int(data_params['cur_dc_tag'].split('of')[-1])
    data_params['time_on'] = int(data_params['cur_dc_tag'].split('of')[0])
    data_params['time_on_in_secs'] = 60*data_params['time_on']
    test_bout_metric_calculation(data_params, bout_params)

    data_params['cur_dc_tag'] = '5of30'
    data_params['cycle_length'] = int(data_params['cur_dc_tag'].split('of')[-1])
    data_params['time_on'] = int(data_params['cur_dc_tag'].split('of')[0])
    data_params['time_on_in_secs'] = 60*data_params['time_on']
    test_bout_metric_calculation(data_params, bout_params)

    data_params['cur_dc_tag'] = '1of6'
    data_params['cycle_length'] = int(data_params['cur_dc_tag'].split('of')[-1])
    data_params['time_on'] = int(data_params['cur_dc_tag'].split('of')[0])
    data_params['time_on_in_secs'] = 60*data_params['time_on']
    test_bout_metric_calculation(data_params, bout_params)


def test_bout_metric_calculation(data_params, bout_params):
    points = 50
    t = np.linspace(0, 1, points, endpoint=False)

    A = 180
    desired_num_bouts = 5
    calls_per_bout = points / desired_num_bouts
    mock_square_ipis = A*(signal.square(2 * np.pi * desired_num_bouts * t, duty=1/np.ceil(calls_per_bout)) + 1)
    call_duration = 0.01
    desired_ipi = 0.09
    expected_dur_of_1bout = (10*call_duration + round(9*desired_ipi, 9))
    mock_square_ipis[0] = desired_ipi
    mock_square_ipis[mock_square_ipis==0] = desired_ipi

    mock_bout_df = create_initial_mock_data_from_ipis(mock_square_ipis)
    mock_bout_df = ss.simulate_dutycycle_on_detections(mock_bout_df, data_params['cycle_length'], data_params['time_on_in_secs'], data_params)
    tagged_dets = bt.classify_bouts_in_bd2_predictions_for_freqgroups(mock_bout_df, bout_params)
    fixed_dets = tagged_dets.groupby('cycle_ref_time').apply(lambda x: bt.add_placeholder_to_tag_dets_wrt_cycle(x, data_params['cycle_length']))
    fixed_dets.reset_index(drop=True, inplace=True)
    bout_metrics = bt.construct_bout_metrics_from_classified_dets(fixed_dets)
    bout_duration = actvt.get_bout_duration_per_cycle(bout_metrics, data_params['cycle_length'])
    assert(bout_duration.sum() == expected_dur_of_1bout*len(bout_metrics))


def test_num_dets_metric_using_simulated_dataset():
    """
    Create a simulated dataset of calls with 5 bouts of calls where within-bout calls are separated by IPI of 90ms.
    Bouts are separated by intervals of 5-min. 
    The BCI is set to be 150ms so we test the bout clustering functions to see if we get 5 bouts exactly.

    The number of detections should be equal to the number of IPIs (including the IPI required before the first call)
    The activity index should be equal to the number of bouts because each bout falls completely within just a single time block.
    The measured bout duration should be equal to the derived bout duration.
    Bout duration derived by calculating the duration of 10 calls and 9 IPIs per bout.
    """

    total_num_dets = 100
    desired_num_bouts = 5

    data_params = dict()
    data_params['resolution_in_min'] = '30'
    data_params["index_time_block_in_secs"] = '5'
    data_params['recording_start'] = '00:00'
    data_params['recording_end'] = '16:00'
    data_params['cur_dc_tag'] = '6of6'
    data_params['cycle_length'] = int(data_params['cur_dc_tag'].split('of')[-1])
    data_params['time_on'] = int(data_params['cur_dc_tag'].split('of')[0])
    data_params['time_on_in_secs'] = 60*data_params['time_on']
    expected_num_dets = total_num_dets
    test_num_dets_metric_calculation(data_params, total_num_dets, desired_num_bouts, expected_num_dets)

    data_params['cur_dc_tag'] = '1of6'
    data_params['cycle_length'] = int(data_params['cur_dc_tag'].split('of')[-1])
    data_params['time_on'] = int(data_params['cur_dc_tag'].split('of')[0])
    data_params['time_on_in_secs'] = 60*data_params['time_on']
    expected_num_dets = total_num_dets
    test_num_dets_metric_calculation(data_params, total_num_dets, desired_num_bouts, expected_num_dets)
    
    data_params['cur_dc_tag'] = '5of30'
    data_params['cycle_length'] = int(data_params['cur_dc_tag'].split('of')[-1])
    data_params['time_on'] = int(data_params['cur_dc_tag'].split('of')[0])
    data_params['time_on_in_secs'] = 60*data_params['time_on']
    calls_per_bout = total_num_dets / desired_num_bouts
    expected_num_dets = calls_per_bout
    test_num_dets_metric_calculation(data_params, total_num_dets, desired_num_bouts, expected_num_dets)


def test_num_dets_metric_calculation(data_params, desired_num_dets, desired_num_bouts, expected_num_dets):
    calls_per_bout = desired_num_dets / desired_num_bouts
    t = np.linspace(0, 1, desired_num_dets, endpoint=False)

    call_duration = 0.01
    desired_ipi = 0.09
    bout_break_time = 360 - ((calls_per_bout*call_duration)+((calls_per_bout-1)*desired_ipi))
    A = bout_break_time/2
    mock_square_ipis = A*(signal.square(2 * np.pi * desired_num_bouts * t, duty=1/desired_num_dets) + 1)
    mock_square_ipis[0] = desired_ipi
    mock_square_ipis[mock_square_ipis==0] = desired_ipi

    mock_df = create_initial_mock_data_from_ipis(mock_square_ipis)
    mock_df = ss.simulate_dutycycle_on_detections(mock_df, data_params['cycle_length'], data_params['time_on_in_secs'], data_params)
    bout_params = dict()
    bout_params['LF_bci'] = 150
    tagged_dets = bt.classify_bouts_in_bd2_predictions_for_freqgroups(mock_df, bout_params)
    test_preds = tagged_dets.copy()
    assert(actvt.get_number_of_detections_per_cycle(test_preds, data_params['cycle_length']).sum() == expected_num_dets)


def test_bout_metric_fixing_on_location_df(location_df, data_params, bout_params):
    dc_applied_df = ss.simulate_dutycycle_on_detections(location_df.copy(), data_params['cycle_length'], data_params['time_on_in_secs'], data_params)

    tagged_dets = bt.classify_bouts_in_bd2_predictions_for_freqgroups(dc_applied_df, bout_params)
    bout_metrics = bt.construct_bout_metrics_from_location_df_for_freqgroups(tagged_dets)
    
    dc_applied_df.reset_index(drop=True, inplace=True)
    bout_metrics_mod = bt.generate_bout_metrics_for_location_and_freq(dc_applied_df, data_params, bout_params)

    assert bout_metrics['bout_duration'].sum() == bout_metrics_mod['bout_duration'].sum()

    return bout_metrics, bout_metrics_mod

def test_bout_calculation_on_location_sums():
    type_keys = ['', 'LF', 'HF']
    for site_key in SITE_NAMES.keys():
        for type_key in type_keys:
            print(site_key, type_key)
            data_params = dict()
            data_params['site_tag'] = site_key
            data_params['type_tag'] = type_key
            data_params['recording_start'] = '00:00'
            data_params['recording_end'] = '16:00'
            data_params['cur_dc_tag'] = '6of6'
            data_params['cycle_length'] = int(data_params['cur_dc_tag'].split('of')[-1])
            data_params['time_on'] = int(data_params['cur_dc_tag'].split('of')[0])
            data_params['time_on_in_secs'] = 60*data_params['time_on']
            file_paths = get_file_paths(data_params)

            if Path(file_paths['raw_SITE_folder']).exists():
                location_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}.csv', index_col=0, low_memory=False)
                bout_params = bt.get_bout_params_from_location(location_df, data_params)

                bout_metrics, bout_metrics_mod = test_bout_metric_fixing_on_location_df(location_df, data_params, bout_params)