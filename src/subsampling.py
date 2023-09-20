import pandas as pd
import data_handling as dh
import bout_clustering as bt_clustering

from pathlib import Path


def simulate_dutycycle_on_detections(location_df, dc_tag):
    """
    Simulates a provided duty-cycling scheme on the provided location summary of concatenated bd2 outputs.
    """

    cycle_length = int(dc_tag.split('of')[1])
    time_on = int(dc_tag.split('of')[0])

    location_df['ref_time'] = pd.DatetimeIndex(location_df['ref_time'])
    location_df['call_end_time'] = pd.DatetimeIndex(location_df['call_end_time'])
    location_df['call_start_time'] = pd.DatetimeIndex(location_df['call_start_time'])
    
    resampled_df = location_df.resample(f'{cycle_length}S', on='ref_time')
    location_df['ref_time'] = resampled_df['ref_time'].transform(lambda x: x.name)
    location_df.insert(0, 'end_time_wrt_ref', (location_df['call_end_time'] - location_df['ref_time']).dt.total_seconds())
    location_df.insert(0, 'start_time_wrt_ref', (location_df['call_start_time'] - location_df['ref_time']).dt.total_seconds())

    dc_applied_df = location_df.loc[(location_df['end_time_wrt_ref'] <= time_on)&(location_df['start_time_wrt_ref'] >= 0)]
    test_for_last_call_within_period(dc_applied_df, cycle_length, time_on)

    return dc_applied_df

def test_for_last_call_within_period(dc_applied_df, cycle_length, time_on):
    calls_grouped_per_cycle = dc_applied_df.resample(f'{cycle_length}S', on='ref_time')
    on_periods = (calls_grouped_per_cycle['ref_time'].first() + pd.to_timedelta(f'{time_on}S')).dropna()
    last_calls_in_periods = (calls_grouped_per_cycle['call_end_time'].max().dropna()) <= on_periods
    first_calls_in_periods = (calls_grouped_per_cycle['call_start_time'].min().dropna()) >= on_periods.index
    assert(not(False in first_calls_in_periods) and not(False in last_calls_in_periods))

def prepare_summary_for_plotting_with_duty_cycle(file_paths, dc_tag):
    """
    Generates a duty-cycled location summary of concatenated bd2 outputs for measuring effects of duty-cycling.
    """

    location_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)
    plottable_location_df = simulate_dutycycle_on_detections(location_df, dc_tag)
    plottable_location_df.to_csv(f'{file_paths["simulated_schemes_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}_{dc_tag}.csv')

    return plottable_location_df

def get_list_of_dc_tags(cycle_lengths=[1800, 360], percent_ons=[0.1667]):
    """
    Takes a list of cycle lengths and the percent ON values to generate a list of tags.
    Cycle length is the period of the duty-cycle when the recording resets.
    Percent ON is the percentage of the cycle length where we simulate recording; only regarding calls within recording periods.
    Each cycle length x percent ON combination will be saved into a list in the format "{percent_on*cycle_lenght}of{cycle_length}"
    """

    dc_tags = []

    cycle_length = 1800
    percent_on = 1.0
    dc_tag = f"{round(percent_on*cycle_length)}of{cycle_length}"
    dc_tags += [dc_tag]

    for cycle_length in cycle_lengths:
        for percent_on in percent_ons:
            dc_tag = f"{round(percent_on*cycle_length)}of{cycle_length}"
            dc_tags += [dc_tag]

    return dc_tags

def construct_activity_dets_arr_from_dc_tags(data_params, file_paths):
    """
    Generates an activity summary for each provided duty-cycling scheme and puts them together for comparison.
    """

    activity_dets_arr = pd.DataFrame()

    for dc_tag in data_params['dc_tags']:

        location_df = prepare_summary_for_plotting_with_duty_cycle(file_paths, dc_tag)
        dc_dets = dh.construct_activity_arr_from_location_summary(location_df, dc_tag, file_paths, data_params)
        dc_dets = dc_dets.set_index("Date_and_Time_UTC")
        activity_dets_arr = pd.concat([activity_dets_arr, dc_dets], axis=1)

    test_subsampled_metrics_less_than_continuous(activity_dets_arr)

    activity_dets_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_dets_TYPE_SITE_summary"]}.csv')

    return activity_dets_arr

def construct_activity_bouts_arr_from_dc_tags(data_params, file_paths):
    """
    Generates an activity summary for each provided duty-cycling scheme and puts them together for comparison.
    """

    activity_bouts_arr = pd.DataFrame()

    for dc_tag in data_params['dc_tags']:

        location_df = prepare_summary_for_plotting_with_duty_cycle(file_paths, dc_tag)
        bout_metrics = bt_clustering.generate_bout_metrics_for_location_and_freq(location_df, data_params, dc_tag)
        dc_bouts = dh.construct_activity_arr_from_bout_metrics(bout_metrics, data_params, file_paths, dc_tag)
        dc_bouts = dc_bouts.set_index("Date_and_Time_UTC")
        activity_bouts_arr = pd.concat([activity_bouts_arr, dc_bouts], axis=1)

    test_subsampled_metrics_less_than_continuous(activity_bouts_arr)

    activity_bouts_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_bouts_TYPE_SITE_summary"]}.csv')

    return activity_bouts_arr

def construct_activity_inds_arr_from_dc_tags(data_params, file_paths):
    """
    Generates an activity summary for each provided duty-cycling scheme and puts them together for comparison.
    """

    activity_inds_arr = pd.DataFrame()

    for dc_tag in data_params['dc_tags']:

        location_df = prepare_summary_for_plotting_with_duty_cycle(file_paths, dc_tag)
        dc_dets = dh.construct_activity_indices_arr(location_df, dc_tag, file_paths, data_params)
        dc_dets = dc_dets.set_index("Date_and_Time_UTC")
        activity_inds_arr = pd.concat([activity_inds_arr, dc_dets], axis=1)

    test_subsampled_metrics_less_than_continuous(activity_inds_arr)

    activity_inds_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_inds_TYPE_SITE_summary"]}.csv')

    return activity_inds_arr

def test_subsampled_metrics_less_than_continuous(activity_arr):
    continuous_column = ''
    for column in activity_arr.columns:
        if ('1800of1800') in column:
            continuous_column = column

    continuous_metrics = activity_arr[continuous_column].dropna()

    for column in activity_arr.columns:
        dc_metrics = activity_arr[column].dropna()
        assertion = (dc_metrics<=continuous_metrics).values
        assert(not(False in assertion))