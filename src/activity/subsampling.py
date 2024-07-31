import pandas as pd
import datetime as dt

def all_last_calls_of_cycle_within_recording(dc_applied_df, cycle_length, time_on_in_secs):
    resampled_cycle_length_df = dc_applied_df.resample(f'{cycle_length}min', on='cycle_ref_time', origin='start_day')
    last_call_of_each_group = resampled_cycle_length_df.last()
    assert last_call_of_each_group['end_time_wrt_ref'].all() < time_on_in_secs

def are_there_expected_number_of_cycles(location_df, num_of_detections, cycle_length, data_params):
    first_date = pd.to_datetime(location_df['call_start_time']).iloc[0].date()
    last_date = (pd.to_datetime(location_df['call_start_time'])).iloc[-1].date()
    first_dt = dt.datetime.combine(first_date, pd.to_datetime(data_params['recording_start'], format="%H:%M").time())
    last_dt = dt.datetime.combine(last_date, pd.to_datetime(data_params['recording_end'], format="%H:%M").time())

    all_cycles = pd.date_range(first_dt, last_dt, freq=f'{cycle_length}t')
    between_time_cycles = all_cycles[(all_cycles.indexer_between_time(data_params['recording_start'], data_params['recording_end']))]
    
    assert num_of_detections.shape[0] <= between_time_cycles.shape[0]

def simulate_dutycycle_on_detections(location_df, data_params):
    """
    Simulates a provided duty-cycling scheme on the provided location summary of concatenated bd2 outputs.
    """ 
    
    cycle_length = int(data_params['cur_dc_tag'].split('of')[1])
    time_on_in_mins = int(data_params['cur_dc_tag'].split('of')[0])
    time_on_in_secs = (60*time_on_in_mins)

    location_df = assign_cycle_groups_to_each_call(location_df, cycle_length, data_params)
    dc_applied_df = gather_calls_existing_in_on_windows(location_df, time_on_in_secs)
    all_last_calls_of_cycle_within_recording(dc_applied_df, cycle_length, time_on_in_secs)

    return dc_applied_df

def gather_calls_existing_in_on_windows(location_df, time_on_in_secs):

    location_df.insert(0, 'end_time_wrt_ref', (location_df['call_end_time'] - location_df['cycle_ref_time']).dt.total_seconds())
    location_df.insert(0, 'start_time_wrt_ref', (location_df['call_start_time'] - location_df['cycle_ref_time']).dt.total_seconds())
    dc_applied_df = location_df.loc[(location_df['end_time_wrt_ref'] <= time_on_in_secs)&(location_df['start_time_wrt_ref'] >= 0)].copy()

    return dc_applied_df

def assign_cycle_groups_to_each_call(location_df, cycle_length, data_params):
    location_df['ref_time'] = pd.DatetimeIndex(location_df['call_start_time'])
    location_df['cycle_ref_time'] = pd.DatetimeIndex(location_df['call_start_time'])
    location_df['call_end_time'] = pd.DatetimeIndex(location_df['call_end_time'])
    location_df['call_start_time'] = pd.DatetimeIndex(location_df['call_start_time'])

    resampled_cycle_length_df = location_df.resample(f'{cycle_length}min', on='cycle_ref_time', origin='start_day')
    first_call_of_each_group = resampled_cycle_length_df.first().between_time(data_params['recording_start'], data_params['recording_end'])
    are_there_expected_number_of_cycles(location_df, first_call_of_each_group, cycle_length, data_params)
    location_df['cycle_ref_time'] = pd.DatetimeIndex(resampled_cycle_length_df['cycle_ref_time'].transform(lambda x: x.name))

    return location_df

def prepare_summary_for_plotting_with_duty_cycle(file_paths, dc_tag, bin_size):
    """
    Generates a duty-cycled location summary of concatenated bd2 outputs for measuring effects of duty-cycling.
    """
    cycle_length = int(dc_tag.split('of')[-1])
    time_on = int(dc_tag.split('of')[0])
    time_on_in_secs = (60*time_on)

    location_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["detector_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)
    plottable_location_df = simulate_dutycycle_on_detections(location_df, cycle_length, time_on_in_secs, bin_size)

    return plottable_location_df

def simulate_dutycycle_on_detections_with_bins(location_df, dc_tag, bin_size):
    """
    Simulates a provided duty-cycling scheme on the provided location summary of concatenated bd2 outputs.
    """
    cycle_length = int(dc_tag.split('of')[1])
    time_on = int(dc_tag.split('of')[0])

    location_df['ref_time'] = pd.DatetimeIndex(location_df['call_start_time'])
    location_df['cycle_ref_time'] = pd.DatetimeIndex(location_df['call_start_time'])
    location_df['call_end_time'] = pd.DatetimeIndex(location_df['call_end_time'])
    location_df['call_start_time'] = pd.DatetimeIndex(location_df['call_start_time'])

    resampled_cycle_length_df = location_df.resample(f'{cycle_length}min', on='cycle_ref_time', origin='start_day')
    location_df['cycle_ref_time'] = pd.DatetimeIndex(resampled_cycle_length_df['cycle_ref_time'].transform(lambda x: x.name))

    resampled_bin_df = location_df.resample(f'{bin_size}min', on='ref_time', origin='start_day')
    location_df['ref_time'] = pd.DatetimeIndex(resampled_bin_df['ref_time'].transform(lambda x: x.name))

    location_df.insert(0, 'end_time_wrt_ref', (location_df['call_end_time'] - location_df['cycle_ref_time']).dt.total_seconds())
    location_df.insert(0, 'start_time_wrt_ref', (location_df['call_start_time'] - location_df['cycle_ref_time']).dt.total_seconds())
    dc_applied_df = location_df.loc[(location_df['end_time_wrt_ref'] <= (60*time_on))&(location_df['start_time_wrt_ref'] >= 0)].copy()

    return dc_applied_df

def prepare_summary_for_plotting_with_duty_cycle_and_bins(file_paths, dc_tag, bin_size):
    """
    Generates a duty-cycled location summary of concatenated bd2 outputs for measuring effects of duty-cycling.
    """

    location_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["detector_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)
    plottable_location_df = simulate_dutycycle_on_detections_with_bins(location_df, dc_tag, bin_size)

    return plottable_location_df

def get_list_of_dc_tags(cycle_lengths=[1800, 360], percent_ons=[0.1667]):
    """
    Takes a list of cycle lengths and the percent ON values to generate a list of tags.
    Cycle length is the period of the duty-cycle when the recording resets.
    Percent ON is the percentage of the cycle length where we simulate recording; only regarding calls within recording periods.
    Each cycle length x percent ON combination will be saved into a list in the format "{percent_on*cycle_lenght}of{cycle_length}"
    """

    dc_tags = []

    cycle_length = 30
    percent_on = 1.0
    dc_tag = f"{round(percent_on*cycle_length)}of{cycle_length}"
    dc_tags += [dc_tag]

    for cycle_length in cycle_lengths:
        for percent_on in percent_ons:
            dc_tag = f"{round(percent_on*cycle_length)}of{cycle_length}"
            dc_tags += [dc_tag]

    return dc_tags