import pandas as pd

def simulate_dutycycle_on_detections(location_df, dc_tag, bin_size):
    """
    Simulates a provided duty-cycling scheme on the provided location summary of concatenated bd2 outputs.
    """
    cycle_length = int(dc_tag.split('of')[1])
    time_on = int(dc_tag.split('of')[0])

    location_df['ref_time'] = pd.DatetimeIndex(location_df['call_start_time'])
    location_df['cycle_ref_time'] = pd.DatetimeIndex(location_df['call_start_time'])
    location_df['call_end_time'] = pd.DatetimeIndex(location_df['call_end_time'])
    location_df['call_start_time'] = pd.DatetimeIndex(location_df['call_start_time'])

    resampled_cycle_length_df = location_df.resample(f'{cycle_length}T', on='cycle_ref_time', origin='start_day')
    location_df['cycle_ref_time'] = pd.DatetimeIndex(resampled_cycle_length_df['cycle_ref_time'].transform(lambda x: x.name))

    resampled_bin_df = location_df.resample(f'{bin_size}T', on='ref_time', origin='start_day')
    location_df['ref_time'] = pd.DatetimeIndex(resampled_bin_df['ref_time'].transform(lambda x: x.name))

    location_df.insert(0, 'end_time_wrt_ref', (location_df['call_end_time'] - location_df['cycle_ref_time']).dt.total_seconds())
    location_df.insert(0, 'start_time_wrt_ref', (location_df['call_start_time'] - location_df['cycle_ref_time']).dt.total_seconds())
    dc_applied_df = location_df.loc[(location_df['end_time_wrt_ref'] <= (60*time_on))&(location_df['start_time_wrt_ref'] >= 0)].copy()

    return dc_applied_df

def prepare_summary_for_plotting_with_duty_cycle(file_paths, dc_tag, bin_size):
    """
    Generates a duty-cycled location summary of concatenated bd2 outputs for measuring effects of duty-cycling.
    """

    location_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)
    plottable_location_df = simulate_dutycycle_on_detections(location_df, dc_tag, bin_size)

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