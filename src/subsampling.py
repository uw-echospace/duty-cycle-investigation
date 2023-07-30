import pandas as pd
import data_handling as dh


def simulate_dutycycle_on_detections(location_df, cycle_length, percent_on):
    location_df['ref_time'] = pd.DatetimeIndex(location_df['ref_time'])
    location_df['call_end_time'] = pd.DatetimeIndex(location_df['call_end_time'])
    location_df['call_start_time'] = pd.DatetimeIndex(location_df['call_start_time'])
    
    resampled_df = location_df.resample(f'{cycle_length}S', on='ref_time')
    location_df['ref_time'] = resampled_df['ref_time'].transform(lambda x: x.name)
    location_df.insert(0, 'end_time_wrt_ref', (location_df['call_end_time'] - location_df['ref_time']).dt.total_seconds())
    location_df.insert(0, 'start_time_wrt_ref', (location_df['call_start_time'] - location_df['ref_time']).dt.total_seconds())

    dc_applied_df = location_df.loc[location_df['end_time_wrt_ref'] <= round(cycle_length*percent_on)]
    return dc_applied_df


def prepare_summary_for_plotting_with_duty_cycle(site_name, cycle_length, percent_on, type_tag):
    location_df = pd.read_csv(f'../data/2022_bd2_summary/{site_name}/bd2__{type_tag}{site_name}_2022.csv', index_col=0)
    plottable_location_df = simulate_dutycycle_on_detections(location_df, cycle_length, percent_on)
    dc_tag = f"{round(percent_on*cycle_length)}e{cycle_length}"
    plottable_location_df.to_csv(f"../data/2022_bd2_summary/{site_name}/duty_cycled/simulated_schemes/bd2__{type_tag}{site_name}_2022_{dc_tag}.csv")

    return plottable_location_df

def get_list_of_dc_tags(cycle_lengths, percent_ons):
    dc_tags = []

    cycle_length = 1800
    percent_on = 1.0
    dc_tag = f"{round(percent_on*cycle_length)}e{cycle_length}"
    dc_tags += [dc_tag]

    for cycle_length in cycle_lengths:
        for percent_on in percent_ons:
            dc_tag = f"{round(percent_on*cycle_length)}e{cycle_length}"
            dc_tags += [dc_tag]

    return dc_tags

def construct_activity_arr_from_dc_tags(data_params):
    activity_arr = pd.DataFrame()

    for dc_tag in data_params['dc_tags']:
        cycle_length = int(dc_tag.split('e')[1])
        percent_on = float(dc_tag.split('e')[0]) / cycle_length

        location_df = prepare_summary_for_plotting_with_duty_cycle(data_params['site_tag'], cycle_length, percent_on, data_params['type_tag'])
        dc_dets = dh.construct_activity_arr_from_location_summary(location_df, dc_tag)
        dc_dets = dc_dets.set_index("Date_and_Time_UTC")
        activity_arr = pd.concat([activity_arr, dc_dets], axis=1)

    activity_arr.to_csv(f'../data/2022_bd2_summary/{data_params["site_tag"]}/duty_cycled/dc__{data_params["type_tag"]}{data_params["site_tag"]}_summary.csv')

    return activity_arr