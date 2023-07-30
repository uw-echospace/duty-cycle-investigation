import pandas as pd
import data_handling as dh

from pathlib import Path


def simulate_dutycycle_on_detections(location_df, dc_tag):
    cycle_length = int(dc_tag.split('e')[1])
    percent_on = float(dc_tag.split('e')[0]) / cycle_length

    location_df['ref_time'] = pd.DatetimeIndex(location_df['ref_time'])
    location_df['call_end_time'] = pd.DatetimeIndex(location_df['call_end_time'])
    location_df['call_start_time'] = pd.DatetimeIndex(location_df['call_start_time'])
    
    resampled_df = location_df.resample(f'{cycle_length}S', on='ref_time')
    location_df['ref_time'] = resampled_df['ref_time'].transform(lambda x: x.name)
    location_df.insert(0, 'end_time_wrt_ref', (location_df['call_end_time'] - location_df['ref_time']).dt.total_seconds())
    location_df.insert(0, 'start_time_wrt_ref', (location_df['call_start_time'] - location_df['ref_time']).dt.total_seconds())

    dc_applied_df = location_df.loc[location_df['end_time_wrt_ref'] <= round(cycle_length*percent_on)]
    return dc_applied_df


def prepare_summary_for_plotting_with_duty_cycle(site_tag, dc_tag, type_tag):
    location_df = pd.read_csv(f'{Path(__file__).resolve().parent}/../data/2022_bd2_summary/{site_tag}/bd2__{type_tag}{site_tag}_2022.csv', index_col=0)
    plottable_location_df = simulate_dutycycle_on_detections(location_df, dc_tag)
    csv_filename = f'bd2__{type_tag}{site_tag}_2022_{dc_tag}.csv'
    plottable_location_df.to_csv(f"{Path(__file__).resolve().parent}/../data/2022_bd2_summary/{site_tag}/duty_cycled/simulated_schemes/{csv_filename}")

    return plottable_location_df

def get_list_of_dc_tags(cycle_lengths=[1800, 360], percent_ons=[0.1667]):
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

        location_df = prepare_summary_for_plotting_with_duty_cycle(data_params['site_tag'], dc_tag, data_params['type_tag'])
        dc_dets = dh.construct_activity_arr_from_location_summary(location_df, dc_tag)
        dc_dets = dc_dets.set_index("Date_and_Time_UTC")
        activity_arr = pd.concat([activity_arr, dc_dets], axis=1)

    csv_filename = f'dc__{data_params["type_tag"]}{data_params["site_tag"]}_summary.csv'
    activity_arr.to_csv(f'{Path(__file__).resolve().parent}/../data/2022_bd2_summary/{data_params["site_tag"]}/duty_cycled/{csv_filename}')

    return activity_arr