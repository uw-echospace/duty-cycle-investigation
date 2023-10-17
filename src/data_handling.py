from pathlib import Path
import glob

import dask.dataframe as dd
import pandas as pd
import numpy as np
import datetime as dt

import subsampling as ss
from core import FREQ_GROUPS


def generate_activity_dets_results(data_params, file_paths):
    """
    Puts together a location summary with all batdetect2 outputs in data/raw and generates a summary of activity.
    A summary of activity is formatted as the number of detected bat calls per time interval.
    """

    return ss.construct_activity_dets_arr_from_dc_tags(data_params, file_paths)


def generate_activity_bouts_results(data_params, file_paths):
    """
    Puts together a location summary with all batdetect2 outputs in data/raw and generates a summary of activity.
    A summary of activity is formatted as the number of detected bat calls per time interval.
    """

    return ss.construct_activity_bouts_arr_from_dc_tags(data_params, file_paths)


def generate_activity_inds_results(data_params, file_paths):
    """
    Puts together a location summary with all batdetect2 outputs in data/raw and generates a summary of activity.
    A summary of activity is formatted as the number of detected bat calls per time interval.
    """

    return ss.construct_activity_inds_arr_from_dc_tags(data_params, file_paths)


def assemble_initial_location_summary(data_params, file_paths, save=True):
    """
    Puts together all bd2 outputs in data/raw and converts detection start_times to datetime objects.
    Returns and saves a summary of bd2-detected bat calls within a desired frequency band.
    """

    location_df = dd.read_csv(f'{file_paths["raw_SITE_folder"]}/*.csv').compute()
    location_df['low_freq'] = location_df['low_freq'].astype('float')
    location_df['high_freq'] = location_df['high_freq'].astype('float')
    file_dts = pd.to_datetime(location_df['input_file'], format='%Y%m%d_%H%M%S', exact=False)
    anchor_start_times = file_dts + pd.to_timedelta(location_df['start_time'].values.astype('float64'), unit='S')
    anchor_end_times = file_dts + pd.to_timedelta(location_df['end_time'].values.astype('float64'), unit='S')

    location_df.insert(0, 'call_end_time', anchor_end_times)
    location_df.insert(0, 'call_start_time', anchor_start_times)
    location_df.insert(0, 'ref_time', anchor_start_times)
    location_df.insert(0, 'freq_group', '')

    groups = FREQ_GROUPS[data_params['site_tag']]
    blue_group = groups['LF1']
    red_group = groups['HF1']
    yellow_group = groups['HF2']

    call_is_yellow = (location_df['low_freq']>=yellow_group[0])&(location_df['high_freq']<=yellow_group[1])
    call_is_red = (location_df['low_freq']>=red_group[0])&(location_df['high_freq']<=red_group[1])
    call_is_blue = (location_df['low_freq']>=blue_group[0])&(location_df['high_freq']<=blue_group[1])

    location_df.loc[call_is_yellow, 'freq_group'] = 'HF2'
    location_df.loc[call_is_red&(~(call_is_yellow)), 'freq_group'] = 'HF1'
    location_df.loc[call_is_blue&(~(call_is_red | call_is_yellow)), 'freq_group'] = 'LF1'

    if data_params['type_tag'] != '':
        location_df = location_df.loc[location_df['freq_group']==data_params['type_tag']]

    if save:
        location_df.to_csv(f'{file_paths["SITE_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}.csv')

    return location_df


def assemble_single_bd2_output(path_to_bd2_output, data_params):
    """
    Adds columns to bd2 output for a single file to be of the same format as the output
    of the assemble_initial_location_summary() method.
    """

    location_df = pd.read_csv(path_to_bd2_output)
    file_dts = pd.to_datetime(location_df['input_file'], format='%Y%m%d_%H%M%S', exact=False)

    anchor_start_times = file_dts + pd.to_timedelta(location_df['start_time'].values.astype('float64'), unit='S')
    anchor_end_times = file_dts + pd.to_timedelta(location_df['end_time'].values.astype('float64'), unit='S') 

    location_df.insert(0, 'call_end_time', anchor_end_times)
    location_df.insert(0, 'call_start_time', anchor_start_times)
    location_df.insert(0, 'ref_time', anchor_start_times)
    location_df.insert(0, 'freq_group', '')

    groups = FREQ_GROUPS[data_params['site_tag']]
    blue_group = groups['LF1']
    red_group = groups['HF1']
    yellow_group = groups['HF2']

    call_is_yellow = (location_df['low_freq']>=yellow_group[0])&(location_df['high_freq']<=yellow_group[1])
    call_is_red = (location_df['low_freq']>=red_group[0])&(location_df['high_freq']<=red_group[1])
    call_is_blue = (location_df['low_freq']>=blue_group[0])&(location_df['high_freq']<=blue_group[1])

    location_df.loc[call_is_yellow, 'freq_group'] = 'HF2'
    location_df.loc[call_is_red&(~(call_is_yellow)), 'freq_group'] = 'HF1'
    location_df.loc[call_is_blue&(~(call_is_red | call_is_yellow)), 'freq_group'] = 'LF1'
    
    return location_df


def construct_activity_arr_from_location_summary(location_df, dc_tag, file_paths, data_params):
    """
    Construct an activity summary for each date and time's number of detected calls. Only looking from 03:00 to 13:00 UTC.
    Will be used later to assembled an activity summary for each duty-cycling scheme to compare effects.
    """
    location_df.insert(0, 'call_durations', (location_df['call_end_time'] - location_df['call_start_time']))
    df_resampled_every_30 = location_df.resample(f"{data_params['resolution_in_min']}T", on='ref_time')
    shortest_call_per_30 = pd.to_numeric(df_resampled_every_30['call_durations'].min())/1e6
    max_allowable_calls_per_30 = np.abs(((int(data_params['resolution_in_min'])*60*60) / shortest_call_per_30))
    num_of_detections = df_resampled_every_30['ref_time'].count()
    test_number_of_detections_less_than_max_per_30(num_of_detections, max_allowable_calls_per_30)

    all_processed_filepaths = sorted(list(map(str, list(Path(f'{file_paths["raw_SITE_folder"]}').glob('*.csv')))))
    all_processed_datetimes = pd.to_datetime(all_processed_filepaths, format="%Y%m%d_%H%M%S", exact=False)
    col_name = f"Number_of_Detections ({dc_tag})"
    incomplete_activity_arr = pd.DataFrame(num_of_detections.values, index=num_of_detections.index, columns=[col_name])
    activity_arr = incomplete_activity_arr.reindex(index=all_processed_datetimes, fill_value=0).resample(f"{data_params['resolution_in_min']}T").first()
    activity_arr = activity_arr.between_time(data_params['recording_start'], data_params['recording_end'], inclusive='left')

    return pd.DataFrame(list(zip(activity_arr.index, activity_arr[col_name].values)), columns=["Date_and_Time_UTC", col_name])

def test_number_of_detections_less_than_max_per_30(num_of_detections, max_allowable_calls_per_30):
    assertion = (num_of_detections <= max_allowable_calls_per_30).values
    assert(not(False in assertion))

def construct_activity_arr_from_bout_metrics(bout_metrics, data_params, file_paths, dc_tag):
    all_processed_filepaths = sorted(list(map(str, list(Path(f'{file_paths["raw_SITE_folder"]}').glob('*.csv')))))
    all_processed_datetimes = pd.to_datetime(all_processed_filepaths, format="%Y%m%d_%H%M%S", exact=False)

    bout_metrics['ref_time'] = pd.DatetimeIndex(bout_metrics['start_time_of_bout'])
    bout_metrics['total_bout_duration_in_secs'] = bout_metrics['bout_duration_in_secs']
    bout_metrics = bout_metrics.set_index('ref_time')

    bout_duration_per_interval = bout_metrics.resample(f"{data_params['resolution_in_min']}T")['total_bout_duration_in_secs'].sum()

    time_occupied_by_bouts  = bout_duration_per_interval.values
    percent_time_occupied_by_bouts = (100*(time_occupied_by_bouts / (60*float(data_params['resolution_in_min']))))
    test_bout_percentages_less_than_100(percent_time_occupied_by_bouts)

    bout_dpi_df = pd.DataFrame(list(zip(bout_duration_per_interval.index, percent_time_occupied_by_bouts)), columns=['ref_time', f'percentage_time_occupied_by_bouts ({dc_tag})'])
    bout_dpi_df = bout_dpi_df.set_index('ref_time')
    bout_dpi_df = bout_dpi_df.reindex(index=all_processed_datetimes, fill_value=0).resample(f"{data_params['resolution_in_min']}T").first().between_time(data_params['recording_start'], data_params['recording_end'], inclusive='left')

    return pd.DataFrame(list(zip(bout_dpi_df.index, bout_dpi_df[f'percentage_time_occupied_by_bouts ({dc_tag})'].values)), columns=["Date_and_Time_UTC", f'percentage_time_occupied_by_bouts ({dc_tag})'])

def test_bout_percentages_less_than_100(percent_time_occupied_by_bouts):
    assert(percent_time_occupied_by_bouts.max() <= 100)

def construct_activity_indices_arr(location_df, dc_tag, file_paths, data_params):
    location_df['ref_time'] = location_df['call_start_time']

    temp = location_df.resample(f'{data_params["index_time_block_in_secs"]}S', on='ref_time')['ref_time'].count()
    temp[temp>0] = 1
    activity_indices = temp.resample(f"{data_params['resolution_in_min']}T").sum()
    test_activity_indices_less_than_max(activity_indices.values, data_params)

    col_name = f"Activity Indices ({dc_tag})"
    incomplete_activity_arr = pd.DataFrame(activity_indices.values, index=activity_indices.index, columns=[col_name])

    all_processed_filepaths = sorted(list(map(str, list(Path(f'{file_paths["raw_SITE_folder"]}').glob('*.csv')))))
    all_processed_datetimes = pd.to_datetime(all_processed_filepaths, format="%Y%m%d_%H%M%S", exact=False)
    
    activity_arr = incomplete_activity_arr.reindex(index=all_processed_datetimes, fill_value=0).resample(f"{data_params['resolution_in_min']}T").first()
    activity_arr = activity_arr.between_time(data_params['recording_start'], data_params['recording_end'], inclusive='left')

    return pd.DataFrame(list(zip(activity_arr.index, activity_arr[col_name].values)), columns=["Date_and_Time_UTC", col_name])

def test_activity_indices_less_than_max(activity_indices, data_params):
    time_block_duration = int(data_params['index_time_block_in_secs'])
    peak_index = (60*int(data_params['resolution_in_min'])/time_block_duration)
    assert(activity_indices.max()<=peak_index)

def construct_activity_grid_for_number_of_dets(activity_arr, dc_tag):
    """
    Reshapes a provided activity summary column to make a grid with date columns and time rows.
    This grid is the one we provide to our activity plotting functions.
    """

    activity_datetimes = pd.to_datetime(activity_arr.index.values)
    raw_dates = activity_datetimes.strftime("%m/%d/%y")
    raw_times = activity_datetimes.strftime("%H:%M")

    col_name = f"Number_of_Detections ({dc_tag})"
    data = list(zip(raw_dates, raw_times, activity_arr[col_name]))
    activity = pd.DataFrame(data, columns=["Date (UTC)", "Time (UTC)", col_name])
    activity_df = activity.pivot(index="Time (UTC)", columns="Date (UTC)", values=col_name)

    return activity_df


def construct_activity_grid_for_bouts(activity_arr, dc_tag):
    activity_datetimes = pd.to_datetime(activity_arr.index.values)
    raw_dates = activity_datetimes.strftime("%m/%d/%y")
    raw_times = activity_datetimes.strftime("%H:%M")

    col_name = f"percentage_time_occupied_by_bouts ({dc_tag})"
    data = list(zip(raw_dates, raw_times, activity_arr[col_name]))
    activity = pd.DataFrame(data, columns=["Date (UTC)", "Time (UTC)", col_name])
    activity_df = activity.pivot(index="Time (UTC)", columns="Date (UTC)", values=col_name)

    return activity_df


def construct_activity_grid_for_inds(activity_arr, dc_tag):
    activity_datetimes = pd.to_datetime(activity_arr.index.values)
    raw_dates = activity_datetimes.strftime("%m/%d/%y")
    raw_times = activity_datetimes.strftime("%H:%M")

    col_name = f"Activity Indices ({dc_tag})"
    data = list(zip(raw_dates, raw_times, activity_arr[col_name]))
    activity = pd.DataFrame(data, columns=["Date (UTC)", "Time (UTC)", col_name])
    activity_df = activity.pivot(index="Time (UTC)", columns="Date (UTC)", values=col_name)

    return activity_df


def construct_presence_grid(activity_arr, dc_tag):
    """
    Constructs a grid similar to construct_activity_grid(). The values are no longer number of calls.
    Instead, the values are 1 if there was a call detected for a date and time, and 0 if calls were absent.
    """

    activity_datetimes = pd.to_datetime(activity_arr.index.values)
    raw_dates = activity_datetimes.strftime("%m/%d/%y")
    raw_times = activity_datetimes.strftime("%H:%M")

    col_name = f"Number_of_Detections ({dc_tag})"
    data = list(zip(raw_dates, raw_times, activity_arr[col_name]))
    presence = pd.DataFrame(data, columns=["Date (UTC)", "Time (UTC)", col_name])
    presence.loc[presence[col_name] > 0, col_name] = 1
    presence_df = presence.pivot(index="Time (UTC)", columns="Date (UTC)", values=col_name)

    return presence_df