from pathlib import Path

import dask.dataframe as dd
import pandas as pd

import sys
sys.path.append("../src/bout")

import bout as bt
import activity.subsampling as ss
from core import FREQ_GROUPS


def generate_activity_dets_results(data_params, file_paths, save=True):
    """
    Generates an activity summary using the detected calls for each provided duty-cycling scheme and puts them together for comparison.
    A summary of activity is formatted as the number of detected bat calls per time interval.
    """

    activity_dets_arr = pd.DataFrame()

    for dc_tag in data_params['dc_tags']:

        location_df = ss.prepare_summary_for_plotting_with_duty_cycle(file_paths, dc_tag, data_params['bin_size'])
        num_of_detections = get_number_of_detections_per_interval(location_df, data_params)
        dc_dets = construct_activity_arr_from_location_summary(num_of_detections, dc_tag, file_paths, data_params)
        dc_dets = dc_dets.set_index("datetime_UTC")
        activity_dets_arr = pd.concat([activity_dets_arr, dc_dets], axis=1)

    if save:
        activity_dets_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_dets_TYPE_SITE_summary"]}.csv')

    return activity_dets_arr

def generate_activity_bouts_results(data_params, file_paths, save=True):
    """
    Generates an activity summary using the activity bouts for each provided duty-cycling scheme and puts them together for comparison.
    A summary of activity is formatted as the % of time occupied by bouts per time interval.
    """

    activity_bouts_arr = pd.DataFrame()

    for dc_tag in data_params['dc_tags']:

        location_df = ss.prepare_summary_for_plotting_with_duty_cycle(file_paths, dc_tag, data_params['bin_size'])
        bout_metrics = bt.generate_bout_metrics_for_location_and_freq(location_df, data_params, dc_tag)
        bout_duration_per_interval = get_bout_duration_per_interval(bout_metrics, data_params)
        dc_bouts = construct_activity_arr_from_bout_metrics(bout_duration_per_interval, data_params, file_paths, dc_tag)
        dc_bouts = dc_bouts.set_index("datetime_UTC")
        activity_bouts_arr = pd.concat([activity_bouts_arr, dc_bouts], axis=1)

    if save:
        activity_bouts_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_bouts_TYPE_SITE_summary"]}.csv')

    return activity_bouts_arr

def generate_activity_inds_results(data_params, file_paths, save=True):
    """
    Generates an activity summary using the Activity Index for each provided duty-cycling scheme and puts them together for comparison.
    A summary of activity is formatted as the activity index per time interval.
    """

    activity_inds_arr = pd.DataFrame()

    for dc_tag in data_params['dc_tags']:

        location_df = ss.prepare_summary_for_plotting_with_duty_cycle(file_paths, dc_tag, data_params['bin_size'])
        activity_indices = get_activity_index_per_interval(location_df, data_params)
        dc_dets = construct_activity_indices_arr(activity_indices, dc_tag, file_paths, data_params)
        dc_dets = dc_dets.set_index("datetime_UTC")
        activity_inds_arr = pd.concat([activity_inds_arr, dc_dets], axis=1)

    if save:
        activity_inds_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_inds_TYPE_SITE_summary"]}.csv')

    return activity_inds_arr


def assemble_initial_location_summary(file_paths):
    """
    Puts together all bd2 outputs in data/raw and converts detection start_times to datetime objects.
    Returns and saves a summary of bd2-detected bat calls within a desired frequency band.
    """

    location_df = dd.read_csv(f'{file_paths["raw_SITE_folder"]}/*.csv', dtype=str).compute()
    location_df['low_freq'] = location_df['low_freq'].astype('float')
    location_df['high_freq'] = location_df['high_freq'].astype('float')
    file_dts = pd.to_datetime(location_df['input_file'], format='%Y%m%d_%H%M%S', exact=False)
    anchor_start_times = file_dts + pd.to_timedelta(location_df['start_time'].values.astype('float64'), unit='S')
    anchor_end_times = file_dts + pd.to_timedelta(location_df['end_time'].values.astype('float64'), unit='S')

    location_df.insert(0, 'call_end_time', anchor_end_times)
    location_df.insert(0, 'call_start_time', anchor_start_times)
    location_df.insert(0, 'ref_time', anchor_start_times)

    return location_df

def add_frequency_groups_to_summary_using_thresholds(location_df, file_paths, data_params, save=True):
    
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

def add_frequency_groups_to_summary_using_kmeans(location_df, file_paths, data_params, save=True):

    location_df.insert(0, 'freq_group', '')
    location_classes = pd.read_csv(Path(file_paths['SITE_classes_file']), index_col=0)
    kept_calls_from_location = location_df.iloc[location_classes['index_in_summary'].values].copy()
    kept_calls_from_location['freq_group'] = location_classes['KMEANS_CLASSES'].values

    if data_params['type_tag'] != '':
        kept_calls_from_location = kept_calls_from_location.loc[kept_calls_from_location['freq_group']==data_params['type_tag']]

    if save:
        kept_calls_from_location.to_csv(f'{file_paths["SITE_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}.csv')

    return kept_calls_from_location


def assemble_single_bd2_output_use_thresholds_to_group(path_to_bd2_output, data_params):
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



def assemble_single_bd2_output_use_kmeans_to_group(path_to_bd2_output, file_paths):
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
    location_classes = pd.read_csv(Path(file_paths['SITE_classes_file']), index_col=0)
    file_classes = location_classes.loc[location_classes['file_name']==Path(location_df['input_file'].unique().item()).name].copy()
    kept_calls_from_location = location_df.iloc[file_classes['index_in_file'].values].copy()
    kept_calls_from_location['freq_group'] = file_classes['KMEANS_CLASSES'].values
    
    return kept_calls_from_location


def get_number_of_detections_per_interval(location_df, data_params):
    """
    Constructs a pandas Series that records the # of detections observed per interval.
    The used interval is the one stored inside data_params['bin_size']
    """

    location_df.insert(0, 'call_durations', (location_df['call_end_time'] - location_df['call_start_time']))
    df_resampled_every_30 = location_df.resample(f"{data_params['bin_size']}T", on='ref_time')
    num_of_detections = df_resampled_every_30['ref_time'].count()

    return num_of_detections

def construct_activity_arr_from_location_summary(num_of_detections, dc_tag, file_paths, data_params):
    """
    Construct an activity summary of the number of detections recorded per date and time interval.
    Will be used later to assemble an activity summary for each duty-cycling scheme to compare effects.
    """

    all_processed_filepaths = sorted(list(map(str, list(Path(f'{file_paths["raw_SITE_folder"]}').glob('*.csv')))))
    all_processed_datetimes = pd.to_datetime(all_processed_filepaths, format="%Y%m%d_%H%M%S", exact=False)
    col_name = f"num_dets ({dc_tag})"
    incomplete_activity_arr = pd.DataFrame(num_of_detections.values, index=num_of_detections.index, columns=[col_name])
    activity_arr = incomplete_activity_arr.reindex(index=all_processed_datetimes, fill_value=0).resample(f"{data_params['bin_size']}T").first()
    activity_arr = activity_arr.between_time(data_params['recording_start'], data_params['recording_end'], inclusive='left')

    return pd.DataFrame(list(zip(activity_arr.index, activity_arr[col_name].values)), columns=["datetime_UTC", col_name])

def get_bout_duration_per_interval(bout_metrics, data_params):
    """
    Constructs a pandas Series that records the duration of time occupied by bouts observed per interval.
    The used interval is the one stored inside data_params['bin_size']
    """

    bout_metrics['ref_time'] = pd.DatetimeIndex(bout_metrics['start_time_of_bout'])
    bout_metrics['total_bout_duration_in_secs'] = bout_metrics['bout_duration_in_secs']
    bout_metrics = bout_metrics.set_index('ref_time')

    bout_duration_per_interval = bout_metrics.resample(f"{data_params['bin_size']}T")['total_bout_duration_in_secs'].sum()

    return bout_duration_per_interval

def construct_activity_arr_from_bout_metrics(bout_duration_per_interval, data_params, file_paths, dc_tag):
    """
    Construct an activity summary of the % of time occupied by bouts per date and time interval.
    Will be used later to assemble an activity summary for each duty-cycling scheme to compare effects.
    """

    time_occupied_by_bouts  = bout_duration_per_interval.values
    percent_time_occupied_by_bouts = (100*(time_occupied_by_bouts / (60*float(data_params['bin_size']))))

    all_processed_filepaths = sorted(list(map(str, list(Path(f'{file_paths["raw_SITE_folder"]}').glob('*.csv')))))
    all_processed_datetimes = pd.to_datetime(all_processed_filepaths, format="%Y%m%d_%H%M%S", exact=False)
    bout_dpi_df = pd.DataFrame(list(zip(bout_duration_per_interval.index, percent_time_occupied_by_bouts)),
                                columns=['ref_time', f'bout_time ({dc_tag})'])
    bout_dpi_df = bout_dpi_df.set_index('ref_time')
    bout_dpi_df = bout_dpi_df.reindex(index=all_processed_datetimes, fill_value=0).resample(f"{data_params['bin_size']}T").first()
    bout_dpi_df = bout_dpi_df.between_time(data_params['recording_start'], data_params['recording_end'], inclusive='left')

    return pd.DataFrame(list(zip(bout_dpi_df.index, bout_dpi_df[f'bout_time ({dc_tag})'].values)), columns=["datetime_UTC", f'bout_time ({dc_tag})'])

def get_activity_index_per_interval(location_df, data_params):
    """
    Constructs a pandas Series that records the activity index observed per interval.
    The used interval is the one stored inside data_params['bin_size']
    The activity index time block is stored inside data_params['index_time_block_in_secs']
    """

    location_df['ref_time'] = location_df['call_start_time']

    temp = location_df.resample(f'{data_params["index_time_block_in_secs"]}S', on='ref_time')['ref_time'].count()
    temp[temp>0] = 1
    activity_indices = temp.resample(f"{data_params['bin_size']}T").sum()
    
    return activity_indices

def construct_activity_indices_arr(activity_indices, dc_tag, file_paths, data_params):
    """
    Construct an activity summary of the activity index per date and time interval.
    Will be used later to assemble an activity summary for each duty-cycling scheme to compare effects.
    """

    col_name = f"activity_index ({dc_tag})"
    incomplete_activity_arr = pd.DataFrame(activity_indices.values, index=activity_indices.index, columns=[col_name])

    all_processed_filepaths = sorted(list(map(str, list(Path(f'{file_paths["raw_SITE_folder"]}').glob('*.csv')))))
    all_processed_datetimes = pd.to_datetime(all_processed_filepaths, format="%Y%m%d_%H%M%S", exact=False)
    
    activity_arr = incomplete_activity_arr.reindex(index=all_processed_datetimes, fill_value=0).resample(f"{data_params['bin_size']}T").first()
    activity_arr = activity_arr.between_time(data_params['recording_start'], data_params['recording_end'], inclusive='left')

    return pd.DataFrame(list(zip(activity_arr.index, activity_arr[col_name].values)), columns=["datetime_UTC", col_name])

def construct_activity_grid_for_number_of_dets(activity_arr, dc_tag):
    """
    Reshapes a provided activity summary column to make a grid with date columns and time rows.
    This grid is the one we provide to our activity plotting functions.
    """

    activity_datetimes = pd.to_datetime(activity_arr.index.values)
    raw_dates = activity_datetimes.strftime("%m/%d/%y")
    raw_times = activity_datetimes.strftime("%H:%M")

    col_name = f"num_dets ({dc_tag})"
    data = list(zip(raw_dates, raw_times, activity_arr[col_name]))
    activity = pd.DataFrame(data, columns=["Date (UTC)", "Time (UTC)", col_name])
    activity_df = activity.pivot(index="Time (UTC)", columns="Date (UTC)", values=col_name)

    return activity_df


def construct_activity_grid_for_bouts(activity_arr, dc_tag):
    """
    Reshapes a provided activity summary column to make a grid with date columns and time rows.
    This grid is the one we provide to our activity plotting functions.
    """

    activity_datetimes = pd.to_datetime(activity_arr.index.values)
    raw_dates = activity_datetimes.strftime("%m/%d/%y")
    raw_times = activity_datetimes.strftime("%H:%M")

    col_name = f"bout_time ({dc_tag})"
    data = list(zip(raw_dates, raw_times, activity_arr[col_name]))
    activity = pd.DataFrame(data, columns=["Date (UTC)", "Time (UTC)", col_name])
    activity_df = activity.pivot(index="Time (UTC)", columns="Date (UTC)", values=col_name)

    return activity_df


def construct_activity_grid_for_inds(activity_arr, dc_tag):
    """
    Reshapes a provided activity summary column to make a grid with date columns and time rows.
    This grid is the one we provide to our activity plotting functions.
    """
    
    activity_datetimes = pd.to_datetime(activity_arr.index.values)
    raw_dates = activity_datetimes.strftime("%m/%d/%y")
    raw_times = activity_datetimes.strftime("%H:%M")

    col_name = f"activity_index ({dc_tag})"
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

    for column in activity_arr.columns:
        if dc_tag in column:
            col_name = column

    data = list(zip(raw_dates, raw_times, activity_arr[col_name]))
    presence = pd.DataFrame(data, columns=["Date (UTC)", "Time (UTC)", col_name])
    presence.loc[presence[col_name] > 0, col_name] = 1
    presence_df = presence.pivot(index="Time (UTC)", columns="Date (UTC)", values=col_name)

    return presence_df