from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import numpy as np
import datetime as dt

import subsampling as ss


def generate_activity_results(data_params, file_paths):
    assemble_initial_location_summary(data_params, file_paths) ## Use to update any bd2__(location summary).csv files
    return ss.construct_activity_arr_from_dc_tags(data_params, file_paths)


def assemble_initial_location_summary(data_params, file_paths, save=True):
    location_df = dd.read_csv(f'{file_paths["raw_SITE_folder"]}/*.csv').compute()
    file_dts = pd.to_datetime(location_df['input_file'], format='%Y%m%d_%H%M%S', exact=False)
    anchor_start_times = file_dts + pd.to_timedelta(location_df['start_time'].values.astype('float64'), unit='S')
    anchor_end_times = file_dts + pd.to_timedelta(location_df['end_time'].values.astype('float64'), unit='S')

    location_df.insert(0, 'call_end_time', anchor_end_times)
    location_df.insert(0, 'call_start_time', anchor_start_times)
    location_df.insert(0, 'ref_time', anchor_start_times)
 
    location_df = location_df.loc[(location_df["high_freq"]).astype('float64') < data_params["freq_tags"][1]]
    location_df = location_df.loc[(location_df["low_freq"]).astype('float64') > data_params["freq_tags"][0]]

    if save:
        location_df.to_csv(f'{file_paths["SITE_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}.csv')

    return location_df


def construct_activity_arr_from_location_summary(location_df, dc_tag, file_paths, resolution="30T"):
    all_processed_filepaths = sorted(list(map(str, list(Path(f'{file_paths["raw_SITE_folder"]}').iterdir()))))
    all_processed_datetimes = pd.to_datetime(all_processed_filepaths, format="%Y%m%d_%H%M%S", exact=False)
    col_name = f"Number_of_Detections ({dc_tag})"

    num_of_detections = location_df.resample(resolution, on='ref_time')['ref_time'].count()
    incomplete_activity_arr = pd.DataFrame(num_of_detections.values, index=num_of_detections.index, columns=[col_name])
    activity_arr = incomplete_activity_arr.reindex(index=all_processed_datetimes, fill_value=0).resample(resolution).first()
    activity_arr = activity_arr.between_time('03:00', '13:00')

    return pd.DataFrame(list(zip(activity_arr.index, activity_arr[col_name].values)), columns=["Date_and_Time_UTC", col_name])


def construct_activity_grid(activity_arr, dc_tag):
    activity_datetimes = pd.to_datetime(activity_arr.index.values)
    raw_dates = activity_datetimes.strftime("%m/%d/%y")
    raw_times = activity_datetimes.strftime("%H:%M")

    col_name = f"Number_of_Detections ({dc_tag})"
    data = list(zip(raw_dates, raw_times, activity_arr[col_name]))
    activity = pd.DataFrame(data, columns=["Date (UTC)", "Time (UTC)", col_name])
    activity_df = activity.pivot(index="Time (UTC)", columns="Date (UTC)", values=col_name)

    return activity_df


def construct_presence_grid(activity_arr, dc_tag):
    activity_datetimes = pd.to_datetime(activity_arr.index.values)
    raw_dates = activity_datetimes.strftime("%m/%d/%y")
    raw_times = activity_datetimes.strftime("%H:%M")

    col_name = f"Number_of_Detections ({dc_tag})"
    data = list(zip(raw_dates, raw_times, activity_arr[col_name]))
    presence = pd.DataFrame(data, columns=["Date (UTC)", "Time (UTC)", col_name])
    presence.loc[presence[col_name] > 0, col_name] = 1
    presence_df = presence.pivot(index="Time (UTC)", columns="Date (UTC)", values=col_name)

    return presence_df