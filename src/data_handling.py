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

    start, end = "03:00:00", "13:00:00"
    num_of_detections = location_df.resample(resolution, on='ref_time')['ref_time'].count()
    incomplete_activity_arr = pd.DataFrame(num_of_detections.values, index=num_of_detections.index, columns=[col_name])

    dates = pd.to_datetime(num_of_detections.index.values).strftime("%Y-%m-%d").unique()
    activity_arr = pd.DataFrame()

    for date in dates:
        start_of_dets_for_date = num_of_detections[pd.DatetimeIndex(num_of_detections.index).strftime("%Y-%m-%d") == date].index[0]
        end_of_dets_for_date = num_of_detections[pd.DatetimeIndex(num_of_detections.index).strftime("%Y-%m-%d") == date].index[-1]

        start_of_recording = dt.datetime.strptime(f"{date} {start}", "%Y-%m-%d %H:%M:%S")
        end_of_recording = dt.datetime.strptime(f"{date} {end}", "%Y-%m-%d %H:%M:%S")

        pad_start, pad_end = pd.Series(), pd.Series()
        if (start_of_recording != start_of_dets_for_date):
            pad_start = pd.Series(pd.date_range(start_of_recording, start_of_dets_for_date, freq=resolution, inclusive='left'))
        if (end_of_recording != end_of_dets_for_date):
            pad_end = pd.Series(pd.date_range(end_of_dets_for_date, end_of_recording, freq=resolution, inclusive='right'))

        all_pad = pd.concat([pad_start, pad_end])
        pad_df = pd.DataFrame(0.0, index=all_pad, columns=[col_name])

        incomplete_activity_arr = pd.concat([incomplete_activity_arr, pad_df])
        incomplete_activity_arr = incomplete_activity_arr.sort_index()

        date_arr = incomplete_activity_arr[pd.DatetimeIndex(incomplete_activity_arr.index).strftime("%Y-%m-%d") == date]
        condition1 = np.logical_and(pd.DatetimeIndex(date_arr.index).hour >= 3, pd.DatetimeIndex(date_arr.index).hour < 13)
        condition2 = np.logical_and(pd.DatetimeIndex(date_arr.index).hour == 13, pd.DatetimeIndex(date_arr.index).minute == 0)
        full_date_arr = date_arr[np.logical_or(condition1, condition2)]
        activity_arr = pd.concat([activity_arr, full_date_arr])
    
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