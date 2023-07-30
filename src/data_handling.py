from pathlib import Path
import dask.dataframe as dd
import pandas as pd
import numpy as np

import subsampling as ss


def generate_activity_results(data_params):
    Path(f'{Path(__file__).resolve().parent}/../data/2022_bd2_summary/{data_params["site_tag"]}/duty_cycled/simulated_schemes/').mkdir(parents=True, exist_ok=True)
    assemble_initial_location_summary(data_params) ## Use to update any bd2__(location summary).csv files
    return ss.construct_activity_arr_from_dc_tags(data_params)

def assemble_initial_location_summary(data_params, save=True):
    location_df = dd.read_csv(f'{Path(__file__).resolve().parent}/../data/raw/{data_params["site_tag"]}/*.csv').compute()
    file_dts = pd.to_datetime(location_df['input_file'], format='%Y%m%d_%H%M%S', exact=False)
    anchor_start_times = file_dts + pd.to_timedelta(location_df['start_time'].values.astype('float64'), unit='S')
    anchor_end_times = file_dts + pd.to_timedelta(location_df['end_time'].values.astype('float64'), unit='S')
    location_df.insert(0, 'call_end_time', anchor_end_times)
    location_df.insert(0, 'call_start_time', anchor_start_times)
    location_df.insert(0, 'ref_time', anchor_start_times)
 
    if data_params["type_tag"] != "":
        location_df = location_df.loc[(location_df["high_freq"]).astype('float64') < data_params["freq_tags"][1]]
        location_df = location_df.loc[(location_df["low_freq"]).astype('float64') > data_params["freq_tags"][0]]

    if save:
        csv_filename = f'bd2__{data_params["type_tag"]}{data_params["site_tag"]}_2022.csv'
        location_df.to_csv(f'{Path(__file__).resolve().parent}/../data/2022_bd2_summary/{data_params["site_tag"]}/{csv_filename}')

    return location_df


def construct_activity_arr_from_location_summary(location_df, dc_tag):
    site_name = location_df['Site name'].values[0].split()[0]
    all_processed_filepaths = sorted(list(map(str, list(Path(f'{Path(__file__).resolve().parent}/../data/raw/{site_name}').iterdir()))))

    all_processed_datetimes = pd.to_datetime(all_processed_filepaths, format="%Y%m%d_%H%M%S", exact=False)
    datetimes_with_calls_detected = pd.to_datetime(location_df["input_file"].unique(), format="%Y%m%d_%H%M%S", exact=False)
    num_of_detections = location_df.groupby(["input_file"])["input_file"].count()
    num_of_detections.index = datetimes_with_calls_detected

    num_of_detections = add_rows_for_absence(num_of_detections, all_processed_datetimes, datetimes_with_calls_detected)
    activity_arr = pd.DataFrame(list(zip(num_of_detections.index, num_of_detections.values)), 
                                columns=["Date_and_Time_UTC", f"Number_of_Detections ({dc_tag})"])
    
    return activity_arr

def construct_activity_grid(activity_arr, dc_tag):
    activity_datetimes = pd.to_datetime(activity_arr.index.values)
    activity_dates = activity_datetimes.strftime("%m/%d/%y").unique()
    activity_times = activity_datetimes.strftime("%H:%M").unique()

    activity = activity_arr[f"Number_of_Detections ({dc_tag})"].values.reshape(len(activity_dates), len(activity_times)).T

    on = int(dc_tag.split('e')[0])
    total = int(dc_tag.split('e')[1])
    if on == total:
        activity_df = pd.DataFrame(activity, index=activity_times, columns=activity_dates)
    else:
        recover_ratio = total / on
        activity_df = pd.DataFrame(recover_ratio*activity, index=activity_times, columns=activity_dates)

    return activity_df

def construct_presence_grid(activity_arr, dc_tag):
    activity_datetimes = pd.to_datetime(activity_arr.index.values)
    activity_dates = activity_datetimes.strftime("%m/%d/%y").unique()
    activity_times = activity_datetimes.strftime("%H:%M").unique()

    presence_arr = activity_arr.copy()
    presence_arr[f"Number_of_Detections ({dc_tag})"].mask(presence_arr[f"Number_of_Detections ({dc_tag})"] > 0, 1, inplace=True)
    
    presence = presence_arr[f"Number_of_Detections ({dc_tag})"].values.reshape(len(activity_dates), len(activity_times)).T
    presence_df = pd.DataFrame(presence, index=activity_times, columns=activity_dates)

    return presence_df

def add_rows_for_absence(num_of_detections, all_processed_datetimes, datetimes_with_calls_detected):
    for file in all_processed_datetimes:
        if (not(file in datetimes_with_calls_detected)):
            num_of_detections[file] = 0
    num_of_detections = num_of_detections.sort_index()

    return num_of_detections