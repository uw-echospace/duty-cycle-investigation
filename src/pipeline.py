import pandas as pd
import data_handling as dh
import plotting

from pathlib import Path

def run(data_params, pipeline_params, file_paths):
    """
    Generate the main result of activity in a location that we use to view duty-cycling effects.
    """

    if not(pipeline_params["read_csv"]):
        activity_arr = dh.generate_activity_results(data_params, file_paths)
    else:
        activity_arr = pd.read_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_TYPE_SITE_summary"]}.csv', index_col=0)
    
    return activity_arr

def plot(activity_arr, data_params, pipeline_params, file_paths):
    """
    Plot various figures to visualize the run() function's results and compare duty-cycling schemes.
    """
    
    plotting.plot_activity_grid(dh.construct_activity_grid(activity_arr, data_params["cur_dc_tag"]), data_params, pipeline_params, file_paths)
    plotting.plot_presence_grid(dh.construct_presence_grid(activity_arr, data_params["cur_dc_tag"]), data_params, pipeline_params, file_paths)
    plotting.plot_dc_comparisons_per_night(activity_arr, data_params, pipeline_params, file_paths)
    plotting.plot_dc_activity_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths)
    plotting.plot_dc_presence_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths)
