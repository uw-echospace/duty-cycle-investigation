import pandas as pd
import data_handling as dh
import plotting

from pathlib import Path

def run_for_dets(data_params, pipeline_params, file_paths):
    """
    Generate the main result of activity in a location that we use to view duty-cycling effects.
    """

    if not(pipeline_params["read_csv"]):
        if (pipeline_params['assemble_location_summary']):
            dh.assemble_initial_location_summary(data_params, file_paths) ## Use to update any bd2__(location summary).csv files
        activity_arr = dh.generate_activity_dets_results(data_params, file_paths)
    else:
        activity_arr = pd.read_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_dets_TYPE_SITE_summary"]}.csv', index_col=0)
    
    return activity_arr

def run_for_bouts(data_params, pipeline_params, file_paths):
    """
    Generate the main result of activity in a location that we use to view duty-cycling effects.
    """

    if not(pipeline_params["read_csv"]):
        if (pipeline_params['assemble_location_summary']):
            dh.assemble_initial_location_summary(data_params, file_paths) ## Use to update any bd2__(location summary).csv files
        activity_arr = dh.generate_activity_bouts_results(data_params, file_paths)
    else:
        activity_arr = pd.read_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_bouts_TYPE_SITE_summary"]}.csv', index_col=0)
    
    return activity_arr

def run_for_inds(data_params, pipeline_params, file_paths):
    """
    Generate the main result of activity in a location that we use to view duty-cycling effects.
    """

    if not(pipeline_params["read_csv"]):
        if (pipeline_params['assemble_location_summary']):
            dh.assemble_initial_location_summary(data_params, file_paths) ## Use to update any bd2__(location summary).csv files
        activity_arr = dh.generate_activity_inds_results(data_params, file_paths)
    else:
        activity_arr = pd.read_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_inds_TYPE_SITE_summary"]}.csv', index_col=0)
    
    return activity_arr

def plot_dets(activity_arr, data_params, pipeline_params, file_paths):
    """
    Plot various figures to visualize the run() function's results and compare duty-cycling schemes.
    """
    
    plotting.plot_activity_grid_for_dets(dh.construct_activity_grid_for_number_of_dets(activity_arr, data_params["cur_dc_tag"]), data_params, pipeline_params, file_paths)
    plotting.plot_presence_grid(dh.construct_presence_grid(activity_arr, data_params["cur_dc_tag"]), data_params, pipeline_params, file_paths)
    plotting.plot_dc_dets_comparisons_per_night(activity_arr, data_params, pipeline_params, file_paths)
    plotting.plot_dc_det_activity_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths)
    plotting.plot_dc_presence_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths)


def plot_bouts(activity_arr, data_params, pipeline_params, file_paths):
    """
    Plot various figures to visualize the run() function's results and compare duty-cycling schemes.
    """
    
    plotting.plot_activity_grid_for_bouts(dh.construct_activity_grid_for_bouts(activity_arr, data_params['cur_dc_tag']), data_params, pipeline_params, file_paths)
    plotting.plot_dc_bouts_comparisons_per_night(activity_arr, data_params, pipeline_params, file_paths)
    plotting.plot_dc_bout_activity_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths)


def compare_bout_and_det_metrics(data_params, pipeline_params, file_paths):
    activity_dets_arr = run_for_dets(data_params, pipeline_params, file_paths)
    activity_bouts_arr = run_for_bouts(data_params, pipeline_params, file_paths)
    plotting.plot_numdets_n_percentbouts(activity_dets_arr, activity_bouts_arr, data_params, pipeline_params, file_paths)


def compare_bout_and_det_metrics(data_params, pipeline_params, file_paths):
    activity_dets_arr = run_for_dets(data_params, pipeline_params, file_paths)
    activity_inds_arr = run_for_inds(data_params, pipeline_params, file_paths)
    plotting.plot_numdets_n_activityinds(activity_dets_arr, activity_inds_arr, data_params, pipeline_params, file_paths)