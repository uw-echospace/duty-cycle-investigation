import pandas as pd

import sys
sys.path.append('../src')

import activity.activity_assembly as actvt
import activity.plot as plot

from pathlib import Path

def run_for_dets(data_params, pipeline_params, file_paths):
    """
    Generate the main result of activity in a location that we use to view duty-cycling effects.
    """

    if not(pipeline_params["read_csv"]):
        if (pipeline_params['assemble_location_summary']):
            actvt.assemble_initial_location_summary(data_params, file_paths) ## Use to update any bd2__(location summary).csv files
        activity_arr = actvt.generate_activity_dets_results(data_params, file_paths)
    else:
        activity_arr = pd.read_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_dets_TYPE_SITE_summary"]}.csv', index_col=0)
    
    return activity_arr

def run_for_bouts(data_params, pipeline_params, file_paths):
    """
    Generate the main result of activity in a location that we use to view duty-cycling effects.
    """

    if not(pipeline_params["read_csv"]):
        if (pipeline_params['assemble_location_summary']):
            actvt.assemble_initial_location_summary(data_params, file_paths) ## Use to update any bd2__(location summary).csv files
        activity_arr = actvt.generate_activity_bouts_results(data_params, file_paths)
    else:
        activity_arr = pd.read_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_bouts_TYPE_SITE_summary"]}.csv', index_col=0)
    
    return activity_arr

def run_for_inds(data_params, pipeline_params, file_paths):
    """
    Generate the main result of activity in a location that we use to view duty-cycling effects.
    """

    if not(pipeline_params["read_csv"]):
        if (pipeline_params['assemble_location_summary']):
            actvt.assemble_initial_location_summary(data_params, file_paths) ## Use to update any bd2__(location summary).csv files
        activity_arr = actvt.generate_activity_inds_results(data_params, file_paths)
    else:
        activity_arr = pd.read_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_inds_TYPE_SITE_summary"]}.csv', index_col=0)
    
    return activity_arr

def plot_dets(activity_arr, data_params, pipeline_params, file_paths):
    """
    Plot various figures to visualize the run() function's results and compare duty-cycling schemes.
    """
    
    plot.plot_activity_grid_for_dets(activity_arr, data_params, pipeline_params, file_paths)
    plot.plot_presence_grid(activity_arr, data_params, pipeline_params, file_paths)
    plot.plot_dc_det_activity_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths)
    plot.plot_dc_presence_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths)


def plot_bouts(activity_arr, data_params, pipeline_params, file_paths):
    """
    Plot various figures to visualize the run() function's results and compare duty-cycling schemes.
    """
    
    plot.plot_activity_grid_for_bouts(activity_arr, data_params, pipeline_params, file_paths)
    plot.plot_presence_grid(activity_arr, data_params, pipeline_params, file_paths)
    plot.plot_dc_bout_activity_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths)
    plot.plot_dc_presence_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths)


def plot_inds(activity_arr, data_params, pipeline_params, file_paths):
    """
    Plot various figures to visualize the run() function's results and compare duty-cycling schemes.
    """
    
    plot.plot_activity_grid_for_inds(activity_arr, data_params, pipeline_params, file_paths)
    plot.plot_presence_grid(activity_arr, data_params, pipeline_params, file_paths)
    plot.plot_dc_indices_activity_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths)
    plot.plot_dc_presence_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths)


def compare_det_and_bout_metrics(data_params, pipeline_params, file_paths):
    """
    Run the pipeline for both % of time by bouts and # of detection metrics and plot a comparison figure
    """
    
    activity_dets_arr = run_for_dets(data_params, pipeline_params, file_paths)
    activity_bouts_arr = run_for_bouts(data_params, pipeline_params, file_paths)
    plot.plot_numdets_n_percentbouts(activity_dets_arr, activity_bouts_arr, data_params, pipeline_params, file_paths)


def compare_det_and_ind_metrics(data_params, pipeline_params, file_paths):
    """
    Run the pipeline for both # of detection and activity index metrics and plot a comparison figure
    """
    
    activity_dets_arr = run_for_dets(data_params, pipeline_params, file_paths)
    activity_inds_arr = run_for_inds(data_params, pipeline_params, file_paths)
    plot.plot_numdets_n_activityinds(activity_dets_arr, activity_inds_arr, data_params, pipeline_params, file_paths)