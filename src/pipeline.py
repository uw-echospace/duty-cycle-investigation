import pandas as pd
import data_handling as dh
import plotting

from pathlib import Path

def run(data_params, cfg):
    if not(cfg["read_csv"]):
        activity_arr = dh.generate_activity_results(data_params)
    else:
        csv_filename = f'dc__{data_params["type_tag"]}{data_params["site_tag"]}_summary.csv'
        rel_filepath = f'../data/2022_bd2_summary/{data_params["site_tag"]}/duty_cycled/{csv_filename}'
        activity_arr = pd.read_csv(f'{Path(__file__).resolve().parent}/{rel_filepath}', index_col=0)

    plotting.plot_activity_grid(dh.construct_activity_grid(activity_arr, data_params["cur_dc_tag"]), data_params, cfg)
    plotting.plot_presence_grid(dh.construct_presence_grid(activity_arr, data_params["cur_dc_tag"]), data_params, cfg)
    plotting.plot_dc_comparisons_per_night(activity_arr, data_params, cfg)
    plotting.plot_dc_activity_comparisons_per_scheme(activity_arr, data_params, cfg)
    plotting.plot_dc_presence_comparisons_per_scheme(activity_arr, data_params, cfg)
    
    return activity_arr
