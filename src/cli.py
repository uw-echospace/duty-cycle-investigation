import pipeline

import sys
sys.path.append('../src/activity')
import activity.subsampling as ss
from core import SITE_NAMES

from pathlib import Path

import argparse

def parse_args():
    """
    Defines the command line interface for the pipeline.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "type_of_calls",
        type=str,
        help="The type of calls we want results for ('lf_', 'hf_', or '')",
    )
    parser.add_argument(
        "site_tag",
        type=str,
        help="The tag of any of the 6 locations",
        default="output",
    )
    parser.add_argument(
        "specific_dc_tag",
        type=str,
        help="The tag of any of specific duty cycle",
        default="1800e1800",
    )
    parser.add_argument(
        '-cycle_lengths', 
        nargs="+", 
        type=int
    )
    parser.add_argument(
        '-percent_ons', 
        nargs="+", 
        type=float
    )
    parser.add_argument(
        "--show_PST",
        action="store_true",
        help="Use to place PST time axis in figures",
    )
    parser.add_argument(
        "--read_csv",
        action="store_true",
        help="Use if data has already been generated",
    )
    parser.add_argument(
        "--save_figures",
        action="store_true",
        help="Use to save all figures",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Use to show all figures",
    )

    return vars(parser.parse_args())

def get_file_paths(data_params):
    """
    Assemble a dictionary for file_paths important for the pipeline.
    """

    file_paths = dict()
    file_paths["raw_SITE_folder"] = f'{Path(__file__).resolve().parent}/../data/raw/{data_params["site_tag"]}'

    file_paths["SITE_folder"] = f'{Path(__file__).resolve().parent}/../data/2022_bd2_summary/{data_params["site_tag"]}'
    Path(f'{file_paths["SITE_folder"]}').mkdir(parents=True, exist_ok=True)
    file_paths['SITE_classes_folder'] = f'{Path(__file__).resolve().parent}/../data/classifications/{data_params["site_tag"]}'
    file_paths['SITE_classes_file'] = f'{file_paths["SITE_classes_folder"]}/2022_{data_params["site_tag"]}_call_classes.csv'
    file_paths["bd2_TYPE_SITE_YEAR"] = f'bd2__{data_params["type_tag"]}{data_params["site_tag"]}_2022'
    file_paths["duty_cycled_folder"] = f'{file_paths["SITE_folder"]}/duty_cycled'
    Path(f'{file_paths["duty_cycled_folder"]}').mkdir(parents=True, exist_ok=True)
    file_paths["dc_dets_TYPE_SITE_summary"] = f'dc_dets_{data_params["type_tag"]}{data_params["site_tag"]}_summary'
    file_paths["dc_bouts_TYPE_SITE_summary"] = f'dc_bouts_{data_params["type_tag"]}{data_params["site_tag"]}_summary'
    file_paths["dc_inds_TYPE_SITE_summary"] = f'dc_inds_{data_params["type_tag"]}{data_params["site_tag"]}_summary'
    file_paths["dc_callrate_TYPE_SITE_summary"] = f'dc_callrate_{data_params["type_tag"]}{data_params["site_tag"]}_summary'
    file_paths["cont_callrate_TYPE_SITE_summary"] = f'cont_callrate_{data_params["type_tag"]}{data_params["site_tag"]}_summary'
    file_paths["dc_btp_TYPE_SITE_summary"] = f'dc_btp_{data_params["type_tag"]}{data_params["site_tag"]}_summary'
    file_paths["cont_btp_TYPE_SITE_summary"] = f'cont_btp_{data_params["type_tag"]}{data_params["site_tag"]}_summary'
    file_paths["dc_actind_TYPE_SITE_summary"] = f'dc_actind_{data_params["type_tag"]}{data_params["site_tag"]}_summary'
    file_paths["cont_actind_TYPE_SITE_summary"] = f'cont_actind_{data_params["type_tag"]}{data_params["site_tag"]}_summary'

    file_paths["figures_SITE_folder"] = f'{Path(__file__).resolve().parent}/../figures/{data_params["site_tag"]}'
    Path(file_paths["figures_SITE_folder"]).mkdir(parents=True, exist_ok=True)
    file_paths["activity_det_comparisons_figname"] = f'activity_det_comparisons_per_dc_{data_params["type_tag"].upper()}{data_params["site_tag"]}'
    file_paths["dc_det_comparisons_figname"] = f'dc_det_comparisons_per_night_{data_params["type_tag"].upper()}{data_params["site_tag"]}'
    file_paths["activity_bout_comparisons_figname"] = f'activity_bout_comparisons_per_dc_{data_params["type_tag"].upper()}{data_params["site_tag"]}'
    file_paths["dc_bout_comparisons_figname"] = f'dc_bout_comparisons_per_night_{data_params["type_tag"].upper()}{data_params["site_tag"]}'
    file_paths["activity_ind_comparisons_figname"] = f'activity_ind_comparisons_per_dc_{data_params["type_tag"].upper()}{data_params["site_tag"]}'
    file_paths["dc_ind_comparisons_figname"] = f'dc_ind_comparisons_per_night_{data_params["type_tag"].upper()}{data_params["site_tag"]}'
    file_paths["dc_metric_comparisons_figname"] = f'metric_comparisons_per_night_{data_params["type_tag"].upper()}{data_params["site_tag"]}'
    file_paths["presence_comparisons_figname"] = f'presence_comparisons_per_dc_{data_params["type_tag"].upper()}{data_params["site_tag"]}'

    file_paths["activity_grid_folder"] = f'{file_paths["figures_SITE_folder"]}/activity_grids'
    Path(file_paths["activity_grid_folder"]).mkdir(parents=True, exist_ok=True)
    file_paths["activity_dets_grid_figname"] = f'{data_params["type_tag"].upper()}{data_params["site_tag"]}_activity_dets_grid'

    file_paths["presence_grid_folder"] = f'{file_paths["figures_SITE_folder"]}/presence_grids'
    Path(file_paths["presence_grid_folder"]).mkdir(parents=True, exist_ok=True)
    file_paths["presence_grid_figname"] = f'{data_params["type_tag"].upper()}{data_params["site_tag"]}_presence_grid'

    return file_paths

if __name__ == "__main__":
    """
    Put together important parahmeters and run the pipeline to generate results
    """
    
    args = parse_args()

    data_params = dict()
    data_params["site_name"] = SITE_NAMES[args['site_tag']]
    data_params["site_tag"] = args['site_tag']
    data_params["type_tag"] = args['type_of_calls']
    data_params["cycle_lengths"] = args['cycle_lengths']
    data_params["percent_ons"] = args['percent_ons']
    dc_tags = ss.get_list_of_dc_tags(data_params["cycle_lengths"], data_params["percent_ons"])
    data_params["dc_tags"] = dc_tags
    data_params["cur_dc_tag"] = args['specific_dc_tag']

    pipeline_params = dict()
    pipeline_params["read_csv"] = args['read_csv']
    pipeline_params["save_activity_grid"] = args['save_figures']
    pipeline_params["save_presence_grid"] = args['save_figures']
    pipeline_params["save_dc_night_comparisons"] = args['save_figures']
    pipeline_params["save_activity_dc_comparisons"] = args['save_figures']
    pipeline_params["save_presence_dc_comparisons"] = args['save_figures']
    pipeline_params["show_plots"] = args['show_plots']
    pipeline_params["show_PST"] = args['show_PST']

    file_paths = get_file_paths(data_params)

    _ = pipeline.run(data_params, pipeline_params, file_paths)