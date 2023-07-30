import pipeline
import subsampling as ss
from cfg import get_config

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

if __name__ == "__main__":
    args = parse_args()

    cfg = get_config()
    data_params = dict()
    data_params["site_name"] = cfg['site_names'][args['site_tag']]
    data_params["site_tag"] = args['site_tag']
    data_params["type_tag"] = args['type_of_calls']
    data_params["freq_tags"] = cfg['freq_groups'][args['type_of_calls']]
    data_params["cycle_lengths"] = args['cycle_lengths']
    data_params["percent_ons"] = args['percent_ons']
    dc_tags = ss.get_list_of_dc_tags(data_params["cycle_lengths"], data_params["percent_ons"])
    data_params["dc_tags"] = dc_tags
    data_params["cur_dc_tag"] = args['specific_dc_tag']

    cfg["read_csv"] = args['read_csv']
    cfg["save_activity_grid"] = args['save_figures']
    cfg["save_presence_grid"] = args['save_figures']
    cfg["save_dc_night_comparisons"] = args['save_figures']
    cfg["save_activity_dc_comparisons"] = args['save_figures']
    cfg["save_presence_dc_comparisons"] = args['save_figures']
    cfg["show_plots"] = args['show_plots']

    _ = pipeline.run(data_params, cfg)