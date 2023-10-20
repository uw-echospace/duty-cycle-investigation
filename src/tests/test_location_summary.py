from pathlib import Path
import pandas as pd

import sys

sys.path.append("../src")
import activity.activity_assembly as actvt
from core import SITE_NAMES

def test_columns_in_location_summary_are_as_expected(location_df):
    """
    Test if the location summary has all the expected columns?
    """
    expected_columns = ['freq_group', 'ref_time', 'call_start_time', 'call_end_time', 'start_time', 'end_time', 'low_freq', 'high_freq', 
        'event', 'class', 'class_prob', 'det_prob', 'individual', 'input_file', 'Site name', 'Recover Folder', 'SD Card']

    existing_columns = list(location_df.columns)
    
    for column in expected_columns:
        assert column in existing_columns


def test_if_calls_exist_in_location_summary(location_df):
    """
    Test if the location summary has at least 1 call
    """
    assert len(location_df) > 0

def test_location_summary_is_dataframe(location_df):
    """
    Test if the location summary is a pandas DataFrame.
    """
    assert isinstance(location_df, pd.DataFrame)

def run_tests_on_all_location_summary_methods():
    """
    Generates a location summary dataframe for each location and runs some basic tests to make sure output is valid
    """

    for site_key in SITE_NAMES.keys():
        type_key = ''
        data_params = dict()
        data_params["site_name"] = SITE_NAMES[site_key]
        data_params["site_tag"] = site_key
        data_params["type_tag"] = type_key

        file_paths = dict()
        file_paths["raw_SITE_folder"] = f'{Path(__file__).resolve().parent}/../../data/raw/{data_params["site_tag"]}'
        if Path(file_paths['raw_SITE_folder']).exists():
            location_df = actvt.assemble_initial_location_summary(data_params, file_paths, save=False)
            test_location_summary_is_dataframe(location_df)
            test_columns_in_location_summary_are_as_expected(location_df)
            test_if_calls_exist_in_location_summary(location_df)


