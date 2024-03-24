from pathlib import Path
import pandas as pd

import sys

sys.path.append("../src")
import pipeline
from core import SITE_NAMES
from cli import get_file_paths

def test_columns_in_location_summary_are_as_expected(location_df):
    """
    Test if the location summary has all the expected columns?
    """
    expected_columns = ['freq_group', 'ref_time', 'call_start_time', 'call_end_time', 'start_time', 'end_time', 'low_freq', 'high_freq', 
        'event', 'class', 'class_prob', 'det_prob', 'individual', 'input_file', 'Site name', 'Recover Folder', 'SD Card']

    existing_columns = list(location_df.columns)
    
    for column in expected_columns:
        assert column in existing_columns


def test_if_good_amount_of_calls_exist_in_location_summary(location_df, data_params):
    """
    Test if the location summary has at least 1 call
    """
    date_of_first_call = location_df['call_start_time'].iloc[0].date()
    date_of_last_call = location_df['call_start_time'].iloc[-1].date()

    mock = pd.date_range(date_of_first_call, date_of_last_call, freq='1S')
    between_time_mock = mock[mock.indexer_between_time(pd.to_datetime(data_params['recording_start'], format='%H:%M').time(), 
                               pd.to_datetime(data_params['recording_end'], format='%H:%M').time())]
    
    assert (400000 < len(location_df))&(len(location_df) < len(between_time_mock))

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
        data_params['cur_dc_tag'] = '30of30'
        data_params["site_name"] = SITE_NAMES[site_key]
        data_params['recording_start'] = '0:00'
        data_params['recording_end'] = '16:00'

        pipeline_params = dict()
        pipeline_params['use_threshold_to_group'] = False
        pipeline_params['use_kmeans_to_group'] = True

        file_paths = get_file_paths(data_params)

        if Path(file_paths['raw_SITE_folder']).exists():
            print(site_key)
            location_df = pipeline.prepare_location_sumary(data_params, pipeline_params, file_paths)
            test_location_summary_is_dataframe(location_df)
            test_columns_in_location_summary_are_as_expected(location_df)
            test_if_good_amount_of_calls_exist_in_location_summary(location_df, data_params)