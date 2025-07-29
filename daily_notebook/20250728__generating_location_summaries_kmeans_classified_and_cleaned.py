import sys
from pathlib import Path

sys.path.append(f"{Path(__file__).parent}/../src")
sys.path.append(f"{Path(__file__).parent}/../src/activity")
from core import SITE_NAMES

from cli import get_file_paths
import pipeline

if __name__ == "__main__":
    data_params = dict()
    data_params["year"] = '2022'
    data_params['detector_tag'] = 'bd2'
    data_params['bin_size'] = '30'
    data_params['recording_start'] = '00:00'
    data_params['recording_end'] = '16:00'
    data_params['assembly_type'] = 'kmeans'
    data_params['training_set'] = 'all_locations'


    for site_key in ['Central', 'Foliage']:
        for type_key in ['']:
            type_name = type_key
            if type_key=='':
                type_name = 'all'
            print(f'Generating location summary dataframe for {type_name}-group from {SITE_NAMES[site_key]}')
            data_params["site_name"] = SITE_NAMES[site_key]
            data_params["site_tag"] = site_key
            data_params["type_tag"] = type_key

            file_paths = get_file_paths(data_params)
            location_df_all = pipeline.prepare_location_sumary(data_params, file_paths) 