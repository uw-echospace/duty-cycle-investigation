import numpy as np
import pandas as pd
import argparse
import re
import math
import matplotlib.pyplot as plt

import fsspec
from sklearn.cluster import KMeans
import scipy

import soundfile as sf
from pathlib import Path

import sys
sys.path.append(f"{Path(__file__).parents[1]}/bout")
sys.path.append(f"{Path(__file__).parents[1]}")
sys.path.append(f"{Path(__file__).parents[0]}")
print(sys.path)

from core import SITE_NAMES
import bout.assembly as bout
import compute_features, call_extraction

from cli import get_file_paths


PADDED_CALL_LENGTH = 0.06
LABEL_FOR_GROUPS = {
                    0: 'LF', 
                    1: 'HF'
                    }


def get_section_of_call_in_file(detection, audio_file):
    fs = audio_file.samplerate
    num_frames = audio_file.frames
    file_length = num_frames/fs

    call_dur = (detection['end_time'] - detection['start_time'])
    pad = min(min(detection['start_time'] - call_dur, file_length - detection['end_time']), 0.006) / 3
    start = detection['start_time'] - call_dur - (3*pad)
    duration = ((2 * call_dur) + (4*pad))

    try:
        audio_file.seek(int(fs*start))
        audio_seg = audio_file.read(int(fs*duration))
        length_of_section = call_dur + (2*pad)
    except sf.LibsndfileError as e:
        print(f'Start time : {start} invalid, detection starts at {detection["start_time"]} in segment:{audio_file.name}')
        audio_seg = None
        length_of_section = 0

    return audio_seg, length_of_section, pad


def gather_features_of_interest(dets, kmean_welch, audio_file):
    fs = audio_file.samplerate
    features_of_interest = dict()
    features_of_interest['call_signals'] = []
    features_of_interest['welch_signals'] = []
    features_of_interest['snrs'] = []
    features_of_interest['peak_freqs_welch'] = []
    features_of_interest['peak_freqs_spec'] = []
    features_of_interest['peak_freq_times_spec'] = []
    features_of_interest['peak_freqs'] = []
    features_of_interest['classes'] = []
    nyquist = fs//2
    for index, row in dets.iterrows():
        call_dur = (row['end_time'] - row['start_time'])
        audio_seg, length_of_section, pad = get_section_of_call_in_file(row, audio_file)
        seg_start = row['start_time'] - call_dur - (3*pad)
        
        freq_pad = 2000
        low_freq_cutoff = row['low_freq']-freq_pad
        high_freq_cutoff = min(nyquist-1, row['high_freq']+freq_pad)
        band_limited_audio_seg = call_extraction.bandpass_audio_signal(audio_seg, fs, low_freq_cutoff, high_freq_cutoff)

        signal_for_peaks = band_limited_audio_seg.copy()
        signal_for_peaks[:int(fs*(length_of_section+pad))] = 0
        signal_for_peaks[-int(fs*pad):] = 0
        mpl_specgram_window = plt.mlab.window_hanning(np.ones(32))
        f, t, Sxx = scipy.signal.spectrogram(signal_for_peaks, fs, detrend=False,
                                    nfft=32, 
                                    window=mpl_specgram_window)
        with np.errstate(divide='ignore'):
            plt_Sxx = 10*np.log10(Sxx)
            max_ind = np.where(plt_Sxx==np.max(plt_Sxx))
            peak_freq = f[max_ind[0]]
            peak_freq_time = t[max_ind[1]]
            if math.isinf(np.max(plt_Sxx)):
                features_of_interest['peak_freqs_spec'].append((row['low_freq']+row['high_freq'])/2)
                features_of_interest['peak_freq_times_spec'].append((row['start_time']+row['end_time'])/2)
            else:
                features_of_interest['peak_freqs_spec'].append(peak_freq[0])
                features_of_interest['peak_freq_times_spec'].append(seg_start+peak_freq_time[0])

            signal = band_limited_audio_seg.copy()
            signal[:int(fs*(length_of_section))] = 0
            noise = band_limited_audio_seg - signal
            snr_call_signal = signal[-int(fs*length_of_section):]
            snr_noise_signal = noise[:int(fs*length_of_section)]
            features_of_interest['call_signals'].append(snr_call_signal)

            snr = call_extraction.get_snr_from_band_limited_signal(snr_call_signal, snr_noise_signal)
            features_of_interest['snrs'].append(snr)

            welch_info = dict()
            welch_info['num_points'] = 100
            max_visible_frequency = 96000
            welch_info['max_freq_visible'] = max_visible_frequency
            welch_signal = compute_features.compute_welch_psd_of_call(snr_call_signal, fs, welch_info)
            features_of_interest['welch_signals'].append(welch_signal)

            peaks = np.argmax(welch_signal)
            features_of_interest['peak_freqs_welch'].append(max_visible_frequency*(peaks/len(welch_signal)))
            
            welch_signal = (welch_signal).reshape(1, len(welch_signal))
            features_of_interest['classes'].append(kmean_welch.predict(welch_signal)[0])

    features_of_interest['call_signals'] = np.array(features_of_interest['call_signals'], dtype='object')

    return features_of_interest


def open_and_get_call_info(audio_file, dets):
    welch_key = 'all_locations'
    output_dir = Path(f'{Path(__file__).parents[2]}/data/generated_welch/{welch_key}')
    output_file_type = 'top1_inbouts_welch_signals'
    welch_data = pd.read_csv(output_dir / f'2022_{welch_key}_{output_file_type}.csv', index_col=0, low_memory=False)
    k = 2
    kmean_welch = KMeans(n_clusters=k, n_init=10, random_state=1).fit(welch_data.values)

    features_of_interest = gather_features_of_interest(dets, kmean_welch, audio_file)

    dets.reset_index(drop=True, inplace=True)

    call_infos = pd.DataFrame()
    call_infos['index_in_file'] = dets['index_in_file']
    call_infos['index_in_summary'] = dets['index_in_summary']
    call_infos['file_name'] = pd.DatetimeIndex(pd.to_datetime(dets['input_file'], format='%Y%m%d_%H%M%S', exact=False)).strftime('%Y%m%d_%H%M%S.WAV')
    call_infos['sampling_rate'] = len(dets) * [audio_file.samplerate]
    call_infos.insert(0, 'SNR', features_of_interest['snrs'])
    call_infos.insert(0, 'peak_frequency_WELCH', features_of_interest['peak_freqs_welch'])
    call_infos.insert(0, 'peak_frequency_SPECTROGRAM', features_of_interest['peak_freqs_spec'])
    call_infos.insert(0, 'peak_frequency_time_SPECTROGRAM', features_of_interest['peak_freq_times_spec'])
    call_infos.insert(0, 'KMEANS_CLASSES', pd.Series(features_of_interest['classes']).map(LABEL_FOR_GROUPS))

    return features_of_interest['call_signals'], call_infos


def classify_calls_from_file(bd2_predictions, data_params):
    file_path = data_params['filesys'].open(path=Path(data_params['audio_file']))
    audio_file = sf.SoundFile(file_path)
    call_signals, dets = open_and_get_call_info(audio_file, bd2_predictions.copy())

    median_peak_HF_freq = dets[dets['KMEANS_CLASSES']=='HF']['peak_frequency_WELCH'].median()
    median_peak_LF_freq = dets[dets['KMEANS_CLASSES']=='LF']['peak_frequency_WELCH'].median()
    print(f'Median LF frequency in File: {median_peak_LF_freq}')
    print(f'Median HF frequency in File: {median_peak_HF_freq}')
    lf_inds = (dets['peak_frequency_WELCH']<median_peak_LF_freq+7000)&(dets['peak_frequency_WELCH']>median_peak_LF_freq-7000)
    hf_inds = (dets['peak_frequency_WELCH']>median_peak_HF_freq-7000)

    lf_dets = dets[lf_inds&(dets['KMEANS_CLASSES']=='LF')]
    hf_dets = dets[hf_inds&(dets['KMEANS_CLASSES']=='HF')]

    fixed_dets = pd.concat([hf_dets, lf_dets]).sort_index()

    return fixed_dets, dets


def open_call_signals_using_summary(location_sum_df, data_params, corrected_classifications, classifications):
    bd2_predictions = location_sum_df.loc[location_sum_df['input_file']==str(data_params['input_file'])].copy()
    
    is_valid_params = len(bd2_predictions)>0 

    if is_valid_params:
        corrected_classifications_in_file, classifications_in_file = classify_calls_from_file(bd2_predictions, data_params)
        corrected_classifications = pd.concat([corrected_classifications, corrected_classifications_in_file])
        classifications = pd.concat([classifications, classifications_in_file])
    
    print(f'There are now {len(corrected_classifications)} rows in call catalogue')

    return corrected_classifications, classifications


def relabel_drivenames_to_mirrors(filepaths):
    drivename = re.compile(r'ubna_data_0[0-9]/')
    for i, fp in enumerate(filepaths):
        if bool(drivename.search(fp)):
            d_name = drivename.search(fp).group()
            replace_d_name = f'{d_name[:-1]}_mir/'
            filepaths[i] = filepaths[i].replace(d_name, replace_d_name)

    return filepaths


def get_params_relevant_to_data_at_location(cfg):
    data_params = dict()
    data_params["type_tag"] = ''
    data_params["cur_dc_tag"] = "30of30"
    data_params["site_tag"] = cfg['site']
    data_params['site_name'] = SITE_NAMES[cfg['site']]
    data_params['assembly_type'] = 'thresh'
    data_params['detector_tag'] = cfg['detector']
    data_params['training_set'] = 'all_locations'
    print(f"Searching for files from {data_params['site_name']}")

    file_paths = get_file_paths(data_params)
    location_sum_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["detector_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)
    location_sum_df.reset_index(inplace=True)
    location_sum_df.rename({'index':'index_in_summary'}, axis='columns', inplace=True)

    data_params['good_audio_files'] = location_sum_df['input_file'].copy().unique()
    print(f"Will be looking at {len(data_params['good_audio_files'])} files from {data_params['site_name']}")

    return location_sum_df, data_params


def sample_calls_and_generate_call_signal_bucket_for_location(cfg):
    corrected_classifications = pd.DataFrame()
    classifications = pd.DataFrame()
    location_sum_df, data_params = get_params_relevant_to_data_at_location(cfg)
    file_raw_title = f'2022_{cfg["detector"]}{data_params["site_tag"]}_call_classes_raw'
    file_corrected_title = f'2022_{cfg["detector"]}{data_params["site_tag"]}_call_classes'
    filesys = fsspec.filesystem('s3', anon=True, client_kwargs={'endpoint_url': 'https://sdsc.osn.xsede.org'})
    data_params['filesys'] = filesys
   
    for input_file in data_params['good_audio_files']:
        file_path = '/'.join(Path(input_file).parts[2:])
        cleaned_path = re.sub(r"(ubna_data_\d+)_mir", r"\1", file_path)
        osn_file_path = Path(f'bio230143-bucket01/{cleaned_path}')
        data_params['input_file'] = Path(input_file)
        data_params['audio_file'] = Path(osn_file_path)
        print(f'Looking at {osn_file_path}')
        corrected_classifications, classifications = open_call_signals_using_summary(location_sum_df, data_params, corrected_classifications, classifications)

    print('Resetting index for call catalogue')
    corrected_classifications.reset_index(inplace=True)
    print(f'Saving call catalogue to {file_corrected_title}.csv')
    corrected_classifications.to_csv(f'{Path(__file__).parents[2]}/data/classifications/{data_params["site_tag"]}/{file_corrected_title}.csv')

    classifications.reset_index(inplace=True)
    classifications.to_csv(f'{Path(__file__).parents[2]}/data/classifications/{data_params["site_tag"]}/{file_raw_title}.csv')

    return corrected_classifications, classifications


def parse_args():
    """
    Defines the command line interface for the pipeline.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site",
        type=str,
        help="the site key",
    )
    parser.add_argument(
        "--recording_start",
        type=str,
        help="the start of recording period",
    )
    parser.add_argument(
        "--recording_end",
        type=str,
        help="the end of recording period",
    )
    parser.add_argument(
        "--detector",
        type=str,
        help="the end of recording period",
    )
    return vars(parser.parse_args())

if __name__ == "__main__":
    args = parse_args()

    cfg= dict()
    cfg['site'] = args['site']
    cfg['recording_start'] = args['recording_start']
    cfg['recording_end'] = args['recording_end']
    cfg['detector'] = args['detector']

    sample_calls_and_generate_call_signal_bucket_for_location(cfg)