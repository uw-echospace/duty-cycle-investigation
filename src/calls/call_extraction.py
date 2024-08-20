import numpy as np
import pandas as pd
import scipy
import dask.dataframe as dd
import argparse
import re
import math

import soundfile as sf
from pathlib import Path

import sys
sys.path.append(f"{Path(__file__).parents[1]}/bout")
sys.path.append(f"{Path(__file__).parents[1]}")
print(sys.path)

from core import SITE_NAMES, FREQ_GROUPS
import bout.assembly as bout
import activity.activity_assembly as actvt
import activity.subsampling as ss
import calls.compute_features as compute_features


from cli import get_file_paths


def get_snr_from_band_limited_signal(snr_call_signal, snr_noise_signal): 

    signal_power_rms = np.sqrt(np.square(snr_call_signal).mean())
    noise_power_rms = np.sqrt(np.square(snr_noise_signal).mean())
    snr = abs(20 * np.log10(signal_power_rms / noise_power_rms))

    return snr


def bandpass_audio_signal(audio_seg, fs, low_freq_cutoff, high_freq_cutoff):
    nyq = fs // 2
    low_cutoff = (low_freq_cutoff) / nyq
    high_cutoff =  (high_freq_cutoff) / nyq
    b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band', analog=False)
    band_limited_audio_seg = scipy.signal.filtfilt(b, a, audio_seg)

    return band_limited_audio_seg


def relabel_drivenames_to_mirrors(filepaths):
    drivename = re.compile(r'ubna_data_0[0-9]/')
    for i, fp in enumerate(filepaths):
        if bool(drivename.search(fp)):
            d_name = drivename.search(fp).group()
            replace_d_name = f'{d_name[:-1]}_mir/'
            filepaths[i] = filepaths[i].replace(d_name, replace_d_name)

    return filepaths


def collect_call_snrs_from_detections_in_audio_file(audio_file, detections):
    fs = audio_file.samplerate
    nyquist = fs//2
    call_snrs = []
    for i, call in detections.iterrows():
        call_dur = (call['end_time'] - call['start_time'])
        pad = 0.002
        start = call['start_time'] - call_dur - (3*pad)
        duration = (2 * call_dur) + (4*pad)
        end = call['end_time']
        if start >=0 and end <= 1795:
            audio_file.seek(int(fs*start))
            audio_seg = audio_file.read(int(fs*duration))
            
            low_freq_cutoff = call['low_freq']-2000
            high_freq_cutoff = min(nyquist-1, call['high_freq']+2000)
            band_limited_audio_seg = bandpass_audio_signal(audio_seg, fs, low_freq_cutoff, high_freq_cutoff)

            sec_length = int(fs*(pad+call_dur+pad))
            signal = band_limited_audio_seg.copy()
            signal[:sec_length] = 0

            noise = band_limited_audio_seg - signal
            snr_call_signal = signal[-sec_length:]
            snr_noise_signal = noise[:sec_length]
            snr = get_snr_from_band_limited_signal(snr_call_signal, snr_noise_signal)
            call_snrs += [snr]
        else:
            call_snrs += [np.NaN]

    return call_snrs


def get_bout_metrics_from_single_bd2_output(bd2_output, data_params):
    dc_tag = data_params['cur_dc_tag']
    cycle_length = int(dc_tag.split('of')[1])
    time_on = int(dc_tag.split('of')[0])

    dc_applied_df = ss.simulate_dutycycle_on_detections_with_bins(bd2_output, dc_tag, cycle_length)
    batdetect2_preds_with_bouttags = bout.classify_bouts_in_detector_preds_for_freqgroups(dc_applied_df, data_params['bout_params'])
    bout_metrics = bout.construct_bout_metrics_from_location_df_for_freqgroups(batdetect2_preds_with_bouttags)

    return bout_metrics


def collect_call_signals_from_detections(audio_file, detections, bucket, data_params):
    fs = audio_file.samplerate
    nyq = fs//2
    sampled_calls_from_bout = pd.DataFrame()

    for i, call in detections.iterrows():
        call_dur = (call['end_time'] - call['start_time'])
        start_pad = min(call['start_time'], data_params['padding'])
        end_pad = min(1795 - call['end_time'], data_params['padding'])
        start = call['start_time'] - start_pad
        duration = call_dur + 2*(start_pad+end_pad)
        end = call['end_time'] + end_pad
        if start >= 0 and end <= 1795:
            audio_file.seek(int(fs*start))
            audio_seg = audio_file.read(int(fs*duration))
            low_freq_cutoff = call['low_freq']-2000
            high_freq_cutoff = min(nyq-1, call['high_freq']+2000)
            if data_params['bandpass']:
                band_limited_audio_seg = bandpass_audio_signal(audio_seg, fs, low_freq_cutoff, high_freq_cutoff)
                cleaned_call_signal = band_limited_audio_seg.copy()
            else:
                cleaned_call_signal = audio_seg.copy()
            bucket.append(cleaned_call_signal)
            sampled_call = pd.DataFrame(columns=call.index)
            sampled_call.loc[len(sampled_call)] = call
            sampled_call.index = [i]
            sampled_calls_from_bout = pd.concat([sampled_calls_from_bout, sampled_call], axis=0)

    return bucket, sampled_calls_from_bout


def select_top_percentage_from_detections(detections, percentage):
    print(f"SNRs in this section: {detections['SNR'].values}")
    # SNR must be at least 1 to be chosen
    top_SNR =  max(1, (1-percentage)*detections['SNR'].max()) 
    print(f"Highest SNR in section: {detections['SNR'].max()}")
    print(f"SNR threshold for section: {top_SNR}")
    if percentage==1:
        selected_set = detections.copy()
    else:
        selected_set = detections.loc[detections['SNR']>=top_SNR].copy()

    return selected_set


def sample_calls_using_bouts(detector_preds, bucket_for_location, data_params):
    bout_metrics = get_bout_metrics_from_single_bd2_output(detector_preds, data_params)
    bout_metrics.reset_index(drop=True, inplace=True)
        
    file_path = Path(data_params['audio_file'])
    audio_file = sf.SoundFile(file_path)
    fs = audio_file.samplerate
    print(f'{len(detector_preds)} calls in this file: {file_path.name}')

    calls_sampled_from_file = pd.DataFrame()
    for bout_index, row in bout_metrics.iterrows():
        group = row['freq_group']
        freq_group = detector_preds.loc[detector_preds['freq_group']==group]
        bat_bout = freq_group.loc[(freq_group['start_time']>=row['start_time'])&(freq_group['end_time']<=row['end_time'])].copy()
        call_snrs = collect_call_snrs_from_detections_in_audio_file(audio_file, bat_bout)
        bat_bout['SNR'] = call_snrs
        print(f"{len(bat_bout)} calls in bout {bout_index}")
        selected_set = select_top_percentage_from_detections(bat_bout, data_params['percent_threshold_for_snr'])
        print(f"{len(selected_set)} high SNR calls in bout {bout_index}")
        if len(selected_set) > 0:
            bucket_for_location, sampled_calls_from_bout = collect_call_signals_from_detections(audio_file, selected_set, 
                                                                                                bucket_for_location, data_params)

            bat_bout_condensed = sampled_calls_from_bout
            bat_bout_condensed['bout_index'] = [bout_index]*len(sampled_calls_from_bout)
            bat_bout_condensed['file_name'] = str(Path(sampled_calls_from_bout['input_file'].values[0]).name)
            bat_bout_condensed['sampling_rate'] = [fs]*len(sampled_calls_from_bout)
            calls_sampled_from_file = pd.concat([calls_sampled_from_file, bat_bout_condensed])
            print(f"{len(bat_bout_condensed)} high SNR calls added to call catalogue")

    return bucket_for_location, calls_sampled_from_file


def sample_calls_from_file(detector_preds, bucket_for_location, data_params):
    file_path = Path(data_params['audio_file'])
    audio_file = sf.SoundFile(file_path)
    fs = audio_file.samplerate
    print(f'{len(detector_preds)} calls in this file: {file_path.name}')

    calls_sampled_from_file = pd.DataFrame()
    for group in detector_preds['freq_group'].unique():
        freq_group = detector_preds.loc[detector_preds['freq_group']==group].copy()
        call_snrs = collect_call_snrs_from_detections_in_audio_file(audio_file, freq_group)
        freq_group['SNR'] = call_snrs
        print(f"{len(freq_group)} {group} calls in file: {file_path.name}")
        selected_set = select_top_percentage_from_detections(freq_group, data_params['percent_threshold_for_snr'])
        print(f"{len(selected_set)} high SNR {group} calls in file: {file_path.name}")
        if len(selected_set) > 0:
            bucket_for_location, sampled_calls_from_bout = collect_call_signals_from_detections(audio_file, selected_set, bucket_for_location)

            detections_condensed = sampled_calls_from_bout
            detections_condensed['file_name'] = str(Path(sampled_calls_from_bout['input_file'].values[0]).name)
            detections_condensed['sampling_rate'] = [fs]*len(sampled_calls_from_bout)
            calls_sampled_from_file = pd.concat([calls_sampled_from_file, detections_condensed])
            print(f"{len(detections_condensed)} high SNR calls added to call catalogue")

    return bucket_for_location, calls_sampled_from_file


def collect_call_signals_from_location_sum(location_sum_df, data_params, bucket_for_location, calls_sampled_from_location):
    location_sum_df['input_file'] = relabel_drivenames_to_mirrors(location_sum_df['input_file'].copy())
    detector_preds = location_sum_df.loc[location_sum_df['input_file']==str(data_params['audio_file'])].copy()
    groups_in_preds = detector_preds['freq_group'].unique()
    valid_group_in_preds = np.logical_or(np.logical_or('LF1' in groups_in_preds, 'HF1' in groups_in_preds), 'HF2' in groups_in_preds)
    print(f"Groups found in this file: {detector_preds['freq_group'].unique()}, valid? {valid_group_in_preds}")
    is_valid_params = len(detector_preds)>0 and valid_group_in_preds

    if is_valid_params:
        if data_params['use_bouts']:
            bucket_for_location, calls_sampled_from_file = sample_calls_using_bouts(detector_preds, bucket_for_location, data_params)
        elif data_params['use_file']:
            bucket_for_location, calls_sampled_from_file = sample_calls_from_file(detector_preds, bucket_for_location, data_params)
        else:
            calls_sampled_from_file = pd.DataFrame()

        calls_sampled_from_location = pd.concat([calls_sampled_from_location, calls_sampled_from_file])
    
    print(f'There are now {len(bucket_for_location)} calls in bucket')
    print(f'There are now {len(calls_sampled_from_location)} rows in call catalogue')

    return bucket_for_location, calls_sampled_from_location


def filter_df_with_location(ubna_data_df, site_name, start_time, end_time):
    site_name_cond = ubna_data_df["site_name"] == site_name

    file_year_cond = ubna_data_df.index.year == 2022
    minute_cond = np.logical_or((ubna_data_df.index).minute == 30, (ubna_data_df.index).minute == 0)
    datetime_cond = np.logical_and((ubna_data_df.index).second == 0, minute_cond)
    file_error_cond = np.logical_and((ubna_data_df["file_duration"]!='File has no comment due to error!'), (ubna_data_df["file_duration"]!='File has no Audiomoth-related comment'))
    all_errors_cond = np.logical_and((ubna_data_df["file_duration"]!='Is empty!'), file_error_cond)

    filtered_location_df = ubna_data_df.loc[site_name_cond&datetime_cond&file_year_cond&all_errors_cond].sort_index()
    filtered_location_nightly_df = filtered_location_df.between_time(start_time, end_time, inclusive="left")

    return filtered_location_nightly_df


def get_params_relevant_to_data_at_location(cfg):
    data_params = dict()
    data_params["type_tag"] = ''
    data_params["cur_dc_tag"] = "30of30"
    data_params["site_tag"] = cfg['site']
    data_params['site_name'] = SITE_NAMES[cfg['site']]
    data_params['percent_threshold_for_snr'] = cfg['percent_threshold_for_snr']
    data_params['padding'] = cfg['padding']
    data_params['bandpass'] = cfg['bandpass']
    data_params['use_bouts'] = cfg['use_bouts']
    data_params['use_file'] = cfg['use_file']
    data_params['use_thresholds'] = True
    data_params['use_kmeans'] = False
    data_params['detector_tag'] = cfg['detector']
    print(f"Searching for files from {data_params['site_name']}")

    file_paths = get_file_paths(data_params)
    location_sum_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["detector_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)
    location_sum_df.reset_index(inplace=True)
    location_sum_df.rename({'index':'index_in_file'}, axis='columns', inplace=True)
    site_filepaths = relabel_drivenames_to_mirrors(location_sum_df['input_file'].copy().unique())
    bout_params = bout.get_bout_params_from_location(location_sum_df, data_params)

    data_params['good_audio_files'] = site_filepaths
    data_params['bout_params'] = bout_params
    print(f"Will be looking at {len(data_params['good_audio_files'])} files from {data_params['site_name']}")

    return location_sum_df, data_params


def sample_calls_and_generate_call_signal_bucket_for_location(cfg):
    bucket_for_location = []
    calls_sampled_from_location = pd.DataFrame()
    location_sum_df, data_params = get_params_relevant_to_data_at_location(cfg)
    year_detector_site_thresh = f'2022_{data_params["detector_tag"]}{data_params["site_tag"]}_top{int(100*data_params["percent_threshold_for_snr"])}'
    padding = f'{int(1000*data_params["padding"])}ms'
    if data_params['bandpass']:
        padding_bandpass = f'{padding}_bandpass'
    else:
        padding_bandpass = f'{padding}_nobandpass'

    if data_params['use_bouts']:
        file_title = f'{year_detector_site_thresh}_inbouts'
    if data_params['use_file']:
        file_title = f'{year_detector_site_thresh}_infile'

    call_signals_file_title = f'{file_title}_{padding_bandpass}_call_signals'
    welch_signals_file_title = f'{file_title}_{padding_bandpass}_welch_signals'
    for filepath in data_params['good_audio_files']:
        data_params['audio_file'] = Path(filepath)
        print(f'Looking at {filepath}')
        bucket_for_location, calls_sampled_from_location = collect_call_signals_from_location_sum(location_sum_df, data_params, bucket_for_location, calls_sampled_from_location)

    print('Resetting index for call catalogue')
    calls_sampled_from_location.reset_index(inplace=True)
    print(f'Saving call catalogue to {call_signals_file_title}.csv')
    calls_sampled_from_location.to_csv(f'{Path(__file__).parents[2]}/data/detected_calls/{data_params["site_tag"]}/{file_title}_{padding_bandpass}.csv')
    print('Converting bucket to np array')
    np_bucket = np.array(bucket_for_location, dtype='object')
    print(f'Saving bucket to {call_signals_file_title}.npy')
    np.save(f'{Path(__file__).parents[2]}/data/detected_calls/{data_params["site_tag"]}/{call_signals_file_title}.npy', np_bucket)

    calls_sampled_from_location['index'] = calls_sampled_from_location.index
    welch_signals = compute_features.generate_welchs_for_calls(calls_sampled_from_location, bucket_for_location)
    welch_data = pd.DataFrame(welch_signals, columns=np.linspace(0, 96000, welch_signals.shape[1]).astype(int))
    welch_data.index.name = 'Call #'
    welch_data.columns.name = 'Frequency (kHz)'
    welch_data.to_csv(f'{Path(__file__).parents[2]}/data/generated_welch/{data_params["site_tag"]}/{welch_signals_file_title}.csv')

    return bucket_for_location, calls_sampled_from_location


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
        "--threshold",
        type=float,
        help="the threshold; the top (100*X)% will be considered in each bout",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.002,
        help="the padding in seconds",
    )
    parser.add_argument(
        "--bandpass",
        action='store_true',
        help="Whether to bandpass collected call signals",
    )
    parser.add_argument(
        "--use_bouts",
        action='store_true',
        help="Collect calls using each bout as a pool",
    )
    parser.add_argument(
        "--use_file",
        action='store_true',
        help="Collect calls using entire file as a pool",
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
    cfg['percent_threshold_for_snr'] = args['threshold']
    cfg['padding'] = args['padding']
    cfg['bandpass'] = args['bandpass']
    cfg['use_bouts'] = args['use_bouts']
    cfg['use_file'] = args['use_file']
    cfg['detector'] = args['detector']

    sample_calls_and_generate_call_signal_bucket_for_location(cfg)