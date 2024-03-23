import pandas as pd
import numpy as np

import sys
sys.path.append('../src/bout')

import clustering as clstr


def construct_bout_metrics_from_classified_dets(fgroups_with_bouttags):
    """
    Reads in the dataframe of detected calls with bout tags.
    Uses these bout tags to create a new dataframe of bout metrics for the start and end times of each bout.
    Also includes the lowest frequency of a call within a bout as the lower bound for the bout
    and the highest frequency of a call within a bout as the upper bound frequency for the bout.
    Now, also included the number of detections captured within each bout.
    """

    location_df = fgroups_with_bouttags.copy()
    location_df.reset_index(drop=True, inplace=True)

    end_times_of_bouts = pd.to_datetime(location_df.loc[location_df['call_status']=='bout end', 'call_end_time'])
    start_times_of_bouts = pd.to_datetime(location_df.loc[location_df['call_status']=='bout start', 'call_start_time'])
    ref_end_times = location_df.loc[location_df['call_status']=='bout end', 'end_time_wrt_ref'].astype('float')
    ref_start_times = location_df.loc[location_df['call_status']=='bout start', 'start_time_wrt_ref'].astype('float')
    end_times = location_df.loc[location_df['call_status']=='bout end', 'end_time'].astype('float')
    start_times = location_df.loc[location_df['call_status']=='bout start', 'start_time'].astype('float')

    if len(start_times_of_bouts) < len(end_times_of_bouts):
        end_times_of_bouts = end_times_of_bouts[:-1]
        end_times = end_times[:-1]
        ref_end_times = ref_end_times[:-1]

    if len(start_times_of_bouts) > len(end_times_of_bouts):
        start_times_of_bouts = start_times_of_bouts[:-1]
        start_times = start_times[:-1]
        ref_start_times = ref_start_times[:-1]

    bout_starts = start_times_of_bouts.index
    bout_ends = end_times_of_bouts.index

    low_freqs = []
    high_freqs = []
    num_calls_per_bout = []
    j=0
    for i in range(len(bout_starts)):
        if (i+j < len(bout_starts)) and bout_ends[i+j] > bout_starts[i]:
            pass_low_freq = np.min((location_df.iloc[bout_starts[i]:bout_ends[i+j]+1]['low_freq']).values)
            pass_high_freq = np.max((location_df.iloc[bout_starts[i]:bout_ends[i+j]+1]['high_freq']).values)
            num_calls = len(location_df.iloc[bout_starts[i]:bout_ends[i+j]]) + 1
            low_freqs += [pass_low_freq]
            high_freqs += [pass_high_freq]
            num_calls_per_bout += [num_calls]
        else:
            j+=1

    bout_metrics = pd.DataFrame()
    bout_metrics['start_time_of_bout'] = start_times_of_bouts.values
    bout_metrics['end_time_of_bout'] = end_times_of_bouts.values
    bout_metrics['start_time_wrt_ref'] = ref_start_times.values
    bout_metrics['end_time_wrt_ref'] = ref_end_times.values
    bout_metrics['start_time'] = start_times.values
    bout_metrics['end_time'] = end_times.values
    bout_metrics['low_freq'] = low_freqs
    bout_metrics['high_freq'] = high_freqs
    bout_metrics['number_of_dets'] = num_calls_per_bout
    bout_metrics['bout_duration'] = end_times_of_bouts.values - start_times_of_bouts.values
    bout_metrics['bout_duration_in_secs'] = bout_metrics['bout_duration'].apply(lambda x : x.total_seconds())
    return bout_metrics

def construct_bout_metrics_from_location_df_for_freqgroups(location_df):
    """
    Given a location summary with tagged bout markers, construct and concatenate together bout metrics for each group
    """

    bout_metrics = pd.DataFrame()
    for group in location_df['freq_group'].unique():
        if group != '':
            freqgroup_bat_preds_with_bouttags = location_df.loc[location_df['freq_group']==group].copy()
            if not(freqgroup_bat_preds_with_bouttags.empty):
                freqgroup_bout_metrics = construct_bout_metrics_from_classified_dets(freqgroup_bat_preds_with_bouttags)
                freqgroup_bout_metrics.insert(0, 'freq_group', group)
                bout_metrics = pd.concat([bout_metrics, freqgroup_bout_metrics])

    return bout_metrics

def classify_bouts_in_bd2_predictions_for_freqgroups(batdetect2_predictions, bout_params):
    """
    Given a location summary and BCIs calculated for each group in location summary, tag each call in summary as the following:
    - Within Bout : Call existing inside a bout
    - Outside Bout: Call that is not a part of any bout
    - Bout Start: Within-bout call that starts a new bout
    - Bout End: Within-bout call that ends the bout
    """

    location_df = batdetect2_predictions.copy()
    location_df.insert(0, 'duration_from_last_call_ms', 0)
    location_df.insert(0, 'bout_tag', 0)
    location_df.insert(0, 'change_markers', 0)
    location_df.insert(0, 'call_status', '')
    result_df = pd.DataFrame()

    for group in location_df['freq_group'].unique():
        if group != '':
            freq_group_df = location_df.loc[location_df['freq_group']==group].copy()
            freq_group_df.reset_index(drop=True, inplace=True)
            if not(freq_group_df.empty):
                intervals = (pd.to_datetime(freq_group_df['call_start_time'].values[1:]) - pd.to_datetime(freq_group_df['call_end_time'].values[:-1]))
                ipis_f = intervals.to_numpy(dtype='float32')/1e6
                ipis_f = np.insert(ipis_f, 0, bout_params[f'{group}_bci'])

                freq_group_df['duration_from_last_call_ms'] =  ipis_f
                freq_group_df.loc[freq_group_df['duration_from_last_call_ms'] < bout_params[f'{group}_bci'], 'bout_tag'] = 1
                freq_group_df.loc[freq_group_df['duration_from_last_call_ms'] >= bout_params[f'{group}_bci'], 'bout_tag'] = 0
                wb_indices = freq_group_df.loc[freq_group_df['bout_tag']==1].index
                ob_indices = freq_group_df.loc[freq_group_df['bout_tag']==0].index
                freq_group_df.loc[wb_indices, 'call_status'] = 'within bout'
                freq_group_df.loc[ob_indices, 'call_status'] = 'outside bout'

                bout_tags = freq_group_df['bout_tag']
                change_markers = bout_tags.shift(-1) - bout_tags
                change_markers[len(change_markers)-1] = 0
                freq_group_df['change_markers'] = change_markers
                be_indices = freq_group_df.loc[freq_group_df['change_markers']==-1].index
                bs_indices = freq_group_df.loc[freq_group_df['change_markers']==1].index
                freq_group_df.loc[be_indices, 'call_status'] = 'bout end'
                freq_group_df.loc[bs_indices, 'call_status'] = 'bout start'

                num_bout_starts = len(freq_group_df.loc[freq_group_df['call_status']=='bout start'])
                num_bout_ends = len(freq_group_df.loc[freq_group_df['call_status']=='bout end'])
                if num_bout_starts != num_bout_ends:
                    freq_group_df.at[len(freq_group_df)-1, 'call_status'] = 'bout end'

                result_df = pd.concat([result_df, freq_group_df])

    return result_df

def generate_bout_metrics_for_location_and_freq(location_sum_df, data_params, time_on):
    """
    Given a location summary of calls dataframe, create an analogous location summary of bouts by:
    1) Calculating the BCI for each frequency group in the summary.
    2) Use the calculated BCI for each group to cluster bouts for that group.
    3) Put together all bout characteristics into the analogous dataframe.
    """

    location_sum_df.reset_index(drop=True, inplace=True)

    bout_params = get_bout_params_from_location(location_sum_df, data_params)

    tagged_dets = classify_bouts_in_bd2_predictions_for_freqgroups(location_sum_df, bout_params)
    bout_metrics = construct_bout_metrics_from_location_df_for_freqgroups(tagged_dets)

    test_bout_end_times_in_period(bout_metrics, time_on)

    return bout_metrics

def get_bout_params_from_location(location_sum_df, data_params):
    """
    Given a location summary and the location it corresponds to, calculate the BCIs for each frequency group in the summary
    """

    bout_params = dict()
    bout_params['site_key'] = data_params['site_tag']

    for group in location_sum_df['freq_group'].unique():
        if group != '':
            freq_group_df = location_sum_df.loc[location_sum_df['freq_group']==group].copy()
            if not(freq_group_df.empty):
                ipis_loc, hist_loc = clstr.get_histogram(freq_group_df, 10)
                intervals_ms, survival = clstr.get_log_survival(hist_loc)
                fast_process = clstr.regress_around_peakIPI(intervals_ms, survival, hist_loc[0])
                fast_process = clstr.calculate_exponential_coefficients(fast_process)
                slow_process = clstr.regress_around_slow_intervals(intervals_ms, survival)
                slow_process = clstr.calculate_exponential_coefficients(slow_process)
                nlin_results, misassigned_points_optim = clstr.get_bci_from_sibly_method(intervals_ms, survival, fast_process, slow_process)
                bout_params[f'{group}_bci'] = nlin_results['bci']

    return bout_params

def test_bout_end_times_in_period(bout_metrics, time_on):
    """
    A test function to see if the duty cycle effects from the location summary of all calls carried over to the location summary of all bouts.
    """
    assert(bout_metrics['end_time_wrt_ref'].max() <= 60*time_on)