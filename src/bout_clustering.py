import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scipy
import scipy.stats as stats
from core import FREQ_GROUPS

def regress_around_peakIPI(intervals_ms, survival, values):
    """
    Use scipy.stats to compute linear regression coefficients around points
    we associate with within-bout intervals.

    These points are chosen around the most common inter-pulse interval.
    We know this interval and neighboring intervals are most likely to be within bout.
    """

    max_val = np.max(values)
    first_peak = np.where(values==max_val)[0][0]
    fast_inds = range(0, int(np.ceil(5*first_peak))+1)
    fast_coeff = stats.linregress(intervals_ms[fast_inds], survival[fast_inds])

    fast_process = dict()
    fast_process['metrics'] = fast_coeff
    fast_process['indices'] = fast_inds
    return fast_process

def regress_around_survival_threshold(intervals_ms, survival):
    """
    Use scipy.stats to compute linear regression coefficients around points
    we associate with within-bout intervals.

    These interval points to regress around are chosen using the top 10% of survival values.
    We can start to guess that these intervals could be within bout.
    """

    fast_inds = np.logical_and(survival >= (survival.max() * 0.90), survival <= (survival.max() * 1.0))
    fast_coeff = stats.linregress(intervals_ms[fast_inds], survival[fast_inds])

    fast_process = dict()
    fast_process['metrics'] = fast_coeff
    fast_process['indices'] = fast_inds
    return fast_process

def regress_around_slow_intervals(intervals_ms, survival):
    """
    Use scipy.stats to compute linear regression coefficients around points
    we associate with between-bout intervals.

    These interval points to regress around are chosen using values between 30-40% of the max survival.
    We have observed that these points have a strong linear relationship.
    They are also among intervals from 20 to 60min. This range is very likely between-bout.
    """

    slow_inds = np.logical_and(survival >= (survival.max() * 0.30), survival <= (survival.max() * 0.40))
    slow_coeff = stats.linregress(intervals_ms[slow_inds], survival[slow_inds])

    slow_process = dict()
    slow_process['metrics'] = slow_coeff
    slow_process['indices'] = slow_inds
    return slow_process

def test_ipis_ms(ipis_ms, location_sum_dates, location_sum_df):
    """
    The # of intervals calculated should be equal to the # of calls - (DATES)
    DATES is a constant here because for each date, the first call is not considered 
    as there is no previous call to calculate an interval for.
    """

    assert(len(ipis_ms) + len(location_sum_dates) == len(location_sum_df))

def get_valid_ipis_ms(bout_params):
    """
    Gets the IPIs (Inter-Pulse Intervals) for a given location and frequency group
    using the 2022_bd2_summary files stored in data.

    Ignores IPIs generated for the first call of every date since we only want to consider intervals within nights.

    Returns a numpy array of IPIs in milliseconds.
    """

    location_sum_df = pd.read_csv(f'../data/2022_bd2_summary/{bout_params["site_key"]}/bd2__{bout_params["freq_key"]}{bout_params["site_key"]}_2022.csv', index_col=0)
    intervals = pd.to_datetime(location_sum_df['call_start_time']) - pd.to_datetime(location_sum_df['call_end_time']).shift(1)
    location_sum_df.insert(0, 'time_from_prev_call_end_time', intervals)

    location_sum_df = location_sum_df.set_index('ref_time')
    location_sum_dates = np.unique(pd.DatetimeIndex(location_sum_df.index).date)
    for date in location_sum_dates:
        ref_times_of_dates = location_sum_df.loc[pd.DatetimeIndex(location_sum_df.index).date == date].index
        location_sum_df.at[ref_times_of_dates[0], 'time_from_prev_call_end_time'] = pd.NaT

    intervals = location_sum_df['time_from_prev_call_end_time'].values
    valid_intervals = intervals[~np.isnan(intervals)]
    ipis_ms = valid_intervals.astype('float32')/1e6

    test_ipis_ms(ipis_ms, location_sum_dates, location_sum_df)

    return ipis_ms

def get_histogram(bout_params, fig_details):
    """
    Uses the IPIs from a location and for a frequency group to compute and return a complete histogram.
    The interval width is set to be 10ms to provide good resolution for the most common IPIs.
    """

    ipis_ms = get_valid_ipis_ms(bout_params)
    hist_loc = np.histogram(ipis_ms, bins=np.arange(0, ipis_ms.max()+fig_details['bin_step'], fig_details['bin_step']))

    return ipis_ms, hist_loc

def get_log_survival(hist_loc):
    """
    Computes and returns a log-survivorship curve provided a histogram.
    """

    values, base = hist_loc[0], hist_loc[1]
    cumulative = (np.cumsum(values[::-1]))[::-1]
    survival = np.log(cumulative)
    intervals_ms = base[:-1]

    return intervals_ms, survival

def calculate_exponential_coefficients(process):
    """
    Computes and returns the exponential coefficients given the slope and intercept of a process.
    Using equations from Sibly et al. (1990) and Slater & Lester (1982)
    """
    
    process['lambda'] = -1*process['metrics'].slope
    process['num_intervals_slater'] = np.exp(process['metrics'].intercept) / process['lambda']

    return process


def get_bci_from_fagenyoung_method(fast_process, slow_process):
    """
    Computes and returns the BCI given the lambda and N value for each process.
    Using the equation from Fagen & Young (1978) derived for minimizing total time misassigned.
    """

    bci = (1/(fast_process['lambda'] - slow_process['lambda'])) * np.log(fast_process['num_intervals_slater']/slow_process['num_intervals_slater'])

    misassigned_points = (fast_process['num_intervals_slater']*np.exp(-1*fast_process['lambda']*bci)) + (slow_process['num_intervals_slater']*(1 - np.exp(-1*slow_process['lambda']*bci)))

    return bci, misassigned_points

def get_bci_from_slater_method(fast_process, slow_process):
    """
    Computes and returns the BCI given the lambda and N value for each process.
    Using the equation from Slater & Lester (1982) derived for minimizing # of events misassigned.
    """

    bci = (1/(fast_process['lambda'] - slow_process['lambda'])) * np.log((fast_process['num_intervals_slater']*fast_process['lambda'])/(slow_process['num_intervals_slater']*slow_process['lambda']))

    misassigned_points = (fast_process['num_intervals_slater']*np.exp(-1*fast_process['lambda']*bci)) + (slow_process['num_intervals_slater']*(1 - np.exp(-1*slow_process['lambda']*bci)))

    return bci, misassigned_points

def model(t, f_intervals, f_lambda, s_intervals, s_lambda):
    return (np.log((f_intervals*f_lambda*np.exp(-1*f_lambda*t))  + (s_intervals*s_lambda*np.exp(-1*s_lambda*t))))

def get_bci_from_sibly_method(intervals_ms, survival, fast_process, slow_process):
    """
    Computes and returns the BCI given the lambda and N value for each process.
    Using the equation from Sibly et al. (1990) derived using NLIN curve-fitting techniques to model the regression lines with a single curve.
    """

    x0 = np.array([fast_process['num_intervals_slater'], fast_process['lambda'], slow_process['num_intervals_slater'], slow_process['lambda']], dtype='float64')
    nlin_inds = np.concatenate([fast_process['indices'], np.where(slow_process['indices']==True)[0]])
    cfit_sols = scipy.optimize.curve_fit(model, intervals_ms[nlin_inds].astype('float64'), survival[nlin_inds].astype('float64'), p0=x0)
    nlin_results = dict()
    nlin_results['solution'] = cfit_sols[0]
    nlin_results['fast_num_intervals'] = nlin_results['solution'][0]
    nlin_results['fast_lambda'] = nlin_results['solution'][1]
    nlin_results['slow_num_intervals'] = nlin_results['solution'][2]
    nlin_results['slow_lambda'] = nlin_results['solution'][3]

    bci_coeff = (1/(nlin_results['fast_lambda'] - nlin_results['slow_lambda'])) 
    nlin_results['bci'] = bci_coeff * np.log((nlin_results['fast_num_intervals']*nlin_results['fast_lambda'])/(nlin_results['slow_num_intervals']*nlin_results['slow_lambda']))

    fast_misassignments = (nlin_results['fast_num_intervals']*np.exp(-1*nlin_results['fast_lambda']*nlin_results['bci']))
    slow_missasignments = (nlin_results['slow_num_intervals']*(1 - np.exp(-1*nlin_results['slow_lambda']*nlin_results['bci'])))
    misassigned_points_optim = fast_misassignments + slow_missasignments

    return nlin_results, misassigned_points_optim

def classify_bouts_in_single_bd2_output(location_df, bout_params):
    """
    Reads in the bd2 output for a single file and assigned bout tags whether a call is:
    within bout, outside bout, a bout start, or a bout end.
    """

    location_df.reset_index(inplace=True)
    location_df = location_df.drop(columns=location_df.columns[0])

    intervals = (pd.to_datetime(location_df['call_start_time'].values[1:]) - pd.to_datetime(location_df['call_end_time'].values[:-1]))
    ipis_f = intervals.to_numpy(dtype='float32')/1e6
    ipis_f = np.insert(ipis_f, 0, bout_params['bci'])

    location_df.insert(0, 'duration_from_last_call_ms', ipis_f)
    location_df.insert(0, 'bout_tag', 0)
    location_df.insert(0, 'call_status', '')
    location_df.loc[location_df['duration_from_last_call_ms'] < bout_params['bci'], 'bout_tag'] = 1
    location_df.loc[location_df['duration_from_last_call_ms'] >= bout_params['bci'], 'bout_tag'] = 0

    bout_tags = location_df['bout_tag'].values
    change_markers = (bout_tags[1:] - bout_tags[:-1])

    location_df.loc[np.where(bout_tags==1)[0], 'call_status'] = 'within bout'
    location_df.loc[np.where(bout_tags==0)[0], 'call_status'] = 'outside bout'
    location_df.loc[np.where(change_markers==-1)[0], 'call_status'] = 'bout end'
    location_df.loc[np.where(change_markers==1)[0], 'call_status'] = 'bout start'

    num_bout_starts = len(location_df.loc[location_df['call_status']=='bout start'])
    num_bout_ends = len(location_df.loc[location_df['call_status']=='bout end'])
    if num_bout_starts != num_bout_ends:
        location_df.at[len(location_df)-1, 'call_status'] = 'bout end'

    return location_df

def classify_bouts_in_location_summary(bout_params):
    """
    Reads in the bd2_summary for a single location and frequency grouping and assigns bout tags whether a call is:
    within bout, outside bout, a bout start, or a bout end.
    """

    location_df = pd.read_csv(f'../data/2022_bd2_summary/{bout_params["site_key"]}/bd2__{bout_params["freq_key"]}{bout_params["site_key"]}_2022.csv')
    location_df.drop(columns=location_df.columns[0], inplace=True)

    intervals = (pd.to_datetime(location_df['call_start_time'].values[1:]) - pd.to_datetime(location_df['call_end_time'].values[:-1]))
    ipis_f = intervals.to_numpy(dtype='float32')/1e6
    ipis_f = np.insert(ipis_f, 0, bout_params['bci'])

    location_df.insert(0, 'duration_from_last_call_ms', ipis_f)
    location_df.insert(0, 'bout_tag', 0)
    location_df.insert(0, 'call_status', '')
    location_df.loc[location_df['duration_from_last_call_ms'] < bout_params['bci'], 'bout_tag'] = 1
    location_df.loc[location_df['duration_from_last_call_ms'] >= bout_params['bci'], 'bout_tag'] = 0

    bout_tags = location_df['bout_tag'].values
    change_markers = (bout_tags[1:] - bout_tags[:-1])

    location_df.loc[np.where(bout_tags==1)[0], 'call_status'] = 'within bout'
    location_df.loc[np.where(bout_tags==0)[0], 'call_status'] = 'outside bout'
    location_df.loc[np.where(change_markers==-1)[0], 'call_status'] = 'bout end'
    location_df.loc[np.where(change_markers==1)[0], 'call_status'] = 'bout start'

    return location_df

def construct_bout_metrics_from_classified_dets(location_df):
    """
    Reads in the dataframe of detected calls with bout tags from above methoods.
    Uses these bout tags to create a new dataframe of bout metrics for the start and end times of each bout.
    Also includes the lowest frequency of a call within a bout as the lower bound for the bout
    and the highest frequency of a call within a bout as the upper bound frequency for the bour.
    """

    end_times_of_bouts = pd.to_datetime(location_df.loc[location_df['call_status']=='bout end', 'call_end_time'])
    start_times_of_bouts = pd.to_datetime(location_df.loc[location_df['call_status']=='bout start', 'call_start_time'])
    end_times = location_df.loc[location_df['call_status']=='bout end', 'end_time'].astype('float')
    start_times = location_df.loc[location_df['call_status']=='bout start', 'start_time'].astype('float')
    if len(start_times_of_bouts) != len(end_times_of_bouts):
        start_times_of_bouts = start_times_of_bouts[:-1]
        start_times = start_times[:-1]
    if len(start_times_of_bouts) != len(end_times_of_bouts):
        start_times_of_bouts = start_times_of_bouts[1:]
        start_times = start_times[1:]

    bout_starts = start_times_of_bouts.index
    bout_ends = end_times_of_bouts.index
    low_freqs = []
    high_freqs = []
    for i in range(len(bout_starts)):
        pass_low_freq = np.min(location_df.iloc[bout_starts[i]:bout_ends[i]]['low_freq'].values)
        pass_high_freq = np.max(location_df.iloc[bout_starts[i]:bout_ends[i]]['high_freq'].values)
        low_freqs += [pass_low_freq]
        high_freqs += [pass_high_freq]

    bout_metrics = pd.DataFrame()
    bout_metrics['start_time_of_bout'] = start_times_of_bouts.values
    bout_metrics['end_time_of_bout'] = end_times_of_bouts.values
    bout_metrics['start_time'] = start_times.values
    bout_metrics['end_time'] = end_times.values
    bout_metrics['low_freq'] = low_freqs
    bout_metrics['high_freq'] = high_freqs
    bout_metrics['bout_duration'] = end_times_of_bouts.values - start_times_of_bouts.values
    bout_metrics['bout_duration_in_secs'] = bout_metrics['bout_duration'].apply(lambda x : x.total_seconds())
    return bout_metrics