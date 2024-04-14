import pandas as pd
import re

import sys

sys.path.append("../src")
sys.path.append("../src/bout")

import bout.assembly as bt
import activity.subsampling as ss
import activity.activity_assembly as actvt

def does_duty_cycled_df_have_less_dets_than_original(dc_applied_df, location_df):
    assert dc_applied_df.shape[0] < location_df.shape[0]

def does_reindexed_match_original_at_original_indices(metric_for_scheme_for_comparison, metric_for_scheme):
    assert (metric_for_scheme_for_comparison.loc[metric_for_scheme.index].compare(metric_for_scheme).empty)

def do_calls_exist_in_reindexed_version(metric_for_scheme_for_comparison):
    assert metric_for_scheme_for_comparison.values.all() >= 0

def get_associated_metric_for_cont_column(metric_for_scheme, cont_column):
    metric_for_scheme_for_comparison = metric_for_scheme.reindex(cont_column.index, fill_value=0)
    does_reindexed_match_original_at_original_indices(metric_for_scheme_for_comparison, metric_for_scheme)
    do_calls_exist_in_reindexed_version(metric_for_scheme_for_comparison)

    return metric_for_scheme_for_comparison

def select_dates_from_metrics(metric_for_scheme_for_comparison, cont_column, data_params):
    plt_dcmetr = metric_for_scheme_for_comparison.loc[data_params['start']:data_params['end']].copy()
    plt_cmetr = cont_column.loc[data_params['start']:data_params['end']].copy()

    return plt_dcmetr, plt_cmetr

def generate_activity_btp_for_dc_schemes_and_cont(data_params, file_paths, save=False):
    activity_arr = pd.DataFrame()
    location_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)
    bout_params = bt.get_bout_params_from_location(location_df, data_params)

    dc_schemes = data_params['dc_tags'][1:]
    cont_scheme = data_params['dc_tags'][0]
    btp_arr = pd.DataFrame()
    prev_cycle = 0
    for dc_tag in dc_schemes:
        metric_col_name = f'{data_params["metric_tag"]} ({dc_tag})'
        cycle_length = int(dc_tag.split('of')[1])
        if prev_cycle != cycle_length:
            prev_cycle = cycle_length
            btp_cont_column = get_continuous_btp_partitioned_for_dc_scheme(metric_col_name, location_df.copy(),
                                                                           data_params, bout_params)
            btp_arr = pd.concat([btp_arr, btp_cont_column], axis=1)

        data_params['cur_dc_tag'] = dc_tag
        cycle_length_in_mins = int(data_params['cur_dc_tag'].split('of')[1])
        time_on_in_mins = int(data_params['cur_dc_tag'].split('of')[0])
        time_on_in_secs = (60*time_on_in_mins)

        dc_applied_df = ss.simulate_dutycycle_on_detections(location_df.copy(), data_params)
        does_duty_cycled_df_have_less_dets_than_original(dc_applied_df, location_df)
        bout_metrics = bt.generate_bout_metrics_for_location_and_freq(dc_applied_df, data_params, bout_params)
        bout_duration = actvt.get_bout_duration_per_cycle(bout_metrics, cycle_length_in_mins)
        bout_time_percentage = actvt.get_btp_per_time_on(bout_duration, time_on_in_secs)
        bout_time_percentage_dc_column = actvt.filter_and_prepare_metric(bout_time_percentage, data_params)
        bout_time_percentage_dc_column = bout_time_percentage_dc_column.set_index("datetime_UTC")
        ss.are_there_expected_number_of_cycles(dc_applied_df, bout_time_percentage_dc_column, cycle_length_in_mins, data_params)

        activity_arr = pd.concat([activity_arr, bout_time_percentage_dc_column], axis=1)

    if save:
        activity_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_btp_TYPE_SITE_summary"]}.csv')
        btp_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["cont_btp_TYPE_SITE_summary"]}.csv')

    return activity_arr, btp_arr

def get_continuous_btp_partitioned_for_dc_scheme(metric_col_name, location_df, data_params, bout_params):
    dc_tag_split = re.findall(r"\d+", metric_col_name)
    cycle_length = int(dc_tag_split[-1])
    cont_tag = f'{cycle_length}of{cycle_length}'
    cycle_length_in_secs = 60*cycle_length
    data_params['cur_dc_tag'] = cont_tag

    dc_applied_df = ss.simulate_dutycycle_on_detections(location_df.copy(), data_params)
    bout_metrics = bt.generate_bout_metrics_for_location_and_freq(dc_applied_df, data_params, bout_params)

    bout_duration = actvt.get_bout_duration_per_cycle(bout_metrics, cycle_length)
    bout_time_percentage = actvt.get_btp_per_time_on(bout_duration, cycle_length_in_secs)
    bout_time_percentage_cont_column = actvt.filter_and_prepare_metric(bout_time_percentage, data_params)
    bout_time_percentage_cont_column = bout_time_percentage_cont_column.set_index("datetime_UTC")
    ss.are_there_expected_number_of_cycles(location_df, bout_time_percentage_cont_column, cycle_length, data_params)

    return bout_time_percentage_cont_column

def generate_activity_call_rate_for_dc_schemes_and_cont(data_params, file_paths, save=False):
    activity_arr = pd.DataFrame()
    location_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)

    dc_schemes = data_params['dc_tags'][1:]
    cont_scheme = data_params['dc_tags'][0]
    callrate_arr = pd.DataFrame()
    prev_cycle = 0
    for dc_tag in dc_schemes:
        metric_col_name = f'{data_params["metric_tag"]} ({dc_tag})'
        cycle_length = int(dc_tag.split('of')[1])
        if prev_cycle != cycle_length:
            prev_cycle = cycle_length
            callrate_cont_column = get_continuous_call_rates_partitioned_for_dc_scheme(metric_col_name, file_paths, data_params)
            callrate_arr = pd.concat([callrate_arr, callrate_cont_column], axis=1)

        data_params['cur_dc_tag'] = dc_tag
        cycle_length_in_mins = int(dc_tag.split('of')[1])
        time_on_in_mins = int(dc_tag.split('of')[0])
        time_on_in_secs = (60*time_on_in_mins)
        
        dc_applied_df = ss.simulate_dutycycle_on_detections(location_df.copy(), data_params)
        does_duty_cycled_df_have_less_dets_than_original(dc_applied_df, location_df)

        num_of_detections = actvt.get_number_of_detections_per_cycle(dc_applied_df, cycle_length_in_mins)        
        call_rate = actvt.get_metric_per_time_on(num_of_detections, time_on_in_mins)
        call_rate_dc_column = actvt.filter_and_prepare_metric(call_rate, data_params)
        call_rate_dc_column = call_rate_dc_column.set_index("datetime_UTC")
        ss.are_there_expected_number_of_cycles(dc_applied_df, call_rate_dc_column, cycle_length_in_mins, data_params)
        
        activity_arr = pd.concat([activity_arr, call_rate_dc_column], axis=1)

    if save:
        activity_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_callrate_TYPE_SITE_summary"]}.csv')
        callrate_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["cont_callrate_TYPE_SITE_summary"]}.csv')

    return activity_arr, callrate_arr

def get_continuous_call_rates_partitioned_for_dc_scheme(metric_col_name, file_paths, data_params):
    dc_tag_split = re.findall(r"\d+", metric_col_name)
    dc_tag = re.findall(r"\d+of\d+", metric_col_name)[0]
    cycle_length = int(dc_tag_split[-1])
    cont_tag = f'{cycle_length}of{cycle_length}'
    data_params['cur_dc_tag'] = cont_tag

    location_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)
    location_df = ss.assign_cycle_groups_to_each_call(location_df, cycle_length, data_params)
    num_of_detections = actvt.get_number_of_detections_per_cycle(location_df, cycle_length)
    call_rate = actvt.get_metric_per_time_on(num_of_detections, cycle_length)
    call_rate_cont_column = actvt.filter_and_prepare_metric(call_rate, data_params)
    call_rate_cont_column = call_rate_cont_column.set_index("datetime_UTC")
    ss.are_there_expected_number_of_cycles(location_df, call_rate_cont_column, cycle_length, data_params)

    return call_rate_cont_column

def generate_activity_index_percent_for_dc_schemes_and_cont(data_params, file_paths, save=False):
    activity_arr = pd.DataFrame()
    location_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)

    dc_schemes = data_params['dc_tags'][1:]
    cont_scheme = data_params['dc_tags'][0]
    actvtind_arr = pd.DataFrame()
    prev_cycle = 0
    for dc_tag in dc_schemes:
        metric_col_name = f'{data_params["metric_tag"]} ({dc_tag})'
        cycle_length = int(dc_tag.split('of')[1])
        if prev_cycle != cycle_length:
            prev_cycle = cycle_length
            actvtind_cont_column = get_continuous_activity_index_partitioned_for_dc_scheme(metric_col_name, file_paths, data_params)
            actvtind_arr = pd.concat([actvtind_arr, actvtind_cont_column], axis=1)

        data_params['cur_dc_tag'] = dc_tag
        cycle_length_in_mins = int(dc_tag.split('of')[1])
        time_on_in_mins = int(dc_tag.split('of')[0])
        time_on_in_secs = (60*time_on_in_mins)
        data_params['cycle_length'] = cycle_length_in_mins
        data_params['time_on_in_secs'] = time_on_in_secs
        
        dc_applied_df = ss.simulate_dutycycle_on_detections(location_df.copy(), data_params)
        does_duty_cycled_df_have_less_dets_than_original(dc_applied_df, location_df)

        num_blocks_of_presence = actvt.get_activity_index_per_cycle(dc_applied_df, data_params)        
        activity_ind_percent = actvt.get_activity_index_per_time_on_index(num_blocks_of_presence, data_params)
        ind_percent_dc_column = actvt.filter_and_prepare_metric(activity_ind_percent, data_params)
        ind_percent_dc_column = ind_percent_dc_column.set_index("datetime_UTC")
        ss.are_there_expected_number_of_cycles(dc_applied_df, ind_percent_dc_column, cycle_length_in_mins, data_params)
        
        activity_arr = pd.concat([activity_arr, ind_percent_dc_column], axis=1)

    if save:
        activity_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["dc_actind_TYPE_SITE_summary"]}.csv')
        actvtind_arr.to_csv(f'{file_paths["duty_cycled_folder"]}/{file_paths["cont_actind_TYPE_SITE_summary"]}.csv')

    return activity_arr, actvtind_arr

def get_continuous_activity_index_partitioned_for_dc_scheme(metric_col_name, file_paths, data_params):
    dc_tag_split = re.findall(r"\d+", metric_col_name)
    cycle_length = int(dc_tag_split[-1])
    cycle_length_in_secs = 60*cycle_length

    cont_tag = f'{cycle_length}of{cycle_length}'
    data_params['cur_dc_tag'] = cont_tag
    data_params['cycle_length'] = cycle_length
    data_params['time_on_in_secs'] = cycle_length_in_secs

    location_df = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["bd2_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)
    location_df = ss.assign_cycle_groups_to_each_call(location_df, cycle_length, data_params)
    num_blocks_of_presence = actvt.get_activity_index_per_cycle(location_df, data_params)
    activity_ind_percent = actvt.get_activity_index_per_time_on_index(num_blocks_of_presence, data_params)
    ind_percent_cont_column = actvt.filter_and_prepare_metric(activity_ind_percent, data_params)
    ind_percent_cont_column = ind_percent_cont_column.set_index("datetime_UTC")
    ss.are_there_expected_number_of_cycles(location_df, ind_percent_cont_column, cycle_length, data_params)

    return ind_percent_cont_column