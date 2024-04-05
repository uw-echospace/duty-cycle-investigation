import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import comparison.data_assembly as comp

def plot_indiv_dc_control_comp_over_cycle_log_scale(plt_dcmetr, plt_cmetr, data_params):
    dc_tag_split = re.findall(r"\d+", plt_dcmetr.columns[0])
    total = int(dc_tag_split[-1])
    on = int(dc_tag_split[0])
    listening_ratio = on/total
    plt_dcmetr.index = pd.DatetimeIndex(plt_dcmetr.index)
    dc_metr1 = pd.pivot_table(plt_dcmetr, index=(plt_dcmetr.index.time), 
                        columns=plt_dcmetr.index.date, 
                        values=plt_dcmetr.columns[0])
    dctag1 = re.search(r"\d+of\d+", plt_dcmetr.columns[0])[0]
    metrictag1 = (plt_dcmetr.columns[0]).split()[0]
    plt_cmetr.index = pd.DatetimeIndex(plt_cmetr.index)
    paired_col_c_metr = plt_cmetr[(~plt_dcmetr.isna()).values]
    c_metr = pd.pivot_table(paired_col_c_metr, index=(paired_col_c_metr.index.time), 
                    columns=paired_col_c_metr.index.date, 
                    values=paired_col_c_metr.columns[0])
    plt.title(f'{dctag1} and continuous scheme')
    
    non_zero_c_metr_min = c_metr.replace(0, np.NaN).min().min()
    c_metr_max = c_metr.max().max()
    if data_params['metric_tag']=='activity_index':
        lower_bound = (100/(60/5))*(1/on)
        plt.axhline(y=lower_bound, linestyle='dashed', color='g', label='low: $on time^{-1}$')
    elif data_params['metric_tag']=='call_rate':
        lower_bound = (1/on)
        plt.axhline(y=lower_bound, linestyle='dashed', color='g', label='low: $on time^{-1}$')

    if data_params['metric_tag']=='bout_time_percentage':
        plt.axhline(y=100, linestyle='dashed', color='g', label='100% line')
    elif data_params['metric_tag']=='activity_index':
        plt.axhline(y=100, linestyle='dashed', color='g', label='100% line')

    bound_c_values = np.logspace(-3, 3, 3)
    plt.plot([non_zero_c_metr_min, c_metr_max], [non_zero_c_metr_min, c_metr_max], linestyle='dashed', color='k')
    plt.plot(bound_c_values, (1/listening_ratio)*bound_c_values, linestyle='dashed', color='g', label='upp: $ratio^{-1}$*c')
   
    colors = np.tile(np.arange(0, c_metr.shape[0]),(c_metr.shape[1],1)).T
    labels = pd.to_datetime(c_metr.index, format='%H:%M:%S').strftime('%H:%M')
    sc = plt.scatter(c_metr, dc_metr1, c=colors, cmap='YlOrRd', edgecolors='k', s=80, alpha=1)
    cbar = plt.colorbar(sc, ticks=[0, (colors.shape[0]-1)//2, colors.shape[0]-1])
    cbar.ax.set_yticklabels([labels[0], labels[len(labels)//2], labels[-1]])
    plt.xlabel(f'Continuous Measured {metrictag1}')
    plt.xscale('log')
    plt.yscale('log')
    if (data_params['metric_tag'] == 'call_rate'):
        plt.xlim(1e-4, 1e3)
        plt.ylim(1e-4, 1e3)
    else:
        plt.xlim(1e-4, 2e2)
        plt.ylim(1e-4, 2e2)
    plt.ylabel(f'DC Measured {metrictag1}')
    plt.grid(which='both')
    plt.legend(loc=3)


def plot_indiv_dc_control_comp_error_over_cycle_log_scale(plt_dcmetr, plt_cmetr, data_params):
    dc_tag_split = re.findall(r"\d+", plt_dcmetr.columns[0])
    total = int(dc_tag_split[-1])
    on = int(dc_tag_split[0])
    listening_ratio = on/total
    dctag1 = re.search(r"\d+of\d+", plt_dcmetr.columns[0])[0]
    cont_metric_col_name = plt_cmetr.columns[0]
    sorted_c_metr = plt_cmetr.sort_values(by=cont_metric_col_name).replace(0, np.NaN).dropna()
    ref_dc_metr = plt_dcmetr.loc[sorted_c_metr.index]
    error_ratio_df = pd.DataFrame(ref_dc_metr.values / sorted_c_metr.values, index=ref_dc_metr.index, columns=[f'error ({dctag1})'])
    dc_tag_split = re.findall(r"\d+", ref_dc_metr.columns[0])
    cycle_length = int(dc_tag_split[-1])
    time_on = int(dc_tag_split[0])
    listening_ratio = time_on/cycle_length

    error_ratio_df.index = pd.DatetimeIndex(error_ratio_df.index)
    error_ratio_table = pd.pivot_table(error_ratio_df, index=(error_ratio_df.index.time), 
                        columns=error_ratio_df.index.date, 
                        values=error_ratio_df.columns[0])

    metrictag1 = (sorted_c_metr.columns[0]).split()[0]
    sorted_c_metr.index = pd.DatetimeIndex(sorted_c_metr.index)
    paired_col_c_metr = sorted_c_metr[(~sorted_c_metr.isna()).values]
    c_metr = pd.pivot_table(paired_col_c_metr, index=(paired_col_c_metr.index.time), 
                    columns=paired_col_c_metr.index.date, 
                    values=paired_col_c_metr.columns[0])
    bound_c_values = np.logspace(-3, 3, 3)
    if data_params['metric_tag']=='activity_index':
        lower_bound = (100/(60/5))*(1/time_on)
        plt.plot(bound_c_values, lower_bound/bound_c_values, 
                 linestyle='dashed', color='g', label='low: $on time^{-1}$/c')
    elif data_params['metric_tag']=='call_rate':
        lower_bound = (1/time_on)
        plt.plot(bound_c_values, lower_bound/bound_c_values, 
                 linestyle='dashed', color='g', label='low: $on time^{-1}$/c')
        
    if data_params['metric_tag']=='bout_time_percentage':
        plt.plot(bound_c_values, 100/bound_c_values, 
                 linestyle='dashed', color='g', label='100% line')
    elif data_params['metric_tag']=='activity_index':
        plt.plot(bound_c_values, 100/bound_c_values, 
                 linestyle='dashed', color='g', label='100% line')

    plt.title(f'{dctag1} and continuous scheme')
    colors = np.tile(np.arange(0, c_metr.shape[0]),(c_metr.shape[1],1)).T
    labels = pd.to_datetime(c_metr.index, format='%H:%M:%S').strftime('%H:%M')
    sc = plt.scatter(c_metr, error_ratio_table, c=colors, cmap='YlOrRd', edgecolors='k', s=80, alpha=1)
    cbar = plt.colorbar(sc, ticks=[0, (colors.shape[0]-1)//2, colors.shape[0]-1])
    cbar.ax.set_yticklabels([labels[0], labels[len(labels)//2], labels[-1]])
    plt.axhline(y=listening_ratio**-1, linestyle='dashed', color='g', label='upp: $ratio^{-1}$')
    plt.axhline(y=1, linestyle='dashed', color='k')
    if (data_params['metric_tag'] == 'call_rate'):
        plt.xlim(1e-3, 1e3)
        plt.ylim(1e-4, 1e1)
    else:
        plt.xlim(1e-4, 2e2)
        plt.ylim(1e-3, 1e1)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(which='both')
    plt.ylabel(f'Error Ratio {dctag1}')
    plt.xlabel(f'{metrictag1}')
    plt.legend(loc=3)


def plot_indiv_dc_control_comp_over_cycle_linear_scale(single_col_dc_metr1, single_col_c_metr, data_params, drop_zero=False):
    dc_tag_split = re.findall(r"\d+", single_col_dc_metr1.columns[0])
    total = int(dc_tag_split[-1])
    on = int(dc_tag_split[0])
    listening_ratio = on/total
    single_col_dc_metr1.index = pd.DatetimeIndex(single_col_dc_metr1.index)
    dc_metr1 = pd.pivot_table(single_col_dc_metr1, index=(single_col_dc_metr1.index.time), 
                        columns=single_col_dc_metr1.index.date, 
                        values=single_col_dc_metr1.columns[0])
    if drop_zero:
        dc_metr1 = dc_metr1.replace(0, np.NaN)
    dctag1 = re.search(r"\d+of\d+", single_col_dc_metr1.columns[0])[0]
    metrictag1 = (single_col_dc_metr1.columns[0]).split()[0]
    single_col_c_metr.index = pd.DatetimeIndex(single_col_c_metr.index)
    paired_col_c_metr = single_col_c_metr[(~single_col_dc_metr1.isna()).values]
    c_metr = pd.pivot_table(paired_col_c_metr, index=(paired_col_c_metr.index.time), 
                    columns=paired_col_c_metr.index.date, 
                    values=paired_col_c_metr.columns[0])
    plt.title(f'{dctag1} and continuous scheme')
    
    non_zero_c_metr_min = c_metr.replace(0, np.NaN).min().min()
    dc_metr_max = dc_metr1.max().max()
    c_metr_max = c_metr.max().max()
    if data_params['metric_tag']=='activity_index':
        lower_bound = (100/(60/5))*(1/on)
        plt.axhline(y=lower_bound, linestyle='dashed', color='g', label='low: $on time^{-1}$')
    elif data_params['metric_tag']=='call_rate':
        lower_bound = (1/on)
        plt.axhline(y=lower_bound, linestyle='dashed', color='g', label='low: $on time^{-1}$')

    if data_params['metric_tag']=='bout_time_percentage':
        plt.axhline(y=100, linestyle='dashed', color='g', label='100% line')
    elif data_params['metric_tag']=='activity_index':
        plt.axhline(y=100, linestyle='dashed', color='g', label='100% line')

    bound_c_values = np.linspace(1e-2, 1e3, 10001)
    plt.plot(bound_c_values, (1/listening_ratio)*bound_c_values, linestyle='dashed', color='g', label='upp: $ratio^{-1}$*c')
    plt.plot([non_zero_c_metr_min, c_metr_max], [non_zero_c_metr_min, c_metr_max], linestyle='dashed', color='k')
   
    colors = np.tile(np.arange(0, c_metr.shape[0]),(c_metr.shape[1],1)).T
    labels = pd.to_datetime(c_metr.index, format='%H:%M:%S').strftime('%H:%M')
    sc = plt.scatter(c_metr, dc_metr1, c=colors, cmap='YlOrRd', edgecolors='k', s=80, alpha=1)
    cbar = plt.colorbar(sc, ticks=[0, (colors.shape[0]-1)//2, colors.shape[0]-1])
    cbar.ax.set_yticklabels([labels[0], labels[len(labels)//2], labels[-1]])
    plt.xlabel(f'Continuous Measured {metrictag1}')
    upper_lim = 1.6*(dc_metr_max)
    if (data_params['metric_tag'] == 'call_rate'):
        plt.xlim(-upper_lim/10, upper_lim)
        plt.ylim(-upper_lim/10, upper_lim)
    else:
        plt.xlim(-upper_lim/10, upper_lim)
        plt.ylim(-upper_lim/10, upper_lim)
        # plt.xlim(-10, 1.6e2)
        # plt.ylim(-10, 1.6e2)
    plt.ylabel(f'DC Measured {metrictag1}')
    plt.grid(which='both')
    plt.legend(loc=1)


def plot_indiv_dc_control_comp_error_over_cycle_linear_scale(plt_dcmetr, plt_cmetr, data_params, drop_zero=False):
    dc_tag_split = re.findall(r"\d+", plt_dcmetr.columns[0])
    total = int(dc_tag_split[-1])
    on = int(dc_tag_split[0])
    listening_ratio = on/total
    dctag1 = re.search(r"\d+of\d+", plt_dcmetr.columns[0])[0]
    cont_metric_col_name = plt_cmetr.columns[0]
    sorted_c_metr = plt_cmetr.sort_values(by=cont_metric_col_name).replace(0, np.NaN).dropna()
    ref_dc_metr = plt_dcmetr.loc[sorted_c_metr.index]
    error_ratio_df = pd.DataFrame(ref_dc_metr.values / sorted_c_metr.values, index=ref_dc_metr.index, columns=[f'error ({dctag1})'])
    dc_tag_split = re.findall(r"\d+", ref_dc_metr.columns[0])
    cycle_length = int(dc_tag_split[-1])
    time_on = int(dc_tag_split[0])
    listening_ratio = time_on/cycle_length

    error_ratio_df.index = pd.DatetimeIndex(error_ratio_df.index)
    error_ratio_table = pd.pivot_table(error_ratio_df, index=(error_ratio_df.index.time), 
                        columns=error_ratio_df.index.date, 
                        values=error_ratio_df.columns[0])
    if drop_zero:
        error_ratio_table = error_ratio_table.replace(0, np.NaN)

    metrictag1 = (sorted_c_metr.columns[0]).split()[0]
    sorted_c_metr.index = pd.DatetimeIndex(sorted_c_metr.index)
    paired_col_c_metr = sorted_c_metr[(~sorted_c_metr.isna()).values]
    c_metr = pd.pivot_table(paired_col_c_metr, index=(paired_col_c_metr.index.time), 
                    columns=paired_col_c_metr.index.date, 
                    values=paired_col_c_metr.columns[0])
    bound_c_values = np.linspace(1e-2, 1e3, 10001)
    if data_params['metric_tag']=='activity_index':
        lower_bound = (100/(60/5))*(1/time_on)
        plt.plot(bound_c_values, lower_bound/bound_c_values, 
                 linestyle='dashed', color='g', label='low: $on time^{-1}$/c')
    elif data_params['metric_tag']=='call_rate':
        lower_bound = (1/time_on)
        plt.plot(bound_c_values, lower_bound/bound_c_values, 
                 linestyle='dashed', color='g', label='low: $on time^{-1}$/c')
        
    if data_params['metric_tag']=='bout_time_percentage':
        plt.plot(bound_c_values, 100/bound_c_values, 
                 linestyle='dashed', color='g', label='100% line')
    elif data_params['metric_tag']=='activity_index':
        plt.plot(bound_c_values, 100/bound_c_values, 
                 linestyle='dashed', color='g', label='100% line')

    plt.title(f'{dctag1} and continuous scheme')
    colors = np.tile(np.arange(0, c_metr.shape[0]),(c_metr.shape[1],1)).T
    labels = pd.to_datetime(c_metr.index, format='%H:%M:%S').strftime('%H:%M')
    sc = plt.scatter(c_metr, error_ratio_table, c=colors, cmap='YlOrRd', edgecolors='k', s=80, alpha=1)
    cbar = plt.colorbar(sc, ticks=[0, (colors.shape[0]-1)//2, colors.shape[0]-1])
    cbar.ax.set_yticklabels([labels[0], labels[len(labels)//2], labels[-1]])
    plt.axhline(y=listening_ratio**-1, linestyle='dashed', color='g', label='upp: $ratio^{-1}$')
    plt.axhline(y=1, linestyle='dashed', color='k')
    plt.ylim(-1, 1e1)
    upper_lim = 1.4*(sorted_c_metr.max().max())
    if (data_params['metric_tag'] == 'call_rate'):
        plt.xlim(-upper_lim/10, upper_lim)
    else:
        plt.xlim(-upper_lim/10, upper_lim)
        # plt.xlim(-1, 1.1e2)
    plt.grid(which='both')
    plt.ylabel(f'Error Ratio {dctag1}')
    plt.xlabel(f'{metrictag1}')
    plt.legend(loc=1)


def plot_all_dc_scheme_comps_log_scale(dc_activity_arr, c_activity_arr, data_params):
    plt.figure(figsize=(6*len(data_params['percent_ons']), 4.5*len(data_params['cycle_lengths'])))
    plt.rcParams.update({'font.size':14})
    for i, dc_col in enumerate(data_params["dc_tags"][1:]):
        plt.subplot(len(data_params['cycle_lengths']), len(data_params['percent_ons']), i+1)
        metric_col_name = f'{data_params["metric_tag"]} ({dc_col})'
        cycle_length = int(dc_col.split('of')[-1])
        cont_tag = f'{cycle_length}of{cycle_length}'
        cont_metric_col_name = f'{data_params["metric_tag"]} ({cont_tag})'
        metric_for_scheme = pd.DataFrame(dc_activity_arr.loc[:,metric_col_name].dropna())
        cont_column = pd.DataFrame(c_activity_arr.loc[:,cont_metric_col_name].dropna())
        metric_for_scheme_for_comparison = comp.get_associated_metric_for_cont_column(metric_for_scheme, cont_column)
        plt_dcmetr, plt_cmetr = comp.select_dates_from_metrics(metric_for_scheme_for_comparison, cont_column, data_params)
        plot_indiv_dc_control_comp_over_cycle_log_scale(plt_dcmetr, plt_cmetr, data_params)

    plt.tight_layout()
    plt.show()


def plot_all_dc_scheme_comp_errors_log_scale(dc_activity_arr, c_activity_arr, data_params):
    plt.figure(figsize=(6*len(data_params['percent_ons']), 4.5*len(data_params['cycle_lengths'])))
    plt.rcParams.update({'font.size':14})
    for i, dc_col in enumerate(data_params["dc_tags"][1:]):
        plt.subplot(len(data_params['cycle_lengths']), len(data_params['percent_ons']), i+1)
        metric_col_name = f'{data_params["metric_tag"]} ({dc_col})'
        cycle_length = int(dc_col.split('of')[-1])
        cont_tag = f'{cycle_length}of{cycle_length}'
        cont_metric_col_name = f'{data_params["metric_tag"]} ({cont_tag})'
        metric_for_scheme = pd.DataFrame(dc_activity_arr.loc[:,metric_col_name].dropna())
        cont_column = pd.DataFrame(c_activity_arr.loc[:,cont_metric_col_name].dropna())
        metric_for_scheme_for_comparison = comp.get_associated_metric_for_cont_column(metric_for_scheme, cont_column)
        plt_dcmetr, plt_cmetr = comp.select_dates_from_metrics(metric_for_scheme_for_comparison, cont_column, data_params)
        plot_indiv_dc_control_comp_error_over_cycle_log_scale(plt_dcmetr, plt_cmetr, data_params)

    plt.tight_layout()
    plt.show()


def plot_all_dc_scheme_comps_linear_scale(dc_activity_arr, c_activity_arr, data_params):
    plt.figure(figsize=(6*len(data_params['percent_ons']), 4.5*len(data_params['cycle_lengths'])))
    plt.rcParams.update({'font.size':14})
    for i, dc_col in enumerate(data_params["dc_tags"][1:]):
        plt.subplot(len(data_params['cycle_lengths']), len(data_params['percent_ons']), i+1)
        metric_col_name = f'{data_params["metric_tag"]} ({dc_col})'
        cycle_length = int(dc_col.split('of')[-1])
        cont_tag = f'{cycle_length}of{cycle_length}'
        cont_metric_col_name = f'{data_params["metric_tag"]} ({cont_tag})'
        metric_for_scheme = pd.DataFrame(dc_activity_arr.loc[:,metric_col_name].dropna())
        cont_column = pd.DataFrame(c_activity_arr.loc[:,cont_metric_col_name].dropna())
        metric_for_scheme_for_comparison = comp.get_associated_metric_for_cont_column(metric_for_scheme, cont_column)
        plt_dcmetr, plt_cmetr = comp.select_dates_from_metrics(metric_for_scheme_for_comparison, cont_column, data_params)
        plot_indiv_dc_control_comp_over_cycle_linear_scale(plt_dcmetr, plt_cmetr, data_params)

    plt.tight_layout()
    plt.show()


def plot_all_dc_scheme_comp_errors_linear_scale(dc_activity_arr, c_activity_arr, data_params):
    plt.figure(figsize=(6*len(data_params['percent_ons']), 4.5*len(data_params['cycle_lengths'])))
    plt.rcParams.update({'font.size':14})
    for i, dc_col in enumerate(data_params["dc_tags"][1:]):
        plt.subplot(len(data_params['cycle_lengths']), len(data_params['percent_ons']), i+1)
        metric_col_name = f'{data_params["metric_tag"]} ({dc_col})'
        cycle_length = int(dc_col.split('of')[-1])
        cont_tag = f'{cycle_length}of{cycle_length}'
        cont_metric_col_name = f'{data_params["metric_tag"]} ({cont_tag})'
        metric_for_scheme = pd.DataFrame(dc_activity_arr.loc[:,metric_col_name].dropna())
        actvtind_cont_column = pd.DataFrame(c_activity_arr.loc[:,cont_metric_col_name].dropna())
        metric_for_scheme_for_comparison = comp.get_associated_metric_for_cont_column(metric_for_scheme, actvtind_cont_column)
        plt_dcmetr, plt_cmetr = comp.select_dates_from_metrics(metric_for_scheme_for_comparison, actvtind_cont_column, data_params)
        plot_indiv_dc_control_comp_error_over_cycle_linear_scale(plt_dcmetr, plt_cmetr, data_params, drop_zero=True)

    plt.tight_layout()
    plt.show()