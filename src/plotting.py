import matplotlib.pyplot as plt
from matplotlib import colors
import data_handling as dh

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append('../')

from core import DC_COLOR_MAPPINGS


def rect(pos):
    """
    Draws boxes in our presence plotting functions to show presence for date and time.
    Function borrowed from https://stackoverflow.com/questions/51432498/add-borders-to-grid-plot-based-on-value.
    """

    r = plt.Rectangle(pos-0.505, 1, 1, facecolor="none", edgecolor="k", linewidth=0.6)
    plt.gca().add_patch(r)

def plot_activity_grid_for_dets(activity_df, data_params, pipeline_params, file_paths):
    """
    Plots an activity grid generated from an activity summary for a specific duty-cycling scheme.
    """

    activity_times = pd.DatetimeIndex(activity_df.index).tz_localize('UTC')
    activity_dates = pd.DatetimeIndex(activity_df.columns).strftime("%m/%d/%y")
    ylabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        ylabel = 'PST'
    activity_times = activity_times.strftime("%H:%M")
    plot_times = [''] * len(activity_times)
    plot_times[::2] = activity_times[::2]
    plot_dates = [''] * len(activity_dates)
    plot_dates[::7] = activity_dates[::7]

    on = int(data_params['cur_dc_tag'].split('of')[0])
    total = int(data_params['cur_dc_tag'].split('of')[1])
    recover_ratio = total / on

    masked_array_for_nodets = np.ma.masked_where(activity_df.values==np.NaN, activity_df.values)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='red')

    plt.rcParams.update({'font.size': 1*len(activity_dates) + 0.5*len(activity_times)})
    plt.figure(figsize=(1.5*len(activity_dates), 1.5*len(activity_times)))
    title = f"{data_params['type_tag']} Activity (# of calls) from {data_params['site_name']} ({data_params['cur_dc_tag']})"
    plt.title(title, fontsize=1.5*len(activity_dates) + 1*len(activity_times))
    plt.imshow(1+(recover_ratio*masked_array_for_nodets), cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=10e3))
    plt.yticks(np.arange(0, len(activity_df.index))-0.5, plot_times, rotation=30)
    plt.xticks(np.arange(0, len(activity_df.columns))-0.5, plot_dates, rotation=30)
    plt.ylabel(f'{ylabel} Time (HH:MM)')
    plt.xlabel('Date (MM/DD/YY)')
    plt.colorbar()
    plt.tight_layout()
    if pipeline_params["save_activity_grid"]:
        plt.savefig(f'{file_paths["activity_grid_folder"]}/{file_paths["activity_dets_grid_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def plot_activity_grid_for_bouts(activity_df, data_params, pipeline_params, file_paths):

    activity_times = pd.DatetimeIndex(activity_df.index).tz_localize('UTC')
    activity_dates = pd.DatetimeIndex(activity_df.columns).strftime("%m/%d/%y")
    ylabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        ylabel = 'PST'
    activity_times = activity_times.strftime("%H:%M")
    plot_times = [''] * len(activity_times)
    plot_times[::2] = activity_times[::2]
    plot_dates = [''] * len(activity_dates)
    plot_dates[::7] = activity_dates[::7]

    on = int(data_params['cur_dc_tag'].split('of')[0])
    total = int(data_params['cur_dc_tag'].split('of')[1])
    recover_ratio = total / on

    masked_array_for_nodets = np.ma.masked_where(activity_df.values==np.NaN, activity_df.values)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='red')

    plt.rcParams.update({'font.size': (len(activity_dates) + 0.5*len(activity_times))})
    plt.figure(figsize=(1.5*len(activity_dates), 1.5*len(activity_times)))
    title = f"{data_params['type_tag']} Activity (% of time occupied by bouts) from {data_params['site_name']} (DC Tag: {data_params['cur_dc_tag']})"
    plt.title(title, fontsize=1.5*len(activity_dates) + 1*len(activity_times))
    plt.imshow(0.1+(recover_ratio*masked_array_for_nodets), cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=100))
    plt.yticks(np.arange(0, len(activity_df.index))-0.5, plot_times, rotation=30)
    plt.xticks(np.arange(0, len(activity_df.columns))-0.5, plot_dates, rotation=30)
    plt.ylabel(f'{ylabel} Time (HH:MM)')
    plt.xlabel('Date (MM/DD/YY)')
    plt.colorbar()
    plt.tight_layout()
    if pipeline_params["save_activity_grid"]:
        plt.savefig(f'{file_paths["activity_grid_folder"]}/{file_paths["activity_bouts_grid_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def plot_activity_grid_for_inds(activity_df, data_params, pipeline_params, file_paths):

    activity_times = pd.DatetimeIndex(activity_df.index).tz_localize('UTC')
    activity_dates = pd.DatetimeIndex(activity_df.columns).strftime("%m/%d/%y")
    ylabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        ylabel = 'PST'
    activity_times = activity_times.strftime("%H:%M")
    plot_times = [''] * len(activity_times)
    plot_times[::2] = activity_times[::2]
    plot_dates = [''] * len(activity_dates)
    plot_dates[::7] = activity_dates[::7]

    on = int(data_params['cur_dc_tag'].split('of')[0])
    total = int(data_params['cur_dc_tag'].split('of')[1])
    recover_ratio = total / on

    masked_array_for_nodets = np.ma.masked_where(activity_df.values==np.NaN, activity_df.values)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='red')

    plt.rcParams.update({'font.size': (len(activity_dates) + 0.5*len(activity_times))})
    plt.figure(figsize=(1.5*len(activity_dates), 1.5*len(activity_times)))
    time_block_duration = int(data_params['index_time_block_in_secs'])
    peak_index = (60*int(data_params['resolution_in_min'])/time_block_duration)
    title = f"{data_params['type_tag']} Activity Indices (time block = {time_block_duration}s) from {data_params['site_name']} (DC Tag: {data_params['cur_dc_tag']})"
    plt.title(title, fontsize=1.5*len(activity_dates) + 1*len(activity_times))
    if (time_block_duration >= 60):
        plt.imshow((recover_ratio*masked_array_for_nodets), cmap=cmap, vmin=0, vmax=peak_index)
    else:
        plt.imshow(1+(recover_ratio*masked_array_for_nodets), cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=1 + peak_index))
    plt.yticks(np.arange(0, len(activity_df.index))-0.5, plot_times, rotation=30)
    plt.xticks(np.arange(0, len(activity_df.columns))-0.5, plot_dates, rotation=30)
    plt.ylabel(f'{ylabel} Time (HH:MM)')
    plt.xlabel('Date (MM/DD/YY)')
    plt.colorbar()
    plt.tight_layout()
    if pipeline_params["save_activity_grid"]:
        plt.savefig(f'{file_paths["activity_grid_folder"]}/{file_paths["activity_inds_grid_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def plot_presence_grid(presence_df, data_params, pipeline_params, file_paths):
    """
    Plots an presence grid generated from an activity summary for a specific duty-cycling scheme.
    """

    presence_df = presence_df.replace(np.NaN, 156)
    activity_times = pd.DatetimeIndex(presence_df.index).tz_localize('UTC')
    activity_dates = pd.DatetimeIndex(presence_df.columns).strftime("%m/%d/%y")
    ylabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        ylabel = 'PST'
    activity_times = activity_times.strftime("%H:%M")
    plot_times = [''] * len(activity_times)
    plot_times[::2] = activity_times[::2]
    plot_dates = [''] * len(activity_dates)
    plot_dates[::7] = activity_dates[::7]

    plt.rcParams.update({'font.size': 1*len(activity_dates) + 0.5*len(activity_times)})
    plt.figure(figsize=(2*len(activity_dates), 2*len(activity_times)))
    title = f"{data_params['type_tag']} Presence/Absence from {data_params['site_name']} ({data_params['cur_dc_tag']})"
    plt.title(title, fontsize=1.5*len(activity_dates) + 1*len(activity_times))
    masked_array = np.ma.masked_where(presence_df == 1, presence_df)
    cmap = plt.get_cmap("Greys")  # Can be any colormap that you want after the cm
    cmap.set_bad(color=DC_COLOR_MAPPINGS[data_params['cur_dc_tag']], alpha=0.75)
    plt.imshow(masked_array, cmap=cmap, vmin=0, vmax=255)
    x, y = np.meshgrid(np.arange(presence_df.shape[1]), np.arange(presence_df.shape[0]))
    m = np.c_[x[presence_df == 1], y[presence_df == 1]]
    for pos in m:
        rect(pos)
    plt.ylabel(f"{ylabel} Time (HH:MM)")
    plt.xlabel('Date (MM/DD/YY)')
    plt.yticks(np.arange(0, len(presence_df.index))-0.5, plot_times, rotation=30)
    plt.xticks(np.arange(0, len(presence_df.columns))-0.5, plot_dates, rotation=30)
    plt.grid(which="both", color='k')
    plt.tight_layout()
    if pipeline_params["save_presence_grid"]:
        plt.savefig(f'{file_paths["presence_grid_folder"]}/{file_paths["presence_grid_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def plot_dc_dets_comparisons_per_night(activity_arr, data_params, pipeline_params, file_paths):
    """
    Plots a bar graph for each date comparing all duty-cycling schemes provided in a given location.
    """

    datetimes = pd.to_datetime(activity_arr.index.values)
    dates = datetimes.strftime("%m/%d/%y").unique()
    times = datetimes.strftime("%H:%M").unique()

    activity_times = pd.DatetimeIndex(times).tz_localize('UTC')
    xlabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        xlabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10*int(np.ceil(np.sqrt(len(dates)))),10*int(np.ceil(np.sqrt(len(dates))))))

    for i, date in enumerate(dates):
        plt.subplot(int(np.ceil(np.sqrt(len(dates)))), int(np.ceil(np.sqrt(len(dates)))), i+1)
        plt.title(f"{data_params['type_tag']} Activity from {data_params['site_name']} (Date : {date})", fontsize=24)
        day_max = 0
        for i, dc_tag in enumerate(data_params["dc_tags"]):
            activity_df = dh.construct_activity_grid_for_number_of_dets(activity_arr, dc_tag)
            on = int(dc_tag.split('of')[0])
            total = int(dc_tag.split('of')[1])
            recover_ratio = total / on
            activity_of_date = recover_ratio*activity_df[date]
            dc_day_max = np.max(activity_of_date)
            if (dc_day_max > day_max):
                day_max = dc_day_max
            bar_width = 1/(len(data_params['dc_tags']))
            plt.bar(np.arange(0, len(activity_df.index))+(bar_width*(i - 1)), height=activity_of_date, width=bar_width, 
                    color=DC_COLOR_MAPPINGS[dc_tag], label=dc_tag, alpha=0.75, edgecolor='k')
        if (day_max > 2000):
            day_yticks = np.arange(0, day_max+1, 1000).astype('int')
            plt.yticks(day_yticks, day_yticks, rotation=50)
        elif (day_max > 500 and day_max <= 2000):
            day_yticks = np.arange(0, day_max+1, 250).astype('int')
            plt.yticks(day_yticks, day_yticks, rotation=50)
        elif (day_max > 100 and day_max <= 500):
            day_yticks = np.arange(0, day_max+1, 50).astype('int')
            plt.yticks(day_yticks, day_yticks, rotation=50)
        elif (day_max > 10 and day_max <= 100):
            day_yticks = np.arange(0, day_max+1, 10).astype('int')
            plt.yticks(day_yticks, day_yticks, rotation=50)
        else:
            day_yticks = np.arange(0, day_max+1, 1).astype('int')
            plt.yticks(day_yticks, day_yticks, rotation=50)
        plt.grid(axis="y")
        plt.xticks(np.arange(0, len(activity_df.index), 2)-0.5, plot_times[::2], rotation=50)
        plt.xlim(plt.xticks()[0][0], plt.xticks()[0][-1])
        plt.ylabel(f'Number of Detections')
        plt.xlabel(f'{xlabel} Time (HH:MM)')
        plt.axvline(1.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(7.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(17.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        ax = plt.gca()
        plt.text(x=(1.5/21), y=0.56, s="Dusk", color='midnightblue', transform=ax.transAxes)
        plt.text(x=7.8/21,  y=0.56, s="Midnight", color='midnightblue', transform=ax.transAxes)
        plt.text(x=(18.3/21),  y=0.56, s="Dawn", color='midnightblue', transform=ax.transAxes)
        plt.legend()

    plt.tight_layout()
    if pipeline_params["save_dc_night_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["dc_det_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def plot_dc_bouts_comparisons_per_night(activity_arr, data_params, pipeline_params, file_paths):
    """
    Plots a bar graph for each date comparing all duty-cycling schemes provided in a given location.
    """

    datetimes = pd.to_datetime(activity_arr.index.values)
    dates = datetimes.strftime("%m/%d/%y").unique()
    times = datetimes.strftime("%H:%M").unique()

    activity_times = pd.DatetimeIndex(times).tz_localize('UTC')
    xlabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        xlabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(10*int(np.ceil(np.sqrt(len(dates)))),10*int(np.ceil(np.sqrt(len(dates))))))

    for i, date in enumerate(dates):
        plt.subplot(int(np.ceil(np.sqrt(len(dates)))), int(np.ceil(np.sqrt(len(dates)))), i+1)
        plt.title(f"{data_params['type_tag']} Activity from {data_params['site_name']} (Date : {date})", fontsize=24)
        for i, dc_tag in enumerate(data_params["dc_tags"]):
            activity_df = dh.construct_activity_grid_for_bouts(activity_arr, dc_tag)
            on = int(dc_tag.split('of')[0])
            total = int(dc_tag.split('of')[1])
            recover_ratio = total / on
            activity_of_date = recover_ratio*activity_df[date]
            bar_width = 1/(len(data_params['dc_tags']))
            plt.bar(np.arange(0, len(activity_df.index))+(bar_width*(i - 1)), height=activity_of_date, width=bar_width, 
                    color=DC_COLOR_MAPPINGS[dc_tag], label=dc_tag, alpha=0.75, edgecolor='k')
        plt.grid(axis="y")
        plt.xticks(np.arange(0, len(activity_df.index), 2)-0.5, plot_times[::2], rotation=50)
        plt.xlim(plt.xticks()[0][0], plt.xticks()[0][-1])
        plt.ylabel(f'% of Time Occupied by Bouts')
        plt.xlabel(f'{xlabel} Time (HH:MM)')
        plt.axvline(1.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(7.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(17.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        ax = plt.gca()
        plt.text(x=(1.5/21), y=0.56, s="Dusk", color='midnightblue', transform=ax.transAxes)
        plt.text(x=7.8/21,  y=0.56, s="Midnight", color='midnightblue', transform=ax.transAxes)
        plt.text(x=(18.3/21),  y=0.56, s="Dawn", color='midnightblue', transform=ax.transAxes)
        plt.legend()

    plt.tight_layout()
    if pipeline_params["save_dc_night_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["dc_bout_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def plot_dc_det_activity_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths):
    """
    Plots an activity grid for each duty-cycling scheme for a given location, looking at all datetimes in data/raw.
    """

    datetimes = pd.to_datetime(activity_arr.index.values)
    dates = datetimes.strftime("%m/%d").unique()
    times = datetimes.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': len(dates) + 0.5*len(times)})
    plt.figure(figsize=((5/3)*len(data_params['dc_tags'])*len(dates), (5/3)*len(data_params['dc_tags'])*len(times)))

    for i, dc_tag in enumerate(data_params['dc_tags']):
        activity_df = (dh.construct_activity_grid_for_number_of_dets(activity_arr, dc_tag))
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        masked_array_for_nodets = np.ma.masked_where(activity_df.values==np.NaN, activity_df.values)
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='red')
        activity_dates = pd.to_datetime(activity_df.columns.values, format='%m/%d/%y').strftime("%m/%d/%y").unique()
        activity_times = pd.to_datetime(activity_df.index.values, format='%H:%M').strftime("%H:%M").unique()
        activity_times = pd.DatetimeIndex(activity_times).tz_localize('UTC')
        xlabel = 'UTC'
        if pipeline_params["show_PST"]:
            activity_times = activity_times.tz_convert(tz='US/Pacific')
            xlabel = 'PST'
        activity_times = activity_times.strftime("%H:%M")
        plot_times = [''] * len(activity_times)
        plot_times[::2] = activity_times[::2]
        plot_dates = [''] * len(activity_dates)
        plot_dates[::7] = activity_dates[::7]
        plt.subplot(len(data_params['dc_tags']), 1, i+1)
        title = f"{data_params['type_tag']} Activity (# of calls) from {data_params['site_name']} (DC Tag : {dc_tag})"
        plt.title(title, fontsize=1.5*len(dates) + len(times))
        plt.imshow(1+(recover_ratio*masked_array_for_nodets), cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=10e3))
        plt.xticks(np.arange(0, len(plot_dates))-0.5, plot_dates, rotation=30)
        plt.yticks(np.arange(0, len(plot_times))-0.5, plot_times, rotation=30)
        plt.xlabel('Date (MM/DD/YY)')
        plt.ylabel(f'{xlabel} Time (HH:MM)')
    plt.tight_layout()
    if pipeline_params["save_activity_dc_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["activity_det_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def plot_dc_bout_activity_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths):
    datetimes = pd.to_datetime(activity_arr.index.values)
    dates = datetimes.strftime("%m/%d").unique()
    times = datetimes.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': len(dates) + 0.5*len(times)})
    plt.figure(figsize=((5/3)*len(data_params['dc_tags'])*len(dates), (5/3)*len(data_params['dc_tags'])*len(times)))

    for i, dc_tag in enumerate(data_params['dc_tags']):
        activity_df = (dh.construct_activity_grid_for_bouts(activity_arr, dc_tag))
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        masked_array_for_nodets = np.ma.masked_where(activity_df.values==np.NaN, activity_df.values)
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='red')
        activity_dates = pd.to_datetime(activity_df.columns.values, format='%m/%d/%y').strftime("%m/%d/%y").unique()
        activity_times = pd.to_datetime(activity_df.index.values, format='%H:%M').strftime("%H:%M").unique()
        activity_times = pd.DatetimeIndex(activity_times).tz_localize('UTC')
        xlabel = 'UTC'
        if pipeline_params["show_PST"]:
            activity_times = activity_times.tz_convert(tz='US/Pacific')
            xlabel = 'PST'
        activity_times = activity_times.strftime("%H:%M")
        plot_times = [''] * len(activity_times)
        plot_times[::2] = activity_times[::2]
        plot_dates = [''] * len(activity_dates)
        plot_dates[::7] = activity_dates[::7]
        plt.subplot(len(data_params['dc_tags']), 1, i+1)
        title = f"{data_params['type_tag']} Activity (% of time occupied by bouts) from {data_params['site_name']} (DC Tag : {dc_tag})"
        plt.title(title, fontsize=1.5*len(dates) + 1*len(times))
        plt.imshow(0.1+(recover_ratio*masked_array_for_nodets), cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=100))
        plt.xticks(np.arange(0, len(plot_dates))-0.5, plot_dates, rotation=30)
        plt.yticks(np.arange(0, len(plot_times))-0.5, plot_times, rotation=30)
        plt.xlabel('Date (MM/DD/YY)')
        plt.ylabel(f'{xlabel} Time (HH:MM)')
    plt.tight_layout()
    if pipeline_params["save_activity_dc_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["activity_bout_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def plot_dc_indices_activity_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths):
    datetimes = pd.to_datetime(activity_arr.index.values)
    dates = datetimes.strftime("%m/%d").unique()
    times = datetimes.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': len(dates) + 0.5*len(times)})
    plt.figure(figsize=((5/3)*len(data_params['dc_tags'])*len(dates), (5/3)*len(data_params['dc_tags'])*len(times)))

    for i, dc_tag in enumerate(data_params['dc_tags']):
        activity_df = (dh.construct_activity_grid_for_inds(activity_arr, dc_tag))
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        masked_array_for_nodets = np.ma.masked_where(activity_df.values==np.NaN, activity_df.values)
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='red')
        activity_dates = pd.to_datetime(activity_df.columns.values, format='%m/%d/%y').strftime("%m/%d/%y").unique()
        activity_times = pd.to_datetime(activity_df.index.values, format='%H:%M').strftime("%H:%M").unique()
        activity_times = pd.DatetimeIndex(activity_times).tz_localize('UTC')
        xlabel = 'UTC'
        if pipeline_params["show_PST"]:
            activity_times = activity_times.tz_convert(tz='US/Pacific')
            xlabel = 'PST'
        activity_times = activity_times.strftime("%H:%M")
        plot_times = [''] * len(activity_times)
        plot_times[::2] = activity_times[::2]
        plot_dates = [''] * len(activity_dates)
        plot_dates[::7] = activity_dates[::7]
        plt.subplot(len(data_params['dc_tags']), 1, i+1)
        time_block_duration = int(data_params['index_time_block_in_secs'])
        peak_index = (60*int(data_params['resolution_in_min'])/time_block_duration)
        title = f"{data_params['type_tag']} Activity Indices (time block = {time_block_duration}s) from {data_params['site_name']} (DC Tag : {dc_tag})"
        plt.title(title, fontsize=1.5*len(dates) + 1*len(times))
        if (time_block_duration >= 60):
            plt.imshow((recover_ratio*masked_array_for_nodets), cmap=cmap, vmin=0, vmax=peak_index)
        else:
            plt.imshow(1+(recover_ratio*masked_array_for_nodets), cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=1 + peak_index))
        plt.xticks(np.arange(0, len(plot_dates))-0.5, plot_dates, rotation=30)
        plt.yticks(np.arange(0, len(plot_times))-0.5, plot_times, rotation=30)
        plt.xlabel('Date (MM/DD/YY)')
        plt.ylabel(f'{xlabel} Time (HH:MM)')
    plt.tight_layout()
    if pipeline_params["save_activity_dc_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["activity_ind_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def plot_dc_presence_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths):
    """
    Plots a presence grid for each duty-cycling scheme for a given location, looking at all datetimes in data/raw.
    """

    datetimes = pd.to_datetime(activity_arr.index.values)
    dates = datetimes.strftime("%m/%d").unique()
    times = datetimes.strftime("%H:%M").unique()

    activity_times = pd.DatetimeIndex(times).tz_localize('UTC')
    xlabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        xlabel = 'PST'
    activity_times = activity_times.strftime("%H:%M")
    plot_times = [''] * len(activity_times)
    plot_times[::2] = activity_times[::2]

    plt.rcParams.update({'font.size': 1*len(dates) + 0.5*len(times)})
    plt.figure(figsize=((5/3)*len(data_params['dc_tags'])*len(dates), (5/3)*len(data_params['dc_tags'])*len(activity_times)))

    for i, dc_tag in enumerate(data_params['dc_tags']):
        presence_df = dh.construct_presence_grid(activity_arr, dc_tag).replace(np.NaN, 156)
        datetimes = pd.to_datetime(presence_df.columns.values, format='%m/%d/%y')
        activity_dates = datetimes.strftime("%m/%d/%y").unique()
        plot_dates = [''] * len(activity_dates)
        plot_dates[::7] = activity_dates[::7]
        plt.subplot(len(data_params["dc_tags"]), 1, i+1)
        title = f"{data_params['type_tag']} Presence/Absence from {data_params['site_name']} (DC : {dc_tag})"
        plt.title(title, fontsize=1.5*len(dates) + 1*len(times))
        masked_array = np.ma.masked_where(presence_df == 1, presence_df)
        cmap = plt.get_cmap("Greys")  # Can be any colormap that you want after the cm
        cmap.set_bad(color=DC_COLOR_MAPPINGS[dc_tag], alpha=0.75)
        plt.imshow(masked_array, cmap=cmap, vmin=0, vmax=255)
        x, y = np.meshgrid(np.arange(presence_df.shape[1]), np.arange(presence_df.shape[0]))
        m = np.c_[x[presence_df == 1], y[presence_df == 1]]
        for pos in m:
            rect(pos)
        plt.ylabel(f"{xlabel} Time (HH:MM)")
        plt.xlabel('Date (MM/DD/YY)')
        plt.xticks(np.arange(0, len(presence_df.columns))-0.5, plot_dates, rotation=30)
        plt.yticks(np.arange(0, len(presence_df.index))-0.5, plot_times, rotation=30)
        plt.grid(which="both", color='k')
    plt.tight_layout()
    if pipeline_params["save_presence_dc_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["presence_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def compare_metrics_per_night(activity_bouts_arr, activity_dets_arr, data_params, pipeline_params, file_paths):
    """
    Plots a bar graph for each date comparing all duty-cycling schemes provided in a given location.
    """

    datetimes = pd.to_datetime(activity_bouts_arr.index.values)
    dates = datetimes.strftime("%m/%d/%y").unique()
    times = datetimes.strftime("%H:%M").unique()

    activity_times = pd.DatetimeIndex(times).tz_localize('UTC')
    xlabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        xlabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 12.5})
    plt.figure(figsize=(5*int(np.ceil(np.sqrt(len(dates)))),5*int(np.ceil(np.sqrt(len(dates))))))

    for i, date in enumerate(dates):
        plt.subplot(int(np.ceil(np.sqrt(len(dates)))), int(np.ceil(np.sqrt(len(dates)))), i+1)
        plt.title(f"{data_params['type_tag']} Activity from {data_params['site_name']} (Date : {date})", fontsize=12.5)
        dc_tag = data_params['cur_dc_tag']

        activity_bouts_df = dh.construct_activity_grid_for_bouts(activity_bouts_arr, dc_tag)
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        bouts_of_date = (recover_ratio*activity_bouts_df[date])
        j=0
        bar_width = 1/2
        plt.bar(np.arange(0, len(activity_bouts_df.index))+(bar_width*(j - 0.5)), height=bouts_of_date, width=bar_width, 
                color='cyan', label=f'% of bout time', alpha=0.75, edgecolor='k')
        
        activity_dets_df = dh.construct_activity_grid_for_number_of_dets(activity_dets_arr, dc_tag)
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        detections_of_date = recover_ratio*activity_dets_df[date]
        normalized_detections_of_date = (100*(detections_of_date / (activity_dets_arr.max()[0])))

        j=1
        bar_width = 1/2
        plt.bar(np.arange(0, len(activity_dets_df.index))+(bar_width*(j - 0.5)), height=normalized_detections_of_date, width=bar_width, 
                color='orange', label=f'% of dets', alpha=0.75, edgecolor='k')
        
        plt.grid(axis="y")
        plt.xticks(np.arange(0, len(activity_bouts_df.index), 2)-0.5, plot_times[::2], rotation=50)
        plt.xlim(plt.xticks()[0][0], plt.xticks()[0][-1])
        plt.ylim(0, 100)
        plt.ylabel(f'Percentage (%)')
        plt.xlabel(f'{xlabel} Time (HH:MM)')
        plt.axvline(1.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(7.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(17.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        ax = plt.gca()
        plt.text(x=(1.5/21), y=0.56, s="Dusk", color='midnightblue', transform=ax.transAxes)
        plt.text(x=7.8/21,  y=0.56, s="Midnight", color='midnightblue', transform=ax.transAxes)
        plt.text(x=(18.3/21),  y=0.56, s="Dawn", color='midnightblue', transform=ax.transAxes)
        plt.legend()

    plt.tight_layout()
    if pipeline_params["save_dc_night_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["dc_metric_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()

def plot_numdets_n_percentbouts(activity_dets_arr, activity_bouts_arr, data_params, pipeline_params, file_paths):
    """
    Plots a bar graph for each date comparing all duty-cycling schemes provided in a given location.
    """

    datetimes = pd.to_datetime(activity_dets_arr.index.values)
    dates = datetimes.strftime("%m/%d/%y").unique()
    times = datetimes.strftime("%H:%M").unique()

    activity_times = pd.DatetimeIndex(times).tz_localize('UTC')
    xlabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        xlabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 12.5})

    plt.figure(figsize=(5*int(np.ceil(np.sqrt(2*len(dates)))),5*int(np.ceil(np.sqrt(2*len(dates))))))

    i = 0
    for date in dates:
        plt.subplot(int(np.ceil(np.sqrt(2*len(dates)))), int(np.ceil(np.sqrt(2*len(dates)))), i+1)
        plt.title(f"{data_params['type_tag']} Activity from {data_params['site_name']} (Date : {date})", fontsize=12.5)
        dc_tag = data_params['cur_dc_tag']

        activity_dets_df = dh.construct_activity_grid_for_number_of_dets(activity_dets_arr, dc_tag)
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        detections_of_date = recover_ratio*activity_dets_df[date]
        normalized_detections_of_date = (100*(detections_of_date / (activity_dets_arr.max()[0])))

        j=0
        bar_width = 1
        plt.bar(np.arange(0, len(activity_dets_df.index))+(bar_width*(j - 0.5)), height=detections_of_date, width=bar_width, 
                color='orange', label=f'% of dets', alpha=0.75, edgecolor='k')

        plt.grid(axis="y")
        plt.xticks(np.arange(0, len(activity_dets_df.index), 2)-0.5, plot_times[::2], rotation=50)
        plt.xlim(plt.xticks()[0][0], plt.xticks()[0][-1])
        plt.ylabel('Number of Detections')
        plt.xlabel(f'{xlabel} Time (HH:MM)')
        plt.axvline(1.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(7.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(17.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        ax = plt.gca()
        plt.text(x=(1.5/21), y=0.56, s="Dusk", color='midnightblue', transform=ax.transAxes)
        plt.text(x=7.8/21,  y=0.56, s="Midnight", color='midnightblue', transform=ax.transAxes)
        plt.text(x=(18.3/21),  y=0.56, s="Dawn", color='midnightblue', transform=ax.transAxes)
        plt.legend()


        plt.subplot(int(np.ceil(np.sqrt(2*len(dates)))), int(np.ceil(np.sqrt(2*len(dates)))), i+2)
        plt.title(f"{data_params['type_tag']} Activity from {data_params['site_name']} (Date : {date})", fontsize=12.5)
        dc_tag = data_params['cur_dc_tag']

        activity_bouts_df = dh.construct_activity_grid_for_bouts(activity_bouts_arr, dc_tag)
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        bouts_of_date = (recover_ratio*activity_bouts_df[date])
        j=0
        bar_width = 1
        plt.bar(np.arange(0, len(activity_bouts_df.index))+(bar_width*(j - 0.5)), height=bouts_of_date, width=bar_width, 
                color='cyan', label=f'% of bout time', alpha=0.75, edgecolor='k')

        plt.grid(axis="y")
        plt.xticks(np.arange(0, len(activity_bouts_df.index), 2)-0.5, plot_times[::2], rotation=50)
        plt.xlim(plt.xticks()[0][0], plt.xticks()[0][-1])
        plt.ylabel(f'Percentage (%)')
        plt.xlabel(f'{xlabel} Time (HH:MM)')
        plt.axvline(1.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(7.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(17.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        ax = plt.gca()
        plt.text(x=(1.5/21), y=0.56, s="Dusk", color='midnightblue', transform=ax.transAxes)
        plt.text(x=7.8/21,  y=0.56, s="Midnight", color='midnightblue', transform=ax.transAxes)
        plt.text(x=(18.3/21),  y=0.56, s="Dawn", color='midnightblue', transform=ax.transAxes)
        plt.legend()
        i+=2

    plt.tight_layout()
    if pipeline_params["save_dc_night_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["dc_metric_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def plot_numdets_n_activityinds(activity_dets_arr, activity_inds_arr, data_params, pipeline_params, file_paths):
    """
    Plots a bar graph for each date comparing all duty-cycling schemes provided in a given location.
    """

    datetimes = pd.to_datetime(activity_dets_arr.index.values)
    dates = datetimes.strftime("%m/%d/%y").unique()
    times = datetimes.strftime("%H:%M").unique()

    activity_times = pd.DatetimeIndex(times).tz_localize('UTC')
    xlabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        xlabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 12.5})

    plt.figure(figsize=(5*int(np.ceil(np.sqrt(2*len(dates)))),5*int(np.ceil(np.sqrt(2*len(dates))))))

    i = 0
    for date in dates:
        plt.subplot(int(np.ceil(np.sqrt(2*len(dates)))), int(np.ceil(np.sqrt(2*len(dates)))), i+1)
        plt.title(f"{data_params['type_tag']} Activity from {data_params['site_name']} (Date : {date})", fontsize=12.5)
        dc_tag = data_params['cur_dc_tag']

        activity_dets_df = dh.construct_activity_grid_for_number_of_dets(activity_dets_arr, dc_tag)
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        detections_of_date = recover_ratio*activity_dets_df[date]
        normalized_detections_of_date = (100*(detections_of_date / (activity_dets_arr.max()[0])))

        j=0
        bar_width = 1
        plt.bar(np.arange(0, len(activity_dets_df.index))+(bar_width*(j - 0.5)), height=detections_of_date, width=bar_width, 
                color='orange', label=f'# of detections', alpha=0.75, edgecolor='k')

        plt.grid(axis="y")
        plt.xticks(np.arange(0, len(activity_dets_df.index), 2)-0.5, plot_times[::2], rotation=50)
        plt.xlim(plt.xticks()[0][0], plt.xticks()[0][-1])
        plt.ylabel('Number of Detections')
        plt.xlabel(f'{xlabel} Time (HH:MM)')
        plt.axvline(1.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(7.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(17.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        ax = plt.gca()
        plt.text(x=(1.5/21), y=0.56, s="Dusk", color='midnightblue', transform=ax.transAxes)
        plt.text(x=7.8/21,  y=0.56, s="Midnight", color='midnightblue', transform=ax.transAxes)
        plt.text(x=(18.3/21),  y=0.56, s="Dawn", color='midnightblue', transform=ax.transAxes)
        plt.legend()


        plt.subplot(int(np.ceil(np.sqrt(2*len(dates)))), int(np.ceil(np.sqrt(2*len(dates)))), i+2)
        plt.title(f"{data_params['type_tag']} Activity from {data_params['site_name']} (Date : {date})", fontsize=12.5)
        dc_tag = data_params['cur_dc_tag']

        activity_inds_df = dh.construct_activity_grid_for_inds(activity_inds_arr, dc_tag)
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        bouts_of_date = (recover_ratio*activity_inds_df[date])
        j=0
        bar_width = 1
        plt.bar(np.arange(0, len(activity_inds_df.index))+(bar_width*(j - 0.5)), height=bouts_of_date, width=bar_width, 
                color='cyan', label=f'activity indices', alpha=0.75, edgecolor='k')

        plt.grid(axis="y")
        plt.xticks(np.arange(0, len(activity_inds_df.index), 2)-0.5, plot_times[::2], rotation=50)
        plt.xlim(plt.xticks()[0][0], plt.xticks()[0][-1])
        plt.ylabel(f'Activity Index')
        plt.xlabel(f'{xlabel} Time (HH:MM)')
        plt.axvline(1.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(7.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(17.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        ax = plt.gca()
        plt.text(x=(1.5/21), y=0.56, s="Dusk", color='midnightblue', transform=ax.transAxes)
        plt.text(x=7.8/21,  y=0.56, s="Midnight", color='midnightblue', transform=ax.transAxes)
        plt.text(x=(18.3/21),  y=0.56, s="Dawn", color='midnightblue', transform=ax.transAxes)
        plt.legend()
        i+=2

    plt.tight_layout()
    if pipeline_params["save_dc_night_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["dc_metric_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()


def plot_percentbouts_n_activityinds(activity_bouts_arr, activity_inds_arr, data_params, pipeline_params, file_paths):
    """
    Plots a bar graph for each date comparing all duty-cycling schemes provided in a given location.
    """

    datetimes = pd.to_datetime(activity_bouts_arr.index.values)
    dates = datetimes.strftime("%m/%d/%y").unique()
    times = datetimes.strftime("%H:%M").unique()

    activity_times = pd.DatetimeIndex(times).tz_localize('UTC')
    xlabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        xlabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 12.5})

    plt.figure(figsize=(5*int(np.ceil(np.sqrt(2*len(dates)))),5*int(np.ceil(np.sqrt(2*len(dates))))))

    i = 0
    for date in dates:
        plt.subplot(int(np.ceil(np.sqrt(2*len(dates)))), int(np.ceil(np.sqrt(2*len(dates)))), i+1)
        plt.title(f"{data_params['type_tag']} Activity from {data_params['site_name']} (Date : {date})", fontsize=12.5)
        dc_tag = data_params['cur_dc_tag']

        activity_bouts_df = dh.construct_activity_grid_for_bouts(activity_bouts_arr, dc_tag)
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        bouts_of_date = (recover_ratio*activity_bouts_df[date])
        j=0
        bar_width = 1
        plt.bar(np.arange(0, len(activity_bouts_df.index))+(bar_width*(j - 0.5)), height=bouts_of_date, width=bar_width, 
                color='orange', label=f'% of bout time', alpha=0.75, edgecolor='k')


        plt.grid(axis="y")
        plt.xticks(np.arange(0, len(activity_bouts_df.index), 2)-0.5, plot_times[::2], rotation=50)
        plt.xlim(plt.xticks()[0][0], plt.xticks()[0][-1])
        plt.ylabel(f'Percentage (%)')
        plt.xlabel(f'{xlabel} Time (HH:MM)')
        plt.axvline(1.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(7.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(17.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        ax = plt.gca()
        plt.text(x=(1.5/21), y=0.56, s="Dusk", color='midnightblue', transform=ax.transAxes)
        plt.text(x=7.8/21,  y=0.56, s="Midnight", color='midnightblue', transform=ax.transAxes)
        plt.text(x=(18.3/21),  y=0.56, s="Dawn", color='midnightblue', transform=ax.transAxes)
        plt.legend()


        plt.subplot(int(np.ceil(np.sqrt(2*len(dates)))), int(np.ceil(np.sqrt(2*len(dates)))), i+2)
        plt.title(f"{data_params['type_tag']} Activity from {data_params['site_name']} (Date : {date})", fontsize=12.5)
        dc_tag = data_params['cur_dc_tag']

        activity_inds_df = dh.construct_activity_grid_for_inds(activity_inds_arr, dc_tag)
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        bouts_of_date = (recover_ratio*activity_inds_df[date])
        j=0
        bar_width = 1
        plt.bar(np.arange(0, len(activity_inds_df.index))+(bar_width*(j - 0.5)), height=bouts_of_date, width=bar_width, 
                color='cyan', label=f'activity indices', alpha=0.75, edgecolor='k')

        plt.grid(axis="y")
        plt.xticks(np.arange(0, len(activity_inds_df.index), 2)-0.5, plot_times[::2], rotation=50)
        plt.xlim(plt.xticks()[0][0], plt.xticks()[0][-1])
        plt.ylabel(f'Activity Index')
        plt.xlabel(f'{xlabel} Time (HH:MM)')
        plt.axvline(1.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(7.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        plt.axvline(17.5, ymax=0.55, linestyle='dashed', color='midnightblue', alpha=0.6)
        ax = plt.gca()
        plt.text(x=(1.5/21), y=0.56, s="Dusk", color='midnightblue', transform=ax.transAxes)
        plt.text(x=7.8/21,  y=0.56, s="Midnight", color='midnightblue', transform=ax.transAxes)
        plt.text(x=(18.3/21),  y=0.56, s="Dawn", color='midnightblue', transform=ax.transAxes)
        plt.legend()
        i+=2

    plt.tight_layout()
    if pipeline_params["save_dc_night_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["dc_metric_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()