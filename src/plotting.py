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

def plot_activity_grid(activity_df, data_params, pipeline_params, file_paths):
    """
    Plots an activity grid generated from an activity summary for a specific duty-cycling scheme.
    """

    activity_times = pd.DatetimeIndex(activity_df.index).tz_localize('UTC')
    activity_dates = pd.DatetimeIndex(activity_df.columns).strftime("%m/%d")
    ylabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        ylabel = 'PST'
    plot_times = np.array(pd.DatetimeIndex(activity_times).strftime("%H:%M").unique())
    plot_times[1::2] = " "
    plot_dates = np.array(activity_dates.unique())
    plot_dates[1::2] = " "

    on = int(data_params['cur_dc_tag'].split('of')[0])
    total = int(data_params['cur_dc_tag'].split('of')[1])
    recover_ratio = total / on

    masked_array_for_nodets = np.ma.masked_where(activity_df.values==np.NaN, activity_df.values)
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='red')

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(len(activity_df.index), len(activity_df.index)//2))
    title = f"{data_params['type_tag'].upper()[:2]} Activity from {data_params['site_name']} ({data_params['cur_dc_tag']})"
    plt.title(title)
    plt.imshow(1+(recover_ratio*masked_array_for_nodets), cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=10e3))
    plt.yticks(np.arange(0, len(activity_df.index))-0.5, plot_times, rotation=50)
    plt.xticks(np.arange(0, len(activity_df.columns))-0.5, plot_dates, rotation=50)
    plt.ylabel(f'{ylabel} Time (HH:MM)')
    plt.xlabel('Date (MM/DD)')
    plt.colorbar()
    plt.tight_layout()
    if pipeline_params["save_activity_grid"]:
        plt.savefig(f'{file_paths["activity_grid_folder"]}/{file_paths["activity_grid_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()

def plot_presence_grid(presence_df, data_params, pipeline_params, file_paths):
    """
    Plots an presence grid generated from an activity summary for a specific duty-cycling scheme.
    """

    presence_df = presence_df.replace(np.NaN, 156)
    activity_times = pd.DatetimeIndex(presence_df.index).tz_localize('UTC')
    activity_dates = pd.DatetimeIndex(presence_df.columns)
    ylabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        ylabel = 'PST'
    plot_times = np.array(activity_times.strftime("%H:%M").unique())
    plot_times[1::2] = " "
    plot_dates = np.array(activity_dates.strftime("%m/%d").unique())
    plot_dates[1::2] = " "

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(len(presence_df.index), len(presence_df.index)//2))
    title = f"{data_params['type_tag'].upper()[:2]} Presence/Absence from {data_params['site_name']} ({data_params['cur_dc_tag']})"
    plt.title(title)
    masked_array = np.ma.masked_where(presence_df == 1, presence_df)
    cmap = plt.get_cmap("Greys")  # Can be any colormap that you want after the cm
    cmap.set_bad(color=DC_COLOR_MAPPINGS[data_params['cur_dc_tag']], alpha=0.75)
    plt.imshow(masked_array, cmap=cmap, vmin=0, vmax=255)
    x, y = np.meshgrid(np.arange(presence_df.shape[1]), np.arange(presence_df.shape[0]))
    m = np.c_[x[presence_df == 1], y[presence_df == 1]]
    for pos in m:
        rect(pos)
    plt.ylabel(f"{ylabel} Time (HH:MM)")
    plt.xlabel('Date (MM/DD)')
    plt.yticks(np.arange(0, len(presence_df.index))-0.5, plot_times, rotation=50)
    plt.xticks(np.arange(0, len(presence_df.columns))-0.5, plot_dates, rotation=50)
    plt.grid(which="both", color='k')
    plt.tight_layout()
    if pipeline_params["save_presence_grid"]:
        plt.savefig(f'{file_paths["presence_grid_folder"]}/{file_paths["presence_grid_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()

def plot_dc_comparisons_per_night(activity_arr, data_params, pipeline_params, file_paths):
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
        for i, dc_tag in enumerate(data_params["dc_tags"]):
            activity_df = dh.construct_activity_grid(activity_arr, dc_tag)
            on = int(dc_tag.split('of')[0])
            total = int(dc_tag.split('of')[1])
            recover_ratio = total / on
            activity_of_date = recover_ratio*activity_df[date]
            bar_width = 1/(len(data_params['dc_tags']))
            plt.title(f"{data_params['type_tag'].upper()[:2]} Activity from {data_params['site_name']} (Date : {date})", fontsize=24)
            plt.bar(np.arange(0, len(activity_df.index))+(bar_width*(i - 1)), height=activity_of_date, width=bar_width, 
                    color=DC_COLOR_MAPPINGS[dc_tag], label=dc_tag, alpha=0.75, edgecolor='k')
        plt.grid(axis="y")
        plt.xticks(np.arange(0, len(activity_df.index), 2)-0.5, plot_times[::2], rotation=50)
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

    plt.tight_layout()
    if pipeline_params["save_dc_night_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["dc_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()

def plot_dc_activity_comparisons_per_scheme(activity_arr, data_params, pipeline_params, file_paths):
    """
    Plots an activity grid for each duty-cycling scheme for a given location, looking at all datetimes in data/raw.
    """

    datetimes = pd.to_datetime(activity_arr.index.values)
    dates = datetimes.strftime("%m/%d").unique()
    times = datetimes.strftime("%H:%M").unique()

    activity_times = pd.DatetimeIndex(times).tz_localize('UTC')
    xlabel = 'UTC'
    if pipeline_params["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        xlabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(len(data_params["dc_tags"])*10, (len(data_params["dc_tags"])**2)*np.sqrt(len(dates))/1.5))

    for i, dc_tag in enumerate(data_params['dc_tags']):
        activity_df = (dh.construct_activity_grid(activity_arr, dc_tag))
        on = int(dc_tag.split('of')[0])
        total = int(dc_tag.split('of')[1])
        recover_ratio = total / on
        masked_array_for_nodets = np.ma.masked_where(activity_df.values==np.NaN, activity_df.values)
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='red')
        plot_dates = pd.to_datetime(activity_df.columns.values, format='%m/%d/%y').strftime("%m/%d").unique()
        plt.subplot(len(data_params['dc_tags']), 1, i+1)
        plt.title(f"{data_params['type_tag'].upper()[:2]} Activity from {data_params['site_name']} (DC Tag : {dc_tag})")
        plt.imshow(1+(recover_ratio*masked_array_for_nodets), cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=10e3))
        plt.xticks(np.arange(0, len(activity_df.columns), 2)-0.5, plot_dates[::2], rotation=45)
        plt.yticks(np.arange(0, len(activity_df.index), 2)-0.5, plot_times[::2], rotation=45)
        plt.xlabel('Date (MM/DD)')
        plt.ylabel(f'{xlabel} Time (HH:MM)')
    plt.tight_layout()
    if pipeline_params["save_activity_dc_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["activity_comparisons_figname"]}.png', bbox_inches='tight')
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
    plot_times = np.array(activity_times.strftime("%H:%M").unique())
    plot_times[1::2] = " "

    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(len(data_params["dc_tags"])*10, (len(data_params["dc_tags"])**2)*np.sqrt(len(dates))/1.5))

    for i, dc_tag in enumerate(data_params['dc_tags']):
        presence_df = dh.construct_presence_grid(activity_arr, dc_tag).replace(np.NaN, 156)
        plot_dates = np.array(pd.to_datetime(presence_df.columns.values, format='%m/%d/%y').strftime("%m/%d").unique())
        plot_dates[1::2] = ' '
        plt.subplot(len(data_params["dc_tags"]), 1, i+1)
        plt.title(f"{data_params['type_tag'].upper()[:2]} Presence/Absence from {data_params['site_name']} (DC : {dc_tag})")
        masked_array = np.ma.masked_where(presence_df == 1, presence_df)
        cmap = plt.get_cmap("Greys")  # Can be any colormap that you want after the cm
        cmap.set_bad(color=DC_COLOR_MAPPINGS[dc_tag], alpha=0.75)
        plt.imshow(masked_array, cmap=cmap, vmin=0, vmax=255)
        x, y = np.meshgrid(np.arange(presence_df.shape[1]), np.arange(presence_df.shape[0]))
        m = np.c_[x[presence_df == 1], y[presence_df == 1]]
        for pos in m:
            rect(pos)
        plt.ylabel(f"{xlabel} Time (HH:MM)")
        plt.xlabel('Date (MM/DD)')
        plt.xticks(np.arange(0, len(presence_df.columns))-0.5, plot_dates, rotation=45)
        plt.yticks(np.arange(0, len(presence_df.index))-0.5, plot_times, rotation=45)
        plt.grid(which="both", color='k')
    plt.tight_layout()
    if pipeline_params["save_presence_dc_comparisons"]:
        plt.savefig(f'{file_paths["figures_SITE_folder"]}/{file_paths["presence_comparisons_figname"]}.png', bbox_inches='tight')
    if pipeline_params["show_plots"]:
        plt.show()