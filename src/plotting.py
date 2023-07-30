import matplotlib.pyplot as plt
from matplotlib import colors
import data_handling as dh

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append('../')


def rect(pos):
    r = plt.Rectangle(pos-0.505, 1, 1, facecolor="none", edgecolor="k", linewidth=0.6)
    plt.gca().add_patch(r)

def plot_activity_grid(activity_df, data_params, cfg):
    activity_times = pd.DatetimeIndex(activity_df.index).tz_localize('UTC')
    activity_dates = pd.DatetimeIndex(activity_df.columns).strftime("%m/%d")
    ylabel = 'UTC'
    if cfg["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        ylabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(20, 10))
    title = f"{data_params['type_tag'].upper()[:2]} Activity from {data_params['site_name']} ({data_params['cur_dc_tag']})"
    plt.title(title, loc='left', y=1.05)
    plt.imshow(1+activity_df, norm=colors.LogNorm(vmin=1, vmax=10e3))
    plt.yticks(np.arange(0, len(activity_df.index), 2)-0.5, plot_times[::2], rotation=50)
    plt.xticks(np.arange(0, len(activity_df.columns))-0.5, activity_dates, rotation=50)
    plt.ylabel(f'{ylabel} Time (HH:MM)')
    plt.xlabel('Date (MM/DD)')
    plt.colorbar()
    plt.tight_layout()
    if cfg["save_activity_grid"]:
        Path(f'{Path(__file__).resolve().parent}/../figures/{data_params["site_tag"]}/activity_grids').mkdir(parents=True, exist_ok=True)
        plot_name = f'{data_params["type_tag"].upper()}{data_params["site_tag"]}_{data_params["cur_dc_tag"]}_activity_grid.png'
        plt.savefig(f'{Path(__file__).resolve().parent}/../figures/{data_params["site_tag"]}/activity_grids/{plot_name}', bbox_inches='tight')
    if cfg["show_plots"]:
        plt.show()

def plot_presence_grid(presence_df, data_params, cfg):
    activity_times = pd.DatetimeIndex(presence_df.index).tz_localize('UTC')
    activity_dates = pd.DatetimeIndex(presence_df.columns).strftime("%m/%d")
    ylabel = 'UTC'
    if cfg["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        ylabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(20, 10))
    title = f"{data_params['type_tag'].upper()[:2]} Presence/Absence from {data_params['site_name']} ({data_params['cur_dc_tag']})"
    plt.title(title, loc='left', y=1.05)
    masked_array = np.ma.masked_where(presence_df == 1, presence_df)
    cmap = plt.get_cmap("Greys")  # Can be any colormap that you want after the cm
    cmap.set_bad(color=cfg["dc_color_mappings"][data_params['cur_dc_tag']], alpha=0.6)
    im = plt.imshow(masked_array, cmap=cmap)
    x, y = np.meshgrid(np.arange(presence_df.shape[1]), np.arange(presence_df.shape[0]))
    m = np.c_[x[presence_df == 1], y[presence_df == 1]]
    for pos in m:
        rect(pos)
    plt.ylabel(f"{ylabel} Time (HH:MM)")
    plt.xlabel('Date (MM/DD)')
    plt.yticks(np.arange(0, len(presence_df.index), 2)-0.5, plot_times[::2], rotation=50)
    plt.xticks(np.arange(0, len(presence_df.columns))-0.5, activity_dates, rotation=50)
    plt.grid(which="both", color='k')
    plt.tight_layout()
    if cfg["save_presence_grid"]:
        Path(f'{Path(__file__).resolve().parent}/../figures/{data_params["site_tag"]}/presence_grids').mkdir(parents=True, exist_ok=True)
        png_name = f'{data_params["type_tag"].upper()}{data_params["site_tag"]}_{data_params["cur_dc_tag"]}_presence_grid.png'
        plot_path = f'{data_params["site_tag"]}/presence_grids/{png_name}'
        plt.savefig(f'{Path(__file__).resolve().parent}/../figures/{plot_path}', bbox_inches='tight')
    if cfg["show_plots"]:
        plt.show()

def plot_dc_comparisons_per_night(activity_arr, data_params, cfg):

    datetimes = pd.to_datetime(activity_arr.index.values)
    dates = datetimes.strftime("%m/%d/%y").unique()
    times = datetimes.strftime("%H:%M").unique()

    activity_times = pd.DatetimeIndex(times).tz_localize('UTC')
    xlabel = 'UTC'
    if cfg["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        xlabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10*int(np.ceil(np.sqrt(len(dates)))), 2.5*len(dates)))

    for i, date in enumerate(dates):
        plt.subplot(int(np.ceil(np.sqrt(len(dates)))), int(np.ceil(np.sqrt(len(dates)))), i+1)
        for i, dc_tag in enumerate(data_params["dc_tags"]):
            activity_df = dh.construct_activity_grid(activity_arr, dc_tag)
            activity_of_date = activity_df[date]
            plt.title(f"{data_params['type_tag'].upper()[:2]} Activity from {data_params['site_name']} (Date : {date})")
            plt.bar(np.arange(0, len(activity_df.index))+i/(len(data_params['dc_tags'])), height=activity_of_date, width=1/(len(data_params['dc_tags'])), 
                    color=cfg["dc_color_mappings"][dc_tag], label=dc_tag, alpha=0.75, edgecolor='k')
        plt.grid(axis="y")
        plt.xticks(np.arange(0, len(activity_df.index))-0.5, plot_times, rotation=50)
        plt.ylabel('Number of Detections')
        plt.xlabel(f'{xlabel} Time (HH:MM)')
        plt.legend()

    plt.tight_layout()
    if cfg["save_dc_night_comparisons"]:
        Path(f'{Path(__file__).resolve().parent}/../figures/{data_params["site_tag"]}').mkdir(parents=True, exist_ok=True)
        plot_name = f'dc_comparisons_per_night_{data_params["type_tag"].upper()}{data_params["site_tag"]}.png'
        plt.savefig(f'{Path(__file__).resolve().parent}/../figures/{data_params["site_tag"]}/{plot_name}', bbox_inches='tight')
    if cfg["show_plots"]:
        plt.show()

def plot_dc_activity_comparisons_per_scheme(activity_arr, data_params, cfg):
    datetimes = pd.to_datetime(activity_arr.index.values)
    dates = datetimes.strftime("%m/%d").unique()
    times = datetimes.strftime("%H:%M").unique()

    activity_times = pd.DatetimeIndex(times).tz_localize('UTC')
    xlabel = 'UTC'
    if cfg["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        xlabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, (len(data_params["dc_tags"])**2)*np.sqrt(len(dates))/1.5))

    for i, dc_tag in enumerate(data_params['dc_tags']):
        activity_df = dh.construct_activity_grid(activity_arr, dc_tag).T
        plt.subplot(len(data_params['dc_tags']), 1, i+1)
        plt.title(f"{data_params['type_tag'].upper().replace('_', '')} Activity from {data_params['site_name']} (DC Tag : {dc_tag})")
        plt.imshow(1+activity_df, norm=colors.LogNorm(vmin=1, vmax=10e3))
        plt.xticks(np.arange(0, len(activity_df.columns))-0.5, plot_times, rotation=50)
        plt.yticks(np.arange(0, len(activity_df.index))-0.5, dates, rotation=50)
        plt.ylabel('Date (MM/DD)')
        plt.xlabel(f'{xlabel} Time (HH:MM)')
    plt.tight_layout()
    if cfg["save_activity_dc_comparisons"]:
        Path(f'{Path(__file__).resolve().parent}/../figures/{data_params["site_tag"]}').mkdir(parents=True, exist_ok=True)
        plot_name = f'activity_comparisons_per_dc_{data_params["type_tag"].upper()}{data_params["site_tag"]}.png'
        plt.savefig(f'{Path(__file__).resolve().parent}/../figures/{data_params["site_tag"]}/{plot_name}', bbox_inches='tight')
    if cfg["show_plots"]:
        plt.show()

def plot_dc_presence_comparisons_per_scheme(activity_arr, data_params, cfg):
    datetimes = pd.to_datetime(activity_arr.index.values)
    dates = datetimes.strftime("%m/%d").unique()
    times = datetimes.strftime("%H:%M").unique()

    activity_times = pd.DatetimeIndex(times).tz_localize('UTC')
    xlabel = 'UTC'
    if cfg["show_PST"]:
        activity_times = activity_times.tz_convert(tz='US/Pacific')
        xlabel = 'PST'
    plot_times = activity_times.strftime("%H:%M").unique()

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, (len(data_params["dc_tags"])**2)*np.sqrt(len(dates))/1.5))
    plot_colors = cfg["dc_color_mappings"]

    for i, dc_tag in enumerate(data_params['dc_tags']):
        presence_df = dh.construct_presence_grid(activity_arr, dc_tag).T
        plt.subplot(len(data_params["dc_tags"]), 1, i+1)
        plt.title(f"{data_params['type_tag'].upper()[:2]} Presence/Absence from {data_params['site_name']} (DC : {dc_tag})", loc='left', y=1.05)
        masked_array = np.ma.masked_where(presence_df == 1, presence_df)
        cmap = plt.get_cmap("Greys")  # Can be any colormap that you want after the cm
        cmap.set_bad(color=plot_colors[dc_tag], alpha=0.75)
        im = plt.imshow(masked_array, cmap=cmap)
        x, y = np.meshgrid(np.arange(presence_df.shape[1]), np.arange(presence_df.shape[0]))
        m = np.c_[x[presence_df == 1], y[presence_df == 1]]
        for pos in m:
            rect(pos)
        plt.ylabel('Date (MM/DD)')
        plt.xticks(np.arange(0, len(presence_df.columns), 1)-0.5, plot_times, rotation=50)
        plt.yticks(np.arange(0, len(presence_df.index))-0.5, dates, rotation=50)
        plt.grid(which="both", color='k')
        plt.xlabel(f"{xlabel} Time (HH:MM)")
    plt.tight_layout()
    if cfg["save_presence_dc_comparisons"]:
        Path(f'{Path(__file__).resolve().parent}/../figures/{data_params["site_tag"]}').mkdir(parents=True, exist_ok=True)
        plot_name = f'presence_comparisons_per_dc_{data_params["type_tag"].upper()}{data_params["site_tag"]}.png'
        plt.savefig(f'{Path(__file__).resolve().parent}/../figures/{data_params["site_tag"]}/{plot_name}', bbox_inches='tight')
    if cfg["show_plots"]:
        plt.show()