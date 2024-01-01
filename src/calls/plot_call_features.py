import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime as dt

import sys
sys.path.append(f"{Path(__file__).parents[0]}")
sys.path.append(f"{Path(__file__).parents[1]}")

from core import SITE_NAMES
import compute_features


def plot_call_spectrogram_centered(calls_sampled, call_signals, audio_info):
    call_info = calls_sampled.loc[audio_info['call_index']]
    fs = call_info['sampling_rate']
    call = call_signals[call_info['index']]
    padded_call = compute_features.pad_call_to_fortyms(call, fs)

    padded_call_dur = round(len(padded_call)/fs, 2)
    plt.title(audio_info['plot_title'], fontsize=12, weight='bold')
    plt.specgram(padded_call, NFFT=132, cmap='jet', vmin=-60)
    time_labels = (np.linspace(0,1000*padded_call_dur, 11).astype('int')).astype('str')
    time_labels[1::2] = ''
    plt.xticks(ticks=np.linspace(0,round(len(padded_call)/2), 11), labels=time_labels, rotation=45)
    ax = plt.gca()
    plt.text(x=0.4, y=0.05, s=f'SNR:{round(call_info["SNR"], 1)}', color='white', fontweight='bold', transform=ax.transAxes)

    low_end = 0
    high_end = 96000
    freq_labels = (np.linspace(low_end/1000, high_end/1000, 11)).astype('int').astype('str')
    if not(audio_info['show_yaxis_fine']):
        freq_labels[1::2] = ''
    plt.yticks(ticks=np.linspace(2*low_end/(fs), 2*high_end/(fs), 11), labels=freq_labels, rotation=45)
    plt.ylim(2*low_end/(fs), 2*high_end/(fs))
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (kHz)')
    plt.grid(which='both')


def plot_call_fft_interpolated(calls_sampled, call_signals, audio_info):
    call_info = calls_sampled.loc[audio_info['call_index']]
    fs = call_info['sampling_rate']
    call = call_signals[call_info['index']]
    
    interpolated_points_from_spectrum = compute_features.compute_fft_of_call(call, fs, audio_info['num_points'])
   
    fft_signal = interpolated_points_from_spectrum
    freqs_of_signal = np.arange(0, len(fft_signal))
    plt.title(audio_info['plot_title'], fontsize=12, weight='bold')
    plt.plot(freqs_of_signal, fft_signal, label=f'SNR:{round(call_info["SNR"], 1)}', color='blue')
    freq_labels = np.linspace(0, 192000/(2000), 9, dtype='int').astype('str')
    if not(audio_info['show_yaxis_fine']):
        freq_labels[1::2] = ''
    plt.xticks(ticks=np.linspace(0, len(fft_signal), 9), labels=freq_labels, rotation=45)
    plt.ylabel("FFT Magnitude (dB)")
    plt.xlabel("Frequency (kHz)")
    plt.xlim(0, len(fft_signal))
    plt.grid(which='both')
    plt.legend(loc='lower center')


def plot_call_welch_interpolated(calls_sampled, call_signals, audio_info):
    call_info = calls_sampled.loc[audio_info['call_index']]
    fs = call_info['sampling_rate']
    call = call_signals[call_info['index']]
    max_visible_frequency = 96000
    audio_info['max_freq_visible'] = max_visible_frequency

    interpolated_points_from_welch = compute_features.compute_welch_psd_of_call(call, fs, audio_info)

    welch_signal = interpolated_points_from_welch
    plt.title(audio_info['plot_title'], fontsize=12, weight='bold')
    plot_freqs = np.linspace(0, max_visible_frequency, len(welch_signal))
    plt.plot(plot_freqs, welch_signal, label=f'SNR:{round(call_info["SNR"], 1)}', color='blue')
    freq_labels = np.linspace(0, max_visible_frequency/(1000), 9, dtype='int').astype('str')
    if not(audio_info['show_yaxis_fine']):
        freq_labels[1::2] = ''
    plt.xticks(ticks=np.linspace(0, max_visible_frequency, 9), labels=freq_labels, rotation=45)
    plt.ylabel("FFT Magnitude (dB)")
    plt.xlabel("Frequency (kHz)")
    plt.xlim(0, max_visible_frequency)
    plt.grid(which='both')
    plt.legend(loc='lower center')


def plot_hundred_calls(calls_sampled, call_signals, site_key):
    side = 10
    call_indices = np.linspace(0, len(calls_sampled)-1, side**2).astype('int')
    plt.figure(figsize=(2.5*side, 2.5*side))
    plt.rcParams.update({'font.size': 12})
    plt.suptitle(f'{SITE_NAMES[site_key]} {side**2} call signals', y=1, fontsize=50)
    for subplot_i, call_index in enumerate(call_indices):
        call_info = calls_sampled.loc[call_index]
        file_name = call_info['file_name']
        datetime = dt.datetime.strptime(file_name, "%Y%m%d_%H%M%S.WAV")

        plt.subplot(side, side, subplot_i+1)
        audio_info = dict()
        audio_info['call_index'] = call_index
        audio_info['plot_title'] = f'{(datetime).strftime("%m/%d/%y %H:%M")}'
        audio_info['show_yaxis_fine'] = False
        plot_call_spectrogram_centered(calls_sampled, call_signals, audio_info)

    plt.tight_layout()
    plt.show()


def plot_hundred_ffts(calls_sampled, call_signals, site_key):
    side = 10
    call_indices = np.linspace(0, len(calls_sampled)-1, side**2).astype('int')
    num_points = 500
    plt.figure(figsize=(2.8*side, 2.8*side))
    plt.rcParams.update({'font.size': 12})
    plt.suptitle(f'{SITE_NAMES[site_key]} {side**2} FFT signals {num_points} points per signal)', y=1, fontsize=50)
    for subplot_i, call_index in enumerate(call_indices):
        call_info = calls_sampled.loc[call_index]
        file_name = call_info['file_name']
        datetime = dt.datetime.strptime(file_name, "%Y%m%d_%H%M%S.WAV")


        plt.subplot(side, side, subplot_i+1)
        audio_info = dict()
        audio_info['call_index'] = call_index
        audio_info['plot_title'] = f'{(datetime).strftime("%m/%d/%y %H:%M")}'
        audio_info['num_points'] = num_points
        audio_info['show_yaxis_fine'] = False
        plot_call_fft_interpolated(calls_sampled, call_signals, audio_info)
        
    plt.tight_layout()
    plt.show()


def plot_hundred_welch(calls_sampled, call_signals, site_key):
    side = 10
    call_indices = np.linspace(0, len(calls_sampled)-1, side**2).astype('int')

    num_points = 100
    plt.figure(figsize=(2.8*side, 2.8*side))
    plt.rcParams.update({'font.size': 12})
    plt.suptitle(f'{SITE_NAMES[site_key]} {side**2} Welch spectrum signals ({num_points} points per signal)', y=1, fontsize=50)
    for subplot_i, call_index in enumerate(call_indices):
        call_info = calls_sampled.loc[call_index]
        file_name = call_info['file_name']
        datetime = dt.datetime.strptime(file_name, "%Y%m%d_%H%M%S.WAV")


        plt.subplot(side, side, subplot_i+1)
        audio_info = dict()
        audio_info['call_index'] = call_index
        audio_info['plot_title'] = f'{(datetime).strftime("%m/%d/%y %H:%M")}'
        audio_info['num_points'] = num_points
        audio_info['show_yaxis_fine'] = False
        plot_call_welch_interpolated(calls_sampled, call_signals, audio_info)
        
    plt.tight_layout()
    plt.show()


def plot_side_by_side_calls_spectra(calls_sampled, call_signals):
    num_calls = 10
    call_indices = np.linspace(0, len(calls_sampled)-1, num_calls).astype('int')
    for call_index in call_indices:
        plt.figure(figsize=(12, 4))

        call_info = calls_sampled.loc[call_index]
        file_name = call_info['file_name']
        datetime = dt.datetime.strptime(file_name, "%Y%m%d_%H%M%S.WAV")
        
        audio_info = dict()
        plt.subplot(1, 3, 1)
        audio_info['call_index'] = call_index
        audio_info['plot_title'] = f'{(datetime).strftime("%m/%d/%y %H:%M")}'
        audio_info['show_yaxis_fine'] = True
        plot_call_spectrogram_centered(calls_sampled, call_signals, audio_info)

        plt.subplot(1, 3, 2)
        audio_info['num_points'] = 500
        audio_info['plot_title'] = f'FFT Spectrum ({audio_info["num_points"]} points)'
        plot_call_fft_interpolated(calls_sampled, call_signals, audio_info)

        plt.subplot(1, 3, 3)
        audio_info['num_points'] = 100
        audio_info['plot_title'] = f'Welch Spectrum ({audio_info["num_points"]} points)'
        plot_call_welch_interpolated(calls_sampled, call_signals, audio_info)
        
        plt.legend()
        plt.tight_layout()
        plt.show()