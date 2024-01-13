import numpy as np
import scipy


def pad_call_ms(call, fs, window_dur):
    window_samples = int(fs*window_dur)
    call_samples = len(call)
    padded_call = np.pad(call, np.ceil((window_samples - call_samples)/2).astype(int), mode='constant', constant_values=1e-6)

    return padded_call


def compute_energy_of_call(call, fs, num_points):
    energy = call**2
    energy_db = 10*np.log10(energy)

    original_freq_vector = np.arange(0, len(energy_db), 1).astype('int')
    common_freq_vector = np.linspace(0, len(energy_db)-1, num_points).astype('int')
    interp_kind = 'linear'
    interpolated_points_from_energy = scipy.interpolate.interp1d(original_freq_vector, energy_db, kind=interp_kind)(common_freq_vector)
    interpolated_points_from_energy = interpolated_points_from_energy - interpolated_points_from_energy.min()

    return interpolated_points_from_energy


def compute_fft_of_call(call, fs, num_points):
    audio_spectrum = scipy.fft.rfft(call)
    freqs = len(audio_spectrum)
    audio_spectrum_mag = np.abs(audio_spectrum[:int(freqs*(192000/fs))])
    audio_spectrum_db =  20*np.log10(audio_spectrum_mag)
    normalized_audio_spectrum_db = audio_spectrum_db - audio_spectrum_db.max()

    thresh = -100
    peak_db = np.zeros(len(normalized_audio_spectrum_db))+thresh
    peak_db[normalized_audio_spectrum_db>=thresh] = normalized_audio_spectrum_db[normalized_audio_spectrum_db>=thresh]

    original_freq_vector = np.arange(0, len(peak_db), 1).astype('int')
    common_freq_vector = np.linspace(0, len(peak_db)-1, num_points).astype('int')
    interp_kind = 'linear'
    interpolated_points_from_spectrum = scipy.interpolate.interp1d(original_freq_vector, peak_db, kind=interp_kind)(common_freq_vector)

    return interpolated_points_from_spectrum


def compute_welch_psd_of_call(call, fs, audio_info):
    freqs, welch = scipy.signal.welch(call, fs=fs, detrend=False, scaling='spectrum')
    cropped_welch = welch[(freqs<=audio_info['max_freq_visible'])]
    audio_spectrum_mag = np.abs(cropped_welch)
    audio_spectrum_db =  10*np.log10(audio_spectrum_mag)
    normalized_audio_spectrum_db = audio_spectrum_db - audio_spectrum_db.max()

    thresh = -100
    peak_db = np.zeros(len(normalized_audio_spectrum_db))+thresh
    peak_db[normalized_audio_spectrum_db>=thresh] = normalized_audio_spectrum_db[normalized_audio_spectrum_db>=thresh]
    
    original_freq_vector = np.arange(0, len(peak_db), 1).astype('int')
    common_freq_vector = np.linspace(0, len(peak_db)-1, audio_info['num_points']).astype('int')
    interp_kind = 'linear'
    interpolated_points_from_welch = scipy.interpolate.interp1d(original_freq_vector, peak_db, kind=interp_kind)(common_freq_vector)

    return interpolated_points_from_welch


def generate_ffts_for_calls(calls_sampled, call_signals):
    fft_signals = []
    for call_index, call_info in calls_sampled.iterrows():
        num_points = 500
        call_info = calls_sampled.loc[call_index]
        fs = call_info['sampling_rate']
        call = call_signals[call_info['index']]
        interpolated_points_from_spectrum = compute_fft_of_call(call, fs, num_points)
        fft_signals.append(interpolated_points_from_spectrum)

    fft_signals = np.vstack(fft_signals)

    return fft_signals


def generate_welchs_for_calls(calls_sampled, call_signals):
    welch_signals = []
    for call_index, call_info in calls_sampled.iterrows():
        audio_info = dict()
        audio_info['max_freq_visible'] = 96000
        audio_info['num_points'] = 100
        call_info = calls_sampled.loc[call_index]
        fs = call_info['sampling_rate']
        call = call_signals[call_info['index']]
        interpolated_points_from_welch = compute_welch_psd_of_call(call, fs, audio_info)
        welch_signals.append(interpolated_points_from_welch)

    welch_signals = np.vstack(welch_signals)

    return welch_signals