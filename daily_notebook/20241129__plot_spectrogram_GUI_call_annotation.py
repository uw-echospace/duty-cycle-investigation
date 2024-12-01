import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import soundfile as sf
import simpleaudio as sa
import pyaudio
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from pathlib import Path
from sklearn.cluster import KMeans
import scipy
from pathlib import Path
import fsspec

import sys
import re

sys.path.append(f"{Path(__file__).parent}/../src")
import activity.activity_assembly as actvt
from cli import get_file_paths

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FREQUENCY_COLOR_MAPPINGS = {
                    'LF' : 'cyan',
                    'HF' : 'orange'
                        }

def plot_spectrogram(ax, row, audio_seg, file_df_orig, fs, duration, start, FREQUENCY_COLOR_MAPPINGS, osn_file_path, row_index, nfft):
    """
    Plots a spectrogram for a given row of data with annotations.

    Parameters:
        ax: The matplotlib axis object.
        row: A row from the DataFrame containing details for the plot.
        location_df_kmeans_raw: DataFrame with detection details.
        fs: Sampling frequency (Hz).
        duration: Duration of the audio segment (seconds).
        start: Start time of the segment (seconds).
        FREQUENCY_COLOR_MAPPINGS: A dictionary mapping frequency groups to colors.
        osn_file_path: The path object for the file being processed.
    """
    # Plot the spectrogram
    ax.clear()
    ax.set_title(f'{osn_file_path.name}', fontsize=8)
    ax.specgram(audio_seg, NFFT=max(128, nfft), cmap='jet', vmin=-60, vmax=0)

    # Filter detection data for the current row
    plot_dets = file_df_orig[(file_df_orig['start_time'] >= start) & 
                             (file_df_orig['end_time'] <= (start + duration))]

    # Add rectangles for detections
    for _, det in plot_dets.iterrows():
        if det['start_time'] == row['start_time']:
            rect = patches.Rectangle(
                ((det['start_time'] - start) * (fs / 2), det['low_freq'] / (fs / 2)),
                (det['end_time'] - det['start_time']) * (fs / 2),
                (det['high_freq'] - det['low_freq']) / (fs / 2),
                linewidth=1, edgecolor='red', facecolor='none', alpha=0.8
            )
        else:
            rect = patches.Rectangle(
                ((det['start_time'] - start) * (fs / 2), det['low_freq'] / (fs / 2)),
                (det['end_time'] - det['start_time']) * (fs / 2),
                (det['high_freq'] - det['low_freq']) / (fs / 2),
                linewidth=1, edgecolor=FREQUENCY_COLOR_MAPPINGS[det['freq_group']], facecolor='none', alpha=0.6
            )
        ax.add_patch(rect)

    # Configure plot labels and ticks
    ax.set_yticks(ticks=np.linspace(0, 1, 6))
    ax.set_yticklabels(labels=np.linspace(0, fs / 2000, 6).astype('int'), fontsize=6)
    ax.set_ylabel("Frequency (kHz)", fontsize=6)

    ax.text(
        x=int(fs * 0.001), y=0.85,
        s=f'Det {row_index+1} ({row["freq_group"]}), SNR:{row["SNR"]:.2f}dB', fontweight='bold', color='white', fontsize=6
    )

    ax.set_xticks(ticks=np.linspace(0, duration * fs / 2, 6))
    ax.set_xticklabels(labels=np.round(np.linspace(0, 0 + duration, 6, dtype=float), 2), fontsize=6)
    ax.set_xlabel(f"Time (s + {start:.2f}s)", fontsize=6)

# GUI Application
class SpectrogramViewer:
    def __init__(self, root, dataframe, location_df_kmeans_raw, FREQUENCY_COLOR_MAPPINGS):
        self.root = root
        self.dataframe = dataframe
        self.index = 0
        self.location_df_kmeans_raw = location_df_kmeans_raw
        self.FREQUENCY_COLOR_MAPPINGS = FREQUENCY_COLOR_MAPPINGS
        self.cur_path = ''
        self.audio_file = ''
        self.fs = 250000

        # Default NFFT value
        self.nfft = tk.IntVar(value=256)

        # Track annotations for each spectrogram
        self.annotations = {idx: "none" for idx in dataframe.index}

        # Custom detections DataFrame
        self.custom_detections = pd.DataFrame(columns=["input_file", "start_time", "end_time", "low_freq", "high_freq", "category"])

        # Store the last drawn rectangle temporarily
        self.pending_rectangle = None

        # Drawing mode state
        self.drawing_mode = tk.BooleanVar(value=False)

        # Track the stack of drawn rectangles for undo functionality
        self.drawing_stack = []  # Holds references to the drawn rectangles

        # Current detection category for all new boxes in drawing mode
        self.current_category = tk.StringVar(value="custom_bat_call")

        # Frame for the plot and buttons
        self.frame = ttk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        # Plotting area
        self.fig, self.ax = plt.subplots(figsize=(2.5, 2.5)) 
        self.canvas = None

        # # Add index navigation box above the plot
        # self.index_frame = ttk.Frame(self.root)
        # self.index_frame.pack(pady=5)

        # Radiobuttons for annotations for the current detection
        self.radio_var = tk.StringVar(value="none")  # Tracks annotation for the current detection
        self.bat_calls_button = ttk.Radiobutton(
            self.frame, text="Bat Call", variable=self.radio_var, value="bat_call", command=self.set_annotation
        )
        self.feeding_buzzes_button = ttk.Radiobutton(
            self.frame, text="Feeding Buzz", variable=self.radio_var, value="feeding_buzz", command=self.set_annotation
        )
        self.neither_button = ttk.Radiobutton(
            self.frame, text="Neither", variable=self.radio_var, value="neither", command=self.set_annotation
        )

        self.bat_calls_button.grid(row=4, column=0, pady=5)
        self.feeding_buzzes_button.grid(row=4, column=1, pady=5)
        self.neither_button.grid(row=4, column=2, pady=5)

        # Dropdown for detection category during drawing mode
        self.category_label = ttk.Label(self.frame, text="Type of detections drawn:")
        self.category_label.grid(row=5, column=0, pady=5)

        self.category_dropdown = ttk.Combobox(
            self.frame, textvariable=self.current_category,
            values=["custom_bat_call", "custom_feeding_buzz", "custom_noise"], state="normal"
        )
        self.category_dropdown.grid(row=5, column=1, columnspan=2, pady=5)
        # Bind the validation method to the Combobox
        self.category_dropdown.bind("<<ComboboxSelected>>", self.validate_category)

        # Slider to adjust NFFT with tick marks
        self.nfft_label = ttk.Label(self.frame, text="NFFT")
        self.nfft_label.grid(row=3, column=1, pady=5, padx=(30, 0))

        self.nfft_slider = tk.Scale(
            self.frame,
            from_=0,
            resolution=128,
            to=4096,
            variable=self.nfft,
            orient="horizontal",
            command=self.update_nfft,
            length=150,
            tickinterval=2048  # Show tick marks at intervals of 256
        )
        self.nfft_slider.grid(row=3, column=1, columnspan=2, pady=5, padx=(110, 0))
        self.nfft_slider.set(256)

        self.playback_speed = tk.DoubleVar(value=1.0)  # Default playback speed
        # Slider for playback speed
        self.playback_speed_label = ttk.Label(self.frame, text="Speed")
        self.playback_speed_label.grid(row=3, column=0, pady=5, padx=(0, 100))

        self.playback_speed_slider = tk.Scale(
            self.frame,
            from_=0,  # Minimum playback speed
            to=2.0,     # Maximum playback speed
            resolution=0.1,  # Step size
            variable=self.playback_speed,
            orient="horizontal",
            length=150,
            tickinterval=0.5
        )
        self.playback_speed_slider.grid(row=3, column=0, columnspan=2, pady=5, padx=(0, 50))
        # Mode state variable
        self.is_drawing_mode = tk.BooleanVar(value=False)  # Default to "Browsing Mode"

        # Checkbutton for toggling Drawing Mode
        self.drawing_mode_toggle = ttk.Checkbutton(
            self.frame, text="Drawing Mode", variable=self.is_drawing_mode, command=self.toggle_mode)
        self.drawing_mode_toggle.grid(row=6, column=0, pady=5)

        # Undo button for drawing mode
        self.undo_button = ttk.Button(self.frame, text="Undo", command=self.undo_last_rectangle)
        self.undo_button.grid(row=6, column=2, pady=5)

        # Navigation Buttons
        self.left_button = ttk.Button(self.frame, text="Previous", command=self.previous_row)
        self.left_button.grid(row=2, column=0, padx=5, pady=5)
        self.right_button = ttk.Button(self.frame, text="Next", command=self.next_row)
        self.right_button.grid(row=2, column=2, padx=5, pady=5)

        # Save Button
        self.save_button = ttk.Button(self.frame, text="Save Annotations", command=self.save_annotations)
        self.save_button.grid(row=6, column=1, pady=5)

        # Add Play Button above the spectrogram area
        self.play_button = ttk.Button(self.frame, text="Play Audio", command=self.play_audio)
        self.play_button.grid(row=2, column=1, pady=5, padx=5)

        # Label for mouse coordinates
        self.coordinates_label = ttk.Label(self.frame, text="(x: , y: )")
        self.coordinates_label.grid(row=1, column=2, pady=5, padx=0)

        self.index_entry = ttk.Entry(self.frame, width=10)
        self.index_entry.grid(row=1, column=1, padx=5)

        self.index_button = ttk.Button(self.frame, text="Go to Detection:", command=self.go_to_index)
        self.index_button.grid(row=1, column=0, padx=5)

        # Initialize the first plot
        self.update_plot()


    def update_index_entry(self):
        """Update the index entry field to reflect the current index."""
        self.index_entry.delete(0, tk.END)
        self.index_entry.insert(0, str(self.index+1))

    def go_to_index(self):
        """Navigate to the specified detection index."""
        try:
            entered_index = int(self.index_entry.get())-1
            if 0 <= entered_index < len(self.dataframe):
                self.index = entered_index
                self.update_plot()  # Refresh the plot for the new index
                self.update_index_entry()  # Sync the entry box with the new index
                print(f"Moved to detection index: {self.index}")
            else:
                print(f"Invalid index: {entered_index}. Must be between 0 and {len(self.dataframe) - 1}.")
        except ValueError:
            print("Please enter a valid integer for the detection index.")

    def validate_category(self):
        """Validate the entered category."""
        user_input = self.category_dropdown.get()
        predefined_categories = ["custom_bat_call", "custom_feeding_buzz", "custom_noise"]

        if user_input not in predefined_categories:
            print(f"Warning: '{user_input}' is not a predefined category. Adding as a custom category.")
        else:
            print(f"Category set to: {user_input}")

    def play_audio(self):
        """Play the current audio segment at the system's default sample rate using PyAudio."""
        if self.audio_segment is None:
            print("No audio segment loaded to play.")
            return

        target_sample_rate = 44100  # Match MacBook Air Speakers sample rate
        playback_speed = max(0.1, self.playback_speed.get())
        stretch_factor = 1/playback_speed

        original_length = len(self.audio_segment)
        stretched_length = int(original_length * stretch_factor)
        original_time = np.linspace(0, original_length - 1, original_length, endpoint=False)
        stretched_time = np.linspace(0, original_length - 1, stretched_length, endpoint=False)
        interpolator = scipy.interpolate.interp1d(original_time, self.audio_segment, kind='linear', fill_value="extrapolate")
        stretched_signal = interpolator(stretched_time)

        # Resample the audio segment to match the target sample rate
        original_length = len(stretched_signal)
        target_length = int(original_length * (target_sample_rate / self.fs))
        resampled_audio = scipy.signal.resample(self.audio_segment, target_length)

        # Scale to int16 if using paInt16
        audio_data = (resampled_audio * 32767).astype(np.int16)

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        try:
            wave_obj = sa.WaveObject(audio_data.tobytes(), num_channels=1, bytes_per_sample=2, sample_rate=target_sample_rate)
            # Play the wave
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait until playback is finished
            print("Playback finished.")

        except Exception as e:
            print(f"Error during playback: {e}")

    def undo_last_rectangle(self):
        """Undo the last rectangle drawn."""
        if not self.drawing_stack:
            print("No rectangles to undo.")
            return
        
        # Pop the last rectangle and its associated index from the stack
        last_rectangle, last_index = self.drawing_stack.pop()
        
        # Remove the rectangle from the plot
        last_rectangle.remove()
        
        # Remove the corresponding entry from the custom_detections DataFrame
        self.custom_detections = self.custom_detections.drop(index=last_index).reset_index(drop=True)
        
        self.canvas.draw()
        print("Undid the last rectangle and removed its detection.")

    def set_annotation(self):
        """Update the annotation for the current spectrogram."""
        self.annotations[self.index] = self.radio_var.get()
        print(f"Set annotation for spectrogram {self.index} to {self.radio_var.get()}")

    def toggle_mode(self):
        """Toggle between Drawing and Browsing modes."""
        self.drawing_mode.set(not self.drawing_mode.get())
        if self.is_drawing_mode.get():
            self.enable_drawing()
            print("Switched to Drawing Mode.")
        else:
            self.disable_drawing()
            print("Switched to Browsing Mode.")

    def enable_drawing(self):
        """Enable interactive drawing on the spectrogram."""
        # Clear any lingering selectors
        self.clear_all_rectangle_selectors()

        # Create a new RectangleSelector
        self.rectangle_selector = RectangleSelector(
            self.ax, self.on_select, interactive=True, useblit=True,
            button=[1], minspanx=0, minspany=0, spancoords="pixels",
            props=dict(facecolor="none", edgecolor="yellow", alpha=0.8, linewidth=0.5),
            handle_props=dict(marker="o", markersize=0.5, markeredgecolor="none", markerfacecolor="none")
        )

        # Connect mouse motion event to update_coordinates
        self.mouse_motion_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.update_coordinates)
        print("Drawing mode enabled with a fresh RectangleSelector.")

    def update_coordinates(self, event):
        """Update the coordinates label with the mouse's current position."""
        if event.inaxes == self.ax:  # Check if the mouse is within the spectrogram plot
            x, y = event.xdata, event.ydata
            row = self.dataframe.iloc[self.index]
            call_dur = (row['end_time'] - row['start_time'])
            pad = min(min(row['start_time'] - call_dur, 1795 - row['end_time']), 9.0) / 3
            start = row['start_time'] - call_dur - (0.5 * pad)

            time = (x / (self.fs/2))
            freq = int(y * (self.fs / 2) / 1000)
            self.coordinates_label.config(text=f"({time:.2f}s, {freq}kHz)")
        else:
            self.coordinates_label.config(text="(x: , y: )")
            
    def disable_drawing(self):
        """Disable interactive drawing on the spectrogram."""
        # Disconnect mouse motion event
        if hasattr(self, 'mouse_motion_cid'):
            self.fig.canvas.mpl_disconnect(self.mouse_motion_cid)
        # Clear the RectangleSelector
        self.clear_all_rectangle_selectors()
        self.drawing_mode.set(False)
        print("Drawing mode disabled.")

    def clear_all_rectangle_selectors(self):
        """Clear all rectangle selectors and associated handles."""
        if hasattr(self, 'rectangle_selector') and self.rectangle_selector is not None:
            # Remove the rectangle itself
            if hasattr(self.rectangle_selector, "artists"):
                for artist in self.rectangle_selector.artists:
                    if artist in self.ax.patches:
                        artist.remove()

            # Remove the handles associated with the rectangle
            if hasattr(self.rectangle_selector, "_handles"):
                for handle in self.rectangle_selector._handles.artists:
                    if handle is not None and handle.axes:  # Check if handle is valid
                        handle.axes.remove_artist(handle)  # Remove the handle

            # Deactivate and reset the selector
            self.rectangle_selector.set_active(False)
            self.rectangle_selector = None
            self.canvas.draw()
            print("Cleared all RectangleSelector elements.")

    def on_select(self, eclick, erelease):
        """Handle rectangle drawing."""
        if not self.drawing_mode.get():
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        row = self.dataframe.iloc[self.index]
        call_dur = (row['end_time'] - row['start_time'])
        pad = min(min(row['start_time'] - call_dur, 1795 - row['end_time']), 9.0) / 3
        start = row['start_time'] - call_dur - (0.5 * pad)

        start_time = start + (min(x1, x2) / (self.fs / 2))
        end_time = start + (max(x1, x2) / (self.fs / 2))
        low_freq = min(y1, y2) * (self.fs / 2)
        high_freq = max(y1, y2) * (self.fs / 2)

        # Add rectangle to custom detections with the current category
        new_detection = pd.DataFrame([{
            "input_file": row['input_file'],
            "start_time": start_time,
            "end_time": end_time,
            "low_freq": low_freq,
            "high_freq": high_freq,
            "category": self.current_category.get(),  # Assign the selected category
        }])
        self.custom_detections = pd.concat([self.custom_detections, new_detection], ignore_index=True)
        detection_index = self.custom_detections.index[-1]

        # Draw green rectangle for confirmed detection
        x = min(x1, x2)
        width = abs(x2 - x1)
        y = min(y1, y2)
        height = abs(y2 - y1)

        rect = patches.Rectangle(
            (x, y), width, height, edgecolor="green", facecolor="none", linewidth=1, alpha=0.8
        )
        self.ax.add_patch(rect)
        self.drawing_stack.append((rect, detection_index))  # Add rectangle and its index to the stack
        self.canvas.draw()

        print(f"Added detection: Start {start_time:.3f}s, End {end_time:.3f}s, "
              f"Low {low_freq:.1f}kHz, High {high_freq:.1f}kHz, Category: {self.current_category.get()}")

    def update_nfft(self, value):
        """Update the NFFT value and redraw the current spectrogram."""
        # Snap the slider value to the nearest power of 2
        self.nfft.set(int(round(float(value) / 256) * 256))
        self.update_plot()

    def update_plot(self):
        """Update the plot and checkbox state for the current spectrogram."""
        print(f"Index: {self.index}, NFFT: {self.nfft.get()}")
        row = self.dataframe.iloc[self.index]
        file_path = '/'.join(Path(row['input_file']).parts[2:])
        cleaned_path = re.sub(r"(ubna_data_\d+)_mir", r"\1", file_path)
        osn_file_path = Path(f'bio230143-bucket01/{cleaned_path}')
        if self.cur_path != osn_file_path:
            self.cur_path = osn_file_path
            print(f"Loading new file {osn_file_path}")
            file = filesys.open(path=self.cur_path)
            self.audio_file = sf.SoundFile(file)
            self.fs = self.audio_file.samplerate
        call_dur = (row['end_time'] - row['start_time'])
        pad = min(min(row['start_time'] - call_dur, 1795 - row['end_time']), 9.0) / 3
        start = row['start_time'] - call_dur - (0.5 * pad)
        duration = (2 * call_dur) + (1 * pad)
        self.audio_file.seek(int(start * self.fs))
        self.audio_segment = self.audio_file.read(int(duration * self.fs))  # Numpy array
        file_df_orig = self.location_df_kmeans_raw[self.location_df_kmeans_raw['input_file'] == row['input_file']]
        
        # Plot the spectrogram
        plot_spectrogram(
            self.ax,
            row,
            self.audio_segment,
            file_df_orig=file_df_orig,
            fs=self.fs,
            duration=duration,
            start=start,
            FREQUENCY_COLOR_MAPPINGS=self.FREQUENCY_COLOR_MAPPINGS,
            osn_file_path=osn_file_path,
            row_index=self.index,
            nfft=self.nfft.get()
        )

        # Draw custom detections
        self.redraw_custom_detections(start, duration)

        # Adjust layout
        self.fig.tight_layout()

        # Redraw the canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, sticky="nsew")
        self.canvas.draw()

        # Update radiobutton state
        self.radio_var.set(self.annotations[self.index])  # Ensure this updates to the annotation for the current detection
        print(f"Updated plot for index: {self.index}, Current Detection Annotation: {self.radio_var.get()}")

        # Update the index_entry field with the current index
        self.update_index_entry()

    def redraw_custom_detections(self, start, duration):
        """Redraw custom detections on the current spectrogram."""
        visible_detections = self.custom_detections[
            (self.custom_detections["start_time"] < (start + duration)) &
            (self.custom_detections["end_time"] > start)
        ]

        for _, detection in visible_detections.iterrows():
            # Calculate rectangle coordinates relative to the current spectrogram
            rect_start = max(detection["start_time"], start)
            rect_end = min(detection["end_time"], start + duration)
            rect_low = detection["low_freq"]
            rect_high = detection["high_freq"]

            # Convert to plot coordinates
            x = (rect_start - start) * (self.fs / 2)
            width = (rect_end - rect_start) * (self.fs / 2)
            y = rect_low / (self.fs / 2)
            height = (rect_high - rect_low) / (self.fs / 2)

            # Draw the rectangle
            self.ax.add_patch(patches.Rectangle(
                (x, y), width, height, edgecolor="green", facecolor="none", linewidth=1, alpha=0.8
            ))

        print(f"Redrew {len(visible_detections)} custom detections.")

    def clear_rectangle_selector(self):
        """Clear the RectangleSelector."""
        if hasattr(self, 'rectangle_selector') and self.rectangle_selector is not None:
            self.rectangle_selector.disconnect_events()  # Disconnect events
            self.rectangle_selector.set_active(False)    # Deactivate
            self.rectangle_selector = None               # Remove reference
            print("Cleared RectangleSelector.")

    def previous_row(self):
        """Navigate to the previous spectrogram."""
        if self.index > 0:
            self.clear_rectangle_selector()  # Clear the current rectangle
            self.index -= 1
            print(f"Moved to previous row: {self.index}")
            self.update_plot()

    def next_row(self):
        """Navigate to the next spectrogram."""
        if self.index < len(self.dataframe) - 1:
            self.clear_rectangle_selector()  # Clear the current rectangle
            self.index += 1
            print(f"Moved to next row: {self.index}")
            self.update_plot()

    def save_annotations(self):
        """Save the annotated rows and custom detections."""
        bat_calls_indices = [idx for idx, annotation in self.annotations.items() if annotation == "bat_call"]
        feeding_buzzes_indices = [idx for idx, annotation in self.annotations.items() if annotation == "feeding_buzz"]
        neither_indices = [idx for idx, annotation in self.annotations.items() if annotation == "neither"]

        # Create DataFrames for each category
        bat_calls_df = self.dataframe.loc[bat_calls_indices]
        feeding_buzzes_df = self.dataframe.loc[feeding_buzzes_indices]
        neither_df = self.dataframe.loc[neither_indices]

        # Save to CSV
        bat_calls_df.to_csv(f"{Path(__file__).parent}/{Path(__file__).stem}_bat_calls.csv", index=False)
        feeding_buzzes_df.to_csv(f"{Path(__file__).parent}/{Path(__file__).stem}_feeding_buzzes.csv", index=False)
        neither_df.to_csv(f"{Path(__file__).parent}/{Path(__file__).stem}_neithers.csv", index=False)
        self.custom_detections.to_csv(f"{Path(__file__).parent}/{Path(__file__).stem}_custom_detections.csv", index=False)

        # Print confirmation
        print(f"Saved {len(bat_calls_indices)} rows to '{Path(__file__).stem}_bat_calls.csv'")
        print(f"Saved {len(feeding_buzzes_indices)} rows to '{Path(__file__).stem}_feeding_buzzes.csv'")
        print(f"Saved {len(neither_indices)} rows to '{Path(__file__).stem}_neithers.csv'")
        print(f"Saved {len(self.custom_detections)} custom detections to 'custom_detections.csv'") 

def add_frequency_group_to_file_dets(file_dets, location_classes):
    file_classes = location_classes[pd.to_datetime(location_classes['file_name'], 
                                                   format='%Y%m%d_%H%M%S.WAV', exact=False)==file_dets.name].copy()
    file_dets.insert(0, 'index_in_summary', file_dets.index)
    file_dets.set_index('index_in_file', inplace=True)
    classified = file_classes['KMEANS_CLASSES']!=''
    file_classes.loc[classified, 'peak_frequency'] = file_classes.loc[classified, 'peak_frequency'].astype('float64')
    file_dets.insert(0, 'peak_frequency', [np.NaN]*len(file_dets))
    file_dets.loc[file_classes['index_in_file'], 'freq_group'] = file_classes['KMEANS_CLASSES'].values
    file_dets.loc[file_classes['index_in_file'], 'peak_frequency'] = file_classes['peak_frequency'].values
    return file_dets

def add_frequency_groups_to_summary_using_kmeans(location_df, file_paths, data_params, save=True):
    location_df.insert(0, 'freq_group', '')
    location_classes = pd.read_csv(Path(file_paths['SITE_classes_file']), index_col=0)
    location_df.insert(0, 'input_file_dt', pd.to_datetime(location_df['input_file'], format='%Y%m%d_%H%M%S.WAV', exact=False))
    location_df_grouped = location_df.groupby('input_file_dt', group_keys=True)
    location_df_classified = location_df_grouped.apply(lambda x: add_frequency_group_to_file_dets(x, location_classes))
    location_df_only_classified = location_df_classified.loc[location_df_classified['freq_group']!='']
    location_df_only_classified = location_df_only_classified.droplevel(level=0)
    location_df_only_classified = location_df_only_classified.reset_index()
    if data_params['type_tag'] != '':
        location_df_only_classified = location_df_only_classified.loc[location_df_only_classified['freq_group']==data_params['type_tag']]
    if save:
        location_df_only_classified.to_csv(f'{file_paths["SITE_folder"]}/{file_paths["detector_TYPE_SITE_YEAR"]}.csv')
    return location_df_only_classified

def get_dropped_by_kmeans(thresh_file_df, all_file_kmeans_df):
    input_file_group_name = thresh_file_df.input_file.values[0]
    thresh_file_df = thresh_file_df.set_index('index_in_file')
    kmeans_file_df = all_file_kmeans_df[all_file_kmeans_df['input_file']==input_file_group_name]
    kmeans_file_df = kmeans_file_df.set_index('index_in_file')
    dropped_inds = sorted(list(set(thresh_file_df.index) - set(kmeans_file_df.index)))
    return thresh_file_df.loc[dropped_inds]


def get_section_of_call_in_file(detection, audio_file):
    fs = audio_file.samplerate
    call_dur = (detection['end_time'] - detection['start_time'])
    pad = min(min(detection['start_time'] - call_dur, 1795 - detection['end_time']), 0.006) / 3
    start = detection['start_time'] - call_dur - (3*pad)
    duration = (2 * call_dur) + (4*pad)
    audio_file.seek(int(fs*start))
    audio_seg = audio_file.read(int(fs*duration))
    length_of_section = call_dur + (2*pad)
    return audio_seg, length_of_section

if __name__ == "__main__":
    data_params = dict()
    data_params["site_name"] = 'Foliage'
    data_params["site_tag"] = 'Foliage'
    data_params["type_tag"] = ''
    data_params["detector_tag"] = 'bd2'
    data_params["assembly_type"] = 'kmeans'

    file_paths = get_file_paths(data_params)
    location_df_kmeans = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["detector_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)
    test_df = location_df_kmeans.reset_index()

    filesys = fsspec.filesystem('s3', anon=True, client_kwargs={'endpoint_url': 'https://sdsc.osn.xsede.org'})

    # Launch the GUI
    root = tk.Tk()
    root.title("Spectrogram Viewer")
    app = SpectrogramViewer(
        root,
        dataframe=test_df,
        location_df_kmeans_raw=location_df_kmeans,
        FREQUENCY_COLOR_MAPPINGS=FREQUENCY_COLOR_MAPPINGS,
    )
    root.mainloop()
