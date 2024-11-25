import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import soundfile as sf
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from pathlib import Path
from sklearn.cluster import KMeans
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
    ax.specgram(audio_seg, NFFT=nfft, cmap='jet', vmin=-60, vmax=0)

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
                linewidth=0.5, edgecolor='red', facecolor='none', alpha=0.8
            )
        else:
            rect = patches.Rectangle(
                ((det['start_time'] - start) * (fs / 2), det['low_freq'] / (fs / 2)),
                (det['end_time'] - det['start_time']) * (fs / 2),
                (det['high_freq'] - det['low_freq']) / (fs / 2),
                linewidth=0.5, edgecolor=FREQUENCY_COLOR_MAPPINGS[det['freq_group']], facecolor='none', alpha=0.8
            )
        ax.add_patch(rect)

    # Configure plot labels and ticks
    ax.set_yticks(ticks=np.linspace(0, 1, 6))
    ax.set_yticklabels(labels=np.linspace(0, fs / 2000, 6).astype('int'), fontsize=6)
    ax.set_ylabel("Frequency (kHz)", fontsize=6)

    ax.text(
        x=int(fs * 0.001), y=0.85,
        s=f'{row["freq_group"]} det{row_index}', fontweight='bold', color='white', fontsize=6
    )

    ax.set_xticks(ticks=np.linspace(0, duration * fs / 2, 6))
    ax.set_xticklabels(labels=np.round(np.linspace(start, start + duration, 6, dtype=float), 1), fontsize=6)
    ax.set_xlabel("Time (s)", fontsize=6)

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
        self.current_category = tk.StringVar(value="bat_calls")

        # Main Header
        self.header_label = ttk.Label(root, text="Spectrogram Viewer", font=("Arial", 20, "bold"))
        self.header_label.pack(pady=10)

        # Frame for the plot and buttons
        self.frame = ttk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        # Plotting area
        self.fig, self.ax = plt.subplots(figsize=(2.5, 2.5)) 
        self.canvas = None

        # Radiobuttons for annotations for the current detection
        self.radio_var = tk.StringVar(value="none")  # Tracks annotation for the current detection
        self.bat_calls_button = ttk.Radiobutton(
            self.frame, text="Bat Calls", variable=self.radio_var, value="bat_calls", command=self.set_annotation
        )
        self.feeding_buzzes_button = ttk.Radiobutton(
            self.frame, text="Feeding Buzzes", variable=self.radio_var, value="feeding_buzzes", command=self.set_annotation
        )
        self.noise_button = ttk.Radiobutton(
            self.frame, text="Noise", variable=self.radio_var, value="noise", command=self.set_annotation
        )

        self.bat_calls_button.grid(row=2, column=0, pady=5)
        self.feeding_buzzes_button.grid(row=2, column=1, pady=5)
        self.noise_button.grid(row=2, column=2, pady=5)

        # Dropdown for detection category during drawing mode
        self.category_label = ttk.Label(self.frame, text="Category:")
        self.category_label.grid(row=3, column=0, pady=5)

        self.category_dropdown = ttk.Combobox(
            self.frame, textvariable=self.current_category,
            values=["bat_calls", "feeding_buzzes", "noise"], state="readonly"
        )
        self.category_dropdown.grid(row=3, column=1, columnspan=2, pady=5)

        # Slider to adjust NFFT with tick marks
        self.nfft_label = ttk.Label(self.frame, text="NFFT:")
        self.nfft_label.grid(row=4, column=0, pady=5)

        self.nfft_slider = tk.Scale(
            self.frame,
            from_=256,
            to=2048,
            variable=self.nfft,
            orient="horizontal",
            command=self.update_nfft,
            length=300,
            tickinterval=256  # Show tick marks at intervals of 256
        )
        self.nfft_slider.grid(row=4, column=1, columnspan=2, pady=5)
        self.nfft_slider.set(256)

        # Button to toggle drawing mode
        self.toggle_drawing_button = ttk.Button(self.frame, text="Toggle Drawing Mode", command=self.toggle_drawing_mode)
        self.toggle_drawing_button.grid(row=5, column=0, pady=5)

        # Undo button for drawing mode
        self.undo_button = ttk.Button(self.frame, text="Undo", command=self.undo_last_rectangle)
        self.undo_button.grid(row=5, column=1, pady=5)

        # Navigation Buttons
        self.left_button = ttk.Button(self.frame, text="Previous", command=self.previous_row)
        self.left_button.grid(row=6, column=0, padx=5, pady=5)

        self.right_button = ttk.Button(self.frame, text="Next", command=self.next_row)
        self.right_button.grid(row=6, column=2, padx=5, pady=5)

        # Save Button
        self.save_button = ttk.Button(self.frame, text="Save Annotations", command=self.save_annotations)
        self.save_button.grid(row=7, column=1, pady=10)

        # Initialize the first plot
        self.update_plot()

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

    def toggle_drawing_mode(self):
        """Toggle drawing mode on or off."""
        self.drawing_mode.set(not self.drawing_mode.get())
        if self.drawing_mode.get():
            print("Drawing mode enabled. Draw rectangles to add detections.")
            self.enable_drawing()
        else:
            print("Drawing mode disabled.")
            self.disable_drawing()

    # def enable_drawing(self):
    #     """Enable interactive drawing on the spectrogram."""
    #     # Remove any lingering RectangleSelector
    #     if hasattr(self, 'rectangle_selector') and self.rectangle_selector is not None:
    #         print('Removing any lingering RectangleSelector when enabling Drawing Mode')
    #         self.clear_rectangle_selector()

    #     # Create a new RectangleSelector
    #     self.rectangle_selector = RectangleSelector(
    #         self.ax, self.on_select, interactive=True, useblit=True,
    #         button=[1], minspanx=0, minspany=0, spancoords="pixels",
    #         props=dict(facecolor="none", edgecolor="yellow", alpha=0.8, linewidth=0.5),
    #         handle_props=dict(marker="o", markersize=0.5, markeredgecolor="yellow", markerfacecolor="yellow")
    #     )
    #     self.canvas.draw()
    #     print("Drawing mode enabled. Fresh RectangleSelector created.")

    # def disable_drawing(self):
    #     """Disable interactive drawing on the spectrogram."""
    #     if hasattr(self, 'rectangle_selector') and self.rectangle_selector is not None:
    #         self.rectangle_selector.disconnect_events()  # Remove all RectangleSelector events
    #         self.rectangle_selector.set_active(False)    # Deactivate the selector
    #         self.rectangle_selector = None               # Remove reference
    #         print("RectangleSelector fully removed.")
        
    #     # Redraw the canvas to remove any lingering yellow rectangles or handles
    #     self.canvas.draw()
    #     print("Drawing mode disabled. Canvas refreshed.")

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
        print("Drawing mode enabled with a fresh RectangleSelector.")

    def disable_drawing(self):
        """Disable interactive drawing on the spectrogram."""
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
        pad = min(min(row['start_time'] - call_dur, 1795 - row['end_time']), 3.0) / 3
        start = row['start_time'] - call_dur - (0.5 * pad)
        duration = (2 * call_dur) + (1 * pad)
        start_time = start + (min(x1, x2) / (self.fs / 2)) * duration
        end_time = start + (max(x1, x2) / (self.fs / 2)) * duration
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
        x = (start_time - start) * (self.fs / 2 / duration)
        width = (end_time - start_time) * (self.fs / 2 / duration)
        y = low_freq / (self.fs / 2)
        height = (high_freq - low_freq) / (self.fs / 2)

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
        pad = min(min(row['start_time'] - call_dur, 1795 - row['end_time']), 3.0) / 3
        start = row['start_time'] - call_dur - (0.5 * pad)
        duration = (2 * call_dur) + (1 * pad)
        self.audio_file.seek(int(self.fs * start))
        audio_seg = self.audio_file.read(int(self.fs * duration))
        file_df_orig = self.location_df_kmeans_raw[self.location_df_kmeans_raw['input_file'] == row['input_file']]
        
        # Plot the spectrogram
        plot_spectrogram(
            self.ax,
            row,
            audio_seg,
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
        bat_calls_indices = [idx for idx, annotation in self.annotations.items() if annotation == "bat_calls"]
        feeding_buzzes_indices = [idx for idx, annotation in self.annotations.items() if annotation == "feeding_buzzes"]
        noise_indices = [idx for idx, annotation in self.annotations.items() if annotation == "noise"]

        # Create DataFrames for each category
        bat_calls_df = self.dataframe.loc[bat_calls_indices]
        feeding_buzzes_df = self.dataframe.loc[feeding_buzzes_indices]
        noise_df = self.dataframe.loc[noise_indices]

        # Save to CSV
        bat_calls_df.to_csv(f"{Path(__file__).parent}/20241116__bat_calls.csv", index=False)
        feeding_buzzes_df.to_csv(f"{Path(__file__).parent}/20241116__feeding_buzzes_spectrograms.csv", index=False)
        noise_df.to_csv(f"{Path(__file__).parent}/20241116__noise_spectrograms.csv", index=False)
        self.custom_detections.to_csv(f"{Path(__file__).parent}/20241116__custom_detections.csv", index=False)

        # Print confirmation
        print(f"Saved {len(bat_calls_indices)} rows to 'bat_calls.csv'")
        print(f"Saved {len(feeding_buzzes_indices)} rows to 'feeding_buzzes_spectrograms.csv'")
        print(f"Saved {len(noise_indices)} rows to 'noise_spectrograms.csv'")
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
    file_paths['SITE_classes_file'] = f"{file_paths['SITE_classes_file'][:-4]}_raw.csv"
    raw_location_df_filepath = Path(f'{Path(__file__).parent}/20241116__location_df_Foliage_kmeans_raw.csv')
    if raw_location_df_filepath.is_file():
        location_df_kmeans_raw = pd.read_csv(raw_location_df_filepath, low_memory=False, index_col=0)
    else:
        init_location_sum = actvt.assemble_initial_location_summary(file_paths) 
        init_location_sum.reset_index(inplace=True)
        init_location_sum.rename({'index':'index_in_file'}, axis='columns', inplace=True)
        location_df_kmeans_raw = add_frequency_groups_to_summary_using_kmeans(init_location_sum.copy(), file_paths, data_params, save=False)
        location_df_kmeans_raw.to_csv(raw_location_df_filepath)

    # file_paths = get_file_paths(data_params)
    # location_df_kmeans = pd.read_csv(f'{file_paths["SITE_folder"]}/{file_paths["detector_TYPE_SITE_YEAR"]}.csv', low_memory=False, index_col=0)

    # all_dropped_calls = location_df_kmeans_raw.groupby(by='input_file', group_keys=False).apply(lambda x : get_dropped_by_kmeans(x, location_df_kmeans))
    # test_df = all_dropped_calls.reset_index().loc[:,['index_in_file', 'freq_group', 'start_time', 'end_time', 'low_freq', 'high_freq', 'input_file']]
    test_df = location_df_kmeans_raw.reset_index()

    filesys = fsspec.filesystem('s3', anon=True, client_kwargs={'endpoint_url': 'https://sdsc.osn.xsede.org'})

    # Launch the GUI
    root = tk.Tk()
    root.title("Spectrogram Viewer")
    app = SpectrogramViewer(
        root,
        dataframe=test_df,
        location_df_kmeans_raw=location_df_kmeans_raw,
        FREQUENCY_COLOR_MAPPINGS=FREQUENCY_COLOR_MAPPINGS,
    )
    root.mainloop()
