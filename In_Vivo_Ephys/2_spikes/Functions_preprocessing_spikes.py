# -*- coding: utf-8 -*-
"""
Functions_preprocessing_spikes.py

Shared helper functions for spike preprocessing.

Functions
---------
load_and_extract_data   Load a raw TDT channel.
filter_data             Lowpass filter (Butterworth or FIR).
plot_raw_and_filtered_data  QC plot of raw vs filtered signal.
downsample_data         Polyphase rational downsampling.
normalize_noise_reference   Z-score normalization of noise reference.

@author: Juliana Groß
"""

# In[] Import the necessary functions:

import matplotlib.pyplot as plt  # standard Python plotting library
import numpy as np  # fundamental package for scientific computing, handles arrays and maths

# import the tdt library
import tdt
import os

# import the pickle library to save variables
import pickle

# import scipy packages
# (even though Python says they are unused, they ARE used and needed)
from scipy import signal
# scipy.signal — only what is actually used in this file
from scipy.signal import butter, sosfiltfilt, firwin, filtfilt

# import pylab


# In[] Function to load and extract data

def load_and_extract_data(block_path, channel):
    # Load the data block
    data = tdt.read_block(block_path)
    
    # Get the sampling rate
    fs = data.streams.RAWs.fs
    
    # Get number of samples
    num_samples = len(data.streams.RAWs.data[0])
    
    # Create data vector for the specified channel
    RAWs_data = data.streams.RAWs.data[channel - 1]
    
    # Create time vector
    RAWs_time = np.linspace(1, num_samples, num_samples) / fs
    
    return RAWs_time, RAWs_data, fs


# In[] Function to filter data with user-selected design parameters:

# ! Mind ! The values specified here are only default values.

# They are specified in the experiments-dictionary for each experiment specifically.

def filter_data(data, fs, filter_type="Butterworth", delta1=0.01, delta2=0.01, 
                transition_width=10, cutoff=100, num_taps=None, order=4):
    if filter_type == "Butterworth":
        # Design Butterworth filter with the specified order and cutoff
        sos = signal.butter(order, cutoff, 'lp', fs=fs, output='sos')
        filtered_data = signal.sosfiltfilt(sos, data)
    
    elif filter_type == "FIR":
        # Calculate num_taps if not provided
        # print(f"num_taps={num_taps}")
        if num_taps is None:
            # print("num_taps none")
            num_taps = int((2 / 3) * np.log10(1 / (10 * delta1 * delta2)) * (fs / transition_width))
            print(f"Calculated number of taps for FIR: {num_taps}")
        
        # Design FIR filter with calculated or given num_taps
        taps = firwin(numtaps=num_taps, cutoff=cutoff, fs=fs, pass_zero='lowpass')
        # Use lfilter or filtfilt as FIR filters do not require SOS
        filtered_data = signal.filtfilt(taps, [1.0], data)
    
    else:
        raise ValueError("Unsupported filter type. Choose 'Butterworth' or 'FIR'.")
    
    return filtered_data

# In[] Function to plot raw and filtered data for comparison

def plot_raw_and_filtered_data(time_vector, raw_data, filtered_data, channel_name, filter_type, reference_type, save_path):
    plt.figure(figsize=(14, 5))
    
    # Plot raw data
    plt.plot(time_vector, raw_data, label='Raw Data', alpha=0.7)
    
    # Plot filtered data
    plt.plot(time_vector, filtered_data, label=f'{filter_type} Filtered Data', alpha=0.7)
    
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(f'Raw and Filtered Signal - Channel {channel_name} ({filter_type} Filter)')
    plt.legend(loc="upper right")
    
    # Save the figure
    plt.savefig(os.path.join(save_path, f'raw_vs_filtered_{filter_type}_{reference_type}_channel_{channel_name}.png'), dpi=300)
    plt.close()
    

# In[] Function to downsample data

def downsample_data(data, original_fs, downsampled_fs):
    gcd = np.gcd(int(original_fs), int(downsampled_fs))
    up_factor = int(downsampled_fs / gcd)
    down_factor = int(original_fs / gcd)
    downsampled_data = signal.resample_poly(data, up_factor, down_factor)
    
    return downsampled_data


# %% Function to normalize the noise reference for adaptive filtering

def normalize_noise_reference(noise_reference_data):
    
    mean = np.mean(noise_reference_data)
    std = np.std(noise_reference_data)
    
    normalized_noise_reference = (noise_reference_data - mean) / std
    
    return normalized_noise_reference