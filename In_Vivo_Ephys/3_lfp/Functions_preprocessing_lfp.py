# -*- coding: utf-8 -*-
"""
Functions_preprocessing_lfp.py

Shared functions for LFP preprocessing pipeline.

Functions
---------
load_block                  Load a TDT block from disk once (use for multi-channel workflows).
extract_channel             Extract a single channel from an already-loaded block.
load_and_extract_data       Legacy single-call wrapper (backward compat).
filter_data                 Lowpass filter (Butterworth or FIR).
downsample_data             Polyphase rational resampling.
normalize_noise_reference   Z-score normalization of a noise reference trace.
plot_raw_and_filtered_data  QC plot comparing raw vs filtered traces.

Notes
-----
- plt.style should be set in the calling script, not here.
- load_and_extract_data is kept as a thin wrapper for backward compatibility.
  Prefer load_block() + extract_channel() when processing multiple channels
  to avoid redundant disk reads.
- The FIR tap estimation formula follows the Rabiner & Gold (1975) approximation.

@author: Juliana Groß (pipeline); refactored for LFP preprocessing
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import firwin

import tdt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_block(block_path: str, store: str = "RAWs") -> Any:
    """
    Load a TDT block from disk once.

    Pass the returned object to extract_channel() for individual channels.
    When processing multiple channels from the same recording, call this
    once and reuse the result — avoids redundant disk I/O.

    Parameters
    ----------
    block_path : str
        Path to the TDT block folder.
    store : str
        TDT store name. Default 'RAWs' for raw electrophysiology.

    Returns
    -------
    tdt data object
        Has .streams.<store>.data (array) and .streams.<store>.fs (float).
    """
    return tdt.read_block(block_path, store=store)


def extract_channel(
    block_data: Any,
    channel: int,
    store: str = "RAWs",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract a single channel from an already-loaded TDT block.

    Parameters
    ----------
    block_data : tdt data object
        Output of load_block().
    channel : int
        1-indexed channel number (matches TDT convention).
    store : str
        TDT store name. Default 'RAWs'.

    Returns
    -------
    time_s : np.ndarray
        Time vector in seconds, starting at 0.0.
    data : np.ndarray
        Raw voltage trace for the channel.
    fs : float
        Sampling rate in Hz.
    """
    stream = getattr(block_data.streams, store)
    fs: float = stream.fs
    raw: np.ndarray = stream.data[channel - 1]   # TDT is 1-indexed
    n = raw.shape[0]
    time_s = np.arange(n) / fs                   # starts at 0.0, not 1/fs
    return time_s, raw, fs


def load_and_extract_data(
    block_path: str,
    channel: int,
    store: str = "RAWs",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Legacy single-call wrapper: load block and extract one channel.

    Kept for backward compatibility. When processing multiple channels
    from the same block, prefer calling load_block() once and then
    extract_channel() in a loop to avoid redundant disk reads.

    Parameters
    ----------
    block_path : str
        Path to the TDT block folder.
    channel : int
        1-indexed channel number.
    store : str
        TDT store name. Default 'RAWs'.

    Returns
    -------
    time_s : np.ndarray
    data : np.ndarray
    fs : float
    """
    block_data = load_block(block_path, store=store)
    return extract_channel(block_data, channel, store=store)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_data(
    data: np.ndarray,
    fs: float,
    filter_type: str = "Butterworth",
    delta1: float = 0.01,
    delta2: float = 0.01,
    transition_width: float = 10.0,
    cutoff: float = 100.0,
    num_taps: int | None = None,
    order: int = 4,
) -> np.ndarray:
    """
    Lowpass filter for LFP preprocessing.

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    fs : float
        Sampling rate in Hz.
    filter_type : str
        'Butterworth' (IIR, zero-phase via sosfiltfilt) or
        'FIR' (zero-phase via filtfilt).
    delta1, delta2 : float
        Pass-band and stop-band ripple tolerances (used for FIR tap estimation).
        Formula: num_taps ≈ (2/3) * log10(1 / (10 * delta1 * delta2)) * (fs / transition_width)
        Source: Rabiner & Gold (1975) approximation.
    transition_width : float
        Width of the transition band in Hz (FIR only).
    cutoff : float
        Cutoff frequency in Hz.
    num_taps : int or None
        Number of FIR taps. If None, estimated from delta1/delta2/transition_width.
    order : int
        Filter order (Butterworth only).

    Returns
    -------
    filtered_data : np.ndarray
    """
    if filter_type == "Butterworth":
        sos = signal.butter(order, cutoff, btype="lp", fs=fs, output="sos")
        return signal.sosfiltfilt(sos, data)

    elif filter_type == "FIR":
        if num_taps is None:
            num_taps = int(
                (2 / 3)
                * np.log10(1 / (10 * delta1 * delta2))
                * (fs / transition_width)
            )
            print(f"Calculated number of FIR taps: {num_taps}")
        taps = firwin(numtaps=num_taps, cutoff=cutoff, fs=fs, pass_zero="lowpass")
        return signal.filtfilt(taps, [1.0], data)

    else:
        raise ValueError(
            f"Unsupported filter_type '{filter_type}'. Choose 'Butterworth' or 'FIR'."
        )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def downsample_data(
    data: np.ndarray,
    original_fs: float,
    downsampled_fs: float,
) -> np.ndarray:
    """
    Polyphase rational resampling (anti-aliasing included).

    Uses GCD reduction to keep up/down factors minimal.

    Parameters
    ----------
    data : np.ndarray
        Input signal at original_fs.
    original_fs : float
        Original sampling rate in Hz.
    downsampled_fs : float
        Target sampling rate in Hz.

    Returns
    -------
    downsampled_data : np.ndarray
    """
    gcd = np.gcd(int(original_fs), int(downsampled_fs))
    up_factor = int(downsampled_fs / gcd)
    down_factor = int(original_fs / gcd)
    return signal.resample_poly(data, up_factor, down_factor)


# ---------------------------------------------------------------------------
# Noise reference
# ---------------------------------------------------------------------------

def normalize_noise_reference(noise_reference_data: np.ndarray) -> np.ndarray:
    """
    Z-score normalize a noise reference trace before subtraction.

    Apply this before subtracting the reference from signal channels
    so that amplitude scaling does not distort the correction.

    Parameters
    ----------
    noise_reference_data : np.ndarray
        Raw noise reference channel.

    Returns
    -------
    normalized : np.ndarray
        Zero-mean, unit-variance noise reference.
    """
    mean = np.mean(noise_reference_data)
    std = np.std(noise_reference_data)
    return (noise_reference_data - mean) / std


# ---------------------------------------------------------------------------
# QC plotting
# ---------------------------------------------------------------------------

def plot_raw_and_filtered_data(
    time_vector: np.ndarray,
    raw_data: np.ndarray,
    filtered_data: np.ndarray,
    channel_name: str,
    filter_type: str,
    reference_type: str,
    save_path: str,
    plot_max_samples: int = 50_000,
) -> None:
    """
    Plot raw vs filtered signal and save to disk.

    Note: plt.style should be set in the calling script, not here,
    to avoid global style side-effects across the pipeline.

    Parameters
    ----------
    time_vector : np.ndarray
        Time axis in seconds.
    raw_data : np.ndarray
        Unfiltered signal.
    filtered_data : np.ndarray
        Filtered signal.
    channel_name : str
        Channel identifier for title and filename.
    filter_type : str
        Filter type label (e.g. 'Butterworth', 'FIR').
    reference_type : str
        Reference label (e.g. 'unreferenced', 'referenced').
    save_path : str
        Directory to save the PNG.
    plot_max_samples : int
        Maximum number of samples to display. If the signal is longer,
        it is uniformly decimated for display only — the exported data
        and all downstream computation are completely unaffected.
        Default 50,000 (sufficient to assess filter quality visually
        across recordings of any length).
    """
    # Decimate for display only — local copies, originals untouched
    n = len(time_vector)
    if n > plot_max_samples:
        step          = n // plot_max_samples
        time_vector   = time_vector[::step]
        raw_data      = raw_data[::step]
        filtered_data = filtered_data[::step]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(time_vector, raw_data, label="Raw", alpha=0.7, linewidth=0.8)
    ax.plot(time_vector, filtered_data,
            label=f"{filter_type} filtered", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Raw vs Filtered — Channel {channel_name} ({filter_type}, {reference_type})"
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(
        save_path,
        f"raw_vs_filtered_{filter_type}_{reference_type}_channel_{channel_name}.png",
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)