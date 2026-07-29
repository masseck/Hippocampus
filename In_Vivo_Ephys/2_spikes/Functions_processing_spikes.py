# -*- coding: utf-8 -*-
"""
Functions_processing_spikes.py

Shared helper functions for spike processing scripts.

Intended to be used by scripts such as:
- process_spikes_isi.py
- process_spikes_psth.py

Responsibilities:
- manifest / file IO helpers
- epoch extraction from Pynapse log
- stimulation window extraction from TDT pTrain epoc
- spike detection (MAD threshold crossing)
- spike train utilities
- waveform QC utilities
- plotting utilities

@author: Juliana Groß
"""

from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# -----------------------------
# Generic helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def list_manifests(export_path_base: str, relative_data_path: str) -> List[str]:
    root = os.path.join(export_path_base, relative_data_path)
    out: List[str] = []
    if not os.path.exists(root):
        return out

    for folder in os.listdir(root):
        m = os.path.join(root, folder, "metadata", "run_manifest.json")
        if os.path.isfile(m):
            out.append(m)

    out.sort()
    return out


def load_spike_continuous_pkls(spikes_continuous_path: str) -> List[str]:
    if not os.path.isdir(spikes_continuous_path):
        return []
    files = [f for f in os.listdir(spikes_continuous_path) if f.endswith(".pkl")]
    files.sort()
    return [os.path.join(spikes_continuous_path, f) for f in files]


def classify_signal_name(base: str) -> str:
    """
    base is filename without extension, e.g.:
      preprocessed_spikes_unreferenced_ch23
      preprocessed_spikes_lmr_ch23

    Returns:
      'unreferenced', 'lmr', or 'other'
    """
    if "preprocessed_spikes_unreferenced_" in base:
        return "unreferenced"
    if "preprocessed_spikes_lmr_" in base:
        return "lmr"
    return "other"


def should_process_signal(
    base: str,
    process_unreferenced: bool = True,
    process_lmr: bool = True,
) -> bool:
    kind = classify_signal_name(base)
    if kind == "unreferenced":
        return process_unreferenced
    if kind == "lmr":
        return process_lmr
    return False

# -----------------------------
# Plot styling helpers
# -----------------------------
def infer_tool_name(relative_data_path: str) -> str:
    path_lower = relative_data_path.lower()
    if "chr2" in path_lower:
        return "ChR2"
    if "vswo" in path_lower:
        return "vSWO"
    if "vlwo" in path_lower:
        return "vLWO"
    return "default"


def get_tool_color(tool_name: str) -> str:
    """
    Default stimulation colors per tool.
    Change these once here, and all scripts can reuse them.
    """
    color_map = {
        "ChR2": "#00fefe",   # cyan
        "vSWO": "#fe00fe",   # pink
        "vLWO": "#feeb00",   # yellow
        "default": "#E69F00" # orange fallback
    }
    return color_map.get(tool_name, color_map["default"])


def get_plot_style(
    theme: str = "dark",
    stim_color: str = "#009E73",
) -> dict:
    """
    Returns a dictionary with consistent plot colors for light/dark themes.
    """
    theme = theme.lower()

    if theme == "dark":
        return {
            "theme": "dark",
            "figure_facecolor": "black",
            "axes_facecolor": "black",
            "text_color": "white",
            "grid_color": "white",
            "spine_color": "white",
            "baseline_color": "gray",
            "stim_color": stim_color,
            "neutral_color": "gray",
            "decrease_color": "#D55E00",
            "increase_color": stim_color,
        }

    # default: light
    return {
        "theme": "light",
        "figure_facecolor": "white",
        "axes_facecolor": "white",
        "text_color": "black",
        "grid_color": "black",
        "spine_color": "black",
        "baseline_color": "gray",
        "stim_color": stim_color,
        "neutral_color": "gray",
        "decrease_color": "#D55E00",
        "increase_color": stim_color,
    }


def apply_axes_style(ax, style: dict) -> None:
    ax.set_facecolor(style["axes_facecolor"])
    ax.tick_params(colors=style["text_color"])
    ax.xaxis.label.set_color(style["text_color"])
    ax.yaxis.label.set_color(style["text_color"])
    ax.title.set_color(style["text_color"])

    for spine in ax.spines.values():
        spine.set_edgecolor(style["spine_color"])


def apply_figure_style(fig, style: dict) -> None:
    fig.patch.set_facecolor(style["figure_facecolor"])


def set_global_plot_style(
    theme: str = "light",
    font_family: str = "Arial",
    base_font_size: float = 12,
    axes_title_size: float = 14,
    axes_label_size: float = 12,
    tick_label_size: float = 11,
    legend_font_size: float = 11,
    axes_linewidth: float = 1.2,
) -> None:
    """
    Set global matplotlib style for all plots in the current script.

    Notes
    -----
    - pdf/ps fonttype 42 keeps text editable in Illustrator.
    - theme controls default matplotlib face/text colors.
    """
    theme = theme.lower()

    mpl.rcParams["font.family"] = font_family
    mpl.rcParams["font.size"] = base_font_size
    mpl.rcParams["axes.titlesize"] = axes_title_size
    mpl.rcParams["axes.labelsize"] = axes_label_size
    mpl.rcParams["xtick.labelsize"] = tick_label_size
    mpl.rcParams["ytick.labelsize"] = tick_label_size
    mpl.rcParams["legend.fontsize"] = legend_font_size
    mpl.rcParams["axes.linewidth"] = axes_linewidth

    # Keep fonts editable in vector graphics software
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    if theme == "dark":
        mpl.rcParams["figure.facecolor"] = "black"
        mpl.rcParams["axes.facecolor"] = "black"
        mpl.rcParams["savefig.facecolor"] = "black"
        mpl.rcParams["text.color"] = "white"
        mpl.rcParams["axes.labelcolor"] = "white"
        mpl.rcParams["axes.edgecolor"] = "white"
        mpl.rcParams["xtick.color"] = "white"
        mpl.rcParams["ytick.color"] = "white"
    else:
        mpl.rcParams["figure.facecolor"] = "white"
        mpl.rcParams["axes.facecolor"] = "white"
        mpl.rcParams["savefig.facecolor"] = "white"
        mpl.rcParams["text.color"] = "black"
        mpl.rcParams["axes.labelcolor"] = "black"
        mpl.rcParams["axes.edgecolor"] = "black"
        mpl.rcParams["xtick.color"] = "black"
        mpl.rcParams["ytick.color"] = "black"
        
def get_figure_size(
    height_mm: float = 40,
    aspect_ratio: float = 2.0,
) -> tuple[float, float]:
    """
    Convert mm to inches for matplotlib figure size.

    Parameters
    ----------
    height_mm : float
        Desired figure height in mm (e.g., 40 mm for journal column figures)
    aspect_ratio : float
        width / height ratio

    Returns
    -------
    (width_inch, height_inch)
    """

    mm_to_inch = 1 / 25.4
    height_in = height_mm * mm_to_inch
    width_in = height_in * aspect_ratio

    return width_in, height_in

def set_publication_fontsizes(height_mm: float = 40) -> None:
    """
    Adjust fonts automatically depending on figure size.
    """

    if height_mm <= 50:
        mpl.rcParams["font.size"] = 7
        mpl.rcParams["axes.labelsize"] = 7
        mpl.rcParams["axes.titlesize"] = 8
        mpl.rcParams["xtick.labelsize"] = 6
        mpl.rcParams["ytick.labelsize"] = 6

    elif height_mm <= 80:
        mpl.rcParams["font.size"] = 9
        mpl.rcParams["axes.labelsize"] = 9
        mpl.rcParams["axes.titlesize"] = 10
        mpl.rcParams["xtick.labelsize"] = 8
        mpl.rcParams["ytick.labelsize"] = 8

    else:
        mpl.rcParams["font.size"] = 12
        mpl.rcParams["axes.labelsize"] = 12
        mpl.rcParams["axes.titlesize"] = 14
        mpl.rcParams["xtick.labelsize"] = 11
        mpl.rcParams["ytick.labelsize"] = 11

# -----------------------------
# Experimental epochs (Pynapse log / pTrain)
# -----------------------------
@dataclass
class Epoch:
    label: str
    start_s: float
    end_s: float


def load_epochs_from_pynapse_csv(
    csv_path: str,
    ignore_label: str = "In start delay",
) -> List:
    """
    Build a list of Epoch(label, start_s, end_s) from Pynapse_call_log.csv.
 
    Duration inference
    ------------------
    Duration is inferred from the FULL dataframe (including rows without
    Output labels such as _onTimeout and Done) BEFORE any filtering.
    This ensures that rows like:
 
        Row 10: Start=125.22, Output="In stimulation state"
        Row 11: Start=605.23, Output=""    ← no label, would be filtered
 
    correctly produce a stimulation duration of 605.23 - 125.22 = 480 s,
    rather than falling back to the Duration column value (0.012 s).
 
    Label handling
    --------------
    All Output labels are normalised (stripped + lowercased) so that
    comparisons via epochs_to_windows() are case-insensitive. Handles
    inconsistencies such as "In stimulation state" vs "In Stimulation State".
 
    Parameters
    ----------
    csv_path : str
    ignore_label : str
        Excluded label (case-insensitive). Default: "In start delay".
 
    Returns
    -------
    List[Epoch]
    """
    df = pd.read_csv(csv_path)
 
    if "Start" not in df.columns:
        raise ValueError("Pynapse_call_log.csv missing required column: 'Start'")
    if "Duration" not in df.columns:
        df["Duration"] = np.nan
 
    # Step 1: infer duration on FULL dataframe before any filtering
    # Unlabelled rows (e.g. _onTimeout, Done) contribute their Start time
    # to the duration of the preceding labelled row.
    df["_next_start"]   = df["Start"].shift(-1)
    df["_dur_inferred"] = df["_next_start"] - df["Start"]
    df["Duration_used"] = np.where(
        np.isfinite(df["_dur_inferred"]) & (df["_dur_inferred"] > 0),
        df["_dur_inferred"],
        df["Duration"].astype(float),
    )
 
    # Step 2: now filter to rows with Output labels
    df = df.dropna(subset=["Output"])
    df["Output_norm"] = df["Output"].astype(str).str.strip().str.lower()
    ignore_norm = ignore_label.strip().lower()
    df = df[df["Output_norm"] != ignore_norm].reset_index(drop=True)
 
    if df.empty:
        return []
 
    epochs = []
    for _, r in df.iterrows():
        start = float(r["Start"])
        dur   = float(r["Duration_used"]) if np.isfinite(r["Duration_used"]) else 0.0
        end   = start + max(dur, 0.0)
        epochs.append(Epoch(label=r["Output_norm"], start_s=start, end_s=end))
 
    return epochs


def epochs_to_windows(
    epochs: List,
    label: str,
) -> List[Tuple[float, float]]:
    """
    Extract (start_s, end_s) windows for epochs matching the given label.
 
    Comparison is case-insensitive and strips whitespace, consistent with
    the normalisation applied in load_epochs_from_pynapse_csv().
 
    Parameters
    ----------
    epochs : List[Epoch]
        Output of load_epochs_from_pynapse_csv().
    label : str
        Label to match, e.g. "In baseline state" or "In stimulation state".
        Case and leading/trailing whitespace are ignored.
 
    Returns
    -------
    List[Tuple[float, float]] : (start_s, end_s) pairs
    """
    label_norm = label.strip().lower()
    return [
        (e.start_s, e.end_s)
        for e in epochs
        if e.label == label_norm
    ]


def load_ptrain_windows(
    block_path: str,
    pulse_event_max_duration_s: float = 0.001,
    collapse_pulse_events: bool = True,
    stim_duration_hint_s: float | None = None,
) -> List[Tuple[float, float]]:
    """
    Reads opto stimulation windows from TDT epoc store for pTrain.

    Handles:
    1) Continuous illumination:
       pTrain stored as one or more real windows with meaningful duration.
    2) Pulse-style storage:
       pTrain stored as many near-instantaneous events.
       In that case, collapse to one stimulation window.

    If stim_duration_hint_s is given, use:
        start = first onset
        end   = first onset + stim_duration_hint_s
    for pulse-style events.

    If no duration hint is available, infer duration from inter-event spacing.
    """
    import tdt

    data = tdt.read_block(block_path, evtype=["epocs"])
    if not hasattr(data, "epocs") or data.epocs is None:
        return []

    for store in ("Pe1_", "Pe1/"):
        if hasattr(data.epocs, store):
            ep = getattr(data.epocs, store)
            on = np.asarray(getattr(ep, "onset", []), dtype=float)
            off = np.asarray(getattr(ep, "offset", []), dtype=float)

            if on.size == 0:
                return []

            if off.size != on.size:
                off = on.copy()

            windows = [(float(a), float(b)) for a, b in zip(on, off) if b >= a]
            if not windows:
                return []

            durations = np.array([b - a for a, b in windows], dtype=float)

            # Pulse-style events: near-zero durations, multiple events
            if collapse_pulse_events and len(windows) > 1:
                max_dur = float(np.max(durations))
                median_dur = float(np.median(durations))

                if max_dur <= pulse_event_max_duration_s:
                    stim_start = float(on[0])

                    # Preferred: use duration hint from call log
                    if stim_duration_hint_s is not None and stim_duration_hint_s > 0:
                        stim_end = stim_start + float(stim_duration_hint_s)
                        print(
                            f" - Detected pulse-style pTrain events "
                            f"(n={len(windows)}, median_dur={median_dur:.6f}s). "
                            f"Using stimulation duration hint: "
                            f"{stim_start:.3f}s to {stim_end:.3f}s."
                        )
                        return [(stim_start, stim_end)]

                    # Fallback: infer from spacing between onsets
                    if len(on) >= 2:
                        isi = np.diff(on)
                        median_spacing = float(np.median(isi))
                        stim_end = stim_start + median_spacing * len(on)
                        print(
                            f" - Detected pulse-style pTrain events "
                            f"(n={len(windows)}, median_dur={median_dur:.6f}s). "
                            f"Inferred stimulation window from pulse spacing: "
                            f"{stim_start:.3f}s to {stim_end:.3f}s."
                        )
                        return [(stim_start, stim_end)]

                    # Final fallback: just use first-to-last
                    stim_end = float(off[-1])
                    print(
                        f" - Detected pulse-style pTrain events but could not infer duration robustly. "
                        f"Using first-to-last event span: {stim_start:.3f}s to {stim_end:.3f}s."
                    )
                    return [(stim_start, stim_end)]

            # Continuous-style real windows
            return [(a, b) for a, b in windows if b > a]

    available = [k for k in dir(data.epocs) if not k.startswith("_")]
    raise KeyError(f"pTrain store not found. Tried Pe1_ / Pe1/. Available epocs: {available}")


def merge_windows(windows: List[Tuple[float, float]], min_gap_s: float = 0.0) -> List[Tuple[float, float]]:
    """Merge overlapping windows (and windows closer than min_gap_s)."""
    if not windows:
        return []

    w = sorted(windows, key=lambda x: x[0])
    out = [w[0]]

    for a, b in w[1:]:
        la, lb = out[-1]
        if a <= lb + min_gap_s:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))

    return out


def summarize_window_durations(windows: List[Tuple[float, float]]) -> Dict[str, float]:
    d = np.array([b - a for a, b in windows if b > a], dtype=float)
    if d.size == 0:
        return {"n": 0, "min_s": np.nan, "median_s": np.nan, "max_s": np.nan}
    return {
        "n": int(len(windows)),
        "min_s": float(np.min(d)),
        "median_s": float(np.median(d)),
        "max_s": float(np.max(d)),
    }


# -----------------------------
# Spike train window utilities
# -----------------------------
def restrict_spikes_to_windows(
    spike_times_s: np.ndarray,
    windows: List[Tuple[float, float]],
) -> np.ndarray:
    if spike_times_s.size == 0 or not windows:
        return np.zeros((0,), dtype=float)

    keep = []
    for a, b in windows:
        m = (spike_times_s >= a) & (spike_times_s < b)
        if np.any(m):
            keep.append(spike_times_s[m])

    return np.concatenate(keep) if keep else np.zeros((0,), dtype=float)


def spikes_in_windows_mask(
    spike_times_s: np.ndarray,
    windows: List[Tuple[float, float]],
) -> np.ndarray:
    if spike_times_s.size == 0 or not windows:
        return np.zeros(spike_times_s.shape, dtype=bool)

    m = np.zeros(spike_times_s.shape, dtype=bool)
    for a, b in windows:
        m |= (spike_times_s >= a) & (spike_times_s < b)
    return m


def aligned_trials_from_windows(
    spike_times_s: np.ndarray,
    windows: List[Tuple[float, float]],
) -> List[np.ndarray]:
    """
    For each window (a,b), return spike times aligned to window onset: spike - a.
    """
    trials: List[np.ndarray] = []
    for a, b in windows:
        st = spike_times_s[(spike_times_s >= a) & (spike_times_s < b)] - a
        trials.append(st)
    return trials


def compute_isi(spike_times_s: np.ndarray) -> np.ndarray:
    if spike_times_s.size < 2:
        return np.zeros((0,), dtype=float)
    return np.diff(np.sort(spike_times_s))


def refractory_violations_fraction(spike_times_s: np.ndarray, refr_ms: float = 1.0) -> float:
    """
    Fraction of ISIs shorter than refr_ms.
    """
    if spike_times_s.size < 2:
        return float("nan")
    isi = np.diff(np.sort(spike_times_s))
    return float(np.mean(isi < (refr_ms / 1000.0)))


# -----------------------------
# Spike detection
# -----------------------------
def mad_sigma(x: np.ndarray) -> float:
    """
    Noise estimate via MAD: median(|x - median(x)|)
    """
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def detect_spikes_threshold(
    x: np.ndarray,
    fs: float,
    b: float = 5.0,
    polarity: str = "neg",
    refractory_ms: float = 1.0,
    refine_peak: bool = True,
    peak_search_ms: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Detect spikes in 1D spike-band filtered continuous trace.

    Returns:
      peaks_samp : spike indices in samples
      metrics    : detection QC/metadata
    """
    sigma = mad_sigma(x)

    if polarity == "neg":
        thr = -b * sigma
        below = (x < thr).astype(np.int8)
    elif polarity == "pos":
        thr = +b * sigma
        below = (x > thr).astype(np.int8)
    else:
        raise ValueError("polarity must be 'neg' or 'pos'")

    crossings = np.flatnonzero(np.diff(below) == 1) + 1

    if refine_peak and crossings.size:
        win = int(round((peak_search_ms / 1000.0) * fs))
        peaks = []
        n = x.size
        for c in crossings:
            start = int(c)
            stop = min(n, start + win)
            if stop <= start + 1:
                continue
            seg = x[start:stop]
            p = start + (int(np.argmin(seg)) if polarity == "neg" else int(np.argmax(seg)))
            peaks.append(p)
        peaks = np.asarray(peaks, dtype=np.int64)
    else:
        peaks = crossings.astype(np.int64)

    if peaks.size:
        ref_samp = int(round((refractory_ms / 1000.0) * fs))
        kept = [int(peaks[0])]
        last = int(peaks[0])
        for p in peaks[1:]:
            p = int(p)
            if p - last >= ref_samp:
                kept.append(p)
                last = p
        peaks = np.asarray(kept, dtype=np.int64)

    metrics = {
        "sigma_mad": float(sigma),
        "threshold": float(thr),
        "b": float(b),
        "refractory_ms": float(refractory_ms),
        "peak_search_ms": float(peak_search_ms),
        "refine_peak": float(1.0 if refine_peak else 0.0),
    }
    return peaks, metrics


# -----------------------------
# Waveform QC
# -----------------------------
def extract_waveforms(
    x: np.ndarray,
    peaks_samp: np.ndarray,
    fs: float,
    pre_ms: float = 1.0,
    post_ms: float = 2.0,
    max_waveforms: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract waveform snippets around detected peaks.

    Returns:
      wf   : shape (n_spikes, n_samples)
      t_ms : waveform time axis in ms
    """
    if peaks_samp.size == 0:
        n = int(round((pre_ms + post_ms) * fs / 1000.0)) + 1
        t = (np.arange(n) - int(round(pre_ms * fs / 1000.0))) / fs * 1000.0
        return np.zeros((0, n), dtype=np.float32), t.astype(np.float32)

    pre = int(round(pre_ms * fs / 1000.0))
    post = int(round(post_ms * fs / 1000.0))
    wlen = pre + post + 1

    peaks = peaks_samp[(peaks_samp >= pre) & (peaks_samp < (x.size - post))].astype(np.int64)

    if peaks.size == 0:
        t = (np.arange(wlen) - pre) / fs * 1000.0
        return np.zeros((0, wlen), dtype=np.float32), t.astype(np.float32)

    if peaks.size > max_waveforms:
        idx = np.random.choice(peaks.size, size=max_waveforms, replace=False)
        peaks = np.sort(peaks[idx])

    wf = np.empty((peaks.size, wlen), dtype=np.float32)
    for i, p in enumerate(peaks):
        wf[i, :] = x[p - pre : p + post + 1].astype(np.float32)

    t_ms = (np.arange(wlen) - pre) / fs * 1000.0
    return wf, t_ms.astype(np.float32)


def split_waveforms_by_condition(
    x: np.ndarray,
    peaks_samp: np.ndarray,
    fs: float,
    baseline_windows: List[Tuple[float, float]],
    stim_windows: List[Tuple[float, float]],
    pre_ms: float = 1.0,
    post_ms: float = 2.0,
    max_waveforms: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Safe waveform splitting:
    1) edge-safe valid peaks
    2) assign peaks to baseline vs stim by time
    3) subsample within each condition
    4) extract waveforms separately

    Returns:
      wf_base, wf_stim, t_ms
    """
    pre = int(round(pre_ms * fs / 1000.0))
    post = int(round(post_ms * fs / 1000.0))

    peaks_valid = peaks_samp[(peaks_samp >= pre) & (peaks_samp < (x.size - post))].astype(np.int64)
    st_valid = peaks_valid / fs

    m_base_valid = spikes_in_windows_mask(st_valid, baseline_windows)
    m_stim_valid = spikes_in_windows_mask(st_valid, stim_windows)

    peaks_base = peaks_valid[m_base_valid]
    peaks_stim = peaks_valid[m_stim_valid]

    if peaks_base.size > max_waveforms:
        idx = np.random.choice(peaks_base.size, size=max_waveforms, replace=False)
        peaks_base = np.sort(peaks_base[idx])

    if peaks_stim.size > max_waveforms:
        idx = np.random.choice(peaks_stim.size, size=max_waveforms, replace=False)
        peaks_stim = np.sort(peaks_stim[idx])

    wf_base, t_ms = extract_waveforms(
        x=x, peaks_samp=peaks_base, fs=fs,
        pre_ms=pre_ms, post_ms=post_ms,
        max_waveforms=max_waveforms,
    )
    wf_stim, _ = extract_waveforms(
        x=x, peaks_samp=peaks_stim, fs=fs,
        pre_ms=pre_ms, post_ms=post_ms,
        max_waveforms=max_waveforms,
    )

    return wf_base, wf_stim, t_ms


def waveform_metrics(wf: np.ndarray, fs: float) -> Dict[str, float]:
    if wf.size == 0:
        return {
            "wf_n": 0,
            "wf_amp_mean": float("nan"),
            "wf_amp_std": float("nan"),
            "wf_width_ms_mean": float("nan"),
            "wf_width_ms_std": float("nan"),
        }

    trough = np.min(wf, axis=1)
    peak = np.max(wf, axis=1)
    amp = peak - trough

    trough_idx = np.argmin(wf, axis=1)
    width_ms = np.full((wf.shape[0],), np.nan, dtype=np.float32)

    for i in range(wf.shape[0]):
        ti = trough_idx[i]
        seg = wf[i, ti:]
        if seg.size < 2:
            continue
        pi = ti + int(np.argmax(seg))
        width_ms[i] = (pi - ti) / fs * 1000.0

    return {
        "wf_n": int(wf.shape[0]),
        "wf_amp_mean": float(np.nanmean(amp)),
        "wf_amp_std": float(np.nanstd(amp)),
        "wf_width_ms_mean": float(np.nanmean(width_ms)),
        "wf_width_ms_std": float(np.nanstd(width_ms)),
    }


# -----------------------------
# Plotting
# -----------------------------
def make_psth(
    spike_times_aligned_s: np.ndarray,
    bin_edges: np.ndarray,
    n_trials: int,
) -> np.ndarray:
    counts, _ = np.histogram(spike_times_aligned_s, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    return counts / (bin_widths * max(n_trials, 1))

def plot_isi_hist(
    isi_baseline: np.ndarray,
    isi_stim: np.ndarray,
    out_png: str,
    title: str,
    max_isi_s: float = 0.5,
    bin_s: float = 0.001,
    style: dict | None = None,
) -> None:
    ensure_dir(os.path.dirname(out_png))
    bins = np.arange(0, max_isi_s + bin_s, bin_s)

    if style is None:
        style = get_plot_style(theme="dark")

    fig, ax = plt.subplots(figsize=(10, 4))
    apply_figure_style(fig, style)
    apply_axes_style(ax, style)

    if isi_baseline.size:
        ax.hist(isi_baseline, bins=bins, alpha=0.6, label="baseline", color=style["baseline_color"])
    if isi_stim.size:
        ax.hist(isi_stim, bins=bins, alpha=0.6, label="stimulation", color=style["stim_color"])

    ax.set_xlabel("Inter-spike interval [s]")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3, color=style["grid_color"])

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, transparent=(style["theme"] == "dark"))
    plt.close(fig)

def plot_raster_and_psth(
    aligned_trials: List[np.ndarray],
    tmin: float,
    tmax: float,
    bin_s: float,
    title: str,
    out_png: str,
    style: dict | None = None,
) -> None:
    ensure_dir(os.path.dirname(out_png))

    if style is None:
        style = get_plot_style(theme="dark")

    fig = plt.figure(figsize=(10, 6))
    apply_figure_style(fig, style)

    ax1 = plt.subplot(2, 1, 1)
    apply_axes_style(ax1, style)
    for i, st in enumerate(aligned_trials):
        st = st[(st >= tmin) & (st <= tmax)]
        if st.size:
            ax1.vlines(st, i + 0.5, i + 1.5, linewidth=0.7, color=style["stim_color"])
    ax1.axvline(0, linestyle="--", color=style["text_color"])
    ax1.set_xlim(tmin, tmax)
    ax1.set_ylabel("Trial")
    ax1.set_title(title)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    apply_axes_style(ax2, style)
    bin_edges = np.arange(tmin, tmax + bin_s, bin_s)
    all_spikes = np.concatenate(aligned_trials) if aligned_trials else np.zeros((0,), dtype=float)
    rate = make_psth(all_spikes, bin_edges, n_trials=len(aligned_trials))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
    ax2.plot(bin_centers, rate, color=style["stim_color"])
    ax2.axvline(0, linestyle="--", color=style["text_color"])
    ax2.set_xlabel("Time from stim onset [s]")
    ax2.set_ylabel("Rate [Hz]")
    ax2.grid(True, linestyle="--", alpha=0.3, color=style["grid_color"])

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, transparent=(style["theme"] == "dark"))
    plt.close(fig)


def plot_two_psths_full_epoch(
    baseline_trials: List[np.ndarray],
    stim_trials: List[np.ndarray],
    tmax: float,
    bin_s: float,
    out_png: str,
    title: str,
    style: dict | None = None,
) -> None:
    ensure_dir(os.path.dirname(out_png))

    if style is None:
        style = get_plot_style(theme="dark")

    bin_edges = np.arange(0.0, tmax + bin_s, bin_s)
    centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0

    def psth_from_trials(trials: List[np.ndarray]) -> np.ndarray:
        if not trials:
            return np.zeros_like(centers)
        all_spikes = np.concatenate([st[(st >= 0) & (st <= tmax)] for st in trials]) if trials else np.zeros((0,))
        return make_psth(all_spikes, bin_edges, n_trials=len(trials))

    r_base = psth_from_trials(baseline_trials)
    r_stim = psth_from_trials(stim_trials)

    fig, ax = plt.subplots(figsize=(10, 4))
    apply_figure_style(fig, style)
    apply_axes_style(ax, style)

    ax.plot(centers, r_base, label="baseline", color=style["baseline_color"])
    ax.plot(centers, r_stim, label="stimulation", color=style["stim_color"])
    ax.set_xlabel("Time within epoch [s]")
    ax.set_ylabel("Rate [Hz]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3, color=style["grid_color"])

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, transparent=(style["theme"] == "dark"))
    plt.close(fig)


def plot_waveforms_baseline_vs_stim(
    wf_base: np.ndarray,
    wf_stim: np.ndarray,
    t_ms: np.ndarray,
    out_png: str,
    title: str,
    max_traces: int = 80,
    style: dict | None = None,
) -> None:
    ensure_dir(os.path.dirname(out_png))

    if style is None:
        style = get_plot_style(theme="dark")

    fig, ax = plt.subplots(figsize=(10, 5))
    apply_figure_style(fig, style)
    apply_axes_style(ax, style)

    def _plot_group(wf: np.ndarray, label: str, color: str):
        if wf.size == 0:
            return
        n = min(max_traces, wf.shape[0])
        ex = wf[np.linspace(0, wf.shape[0] - 1, n).astype(int)]
        for w in ex:
            ax.plot(t_ms, w, alpha=0.15, color=color)
        m = np.mean(wf, axis=0)
        s = np.std(wf, axis=0)
        ax.plot(t_ms, m, linewidth=2, label=f"{label} mean", color=color)
        ax.fill_between(t_ms, m - s, m + s, alpha=0.2, color=color)

    _plot_group(wf_base, "baseline", style["baseline_color"])
    _plot_group(wf_stim, "stim", style["stim_color"])

    ax.axvline(0, linestyle="--", linewidth=1, color=style["text_color"])
    ax.set_xlabel("Time around peak [ms]")
    ax.set_ylabel("Amplitude (filtered units)")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, transparent=(style["theme"] == "dark"))
    plt.close(fig)