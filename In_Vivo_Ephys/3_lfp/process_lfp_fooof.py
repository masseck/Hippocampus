# -*- coding: utf-8 -*-
"""
process_lfp_fooof.py

FOOOF (Fitting Oscillations & One Over F) analysis of LFP power spectra.

What FOOOF does
---------------
FOOOF parameterizes a power spectrum into two components:

  1. Aperiodic component — the 1/f-like background, described by:
       - offset   : overall power level (log scale)
       - exponent : slope of the 1/f component (steeper = more negative)
       - [knee]   : optional bend in the aperiodic component (use
                    aperiodic_mode='knee' for data with a clear bend)

  2. Periodic components — peaks rising above the aperiodic background,
     each described by:
       - center frequency (CF)  : peak frequency in Hz
       - power (PW)             : peak height above aperiodic
       - bandwidth (BW)         : peak width in Hz

This allows you to ask: does stimulation change the aperiodic slope (general
excitability/inhibition balance), or does it change specific oscillatory peaks,
or both?

Workflow
--------
1. Load preprocessed LFP pkl files (same as process_lfp_psd.py)
2. Apply NLMS noise filtering if noise reference is available
3. Segment into baseline and stimulation epochs
4. Compute Welch PSD per channel per epoch
5. Average channels within animal → one PSD per animal
6. Fit FOOOF model to each animal's baseline and stimulation PSD
7. Extract and compare aperiodic and periodic parameters
8. Export:
   - Per-animal FOOOF model plots (QC)
   - Group CSV with all parameters
   - Summary plots: aperiodic exponent baseline vs stim (paired),
     peak CF distribution

Installation
------------
pip install fooof

Reference
---------
Donoghue et al. (2020). Parameterizing neural power spectra into periodic
and aperiodic components. Nature Neuroscience, 23, 1655-1665.
DOI: 10.1038/s41593-020-00744-x

Signal kinds
------------
PROCESS_REFERENCED     : preprocessed_lfp_referenced_*
PROCESS_UNREFERENCED   : preprocessed_lfp_unreferenced_*
PROCESS_REFERENCED_CAR : preprocessed_lfp_referenced_car_*  (early ChR2 animals)
"""

from __future__ import annotations

import os
import pickle
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

from specparam import SpectralModel as FOOOF, SpectralGroupModel as FOOOFGroup

from Functions_processing_spikes import (
    ensure_dir,
    load_json,
    list_manifests,
    load_epochs_from_pynapse_csv,
    epochs_to_windows,
    load_ptrain_windows,
    merge_windows,
    summarize_window_durations,
    infer_tool_name,
    get_tool_color,
    get_plot_style,
    apply_axes_style,
    apply_figure_style,
    set_global_plot_style,
    get_figure_size,
    set_publication_fontsizes,
)

def _prompt_path(label: str, default: str) -> str:
    """Prompt the user to confirm or override a path at startup."""
    user_input = input(f"{label}\n  [{default}]: ").strip()
    return user_input if user_input else default

# -----------------------------
# Settings
# -----------------------------
_DEFAULT_EXPORT_PATH_BASE  = r"C:\Users\Juliana\Documents\_PhD\Data\_Processed"
_DEFAULT_DATA_PATH_BASE    = r"C:\Users\Juliana\Documents\_PhD\Data\_Raw"

_DEFAULT_RELATIVE_DATA_PATH = r"Jills_paper\ChR2_cohort\OFT_experimental"  # adjust per cohort

relative_data_path = r"Jills_paper\ChR2_cohort\OFT_experimental"
# relative_data_path = r"Jills_paper\vSWO_cohort\OFT_experimental"
# relative_data_path = r"Jills_paper\vLWO_cohort\OFT_experimental"

only_folder: Optional[str] = None
only_regions: Optional[List[str]] = ["CA1_L"]

# Signal kinds to process — set one True per run
PROCESS_REFERENCED     = True    # preprocessed_lfp_referenced_*
PROCESS_UNREFERENCED   = False   # preprocessed_lfp_unreferenced_*
PROCESS_REFERENCED_CAR = False   # preprocessed_lfp_referenced_car_* (early ChR2)

BASELINE_LABEL = "In baseline state"
STIM_LABEL     = "In stimulation state"
IGNORE_LABEL   = "In start delay"

# PSD settings (must match process_lfp_psd.py)
NPERSEG_S = 2.0

# FOOOF settings
# -----------------------------
# FOOOF_FREQ_RANGE : [lower, upper] in Hz for model fitting.
#
#   Lower bound: must be > 0 Hz — FOOOF fits in log space and log(0) is
#                undefined. 1 Hz is a safe minimum for LFP data.
#   Upper bound: choose based on your signal of interest. Common choices:
#                40 Hz  — covers delta, theta, beta and low gamma
#                100 Hz — full LFP range including high gamma
#
#   Note: FOOOF_FREQ_RANGE controls the fitting range only. The PSD is
#   still computed over the full range (up to fs/2) before FOOOF is called.
#   The window_suffix in the output filename will reflect these bounds.
#
FOOOF_FREQ_RANGE = [1.0, 40.0]   # Hz — adjust upper bound as needed

# peak_width_limits: [min, max] width of peaks in Hz.
# min should be >= frequency resolution (0.5 Hz with 2s Welch).
FOOOF_PEAK_WIDTH_LIMITS = [1.0, 12.0]

# max_n_peaks: maximum number of peaks to extract per spectrum.
FOOOF_MAX_N_PEAKS = 6

# min_peak_height: minimum peak height (in log power units) above aperiodic.
# 0.05 is permissive — increase to 0.1 or 0.2 if you see too many spurious peaks.
FOOOF_MIN_PEAK_HEIGHT = 0.05

# peak_threshold: relative threshold (in SDs of residuals) for peak detection.
FOOOF_PEAK_THRESHOLD = 2.0

# aperiodic_mode: 'fixed' (default) or 'knee'.
# Use 'knee' if your data shows a clear bend in the aperiodic component.
# For LFP in freely moving mice, 'fixed' is usually appropriate.
FOOOF_APERIODIC_MODE = "fixed"

# Frequency bands for periodic peak extraction summary.
# Bands outside FOOOF_FREQ_RANGE will have no peaks assigned to them —
# adjust to match your fitting range.
FREQ_BANDS = {
    "delta":  (1.0,   4.0),
    "theta":  (4.0,  12.0),
    "beta":   (12.0, 30.0),
    "gamma":  (30.0, 40.0),   # upper bound matches FOOOF_FREQ_RANGE
}

# NLMS adaptive filter settings (same as process_lfp_psd.py)
NLMS_MU            = 0.015
NLMS_N_TAPS_FACTOR = 3

# Adjusted window settings
USE_ADJUSTED_WINDOWS = True    # True = adjusted | False = full epochs
ADJUSTED_BASELINE_S  = 120.0
ADJUSTED_STIM_S      = 300.0

# Plot settings
PLOT_THEME          = "light"
STIM_COLOR_OVERRIDE = None

FIG_HEIGHT_MM = 50
FIG_ASPECT    = 1.6

tool_name  = infer_tool_name(relative_data_path)
stim_color = STIM_COLOR_OVERRIDE if STIM_COLOR_OVERRIDE is not None \
    else get_tool_color(tool_name)
plot_style = get_plot_style(theme=PLOT_THEME, stim_color=stim_color)

# ChR2 uses call log; vSWO/vLWO use pTrain
USE_PTRAIN = tool_name != "ChR2"

set_global_plot_style(
    theme=PLOT_THEME, font_family="Arial",
    base_font_size=6, axes_title_size=7, axes_label_size=6,
    tick_label_size=5.5, legend_font_size=5.5, axes_linewidth=1.0,
)
set_publication_fontsizes(FIG_HEIGHT_MM)
FIG_SIZE = get_figure_size(height_mm=FIG_HEIGHT_MM, aspect_ratio=FIG_ASPECT)

print(
    f"Tool: {tool_name} | USE_PTRAIN: {USE_PTRAIN} | "
    f"Adjusted windows: {USE_ADJUSTED_WINDOWS}"
    + (f" (baseline={ADJUSTED_BASELINE_S:.0f} s, stim={ADJUSTED_STIM_S:.0f} s)"
       if USE_ADJUSTED_WINDOWS else "")
)


# -----------------------------
# File loading helpers
# -----------------------------
def load_lfp_pkls(lfp_cont_path: str, kind: str) -> List[str]:
    if not os.path.isdir(lfp_cont_path):
        return []
    prefix = f"preprocessed_lfp_{kind}_"
    files  = sorted(
        f for f in os.listdir(lfp_cont_path)
        if f.startswith(prefix) and f.endswith(".pkl")
    )
    if kind == "referenced":
        files = [f for f in files if not f.startswith("preprocessed_lfp_referenced_car_")]
    return [os.path.join(lfp_cont_path, f) for f in files]


def load_noise_reference(processed_folder_path: str) -> Optional[np.ndarray]:
    noise_lfp_path = os.path.join(processed_folder_path, "Noise_reference", "LFP")
    if not os.path.isdir(noise_lfp_path):
        return None
    ref_files = [
        f for f in os.listdir(noise_lfp_path)
        if f.startswith("preprocessed_lfp_noise_reference_") and f.endswith(".pkl")
    ]
    if not ref_files:
        return None
    with open(os.path.join(noise_lfp_path, ref_files[0]), "rb") as f:
        ref_obj = pickle.load(f)
    return np.asarray(ref_obj["downsampled_data"], dtype=float)


# -----------------------------
# NLMS adaptive filter
# -----------------------------
def apply_adaptive_filter_nlms(
    data: np.ndarray, reference: np.ndarray, fs: float,
) -> np.ndarray:
    if len(reference) != len(data):
        reference = np.interp(
            np.linspace(0, 1, len(data)),
            np.linspace(0, 1, len(reference)),
            reference,
        )
    n_taps        = int(fs / 50) * NLMS_N_TAPS_FACTOR
    weights       = np.zeros(n_taps)
    filtered_data = np.zeros_like(data)
    ref_buffer    = np.zeros(n_taps)
    mu            = NLMS_MU
    for i in range(len(data)):
        ref_buffer[:-1] = ref_buffer[1:]
        ref_buffer[-1]  = reference[i]
        estimate        = np.dot(weights, ref_buffer)
        error           = data[i] - estimate
        norm_factor     = np.dot(ref_buffer, ref_buffer) + 1e-6
        weights        += (mu / norm_factor) * error * ref_buffer
        filtered_data[i] = error
    return filtered_data


# -----------------------------
# Window helpers
# -----------------------------
def adjust_baseline_window(wins, dur):
    return [(end - min(end - start, dur), end) for start, end in wins]


def adjust_stim_window(wins, dur):
    return [(start, start + min(end - start, dur)) for start, end in wins]


def build_stim_windows(epochs, block_path):
    calllog = epochs_to_windows(epochs, STIM_LABEL)
    if not USE_PTRAIN or block_path is None:
        return calllog
    hint = float(np.median([b - a for a, b in calllog])) if calllog else None
    try:
        ptrain = merge_windows(
            load_ptrain_windows(block_path, stim_duration_hint_s=hint), min_gap_s=0.0
        )
        if ptrain:
            print(f" - Using pTrain: {len(ptrain)} epoch(s).")
            return ptrain
        print(" - pTrain empty; falling back to call log.")
    except FileNotFoundError:
        print(" - Raw path not found; falling back to call log.")
    except Exception as e:
        print(f" - pTrain error ({e}); falling back to call log.")
    return calllog


def segment_data(data, fs, windows):
    n, segments = len(data), []
    for a, b in windows:
        s, e = max(0, int(round(a * fs))), min(n, int(round(b * fs)))
        if e > s:
            segments.append(data[s:e])
    return np.concatenate(segments) if segments else np.zeros(0)


# -----------------------------
# PSD
# -----------------------------
def compute_psd(data, fs):
    nperseg = int(round(NPERSEG_S * fs))
    return welch(data, fs=fs, nperseg=nperseg)


# -----------------------------
# FOOOF fitting helpers
# -----------------------------
def make_fooof() -> FOOOF:
    return FOOOF(
        peak_width_limits=FOOOF_PEAK_WIDTH_LIMITS,
        max_n_peaks=FOOOF_MAX_N_PEAKS,
        min_peak_height=FOOOF_MIN_PEAK_HEIGHT,
        peak_threshold=FOOOF_PEAK_THRESHOLD,
        aperiodic_mode=FOOOF_APERIODIC_MODE,
        verbose=False,
    )


def fit_fooof_single(
    freqs: np.ndarray,
    power: np.ndarray,
    label: str,
    out_plot_path: Optional[str] = None,
) -> Optional[dict]:
    """
    Fit FOOOF to a single power spectrum and return extracted parameters.

    Parameters
    ----------
    freqs : np.ndarray  — 1D frequency array (linear Hz)
    power : np.ndarray  — 1D power array (linear µV²/Hz)
    label : str         — label for the QC plot title
    out_plot_path : str — if given, save QC plot here

    Returns
    -------
    dict with aperiodic and peak parameters, or None if fit failed.
    """
    fm = make_fooof()
    try:
        fm.fit(freqs, power, FOOOF_FREQ_RANGE)
    except Exception as e:
        print(f"   ! FOOOF fit failed for {label}: {e}")
        return None

    # specparam new API: results are stored in fm.results.params and fm.results.metrics
    res = fm.results

    # Try new specparam API first, fall back gracefully
    try:
        metrics = res.metrics.results
        r_sq    = float(metrics.get("gof_rsquared", float("nan")))
        error   = float(metrics.get("error_mae",    float("nan")))
        # Sanity check — R² must be between -1 and 1
        if not (-1.0 <= r_sq <= 1.0):
            raise ValueError(f"R² out of range: {r_sq}")
    except (AttributeError, ValueError, TypeError):
        # Fall back to direct attribute access (classic FOOOF API)
        try:
            r_sq  = float(fm.r_squared_)
            error = float(fm.error_)
        except AttributeError:
            r_sq  = float("nan")
            error = float("nan")

    # Aperiodic params: fm.results.params.aperiodic
    # ComponentParameters stores values as a dict or array — try both
    # specparam stores aperiodic as ComponentParameters with asdict()
    # asdict() returns a dict of arrays, one per parameter name.
    # fixed mode: {'offset': array([val, nan]), 'exponent': array([val, nan])}
    # The first element of each array is the actual value; nan = unused mode slot.
    # specparam new API:
    # ap_dict["aperiodic_fit"] = array([offset, exponent])  (fixed mode)
    #                          = array([offset, knee, exponent])  (knee mode)
    # pk_dict["peak_fit"]      = array of shape (n_peaks, 3): [cf, pw, bw] per row
    ap_obj  = res.params.aperiodic
    ap_dict = ap_obj.asdict()
    ap_fit  = np.asarray(ap_dict.get("aperiodic_fit", []), dtype=float).ravel()
    ap_fit  = ap_fit[np.isfinite(ap_fit)]  # remove nan padding

    if len(ap_fit) == 2:
        ap_offset, ap_exponent, ap_knee = float(ap_fit[0]), float(ap_fit[1]), float("nan")
    elif len(ap_fit) == 3:
        ap_offset, ap_knee, ap_exponent = float(ap_fit[0]), float(ap_fit[1]), float(ap_fit[2])
    else:
        ap_offset = ap_exponent = ap_knee = float("nan")

    # Peak params: peak_fit is (n_peaks, 3) array: [cf, pw, bw]
    pk_obj  = res.params.periodic
    pk_dict = pk_obj.asdict()
    pk_fit  = np.asarray(pk_dict.get("peak_fit", []), dtype=float)
    if pk_fit.ndim == 2 and pk_fit.shape[1] == 3 and pk_fit.shape[0] > 0:
        # Keep only rows with finite cf (first column)
        valid_rows = np.isfinite(pk_fit[:, 0])
        peaks = pk_fit[valid_rows]
    else:
        peaks = np.empty((0, 3))


    if np.isfinite(r_sq) and r_sq < 0.7:
        print(f"   ! Low R² ({r_sq:.3f}) for {label} — check QC plot.")

    # Save QC plot if path given
    if out_plot_path is not None:
        ensure_dir(os.path.dirname(out_plot_path))
        try:
            fm.plot(plot_peaks="shade", save_fig=True, file_name=out_plot_path,
                    file_path="")
            plt.close("all")
        except Exception as e:
            print(f"   ! FOOOF plot failed: {e}")

    # Build result dict from extracted params
    result = {
        "r_squared":          float(r_sq),
        "error":              float(error),
        "aperiodic_offset":   ap_offset,
        "aperiodic_exponent": ap_exponent,
    }
    if FOOOF_APERIODIC_MODE == "knee":
        result["aperiodic_knee"] = ap_knee

    # Extract peak params (CF, PW, BW per peak, up to max_n_peaks)
    for i in range(FOOOF_MAX_N_PEAKS):
        if i < len(peaks):
            result[f"peak{i+1}_cf"] = float(peaks[i, 0])
            result[f"peak{i+1}_pw"] = float(peaks[i, 1])
            result[f"peak{i+1}_bw"] = float(peaks[i, 2])
        else:
            result[f"peak{i+1}_cf"] = float("nan")
            result[f"peak{i+1}_pw"] = float("nan")
            result[f"peak{i+1}_bw"] = float("nan")

    return result


# -----------------------------
# Summary plots
# -----------------------------
def plot_aperiodic_comparison(
    df: pd.DataFrame,
    out_png: str,
    out_svg: str,
    title: str,
    style: dict,
) -> None:
    """
    Paired plot of aperiodic exponent: baseline vs stimulation per animal.
    One line per animal, mean ± SEM overlaid.
    """
    ensure_dir(os.path.dirname(out_png))

    base_vals = df["aperiodic_exponent_baseline"].dropna().to_numpy(dtype=float)
    stim_vals = df["aperiodic_exponent_stim"].dropna().to_numpy(dtype=float)
    animals   = df["folder"].to_numpy()

    if len(base_vals) == 0:
        print(f"  ! No valid aperiodic data for {title}, skipping.")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    apply_figure_style(fig, style)
    apply_axes_style(ax, style)

    for b, s in zip(base_vals, stim_vals):
        ax.plot([0, 1], [b, s], color=style["neutral_color"],
                linewidth=0.8, alpha=0.6, zorder=2)
        ax.scatter([0], [b], color=style["baseline_color"],
                   s=18, zorder=3, edgecolors=style["text_color"], linewidths=0.5)
        ax.scatter([1], [s], color=style["stim_color"],
                   s=18, zorder=3, edgecolors=style["text_color"], linewidths=0.5)

    # Mean ± SEM
    for x_pos, vals, color in [(0, base_vals, style["baseline_color"]),
                                (1, stim_vals, style["stim_color"])]:
        mean = np.nanmean(vals)
        sem  = np.nanstd(vals, ddof=1) / np.sqrt(np.sum(np.isfinite(vals)))
        ax.errorbar(x_pos, mean, yerr=sem, fmt="o", color=style["text_color"],
                    markerfacecolor=color, markeredgecolor=style["text_color"],
                    markersize=7, elinewidth=1.2, capsize=3, zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Stimulation"])
    ax.set_ylabel("Aperiodic exponent")
    ax.set_title(f"{title} | n={len(base_vals)} animals")
    ax.set_xlim(-0.3, 1.3)
    ax.grid(False)

    # Print mean ± SEM for convenience
    print(
        f"   Aperiodic exponent: "
        f"baseline={np.nanmean(base_vals):.3f}±{np.nanstd(base_vals, ddof=1)/np.sqrt(len(base_vals)):.3f} | "
        f"stim={np.nanmean(stim_vals):.3f}±{np.nanstd(stim_vals, ddof=1)/np.sqrt(len(stim_vals)):.3f}"
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    plt.close(fig)



def plot_aperiodic_offset(
    df: pd.DataFrame,
    out_png: str,
    out_svg: str,
    title: str,
    style: dict,
) -> None:
    """
    Paired plot of aperiodic offset: baseline vs stimulation per animal.

    The offset is the y-intercept of the aperiodic fit in log-log space and
    reflects the overall power level of the aperiodic background, independent
    of the slope (exponent). It is more robust than the exponent when spectra
    contain dominant peaks that distort the aperiodic fit.
    """
    ensure_dir(os.path.dirname(out_png))

    base_vals = df["aperiodic_offset_baseline"].dropna().to_numpy(dtype=float)
    stim_vals = df["aperiodic_offset_stim"].dropna().to_numpy(dtype=float)

    if len(base_vals) == 0:
        print(f"  ! No valid aperiodic offset data for {title}, skipping.")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    apply_figure_style(fig, style)
    apply_axes_style(ax, style)

    for b, s in zip(base_vals, stim_vals):
        ax.plot([0, 1], [b, s], color=style["neutral_color"],
                linewidth=0.8, alpha=0.6, zorder=2)
        ax.scatter([0], [b], color=style["baseline_color"],
                   s=18, zorder=3, edgecolors=style["text_color"], linewidths=0.5)
        ax.scatter([1], [s], color=style["stim_color"],
                   s=18, zorder=3, edgecolors=style["text_color"], linewidths=0.5)

    for x_pos, vals, color in [(0, base_vals, style["baseline_color"]),
                                (1, stim_vals, style["stim_color"])]:
        mean = np.nanmean(vals)
        sem  = np.nanstd(vals, ddof=1) / np.sqrt(np.sum(np.isfinite(vals)))
        ax.errorbar(x_pos, mean, yerr=sem, fmt="o", color=style["text_color"],
                    markerfacecolor=color, markeredgecolor=style["text_color"],
                    markersize=7, elinewidth=1.2, capsize=3, zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Stimulation"])
    ax.set_ylabel("Aperiodic offset (log power)")
    ax.set_title(f"{title} | n={len(base_vals)} animals")
    ax.set_xlim(-0.3, 1.3)
    ax.grid(False)

    print(
        f"   Aperiodic offset: "
        f"baseline={np.nanmean(base_vals):.3f}\u00b1{np.nanstd(base_vals, ddof=1)/np.sqrt(len(base_vals)):.3f} | "
        f"stim={np.nanmean(stim_vals):.3f}\u00b1{np.nanstd(stim_vals, ddof=1)/np.sqrt(len(stim_vals)):.3f}"
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    plt.close(fig)

def plot_peak_cf_distribution(
    df_peaks: pd.DataFrame,
    out_png: str,
    out_svg: str,
    title: str,
    style: dict,
) -> None:
    """
    Scatter/strip plot of all extracted peak center frequencies,
    baseline vs stimulation, coloured by frequency band.
    """
    ensure_dir(os.path.dirname(out_png))

    band_colors = {
        "delta": "#4477AA",
        "theta": "#66CCEE",
        "beta":  "#CCBB44",
        "gamma": "#EE6677",
        "other": "#AAAAAA",
    }

    def _band(cf):
        for name, (lo, hi) in FREQ_BANDS.items():
            if lo <= cf < hi:
                return name
        return "other"

    fig, axes = plt.subplots(1, 2, figsize=(FIG_SIZE[0] * 1.4, FIG_SIZE[1]),
                             sharey=True)
    apply_figure_style(fig, style)

    for ax, condition in zip(axes, ["baseline", "stim"]):
        apply_axes_style(ax, style)
        col = f"peak_cf_{condition}"
        if col not in df_peaks.columns:
            continue
        cfs = df_peaks[col].dropna().to_numpy(dtype=float)
        if cfs.size == 0:
            continue
        colors = [band_colors[_band(cf)] for cf in cfs]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(cfs))
        ax.scatter(jitter, cfs, c=colors, s=10, alpha=0.7, zorder=3,
                   edgecolors="none")
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_xlabel(condition.capitalize())
        ax.set_ylabel("Peak CF (Hz)" if ax == axes[0] else "")
        ax.set_ylim(FOOOF_FREQ_RANGE)
        ax.grid(False)
        # Band boundary lines
        for _, (lo, hi) in FREQ_BANDS.items():
            ax.axhline(lo, color=style["neutral_color"], linewidth=0.4,
                       linestyle=":", alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=b) for b, c in band_colors.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=5, frameon=False,
               bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{title} | extracted peaks", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    plt.close(fig)




def plot_peak_cf_violin(
    df_peaks: pd.DataFrame,
    out_png: str,
    out_svg: str,
    title: str,
    style: dict,
) -> None:
    """
    Violin plot of peak center frequencies across the full frequency range,
    baseline vs stimulation, with individual data points overlaid.

    Shows the full distribution of extracted peak CFs across all animals
    and peaks. Each point is one peak from one animal. The violin shows
    the kernel density estimate of the distribution.
    Coloured by frequency band per the FREQ_BANDS setting.
    """
    ensure_dir(os.path.dirname(out_png))

    band_colors = {
        "delta": "#4477AA",
        "theta": "#66CCEE",
        "beta":  "#CCBB44",
        "gamma": "#EE6677",
        "other": "#AAAAAA",
    }

    def _band(cf):
        for name, (lo, hi) in FREQ_BANDS.items():
            if lo <= cf < hi:
                return name
        return "other"

    cfs_base = df_peaks["peak_cf_baseline"].dropna().to_numpy(dtype=float)
    cfs_stim = df_peaks["peak_cf_stim"].dropna().to_numpy(dtype=float)

    if cfs_base.size == 0 and cfs_stim.size == 0:
        print(f"  ! No peak CF data for violin plot {title}, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(FIG_SIZE[0] * 1.4, FIG_SIZE[1]),
                             sharey=True)
    apply_figure_style(fig, style)

    for ax, cfs, condition in zip(axes,
                                   [cfs_base, cfs_stim],
                                   ["Baseline", "Stimulation"]):
        apply_axes_style(ax, style)

        if cfs.size == 0:
            ax.set_xlabel(condition)
            continue

        # Violin
        parts = ax.violinplot(cfs, positions=[0], widths=0.6,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(
                style["baseline_color"] if condition == "Baseline"
                else style["stim_color"]
            )
            pc.set_alpha(0.35)
            pc.set_edgecolor(style["text_color"])
            pc.set_linewidth(0.8)
        parts["cmedians"].set_color(style["text_color"])
        parts["cmedians"].set_linewidth(1.2)

        # Jittered data points coloured by band
        rng    = np.random.default_rng(42)
        jitter = rng.uniform(-0.12, 0.12, size=len(cfs))
        colors = [band_colors[_band(cf)] for cf in cfs]
        ax.scatter(jitter, cfs, c=colors, s=12, alpha=0.8,
                   zorder=4, edgecolors=style["text_color"], linewidths=0.3)

        # Band boundary lines
        for _, (lo, hi) in FREQ_BANDS.items():
            ax.axhline(lo, color=style["neutral_color"], linewidth=0.4,
                       linestyle=":", alpha=0.5)

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_xlabel(condition, fontsize=6)
        ax.set_ylim(FOOOF_FREQ_RANGE)
        ax.grid(False)

    axes[0].set_ylabel("Peak CF (Hz)")

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=b) for b, c in band_colors.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=5,
               frameon=False, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{title} | peak CF violin", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    t0 = time.perf_counter()

    print("=" * 60)
    print("process_lfp_fooof.py — path configuration")
    print("Press Enter to accept the default path shown in brackets.")
    print("=" * 60)
    export_path_base   = _prompt_path("Processed data root:", _DEFAULT_EXPORT_PATH_BASE)
    relative_data_path = _prompt_path("Relative cohort path:", _DEFAULT_RELATIVE_DATA_PATH)
    print()

    manifests = list_manifests(export_path_base, relative_data_path)
    if not manifests:
        raise FileNotFoundError("No manifests found. Run common_ingest.py first.")

    if only_folder is not None:
        manifests = [
            m for m in manifests
            if os.path.basename(os.path.dirname(os.path.dirname(m))) == only_folder
        ]
        if not manifests:
            raise FileNotFoundError(f"No manifest found for only_folder='{only_folder}'.")

    print(f"Processing {len(manifests)} folder(s) for FOOOF analysis...")

    window_suffix = (
        f"adj_base{int(ADJUSTED_BASELINE_S)}s_stim{int(ADJUSTED_STIM_S)}s"
        if USE_ADJUSTED_WINDOWS else "full"
    )

    # Collect results: keyed by (region_name, kind)
    all_results: Dict[Tuple[str, str], List[dict]] = {}

    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        folder                = manifest["folder"]
        processed_folder_path = manifest["processed_folder_path"]
        block_path            = manifest.get("block_path", None)

        print(f"\n==============================")
        print(f"Folder: {folder}")

        pynapse_csv = os.path.join(
            processed_folder_path, "metadata", "Pynapse_call_log.csv"
        )
        if not os.path.isfile(pynapse_csv):
            print(f" - Missing Pynapse_call_log.csv, skipping.")
            continue

        epochs           = load_epochs_from_pynapse_csv(pynapse_csv, ignore_label=IGNORE_LABEL)
        baseline_windows = epochs_to_windows(epochs, BASELINE_LABEL)
        stim_windows     = build_stim_windows(epochs, block_path)

        if USE_ADJUSTED_WINDOWS:
            baseline_windows = adjust_baseline_window(baseline_windows, ADJUSTED_BASELINE_S)
            stim_windows     = adjust_stim_window(stim_windows, ADJUSTED_STIM_S)

        print(" - baseline:", summarize_window_durations(baseline_windows))
        print(" - stim:    ", summarize_window_durations(stim_windows))

        if not baseline_windows or not stim_windows:
            print(" - Missing windows, skipping.")
            continue

        noise_ref_data = load_noise_reference(processed_folder_path)
        if noise_ref_data is None:
            print(" - No noise reference; NLMS skipped.")

        regions = manifest.get("regions", [])
        if only_regions is not None:
            regions = [r for r in regions if r.get("region_name") in only_regions]

        for region in regions:
            region_name   = region["region_name"]
            print(f"\nRegion: {region_name}")

            lfp_cont_path = os.path.join(
                processed_folder_path, region_name, "LFP", "continuous"
            )

            kinds_to_process = []
            if PROCESS_REFERENCED:
                kinds_to_process.append("referenced")
            if PROCESS_UNREFERENCED:
                kinds_to_process.append("unreferenced")
            if PROCESS_REFERENCED_CAR:
                kinds_to_process.append("referenced_car")

            for kind in kinds_to_process:
                pkl_files = load_lfp_pkls(lfp_cont_path, kind)
                if not pkl_files:
                    print(f" - No {kind} LFP pkls.")
                    continue

                # Collect per-channel PSDs, then average within animal
                ch_psd_base: List[np.ndarray] = []
                ch_psd_stim: List[np.ndarray] = []
                ref_freqs: Optional[np.ndarray] = None

                for pkl_path in pkl_files:
                    with open(pkl_path, "rb") as f:
                        obj = pickle.load(f)

                    data    = np.asarray(obj["downsampled_data"], dtype=float)
                    fs      = float(obj["settings"]["downsampling_Hz"])
                    channel = obj.get("channel", "?")

                    if noise_ref_data is not None:
                        data = apply_adaptive_filter_nlms(data, noise_ref_data, fs)

                    seg_base = segment_data(data, fs, baseline_windows)
                    seg_stim = segment_data(data, fs, stim_windows)

                    min_len = int(fs * 2)
                    if seg_base.size < min_len or seg_stim.size < min_len:
                        print(f"   ! ch{channel}: epoch too short, skipping.")
                        continue

                    freqs_full, psd_base = compute_psd(seg_base, fs)
                    _,          psd_stim = compute_psd(seg_stim, fs)

                    # FOOOF requires linear power in a frequency range
                    # that starts above 0 Hz — keep full resolution
                    freq_mask = freqs_full >= FOOOF_FREQ_RANGE[0]
                    freqs_r   = freqs_full[freq_mask]

                    if ref_freqs is None:
                        ref_freqs = freqs_r
                    elif not np.allclose(freqs_r, ref_freqs, atol=1e-6):
                        print(f"   ! ch{channel}: freq axis mismatch, skipping.")
                        continue

                    ch_psd_base.append(psd_base[freq_mask])
                    ch_psd_stim.append(psd_stim[freq_mask])

                if not ch_psd_base or ref_freqs is None:
                    print(f" - No valid channels for {folder}/{region_name}/{kind}")
                    continue

                # Animal-mean PSD (one data point per animal per condition)
                mean_psd_base = np.mean(np.vstack(ch_psd_base), axis=0)
                mean_psd_stim = np.mean(np.vstack(ch_psd_stim), axis=0)

                print(
                    f" - Fitting FOOOF for {folder}/{region_name}/{kind} "
                    f"(n_channels={len(ch_psd_base)})"
                )

                # Output folder for QC plots
                out_root_animal = os.path.join(
                    export_path_base, relative_data_path,
                    "Postprocessing", "LFP_FOOOF",
                    region_name, kind, window_suffix, folder,
                )
                ensure_dir(out_root_animal)

                result_base = fit_fooof_single(
                    ref_freqs, mean_psd_base, label=f"{folder} baseline",
                    out_plot_path=os.path.join(
                        out_root_animal, f"FOOOF_baseline_{folder}.png"
                    ),
                )
                result_stim = fit_fooof_single(
                    ref_freqs, mean_psd_stim, label=f"{folder} stim",
                    out_plot_path=os.path.join(
                        out_root_animal, f"FOOOF_stim_{folder}.png"
                    ),
                )

                if result_base is None or result_stim is None:
                    print(f" - FOOOF fit failed for {folder}, skipping.")
                    continue

                # Build combined row
                row = {
                    "folder":      folder,
                    "region":      region_name,
                    "kind":        kind,
                    "window_mode": window_suffix,
                    "n_channels":  len(ch_psd_base),
                }
                for suffix, res in [("baseline", result_base), ("stim", result_stim)]:
                    for k, v in res.items():
                        row[f"{k}_{suffix}"] = v

                # Delta aperiodic exponent
                row["delta_aperiodic_exponent"] = (
                    result_stim["aperiodic_exponent"]
                    - result_base["aperiodic_exponent"]
                )

                key = (region_name, kind)
                all_results.setdefault(key, []).append(row)
                print(
                    f"   Exponent: baseline={result_base['aperiodic_exponent']:.3f} | "
                    f"stim={result_stim['aperiodic_exponent']:.3f} | "
                    f"Δ={row['delta_aperiodic_exponent']:.3f}"
                )

    # ---------------------------
    # Export group results
    # ---------------------------
    out_root = os.path.join(
        export_path_base, relative_data_path,
        "Postprocessing", "LFP_FOOOF",
    )
    ensure_dir(out_root)

    for (region_name, kind), rows in all_results.items():
        if not rows:
            continue

        df               = pd.DataFrame(rows)
        region_kind_root = os.path.join(out_root, region_name, kind, window_suffix)
        ensure_dir(region_kind_root)

        # Save full results CSV
        out_csv = os.path.join(
            region_kind_root,
            f"FOOOF_results_{region_name}_{kind}_{window_suffix}.csv",
        )
        df.to_csv(out_csv, index=False)
        print(f"\n✓ Saved FOOOF results: {out_csv}")

        # Aperiodic exponent paired plot
        plot_aperiodic_comparison(
            df=df,
            out_png=os.path.join(
                region_kind_root,
                f"FOOOF_aperiodic_exponent_{region_name}_{kind}_{window_suffix}.png",
            ),
            out_svg=os.path.join(
                region_kind_root,
                f"FOOOF_aperiodic_exponent_{region_name}_{kind}_{window_suffix}.svg",
            ),
            title=f"Aperiodic exponent | {region_name} | {kind} | {tool_name}",
            style=plot_style,
        )

        # Peak CF distribution (collect all peaks from all animals)
        peak_rows = []
        for _, row in df.iterrows():
            for i in range(1, FOOOF_MAX_N_PEAKS + 1):
                cf_b = row.get(f"peak{i}_cf_baseline", float("nan"))
                cf_s = row.get(f"peak{i}_cf_stim",     float("nan"))
                if np.isfinite(cf_b) or np.isfinite(cf_s):
                    peak_rows.append({
                        "folder":         row["folder"],
                        "peak_cf_baseline": cf_b,
                        "peak_cf_stim":     cf_s,
                    })

        if peak_rows:
            df_peaks = pd.DataFrame(peak_rows)
            df_peaks.to_csv(
                os.path.join(
                    region_kind_root,
                    f"FOOOF_peaks_{region_name}_{kind}_{window_suffix}.csv",
                ),
                index=False,
            )
            plot_peak_cf_distribution(
                df_peaks=df_peaks,
                out_png=os.path.join(
                    region_kind_root,
                    f"FOOOF_peak_CF_{region_name}_{kind}_{window_suffix}.png",
                ),
                out_svg=os.path.join(
                    region_kind_root,
                    f"FOOOF_peak_CF_{region_name}_{kind}_{window_suffix}.svg",
                ),
                title=f"Peak CFs | {region_name} | {kind} | {tool_name}",
                style=plot_style,
            )


        # Aperiodic offset paired plot
        plot_aperiodic_offset(
            df=df,
            out_png=os.path.join(
                region_kind_root,
                f"FOOOF_aperiodic_offset_{region_name}_{kind}_{window_suffix}.png",
            ),
            out_svg=os.path.join(
                region_kind_root,
                f"FOOOF_aperiodic_offset_{region_name}_{kind}_{window_suffix}.svg",
            ),
            title=f"Aperiodic offset | {region_name} | {kind} | {tool_name}",
            style=plot_style,
        )

        if peak_rows:
            # Violin plot of peak CF distribution
            plot_peak_cf_violin(
                df_peaks=df_peaks,
                out_png=os.path.join(
                    region_kind_root,
                    f"FOOOF_peak_CF_violin_{region_name}_{kind}_{window_suffix}.png",
                ),
                out_svg=os.path.join(
                    region_kind_root,
                    f"FOOOF_peak_CF_violin_{region_name}_{kind}_{window_suffix}.svg",
                ),
                title=f"Peak CFs | {region_name} | {kind} | {tool_name}",
                style=plot_style,
            )

        n_animals = df["folder"].nunique()
        print(f"✓ Done: {region_name}/{kind}/{window_suffix} (n={n_animals} animals)")

    dt = time.perf_counter() - t0
    print(f"\nDone. Total processing time: {dt:.1f}s")


if __name__ == "__main__":
    main()