# -*- coding: utf-8 -*-
"""
postprocess_spikes_psth_mainfigure.py

Postprocessing script for main-figure PSTH analysis.

Outputs
-------
For each (region, signal_kind):

1) Onset-aligned PSTH (group average ± SEM across animals)
   - Window: ALIGN_WINDOW_S around stim onset
   - Bin: ONSET_BIN_S (100 ms), no smoothing
   - Baseline period trace vs stim period trace
   - Vertical dashed line at t=0

2) Full-epoch PSTH (group average ± SEM across animals)
   - Baseline epoch vs full stimulation epoch
   - Bin: FULL_EPOCH_BIN_S (5 s), Gaussian-smoothed for display only
   - Dotted vertical marker at time of maximum suppression

pTrain vs call log
------------------
ChR2 stimulation uses short trigger pulses — call log is the reliable source.
vSWO and vLWO use continuous stimulation — pTrain is preferred.
USE_PTRAIN is set automatically from the inferred tool name.

Adjusted window mode
--------------------
For cohorts where the recording protocol differed (e.g. ChR2 animals with
120 s baseline / 480 s stimulation vs. 300 s / 300 s), windows can be
adjusted to a common length for pooling:

  USE_ADJUSTED_WINDOWS = True
  ADJUSTED_BASELINE_S  = 120.0   # last N seconds before stim onset
  ADJUSTED_STIM_S      = 300.0   # first N seconds after stim onset

When USE_ADJUSTED_WINDOWS = False, the full epoch durations from the
Pynapse call log are used (default behaviour).

Maximum deviation
-----------------
Detected from the UNSMOOTHED group-mean PSTH (5 s bins), as the bin where
(stim_mean - baseline_mean) is most negative. Independent of smoothing.

Aggregation hierarchy
---------------------
- Channels averaged within animal → one PSTH per animal
- Animals averaged across group → group mean ± SEM
"""

from __future__ import annotations

import os
import time
import pickle
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import matplotlib.pyplot as plt

from Functions_processing_spikes import (
    ensure_dir,
    load_json,
    list_manifests,
    load_spike_continuous_pkls,
    classify_signal_name,
    load_epochs_from_pynapse_csv,
    epochs_to_windows,
    load_ptrain_windows,
    merge_windows,
    summarize_window_durations,
    detect_spikes_threshold,
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
_DEFAULT_EXPORT_PATH_BASE = r"C:\Users\Juliana\Documents\_PhD\Data\_Processed"
_DEFAULT_RELATIVE_DATA_PATH = r"Jills_paper\ChR2_cohort\OFT_experimental"
# Alias for module-level calls (e.g. infer_tool_name).
relative_data_path = _DEFAULT_RELATIVE_DATA_PATH
# relative_data_path = r"Jills_paper\vSWO_cohort\OFT_experimental"
# relative_data_path = r"Jills_paper\vLWO_cohort\OFT_experimental"

only_folder: Optional[str] = None
only_regions: Optional[List[str]] = ["CA1_L"]  # CA1_L = ipsilateral stimulation site

PROCESS_UNREFERENCED = False
PROCESS_LMR          = True

# Detection — must match process_spikes_psth.py
THRESH_MULT    = 5.0
POLARITY       = "neg"
REFRACTORY_MS  = 1.0
REFINE_PEAK    = True
PEAK_SEARCH_MS = 2.0

# Epoch labels
BASELINE_LABEL = "In baseline state"
STIM_LABEL     = "In stimulation state"
IGNORE_LABEL   = "In start delay"

# PSTH settings
ALIGN_WINDOW_S   = (-2.0, 5.0)  # seconds around stim onset for onset PSTH
ONSET_BIN_S      = 0.1           # 100 ms bins (no smoothing)
FULL_EPOCH_BIN_S = 5.0           # 5 s bins — detection and display
SMOOTH_SIGMA_BINS = 2.0          # Gaussian sigma in bins (display only)
MIN_CHANNELS_PER_ANIMAL = 1

ONSET_YLIM:      Optional[Tuple[float, float]] = None
FULL_EPOCH_YLIM: Optional[Tuple[float, float]] = None

# Plot settings
PLOT_THEME          = "light"
STIM_COLOR_OVERRIDE = None

FIG_HEIGHT_MM = 45
FIG_ASPECT    = 1.8

# -----------------------------
# Adjusted window settings
# -----------------------------
# Set USE_ADJUSTED_WINDOWS = True to use a common window length across all
# animals regardless of their original recording protocol.
# Set to False to use the full epoch durations (default).
USE_ADJUSTED_WINDOWS = False   # True = adjusted | False = full epochs

ADJUSTED_BASELINE_S  = 120.0  # last N seconds before stimulation onset
ADJUSTED_STIM_S      = 300.0  # first N seconds after stimulation onset

tool_name  = infer_tool_name(relative_data_path)
stim_color = STIM_COLOR_OVERRIDE if STIM_COLOR_OVERRIDE is not None \
    else get_tool_color(tool_name)
plot_style = get_plot_style(theme=PLOT_THEME, stim_color=stim_color)

# ChR2 uses short trigger pulses in pTrain — call log is the reliable source.
# vSWO and vLWO use continuous stimulation — pTrain is preferred.
USE_PTRAIN = tool_name != "ChR2"

set_global_plot_style(
    theme=PLOT_THEME,
    font_family="Arial",
    base_font_size=6,
    axes_title_size=7,
    axes_label_size=6,
    tick_label_size=5.5,
    legend_font_size=5.5,
    axes_linewidth=1.0,
)
set_publication_fontsizes(FIG_HEIGHT_MM)
FIG_SIZE = get_figure_size(height_mm=FIG_HEIGHT_MM, aspect_ratio=FIG_ASPECT)

print(
    f"Tool: {tool_name} | USE_PTRAIN: {USE_PTRAIN} | "
    f"Adjusted windows: {USE_ADJUSTED_WINDOWS}"
    + (
        f" (baseline={ADJUSTED_BASELINE_S:.0f} s, stim={ADJUSTED_STIM_S:.0f} s)"
        if USE_ADJUSTED_WINDOWS else ""
    )
)


# -----------------------------
# Helpers
# -----------------------------
def adjust_baseline_window(
    baseline_windows: List[Tuple[float, float]],
    duration_s: float,
) -> List[Tuple[float, float]]:
    """Last `duration_s` seconds of the baseline epoch."""
    adjusted = []
    for start, end in baseline_windows:
        dur = end - start
        adjusted.append((end - min(dur, duration_s), end))
    return adjusted


def adjust_stim_window(
    stim_windows: List[Tuple[float, float]],
    duration_s: float,
) -> List[Tuple[float, float]]:
    """First `duration_s` seconds of the stimulation epoch."""
    adjusted = []
    for start, end in stim_windows:
        dur = end - start
        adjusted.append((start, start + min(dur, duration_s)))
    return adjusted


def build_stim_windows(
    epochs,
    block_path: Optional[str],
) -> List[Tuple[float, float]]:
    """Return stim windows from pTrain (vSWO/vLWO) or call log (ChR2)."""
    stim_windows_calllog = epochs_to_windows(epochs, STIM_LABEL)

    if not USE_PTRAIN or block_path is None:
        return stim_windows_calllog

    stim_duration_hint_s = None
    if stim_windows_calllog:
        stim_duration_hint_s = float(
            np.median([b - a for a, b in stim_windows_calllog])
        )

    try:
        ptrain = merge_windows(
            load_ptrain_windows(
                block_path, stim_duration_hint_s=stim_duration_hint_s
            ),
            min_gap_s=0.0,
        )
        if ptrain:
            print(f" - Using pTrain: {len(ptrain)} epoch(s).")
            return ptrain
        else:
            print(" - pTrain empty; falling back to call log.")
    except FileNotFoundError:
        print(" - Raw path not found; falling back to call log.")
    except Exception as e:
        print(f" - pTrain error ({e}); falling back to call log.")
        traceback.print_exc()

    return stim_windows_calllog


def make_onset_psth(
    spike_times_s: np.ndarray,
    stim_windows: List[Tuple[float, float]],
    baseline_windows: List[Tuple[float, float]],
    bin_edges: np.ndarray,
    align_window_s: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    tmin, tmax = align_window_s
    bin_widths = np.diff(bin_edges)

    def _rate(onsets: List[float]) -> np.ndarray:
        spikes = []
        for o in onsets:
            st = spike_times_s[
                (spike_times_s >= o + tmin) & (spike_times_s <= o + tmax)
            ] - o
            spikes.append(st)
        n = max(len(spikes), 1)
        counts, _ = np.histogram(
            np.concatenate(spikes) if spikes else np.zeros(0), bins=bin_edges
        )
        return counts / (bin_widths * n)

    return (
        _rate([b for _, b in baseline_windows]),
        _rate([a for a, _ in stim_windows]),
    )


def make_full_epoch_psth(
    spike_times_s: np.ndarray,
    windows: List[Tuple[float, float]],
    bin_edges: np.ndarray,
) -> np.ndarray:
    if not windows:
        return np.zeros(len(bin_edges) - 1)
    bin_widths = np.diff(bin_edges)
    tmax = bin_edges[-1]
    spikes = []
    for (a, b) in windows:
        st = spike_times_s[(spike_times_s >= a) & (spike_times_s < b)] - a
        spikes.append(st[st <= tmax])
    counts, _ = np.histogram(
        np.concatenate(spikes) if spikes else np.zeros(0), bins=bin_edges
    )
    return counts / (bin_widths * max(len(windows), 1))


def smooth_psth(rate: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return rate.copy()
    return gaussian_filter1d(rate.astype(float), sigma=sigma_bins)


def mean_sem(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    n    = np.sum(np.isfinite(x), axis=0)
    sem  = np.nanstd(x, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    sem[n < 2] = np.nan
    if np.any(n < 2):
        print(f"   ! mean_sem: {int(np.sum(n < 2))} bin(s) with n<2 — SEM=nan.")
    return mean, sem


def detect_max_deviation(
    mean_stim: np.ndarray,
    mean_base: np.ndarray,
    bin_centers: np.ndarray,
) -> Tuple[float, float]:
    """
    Time and magnitude of maximum suppression from UNSMOOTHED data.
    Defined as the bin where (stim - baseline) is most negative.
    """
    delta = mean_stim - mean_base
    idx   = int(np.argmin(delta))
    return float(bin_centers[idx]), float(delta[idx])


def plot_onset_psth(
    bin_edges, mean_base, sem_base, mean_stim, sem_stim,
    out_png, out_svg, title, style, n_animals, ylim,
) -> None:
    ensure_dir(os.path.dirname(out_png))
    centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    apply_figure_style(fig, style); apply_axes_style(ax, style)

    ax.plot(centers, mean_base, color=style["baseline_color"],
            linewidth=1.8, label="Baseline period")
    ax.fill_between(centers, mean_base - sem_base, mean_base + sem_base,
                    color=style["baseline_color"], alpha=0.25)
    ax.plot(centers, mean_stim, color=style["stim_color"],
            linewidth=1.8, label="Stimulation period")
    ax.fill_between(centers, mean_stim - sem_stim, mean_stim + sem_stim,
                    color=style["stim_color"], alpha=0.25)
    ax.axvline(0.0, color=style["text_color"], linewidth=1.0,
               linestyle="--", label="Stim onset")
    ax.axvspan(bin_edges[0], 0.0, alpha=0.05, color=style["baseline_color"])

    ax.set_xlim(bin_edges[0], bin_edges[-1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Time from stimulation onset (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(f"{title} | n={n_animals} animals")
    ax.legend(frameon=False)
    ax.grid(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    plt.close(fig)


def plot_full_epoch_psth(
    bin_edges, mean_base, sem_base, mean_stim, sem_stim,
    t_max_dev_s, max_dev_hz,
    out_png, out_svg, title, style, n_animals, ylim,
) -> None:
    ensure_dir(os.path.dirname(out_png))
    centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0

    # Gaussian smoothing for display only
    mean_base_sm = smooth_psth(mean_base, SMOOTH_SIGMA_BINS)
    sem_base_sm  = smooth_psth(sem_base,  SMOOTH_SIGMA_BINS)
    mean_stim_sm = smooth_psth(mean_stim, SMOOTH_SIGMA_BINS)
    sem_stim_sm  = smooth_psth(sem_stim,  SMOOTH_SIGMA_BINS)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    apply_figure_style(fig, style); apply_axes_style(ax, style)

    ax.plot(centers, mean_base_sm, color=style["baseline_color"],
            linewidth=1.8, label="Baseline")
    ax.fill_between(centers, mean_base_sm - sem_base_sm, mean_base_sm + sem_base_sm,
                    color=style["baseline_color"], alpha=0.25)
    ax.plot(centers, mean_stim_sm, color=style["stim_color"],
            linewidth=1.8, label="Stimulation")
    ax.fill_between(centers, mean_stim_sm - sem_stim_sm, mean_stim_sm + sem_stim_sm,
                    color=style["stim_color"], alpha=0.25)

    # Marker from unsmoothed detection — stable across smoothing parameters
    ax.axvline(
        t_max_dev_s,
        color=style["stim_color"], linewidth=1.2, linestyle=":",
        label=f"Max suppression ({t_max_dev_s:.0f} s)",
    )

    ax.set_xlim(bin_edges[0], bin_edges[-1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Time within epoch (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(f"{title} | n={n_animals} animals")
    ax.legend(frameon=False)
    ax.grid(False)

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
    print("postprocess_spikes_psth_mainfigure.py — path configuration")
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

    print(f"Postprocessing {len(manifests)} folder(s) for main-figure PSTH plots...")

    window_suffix = (
        f"adj_base{int(ADJUSTED_BASELINE_S)}s_stim{int(ADJUSTED_STIM_S)}s"
        if USE_ADJUSTED_WINDOWS else "full"
    )

    onset_bin_edges = np.arange(
        ALIGN_WINDOW_S[0], ALIGN_WINDOW_S[1] + ONSET_BIN_S, ONSET_BIN_S
    )

    grouped_onset:     Dict[Tuple[str, str], Dict[str, List]] = {}
    grouped_full:      Dict[Tuple[str, str], Dict[str, List]] = {}
    grouped_durations: Dict[Tuple[str, str], List[float]]     = {}

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

        # Apply window adjustments if requested
        if USE_ADJUSTED_WINDOWS:
            baseline_windows = adjust_baseline_window(baseline_windows, ADJUSTED_BASELINE_S)
            stim_windows     = adjust_stim_window(stim_windows, ADJUSTED_STIM_S)

        print(" - baseline:", summarize_window_durations(baseline_windows))
        print(" - stim:    ", summarize_window_durations(stim_windows))

        if not baseline_windows or not stim_windows:
            print(" - Missing baseline or stim windows, skipping.")
            continue

        dur_base   = float(np.median([b - a for a, b in baseline_windows if b > a]))
        dur_stim   = float(np.median([b - a for a, b in stim_windows    if b > a]))
        tmax_full  = min(dur_base, dur_stim)

        if tmax_full <= 0:
            print(" - Zero-duration epoch, skipping.")
            continue

        full_bin_edges = np.arange(0.0, tmax_full + FULL_EPOCH_BIN_S, FULL_EPOCH_BIN_S)

        regions = manifest.get("regions", [])
        if only_regions is not None:
            regions = [r for r in regions if r.get("region_name") in only_regions]

        for region in regions:
            region_name = region["region_name"]
            print(f"\nRegion: {region_name}")

            spikes_cont = os.path.join(
                processed_folder_path, region_name, "Spikes", "continuous"
            )
            pkl_files = load_spike_continuous_pkls(spikes_cont)
            if not pkl_files:
                print(f" - No continuous pkls in {spikes_cont}")
                continue

            for kind in ["unreferenced", "lmr"]:
                if kind == "unreferenced" and not PROCESS_UNREFERENCED:
                    continue
                if kind == "lmr" and not PROCESS_LMR:
                    continue

                kind_pkls = [
                    p for p in pkl_files
                    if classify_signal_name(
                        os.path.splitext(os.path.basename(p))[0]
                    ) == kind
                ]
                if not kind_pkls:
                    print(f" - No {kind} pkls found.")
                    continue

                ch_on_base,   ch_on_stim   = [], []
                ch_full_base, ch_full_stim = [], []

                for pkl_path in kind_pkls:
                    with open(pkl_path, "rb") as f:
                        obj = pickle.load(f)

                    x  = obj.get("filtered_data", None)
                    fs = obj.get("fs_original",  None)
                    if x is None or fs is None:
                        continue

                    peaks_samp, _ = detect_spikes_threshold(
                        x=np.asarray(x), fs=float(fs),
                        b=THRESH_MULT, polarity=POLARITY,
                        refractory_ms=REFRACTORY_MS,
                        refine_peak=REFINE_PEAK,
                        peak_search_ms=PEAK_SEARCH_MS,
                    )
                    spike_times_s = peaks_samp / float(fs)

                    rb_on, rs_on = make_onset_psth(
                        spike_times_s, stim_windows, baseline_windows,
                        onset_bin_edges, ALIGN_WINDOW_S,
                    )
                    rb_full = make_full_epoch_psth(
                        spike_times_s, baseline_windows, full_bin_edges
                    )
                    rs_full = make_full_epoch_psth(
                        spike_times_s, stim_windows, full_bin_edges
                    )

                    ch_on_base.append(rb_on)
                    ch_on_stim.append(rs_on)
                    ch_full_base.append(rb_full)
                    ch_full_stim.append(rs_full)

                if len(ch_on_base) < MIN_CHANNELS_PER_ANIMAL:
                    print(f" - Not enough channels for {folder}/{region_name}/{kind}")
                    continue

                key = (region_name, kind)

                grouped_onset.setdefault(key, {"base": [], "stim": [], "animals": []})
                grouped_onset[key]["base"].append(
                    np.mean(np.vstack(ch_on_base), axis=0)
                )
                grouped_onset[key]["stim"].append(
                    np.mean(np.vstack(ch_on_stim), axis=0)
                )
                grouped_onset[key]["animals"].append(folder)

                grouped_full.setdefault(key, {"base": [], "stim": [], "animals": []})
                grouped_full[key]["base"].append(
                    np.mean(np.vstack(ch_full_base), axis=0)
                )
                grouped_full[key]["stim"].append(
                    np.mean(np.vstack(ch_full_stim), axis=0)
                )
                grouped_full[key]["animals"].append(folder)

                grouped_durations.setdefault(key, []).append(tmax_full)

                print(
                    f" - Added {folder}/{region_name}/{kind} "
                    f"(n_ch={len(ch_on_base)}, tmax={tmax_full:.1f}s)"
                )

    # -----------------------------
    # Export
    # -----------------------------
    out_root = os.path.join(
        export_path_base, relative_data_path,
        "Postprocessing", "Spike_PSTH_MainFigure",
    )
    ensure_dir(out_root)

    for (region_name, kind), d in grouped_onset.items():
        if not d["base"]:
            continue

        n_animals        = len(d["animals"])
        region_kind_root = os.path.join(out_root, region_name, kind, window_suffix)
        ensure_dir(region_kind_root)

        # Onset-aligned PSTH (no smoothing)
        arr_ob = np.vstack(d["base"])
        arr_os = np.vstack(d["stim"])
        mean_ob, sem_ob = mean_sem(arr_ob)
        mean_os, sem_os = mean_sem(arr_os)

        centers_on = onset_bin_edges[:-1] + np.diff(onset_bin_edges) / 2.0
        pd.DataFrame({
            "bin_center_s":     centers_on,
            "baseline_mean_hz": mean_ob,
            "baseline_sem_hz":  sem_ob,
            "stim_mean_hz":     mean_os,
            "stim_sem_hz":      sem_os,
            "window_mode":      window_suffix,
        }).to_csv(
            os.path.join(region_kind_root, "group_onset_psth.csv"), index=False
        )

        plot_onset_psth(
            bin_edges=onset_bin_edges,
            mean_base=mean_ob, sem_base=sem_ob,
            mean_stim=mean_os, sem_stim=sem_os,
            out_png=os.path.join(
                region_kind_root,
                f"OnsetPSTH_{region_name}_{kind}_{window_suffix}.png",
            ),
            out_svg=os.path.join(
                region_kind_root,
                f"OnsetPSTH_{region_name}_{kind}_{window_suffix}.svg",
            ),
            title=f"Onset PSTH | {region_name} | {kind} | {window_suffix}",
            style=plot_style, n_animals=n_animals, ylim=ONSET_YLIM,
        )
        print(f"\u2713 Saved onset PSTH: {region_name}/{kind}/{window_suffix} (n={n_animals})")

        # Full-epoch PSTH
        full_d = grouped_full.get((region_name, kind), {})
        if not full_d.get("base"):
            continue

        tmax_common = float(np.min(grouped_durations[(region_name, kind)]))
        full_bin_edges_common = np.arange(
            0.0, tmax_common + FULL_EPOCH_BIN_S, FULL_EPOCH_BIN_S
        )
        n_bins = len(full_bin_edges_common) - 1

        arr_fb = np.vstack([r[:n_bins] for r in full_d["base"]])
        arr_fs = np.vstack([r[:n_bins] for r in full_d["stim"]])
        mean_fb, sem_fb = mean_sem(arr_fb)
        mean_fs, sem_fs = mean_sem(arr_fs)

        centers_full = (
            full_bin_edges_common[:-1] + np.diff(full_bin_edges_common) / 2.0
        )

        # Detect max deviation from UNSMOOTHED data
        t_max_dev_s, max_dev_hz = detect_max_deviation(mean_fs, mean_fb, centers_full)
        print(
            f"   - Max suppression (unsmoothed {FULL_EPOCH_BIN_S:.0f}s bins): "
            f"t = {t_max_dev_s:.1f} s, Δ = {max_dev_hz:.2f} Hz"
        )

        pd.DataFrame({
            "bin_center_s":         centers_full,
            "baseline_mean_hz":     mean_fb,
            "baseline_sem_hz":      sem_fb,
            "stim_mean_hz":         mean_fs,
            "stim_sem_hz":          sem_fs,
            "delta_hz":             mean_fs - mean_fb,
            "window_mode":          window_suffix,
            "max_deviation_time_s": [t_max_dev_s] + [np.nan] * (n_bins - 1),
            "max_deviation_hz":     [max_dev_hz]  + [np.nan] * (n_bins - 1),
        }).to_csv(
            os.path.join(region_kind_root, "group_full_epoch_psth.csv"), index=False
        )

        plot_full_epoch_psth(
            bin_edges=full_bin_edges_common,
            mean_base=mean_fb, sem_base=sem_fb,
            mean_stim=mean_fs, sem_stim=sem_fs,
            t_max_dev_s=t_max_dev_s,
            max_dev_hz=max_dev_hz,
            out_png=os.path.join(
                region_kind_root,
                f"FullEpochPSTH_{region_name}_{kind}_{window_suffix}.png",
            ),
            out_svg=os.path.join(
                region_kind_root,
                f"FullEpochPSTH_{region_name}_{kind}_{window_suffix}.svg",
            ),
            title=f"Full-epoch PSTH | {region_name} | {kind} | {window_suffix}",
            style=plot_style, n_animals=n_animals, ylim=FULL_EPOCH_YLIM,
        )
        print(
            f"\u2713 Saved full-epoch PSTH: {region_name}/{kind}/{window_suffix} "
            f"(n={n_animals}, tmax={tmax_common:.1f}s)"
        )

    dt = time.perf_counter() - t0
    print(f"\nDone. Total processing time: {dt:.1f}s")


if __name__ == "__main__":
    main()