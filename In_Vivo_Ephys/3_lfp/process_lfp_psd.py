# -*- coding: utf-8 -*-
"""
process_lfp_psd.py

Manifest-driven LFP power spectral density (PSD) analysis.

Workflow
--------
1. Load preprocessed LFP pkl files (referenced and/or unreferenced)
2. Load noise reference pkl and apply NLMS adaptive filtering per channel
3. Segment into baseline and stimulation epochs using Pynapse call log
4. Compute Welch PSD per channel per epoch
5. Average channels within animal -> one PSD per animal
6. Average animals -> group mean +/- SEM
7. Statistics: paired t-test + FDR (saved to CSV only, not plotted)
8. Effect size: Cohen's d per frequency bin
9. Delta power +/- 95% bootstrap CI per frequency bin
10. Export: group PSD CSV, 4-panel main figure, supplement table PNG/SVG

Signal kinds
------------
PROCESS_REFERENCED     : preprocessed_lfp_referenced_*      (differential pairs)
PROCESS_UNREFERENCED   : preprocessed_lfp_unreferenced_*     (single-wire, no reference)
PROCESS_REFERENCED_CAR : preprocessed_lfp_referenced_car_*  (CAR-corrected, early ChR2
                         animals 0714/0895/0896 with floating reference headstage)

Set the corresponding flag to True and run the script once per kind.
Results are saved in separate subfolders per kind.

Aggregation hierarchy
---------------------
  Channel PSDs averaged within animal -> one animal-mean PSD
  Animal PSDs averaged across group   -> group mean +/- SEM
  Statistics run across animals (n = number of animals), not channels

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

Cohen's d thresholds
--------------------
Classical Cohen (1988) benchmarks: small=0.2, medium=0.5, large=0.8.
Direction set per cohort via COHENS_D_DIRECTION:
  ChR2, vSWO -> "suppression" (d < 0): thresholds are negative
  vLWO       -> "increase"   (d > 0): thresholds are positive
PSD panel shades at medium threshold. All three levels reported in
supplement table and band summary panel.
FDR p-values saved to CSV only.
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
import matplotlib.patches as mpatches
from scipy.signal import welch
from scipy.stats import ttest_rel
from scipy.ndimage import label as ndlabel
from statsmodels.stats.multitest import multipletests

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
only_regions: Optional[List[str]] = ["CA1_R"]

# Signal kinds to process — set one True per run to save memory
PROCESS_REFERENCED     = False   # preprocessed_lfp_referenced_*      (standard LMR)
PROCESS_UNREFERENCED   = False   # preprocessed_lfp_unreferenced_*     (no referencing)
PROCESS_REFERENCED_CAR = True    # preprocessed_lfp_referenced_car_*   (CAR-corrected, use for early ChR2 animals 0714/0895/0896)

BASELINE_LABEL = "In baseline state"
STIM_LABEL     = "In stimulation state"
IGNORE_LABEL   = "In start delay"

# PSD settings
NPERSEG_S  = 2.0
FREQ_RANGE = (0.0, 100.0)

# NLMS adaptive filter settings
NLMS_MU            = 0.015
NLMS_N_TAPS_FACTOR = 3

# Cohen's d thresholds — classical benchmarks (Cohen, 1988)
# ChR2 / vSWO — suppression (negative d):
COHENS_D_THRESHOLDS: Dict[str, float] = {
    "small":  -0.2,
    "medium": -0.5,
    "large":  -0.8,
}
COHENS_D_DIRECTION = "suppression"   # "suppression" or "increase"

# vLWO — increase (uncomment and switch COHENS_D_DIRECTION):
# COHENS_D_THRESHOLDS: Dict[str, float] = {
#     "small":  0.2,
#     "medium": 0.5,
#     "large":  0.8,
# }
# COHENS_D_DIRECTION = "increase"

# Bootstrap CI settings
N_BOOTSTRAP    = 5000
BOOTSTRAP_CI   = 95
BOOTSTRAP_SEED = 42

# Plot settings
PLOT_THEME          = "light"
STIM_COLOR_OVERRIDE = None

FIG_HEIGHT_MM = 50
FIG_ASPECT    = 2.0

# -----------------------------
# Adjusted window settings
# -----------------------------
USE_ADJUSTED_WINDOWS = True    # True = adjusted | False = full epochs | use for ChR cohort
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

THRESHOLD_STYLES = {
    "small":  {"color": "#AAAAAA", "linewidth": 0.8, "linestyle": "--",
               "label": "Small (|d|=0.2)"},
    "medium": {"color": "#FF8800", "linewidth": 1.0, "linestyle": "--",
               "label": "Medium (|d|=0.5)"},
    "large":  {"color": "#CC0000", "linewidth": 1.2, "linestyle": ":",
               "label": "Large (|d|=0.8)"},
}
BAND_ALPHA = {"small": 0.40, "medium": 0.55, "large": 0.70}


# -----------------------------
# File loading
# -----------------------------
def load_lfp_pkls(lfp_cont_path: str, kind: str) -> List[str]:
    """
    Load LFP pkl files for a given signal kind.

    The kind string maps directly to the file prefix:
      "referenced"     -> preprocessed_lfp_referenced_*
      "unreferenced"   -> preprocessed_lfp_unreferenced_*
      "referenced_car" -> preprocessed_lfp_referenced_car_*

    Note: "referenced" uses an exact prefix match to avoid accidentally
    loading "referenced_car" files when PROCESS_REFERENCED is True.
    """
    if not os.path.isdir(lfp_cont_path):
        return []

    prefix = f"preprocessed_lfp_{kind}_"
    files  = sorted(
        f for f in os.listdir(lfp_cont_path)
        if f.startswith(prefix) and f.endswith(".pkl")
    )

    # Guard: when kind="referenced", exclude referenced_car files
    # (their prefix "preprocessed_lfp_referenced_car_" also starts with
    #  "preprocessed_lfp_referenced_", so we filter them out explicitly)
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
    data: np.ndarray,
    reference: np.ndarray,
    mu: float = 0.01,
    n_taps: int = 16,
) -> np.ndarray:
    if len(reference) != len(data):
        reference = np.interp(
            np.linspace(0, 1, len(data)),
            np.linspace(0, 1, len(reference)),
            reference,
        )
    weights       = np.zeros(n_taps)
    filtered_data = np.zeros_like(data)
    ref_buffer    = np.zeros(n_taps)
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
    stim_windows_calllog: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Return stim windows from pTrain (vSWO/vLWO) or call log (ChR2)."""
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


def segment_data(
    data: np.ndarray,
    fs: float,
    windows: List[Tuple[float, float]],
) -> np.ndarray:
    segments = []
    n = len(data)
    for a, b in windows:
        start = max(0, int(round(a * fs)))
        end   = min(n, int(round(b * fs)))
        if end > start:
            segments.append(data[start:end])
    return np.concatenate(segments) if segments else np.zeros(0)


# -----------------------------
# PSD helpers
# -----------------------------
def compute_psd(
    data: np.ndarray,
    fs: float,
    nperseg_s: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = int(round(nperseg_s * fs))
    return welch(data, fs=fs, nperseg=nperseg)


# -----------------------------
# Statistics helpers
# -----------------------------
def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    diff      = np.mean(y) - np.mean(x)
    pooled_sd = np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2)
    if pooled_sd < 1e-10:
        return 0.0
    return float(diff / pooled_sd)


def bootstrap_ci_diff(
    data_base: np.ndarray,
    data_stim: np.ndarray,
    n_bootstrap: int = 5000,
    ci: float = 95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng   = np.random.default_rng(seed=seed)
    diffs = np.array([
        np.mean(data_stim[idx] - data_base[idx])
        for idx in (rng.integers(0, len(data_base), size=len(data_base))
                    for _ in range(n_bootstrap))
    ])
    lower = float(np.percentile(diffs, (100 - ci) / 2))
    upper = float(np.percentile(diffs, 100 - (100 - ci) / 2))
    return float(np.mean(diffs)), lower, upper


def extract_significant_bands(
    freqs: np.ndarray,
    cohens_d_arr: np.ndarray,
    threshold: float,
    direction: str = "suppression",
) -> List[Tuple[float, float]]:
    mask = cohens_d_arr < threshold if direction == "suppression" \
        else cohens_d_arr > threshold
    labeled, n_features = ndlabel(mask)
    return [
        (float(freqs[np.where(labeled == i)[0][0]]),
         float(freqs[np.where(labeled == i)[0][-1]]))
        for i in range(1, n_features + 1)
    ]


def fdr_correct(p_values: np.ndarray) -> np.ndarray:
    valid  = np.isfinite(p_values)
    p_corr = np.full_like(p_values, np.nan)
    if np.any(valid):
        _, p_corr[valid], _, _ = multipletests(p_values[valid], method="fdr_bh")
    return p_corr


# -----------------------------
# Main figure
# -----------------------------
def plot_psd_group(
    freqs, mean_base, sem_base, mean_stim, sem_stim,
    p_corr, cohens_d_arr, delta_mean, delta_ci_lower, delta_ci_upper,
    out_png, out_svg, title, style, n_animals, freq_range,
    cohens_d_thresholds, cohens_d_direction="suppression",
) -> None:
    ensure_dir(os.path.dirname(out_png))

    freq_mask    = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    f            = freqs[freq_mask]
    all_bands    = {
        level: extract_significant_bands(
            freqs, cohens_d_arr, thr, direction=cohens_d_direction
        )
        for level, thr in cohens_d_thresholds.items()
    }
    medium_bands = all_bands.get("medium", [])
    large_bands  = all_bands.get("large",  [])

    fig, axes = plt.subplots(
        1, 4, figsize=(FIG_SIZE[0] * 2.7, FIG_SIZE[1]),
        gridspec_kw={"width_ratios": [2, 1.2, 1.2, 1.4]},
    )
    for ax in axes:
        apply_figure_style(fig, style)
        apply_axes_style(ax, style)

    ax1 = axes[0]
    for fl, fh in medium_bands:
        ax1.axvspan(fl, fh, color=style["stim_color"], alpha=0.12, zorder=0)
    ax1.plot(f, mean_base[freq_mask], color=style["baseline_color"],
             linewidth=1.8, label="Baseline")
    ax1.fill_between(f, (mean_base - sem_base)[freq_mask],
                     (mean_base + sem_base)[freq_mask],
                     color=style["baseline_color"], alpha=0.25)
    ax1.plot(f, mean_stim[freq_mask], color=style["stim_color"],
             linewidth=1.8, label="Stimulation")
    ax1.fill_between(f, (mean_stim - sem_stim)[freq_mask],
                     (mean_stim + sem_stim)[freq_mask],
                     color=style["stim_color"], alpha=0.25)
    ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("Power (\u03bcV\u00b2/Hz)")
    ax1.set_title(f"PSD | n={n_animals}"); ax1.legend(frameon=False, fontsize=5)
    ax1.set_xlim(freq_range); ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    for fl, fh in medium_bands:
        ax2.axvspan(fl, fh, color=style["stim_color"], alpha=0.10, zorder=0)
    ax2.plot(f, delta_mean[freq_mask], color=style["stim_color"],
             linewidth=1.8, label="\u0394 power")
    ax2.fill_between(f, delta_ci_lower[freq_mask], delta_ci_upper[freq_mask],
                     color=style["neutral_color"], alpha=0.35, label="95% CI")
    ax2.axhline(0, color=style["text_color"], linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("\u0394 power (\u03bcV\u00b2/Hz)")
    ax2.set_title("\u0394 Power \u00b1 95% CI"); ax2.legend(frameon=False, fontsize=5)
    ax2.set_xlim(freq_range)

    ax3 = axes[2]
    for fl, fh in large_bands:
        ax3.axvspan(fl, fh, color=style["stim_color"], alpha=0.12, zorder=0)
    ax3.plot(f, cohens_d_arr[freq_mask], color=style["stim_color"],
             linewidth=1.8, label="Cohen's d")
    ax3.axhline(0, color=style["text_color"], linewidth=0.8, linestyle="--")
    for level, thr in cohens_d_thresholds.items():
        ts = THRESHOLD_STYLES[level]
        ax3.axhline(thr, color=ts["color"], linewidth=ts["linewidth"],
                    linestyle=ts["linestyle"], label=ts["label"], zorder=3)
        ax3.axhline(-thr, color=ts["color"], linewidth=ts["linewidth"] * 0.6,
                    linestyle=ts["linestyle"], alpha=0.35, zorder=3)
    ax3.set_xlabel("Frequency (Hz)"); ax3.set_ylabel("Cohen's d")
    ax3.set_title("Effect size")
    ax3.legend(frameon=False, fontsize=4.5, loc="lower right")
    ax3.set_xlim(freq_range)

    ax4         = axes[3]
    level_order  = ["large", "medium", "small"]
    level_labels = {"large": "Large", "medium": "Medium", "small": "Small"}
    y_cursor     = 0; y_ticks = []; y_tick_labels = []; any_bands = False

    for level in level_order:
        bands = all_bands.get(level, [])
        thr   = cohens_d_thresholds[level]
        ts    = THRESHOLD_STYLES[level]
        if not bands:
            ax4.text((freq_range[0] + freq_range[1]) / 2, y_cursor, "None",
                     ha="center", va="center", fontsize=4.5,
                     color=style["neutral_color"])
            y_ticks.append(y_cursor)
            y_tick_labels.append(f"{level_labels[level]}\n(|d|={abs(thr):.1f})")
            y_cursor += 1.2
            continue
        any_bands    = True
        band_y_start = y_cursor
        for fl, fh in bands:
            idx    = (freqs >= fl) & (freqs <= fh)
            d_mean = float(np.mean(cohens_d_arr[idx]))
            ax4.barh(y=y_cursor, width=fh - fl, left=fl, height=0.6,
                     color=ts["color"], alpha=BAND_ALPHA[level],
                     edgecolor=ts["color"], linewidth=0.8)
            ax4.text((fl + fh) / 2, y_cursor,
                     f"{fl:.1f}-{fh:.1f} Hz\nd={d_mean:.2f}",
                     ha="center", va="center", fontsize=4.0,
                     color=style["text_color"])
            y_cursor += 0.9
        group_mid = (band_y_start + y_cursor - 0.9) / 2
        y_ticks.append(group_mid)
        y_tick_labels.append(f"{level_labels[level]}\n(|d|={abs(thr):.1f})")
        y_cursor += 0.5

    ax4.set_xlim(freq_range); ax4.set_ylim(-0.6, max(y_cursor, 1.0))
    ax4.set_xlabel("Frequency (Hz)"); ax4.set_yticks(y_ticks)
    ax4.set_yticklabels(y_tick_labels, fontsize=4.5)
    ax4.set_title("Effect size bands"); ax4.spines["left"].set_visible(False)
    ax4.tick_params(axis="y", length=0)
    if not any_bands:
        ax4.text(0.5, 0.5, "No bands detected", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=6, color=style["neutral_color"])

    fig.suptitle(title, fontsize=7, y=1.01)
    plt.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    plt.close(fig)


# -----------------------------
# Supplement table
# -----------------------------
def plot_supplement_table(
    band_rows: List[Dict],
    out_png: str,
    out_svg: str,
    title: str,
    style: dict,
    cohens_d_direction: str = "suppression",
) -> None:
    ensure_dir(os.path.dirname(out_png))

    if not band_rows:
        fig, ax = plt.subplots(figsize=(8, 1.5))
        apply_figure_style(fig, style); ax.axis("off")
        ax.text(0.5, 0.5, "No frequency bands detected at any threshold level.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=style["text_color"])
        fig.suptitle(title, fontsize=9, y=1.02)
        for path, fmt in [(out_png, None), (out_svg, "svg")]:
            kwargs = {"dpi": 300, "bbox_inches": "tight",
                      "transparent": (style["theme"] == "dark")}
            if fmt:
                kwargs["format"] = fmt
            fig.savefig(path, **kwargs)
        plt.close(fig)
        return

    df           = pd.DataFrame(band_rows)
    level_order  = ["large", "medium", "small"]
    level_labels = {
        "large":  "Large (|d| >= 0.8)",
        "medium": "Medium (|d| >= 0.5)",
        "small":  "Small (|d| >= 0.2)",
    }

    rows = []; row_colors = []
    for level in level_order:
        subset = df[df["threshold_level"] == level].copy()
        ts     = THRESHOLD_STYLES[level]
        bg     = ts["color"] + "33"
        if subset.empty:
            rows.append([level_labels[level], "-", "-", "-", "-", "-"])
            row_colors.append([bg] * 6)
            continue
        for _, r in subset.iterrows():
            peak_d = r["cohens_d_min"] if cohens_d_direction == "suppression" \
                else r["cohens_d_max"]
            rows.append([
                level_labels[level],
                r["band_hz"] + " Hz",
                f"{r['cohens_d_mean']:.2f}",
                f"{peak_d:.2f}",
                f"{r['delta_power_mean_uv2hz']:.5f}",
                str(int(r["n_freq_bins"])),
            ])
            row_colors.append([bg] * 6)

    col_labels = ["Threshold level", "Band (Hz)", "Mean d", "Peak d",
                  "Mean Delta Power\n(uV2/Hz)", "N freq bins"]
    col_widths = [0.24, 0.18, 0.10, 0.10, 0.22, 0.10]
    n_rows     = len(rows)
    fig_h      = max(1.8, 0.38 * n_rows + 1.0)

    fig, ax = plt.subplots(figsize=(9, fig_h))
    apply_figure_style(fig, style); ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)

    for col_idx in range(len(col_labels)):
        cell = tbl[0, col_idx]
        cell.set_facecolor(style.get("axes_facecolor", "#EEEEEE"))
        cell.set_text_props(weight="bold", color=style["text_color"])
        cell.set_edgecolor(style["text_color"]); cell.set_height(0.12)

    for row_idx, colors in enumerate(row_colors, start=1):
        for col_idx in range(len(col_labels)):
            cell = tbl[row_idx, col_idx]
            cell.set_facecolor(colors[col_idx])
            cell.set_text_props(color=style["text_color"])
            cell.set_edgecolor(style.get("neutral_color", "#CCCCCC"))

    ax.legend(
        handles=[
            mpatches.Patch(color=THRESHOLD_STYLES["large"]["color"],
                           alpha=0.6, label="Large (|d| >= 0.8)"),
            mpatches.Patch(color=THRESHOLD_STYLES["medium"]["color"],
                           alpha=0.6, label="Medium (|d| >= 0.5)"),
            mpatches.Patch(color=THRESHOLD_STYLES["small"]["color"],
                           alpha=0.6, label="Small (|d| >= 0.2)"),
        ],
        loc="lower right", fontsize=6, frameon=False,
        bbox_to_anchor=(1.0, -0.08),
    )

    direction_str = "suppression (negative d)" if cohens_d_direction == "suppression" \
        else "increase (positive d)"
    fig.suptitle(f"{title} | Effect direction: {direction_str}", fontsize=8, y=1.03)
    plt.tight_layout()

    for path, fmt in [(out_png, None), (out_svg, "svg")]:
        kwargs = {"dpi": 300, "bbox_inches": "tight",
                  "transparent": (style["theme"] == "dark")}
        if fmt:
            kwargs["format"] = fmt
        fig.savefig(path, **kwargs)
    plt.close(fig)


# -----------------------------
# Per-kind processing
# -----------------------------
def process_kind(
    kind: str,
    pkl_files: List[str],
    baseline_windows: List[Tuple[float, float]],
    stim_windows: List[Tuple[float, float]],
    noise_ref_data: Optional[np.ndarray],
    folder: str,
    region_name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
            n_taps = int(fs / 50 * NLMS_N_TAPS_FACTOR)
            data   = apply_adaptive_filter_nlms(
                data, noise_ref_data, mu=NLMS_MU, n_taps=n_taps
            )

        seg_base = segment_data(data, fs, baseline_windows)
        seg_stim = segment_data(data, fs, stim_windows)

        min_len = int(fs * 2)
        if seg_base.size < min_len or seg_stim.size < min_len:
            print(f"   ! Channel {channel}: epoch too short, skipping.")
            continue

        freqs_full, psd_base = compute_psd(seg_base, fs, nperseg_s=NPERSEG_S)
        _,          psd_stim = compute_psd(seg_stim, fs, nperseg_s=NPERSEG_S)

        freq_mask = (freqs_full >= FREQ_RANGE[0]) & (freqs_full <= FREQ_RANGE[1])
        freqs_r   = freqs_full[freq_mask]

        if ref_freqs is None:
            ref_freqs = freqs_r
        elif not np.allclose(freqs_r, ref_freqs, atol=1e-6):
            print(f"   ! Channel {channel}: frequency axis mismatch, skipping.")
            continue

        ch_psd_base.append(psd_base[freq_mask])
        ch_psd_stim.append(psd_stim[freq_mask])

    if not ch_psd_base:
        print(f" - No valid channels for {folder}/{region_name}/{kind}")
        return None

    print(f" - Added {folder}/{region_name}/{kind} (n_channels={len(ch_psd_base)})")
    return (
        np.mean(np.vstack(ch_psd_base), axis=0),
        np.mean(np.vstack(ch_psd_stim), axis=0),
        ref_freqs,
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    t0 = time.perf_counter()

    print("=" * 60)
    print("process_lfp_psd.py — path configuration")
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

    print(f"Processing {len(manifests)} folder(s) for LFP PSD analysis...")

    window_suffix = (
        f"adj_base{int(ADJUSTED_BASELINE_S)}s_stim{int(ADJUSTED_STIM_S)}s"
        if USE_ADJUSTED_WINDOWS else "full"
    )

    grouped_base:    Dict[Tuple[str, str], List[np.ndarray]] = {}
    grouped_stim:    Dict[Tuple[str, str], List[np.ndarray]] = {}
    grouped_freqs:   Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    grouped_animals: Dict[Tuple[str, str], List[str]] = {}

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

        epochs               = load_epochs_from_pynapse_csv(pynapse_csv, ignore_label=IGNORE_LABEL)
        baseline_windows     = epochs_to_windows(epochs, BASELINE_LABEL)
        stim_windows_calllog = epochs_to_windows(epochs, STIM_LABEL)
        stim_windows         = build_stim_windows(epochs, block_path, stim_windows_calllog)

        if USE_ADJUSTED_WINDOWS:
            baseline_windows = adjust_baseline_window(baseline_windows, ADJUSTED_BASELINE_S)
            stim_windows     = adjust_stim_window(stim_windows, ADJUSTED_STIM_S)

        print(" - baseline:", summarize_window_durations(baseline_windows))
        print(" - stim:    ", summarize_window_durations(stim_windows))

        if not baseline_windows or not stim_windows:
            print(" - Missing baseline or stim windows, skipping.")
            continue

        noise_ref_data = load_noise_reference(processed_folder_path)
        if noise_ref_data is None:
            print(" - No noise reference found; NLMS filtering skipped.")

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
                    print(f" - No {kind} LFP pkls in {lfp_cont_path}")
                    continue

                result = process_kind(
                    kind=kind,
                    pkl_files=pkl_files,
                    baseline_windows=baseline_windows,
                    stim_windows=stim_windows,
                    noise_ref_data=noise_ref_data,
                    folder=folder,
                    region_name=region_name,
                )
                if result is None:
                    continue

                animal_psd_base, animal_psd_stim, ref_freqs = result
                key = (region_name, kind)

                grouped_base.setdefault(key, []).append(animal_psd_base)
                grouped_stim.setdefault(key, []).append(animal_psd_stim)
                grouped_animals.setdefault(key, []).append(folder)
                if key not in grouped_freqs:
                    grouped_freqs[key] = ref_freqs

    # -----------------------------
    # Export
    # -----------------------------
    out_root = os.path.join(
        export_path_base, relative_data_path, "Postprocessing", "LFP_PSD"
    )
    ensure_dir(out_root)

    for (region_name, kind), animal_bases in grouped_base.items():
        animal_stims = grouped_stim[(region_name, kind)]
        animals      = grouped_animals[(region_name, kind)]
        freqs        = grouped_freqs[(region_name, kind)]
        n_animals    = len(animals)

        if freqs is None or n_animals == 0:
            continue

        region_kind_root = os.path.join(out_root, region_name, kind, window_suffix)
        ensure_dir(region_kind_root)

        arr_base  = np.vstack(animal_bases)
        arr_stim  = np.vstack(animal_stims)
        mean_base = np.mean(arr_base, axis=0)
        sem_base  = np.std(arr_base,  axis=0, ddof=1) / np.sqrt(n_animals)
        mean_stim = np.mean(arr_stim, axis=0)
        sem_stim  = np.std(arr_stim,  axis=0, ddof=1) / np.sqrt(n_animals)

        if n_animals >= 3:
            _, p_vals = ttest_rel(arr_base, arr_stim, axis=0)
            p_corr    = fdr_correct(p_vals)
        else:
            print(f"   ! {region_name}/{kind}: n={n_animals} < 3, skipping t-test.")
            p_vals = np.full(len(freqs), np.nan)
            p_corr = np.full(len(freqs), np.nan)

        cohens_d_arr = np.array([
            cohen_d(arr_base[:, i], arr_stim[:, i]) for i in range(len(freqs))
        ])

        print(f"   Computing bootstrap CI for {region_name}/{kind} ({n_animals} animals)...")
        delta_mean  = np.zeros(len(freqs))
        delta_lower = np.zeros(len(freqs))
        delta_upper = np.zeros(len(freqs))
        for i in range(len(freqs)):
            m, lo, hi = bootstrap_ci_diff(
                arr_base[:, i], arr_stim[:, i],
                n_bootstrap=N_BOOTSTRAP, ci=BOOTSTRAP_CI, seed=BOOTSTRAP_SEED,
            )
            delta_mean[i]  = m
            delta_lower[i] = lo
            delta_upper[i] = hi

        band_rows = []
        for level, thr in COHENS_D_THRESHOLDS.items():
            bands = extract_significant_bands(
                freqs, cohens_d_arr, thr, direction=COHENS_D_DIRECTION
            )
            for fl, fh in bands:
                idx    = (freqs >= fl) & (freqs <= fh)
                peak_d = float(np.min(cohens_d_arr[idx])) \
                    if COHENS_D_DIRECTION == "suppression" \
                    else float(np.max(cohens_d_arr[idx]))
                band_rows.append({
                    "region":                 region_name,
                    "kind":                   kind,
                    "window_mode":            window_suffix,
                    "threshold_level":        level,
                    "threshold_value":        thr,
                    "band_hz":                f"{fl:.1f}-{fh:.1f}",
                    "cohens_d_mean":          float(np.mean(cohens_d_arr[idx])),
                    "cohens_d_min":           float(np.min(cohens_d_arr[idx])),
                    "cohens_d_max":           float(np.max(cohens_d_arr[idx])),
                    "delta_power_mean_uv2hz": float(np.mean(delta_mean[idx])),
                    "n_freq_bins":            int(np.sum(idx)),
                })

        pd.DataFrame(band_rows).to_csv(
            os.path.join(
                region_kind_root,
                f"SignificantBands_{region_name}_{kind}_{window_suffix}.csv",
            ),
            index=False,
        )
        pd.DataFrame({
            "freq_hz":        freqs,
            "baseline_mean":  mean_base, "baseline_sem":  sem_base,
            "stim_mean":      mean_stim, "stim_sem":      sem_stim,
            "delta_mean":     delta_mean,
            "delta_ci_lower": delta_lower, "delta_ci_upper": delta_upper,
            "cohens_d":       cohens_d_arr,
            "p_raw":          p_vals, "p_fdr": p_corr,
            "window_mode":    window_suffix,
        }).to_csv(
            os.path.join(
                region_kind_root,
                f"GroupPSD_{region_name}_{kind}_{window_suffix}.csv",
            ),
            index=False,
        )

        plot_psd_group(
            freqs=freqs, mean_base=mean_base, sem_base=sem_base,
            mean_stim=mean_stim, sem_stim=sem_stim, p_corr=p_corr,
            cohens_d_arr=cohens_d_arr, delta_mean=delta_mean,
            delta_ci_lower=delta_lower, delta_ci_upper=delta_upper,
            out_png=os.path.join(
                region_kind_root,
                f"PSD_{region_name}_{kind}_{window_suffix}.png",
            ),
            out_svg=os.path.join(
                region_kind_root,
                f"PSD_{region_name}_{kind}_{window_suffix}.svg",
            ),
            title=(
                f"LFP PSD | {region_name} | {kind} | {tool_name} | "
                f"n={n_animals} | {window_suffix}"
            ),
            style=plot_style, n_animals=n_animals, freq_range=FREQ_RANGE,
            cohens_d_thresholds=COHENS_D_THRESHOLDS,
            cohens_d_direction=COHENS_D_DIRECTION,
        )

        plot_supplement_table(
            band_rows=band_rows,
            out_png=os.path.join(
                region_kind_root,
                f"SupplementTable_Bands_{region_name}_{kind}_{window_suffix}.png",
            ),
            out_svg=os.path.join(
                region_kind_root,
                f"SupplementTable_Bands_{region_name}_{kind}_{window_suffix}.svg",
            ),
            title=(
                f"Effect size bands | {tool_name} | {region_name} | "
                f"{kind} | n={n_animals} | {window_suffix}"
            ),
            style=plot_style,
            cohens_d_direction=COHENS_D_DIRECTION,
        )

        print(
            f"\u2713 Saved PSD + supplement table: "
            f"{region_name}/{kind}/{window_suffix} (n={n_animals})"
        )
        for row in band_rows:
            peak_key = "cohens_d_min" if COHENS_D_DIRECTION == "suppression" \
                else "cohens_d_max"
            print(
                f"   [{row['threshold_level']}] {row['band_hz']} Hz: "
                f"mean d={row['cohens_d_mean']:.2f}, "
                f"peak d={row[peak_key]:.2f}"
            )

    dt = time.perf_counter() - t0
    print(f"\nDone. Total processing time: {dt:.1f}s")


if __name__ == "__main__":
    main()