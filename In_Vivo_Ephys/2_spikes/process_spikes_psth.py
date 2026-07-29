# -*- coding: utf-8 -*-
"""
process_spikes_psth.py

Spike processing focused on:
- spike detection
- raster + onset-aligned PSTH
- full-epoch baseline vs stimulation PSTH

pTrain vs call log
------------------
ChR2 stimulation uses short trigger pulses in the pTrain store, not a
continuous signal. load_ptrain_windows() cannot reliably recover the epoch
boundaries from pulse-style events for this tool. The Pynapse call log is
therefore used as the primary source for ChR2 stimulation windows.

For vSWO and vLWO, stimulation is delivered as a continuous signal and
pTrain is the more precise source. The call log is used as fallback.

USE_PTRAIN is set automatically based on the tool inferred from
relative_data_path. No manual changes needed when switching cohorts.

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

Note: early/late stim subwindows (EARLY_STIM_S, LATE_STIM_S) are computed
relative to the (possibly adjusted) stimulation window.
"""

from __future__ import annotations

import os
import time
import pickle
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Functions_processing_spikes import (
    ensure_dir,
    load_json,
    list_manifests,
    load_spike_continuous_pkls,
    classify_signal_name,
    should_process_signal,
    load_epochs_from_pynapse_csv,
    epochs_to_windows,
    load_ptrain_windows,
    merge_windows,
    summarize_window_durations,
    detect_spikes_threshold,
    aligned_trials_from_windows,
    restrict_spikes_to_windows,
    plot_raster_and_psth,
    plot_two_psths_full_epoch,
    infer_tool_name,
    get_tool_color,
    get_plot_style,
    apply_axes_style,
    apply_figure_style,
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
# Alias used by module-level calls (e.g. infer_tool_name).
# The actual path used during processing is confirmed via prompt in main().
relative_data_path = _DEFAULT_RELATIVE_DATA_PATH
# relative_data_path = r"Jills_paper\vSWO_cohort\OFT_experimental"
# relative_data_path = r"Jills_paper\vLWO_cohort\OFT_experimental"

only_folder: Optional[str] = None
only_regions: Optional[List[str]] = ["CA1_R"]  # ["CA1_L"] or ["CA1_R"]

PROCESS_UNREFERENCED = False
PROCESS_LMR          = True

THRESH_MULT    = 5.0
POLARITY       = "neg"
REFRACTORY_MS  = 1.0
REFINE_PEAK    = True
PEAK_SEARCH_MS = 2.0

ALIGN_WINDOW_S   = (-2.0, 5.0)  # seconds around stim onset
PSTH_BIN_S       = 0.01          # 10 ms bins
FULL_EPOCH_BIN_S = 1.0

EARLY_STIM_S = 10.0
LATE_STIM_S  = 60.0

BASELINE_LABEL = "In baseline state"
STIM_LABEL     = "In stimulation state"
IGNORE_LABEL   = "In start delay"

PLOT_THEME          = "light"
STIM_COLOR_OVERRIDE = None

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
    """
    Return stimulation windows from pTrain (vSWO/vLWO) or call log (ChR2).
    """
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
                block_path,
                stim_duration_hint_s=stim_duration_hint_s,
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


def rate_in_windows(spk: np.ndarray, windows: List[Tuple[float, float]]) -> float:
    if not windows:
        return float("nan")
    total_t = sum(max(0.0, b - a) for a, b in windows)
    if total_t <= 0:
        return float("nan")
    return float(spk.size / total_t)


def median_duration(wins: List[Tuple[float, float]]) -> float:
    d = [b - a for a, b in wins if b > a]
    return float(np.median(d)) if d else 0.0


def spike_count_in_windows(spk: np.ndarray, windows: List[Tuple[float, float]]) -> int:
    return int(spk.size)


def make_early_stim_windows(
    stim_windows: List[Tuple[float, float]],
    early_duration_s: float = 10.0,
) -> List[Tuple[float, float]]:
    return [(a, min(a + early_duration_s, b)) for a, b in stim_windows if b > a]


def make_late_stim_windows(
    stim_windows: List[Tuple[float, float]],
    late_duration_s: float = 60.0,
) -> List[Tuple[float, float]]:
    return [(max(a, b - late_duration_s), b) for a, b in stim_windows if b > a]


def plot_stim_timecourse(
    stim_trials: List[np.ndarray],
    tmax: float,
    bin_s: float,
    title: str,
    out_png: str,
    style: dict | None = None,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(out_png))
    if style is None:
        style = get_plot_style(theme="dark")

    bin_edges = np.arange(0.0, tmax + bin_s, bin_s)
    if len(bin_edges) < 2:
        return

    centers = bin_edges[:-1] + np.diff(bin_edges) / 2.0
    if stim_trials:
        all_spikes = np.concatenate([st[(st >= 0) & (st <= tmax)] for st in stim_trials])
        counts, _ = np.histogram(all_spikes, bins=bin_edges)
        rate = counts / (np.diff(bin_edges) * max(len(stim_trials), 1))
    else:
        rate = np.zeros_like(centers)

    fig, ax = plt.subplots(figsize=(8, 4))
    apply_figure_style(fig, style)
    apply_axes_style(ax, style)
    ax.plot(centers, rate, color=style["stim_color"])
    ax.set_xlabel("Time from stimulation onset [s]")
    ax.set_ylabel("Rate [Hz]")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3, color=style["grid_color"])
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, transparent=(style["theme"] == "dark"))
    plt.close(fig)


def plot_temporal_rate_summary(
    df: pd.DataFrame,
    out_png: str,
    title: str,
    style: dict | None = None,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(out_png))
    if style is None:
        style = get_plot_style(theme="dark")

    required_cols = [
        "baseline_rate_hz", "early_stim_rate_hz",
        "late_stim_rate_hz", "full_stim_rate_hz",
    ]
    if df.empty or any(c not in df.columns for c in required_cols):
        return

    x      = np.arange(4)
    labels = ["Baseline", "Early stim", "Late stim", "Full stim"]

    fig, ax = plt.subplots(figsize=(7, 5))
    apply_figure_style(fig, style); apply_axes_style(ax, style)

    for _, row in df.iterrows():
        y = [row[c] for c in required_cols]
        if np.all(np.isfinite(y)):
            ax.plot(x, y, alpha=0.35, linewidth=1, color=style["neutral_color"])

    ax.plot(x, [df[c].mean() for c in required_cols],
            linewidth=3, marker="o", label="mean", color=style["stim_color"])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Firing rate [Hz]"); ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3, color=style["grid_color"])
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, transparent=(style["theme"] == "dark"))
    plt.close(fig)


def plot_temporal_delta_summary(
    df: pd.DataFrame,
    out_png: str,
    title: str,
    style: dict | None = None,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(out_png))
    if style is None:
        style = get_plot_style(theme="dark")

    required_cols = ["delta_rate_early_hz", "delta_rate_late_hz", "delta_rate_full_hz"]
    if df.empty or any(c not in df.columns for c in required_cols):
        return

    x      = np.arange(3)
    labels = ["Early - Base", "Late - Base", "Full - Base"]

    fig, ax = plt.subplots(figsize=(7, 5))
    apply_figure_style(fig, style); apply_axes_style(ax, style)

    for _, row in df.iterrows():
        y = [row[c] for c in required_cols]
        if np.all(np.isfinite(y)):
            ax.plot(x, y, alpha=0.35, linewidth=1, color=style["neutral_color"])

    ax.plot(x, [df[c].mean() for c in required_cols],
            linewidth=3, marker="o", label="mean", color=style["stim_color"])
    ax.axhline(0, linestyle="--", linewidth=1, color=style["text_color"])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Δ firing rate [Hz]"); ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3, color=style["grid_color"])
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, transparent=(style["theme"] == "dark"))
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    t0 = time.perf_counter()

    print("=" * 60)
    print("process_spikes_psth.py — path configuration")
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

    print(f"Processing {len(manifests)} folder(s) for spike PSTH analysis...")

    window_suffix = (
        f"adj_base{int(ADJUSTED_BASELINE_S)}s_stim{int(ADJUSTED_STIM_S)}s"
        if USE_ADJUSTED_WINDOWS else "full"
    )

    for manifest_path in manifests:
        manifest = load_json(manifest_path)

        folder                = manifest["folder"]
        processed_folder_path = manifest["processed_folder_path"]
        block_path            = manifest.get("block_path", None)

        print(f"\n==============================")
        print(f"Folder: {folder}")
        print(f"Manifest: {manifest_path}")

        pynapse_csv = os.path.join(
            processed_folder_path, "metadata", "Pynapse_call_log.csv"
        )
        if not os.path.isfile(pynapse_csv):
            print(f" - Missing Pynapse_call_log.csv, skipping folder.")
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
            print(" - Missing baseline or stim windows, skipping folder.")
            continue

        # Early/late subwindows relative to (possibly adjusted) stim windows
        early_stim_windows = make_early_stim_windows(stim_windows, EARLY_STIM_S)
        late_stim_windows  = make_late_stim_windows(stim_windows, LATE_STIM_S)

        tmax_base  = median_duration(baseline_windows)
        tmax_stim  = median_duration(stim_windows)
        tmax_epoch = min(tmax_base, tmax_stim)

        regions = manifest.get("regions", [])
        if only_regions is not None:
            regions = [r for r in regions if r.get("region_name") in only_regions]

        for region in regions:
            region_name = region["region_name"]
            print(f"\nRegion: {region_name}")

            spikes_root = os.path.join(processed_folder_path, region_name, "Spikes")
            spikes_cont = os.path.join(spikes_root, "continuous")

            # QC output subfolder reflects window mode
            qc_root = os.path.join(
                processed_folder_path, region_name, "QC",
                "Spikes_PSTH", window_suffix,
            )
            ensure_dir(qc_root)

            pkl_files = load_spike_continuous_pkls(spikes_cont)
            if not pkl_files:
                print(f" - No spike continuous pkls in {spikes_cont}")
                continue

            summary_rows_by_kind: Dict[str, List[Dict[str, Any]]] = {
                "unreferenced": [],
                "lmr":          [],
            }

            for pkl_path in pkl_files:
                base = os.path.splitext(os.path.basename(pkl_path))[0]
                kind = classify_signal_name(base)

                if not should_process_signal(
                    base,
                    process_unreferenced=PROCESS_UNREFERENCED,
                    process_lmr=PROCESS_LMR,
                ):
                    print(f"  Skipping ({kind}): {base}")
                    continue

                print(f"  Processing ({kind}): {base}")

                qc_kind_root = os.path.join(qc_root, kind)
                ensure_dir(qc_kind_root)

                with open(pkl_path, "rb") as f:
                    obj = pickle.load(f)

                x  = obj.get("filtered_data", None)
                fs = obj.get("fs_original", None)
                if x is None or fs is None:
                    print("   ! Missing filtered_data/fs_original in pkl, skipping.")
                    continue

                x  = np.asarray(x)
                fs = float(fs)

                peaks_samp, det_metrics = detect_spikes_threshold(
                    x=x, fs=fs,
                    b=THRESH_MULT, polarity=POLARITY,
                    refractory_ms=REFRACTORY_MS,
                    refine_peak=REFINE_PEAK,
                    peak_search_ms=PEAK_SEARCH_MS,
                )
                spike_times_s = peaks_samp / fs

                # Onset-aligned raster / PSTH
                aligned_trials: List[np.ndarray] = []
                for (a, b) in stim_windows:
                    tmin_w = a + ALIGN_WINDOW_S[0]
                    tmax_w = a + ALIGN_WINDOW_S[1]
                    st = spike_times_s[
                        (spike_times_s >= tmin_w) & (spike_times_s <= tmax_w)
                    ] - a
                    aligned_trials.append(st)

                if stim_windows:
                    plot_raster_and_psth(
                        style=plot_style,
                        aligned_trials=aligned_trials,
                        tmin=ALIGN_WINDOW_S[0],
                        tmax=ALIGN_WINDOW_S[1],
                        bin_s=PSTH_BIN_S,
                        title=f"Raster + PSTH aligned to stim onset ({base})",
                        out_png=os.path.join(qc_kind_root, f"raster_psth_{base}.png"),
                    )

                # Full-epoch PSTH: baseline vs stimulation
                base_trials      = aligned_trials_from_windows(spike_times_s, baseline_windows)
                stim_trials_full = aligned_trials_from_windows(spike_times_s, stim_windows)

                if tmax_epoch > 0:
                    plot_two_psths_full_epoch(
                        style=plot_style,
                        baseline_trials=base_trials,
                        stim_trials=stim_trials_full,
                        tmax=tmax_epoch,
                        bin_s=FULL_EPOCH_BIN_S,
                        out_png=os.path.join(
                            qc_kind_root,
                            f"psth_full_epoch_base_vs_stim_{base}.png",
                        ),
                        title=f"Full-epoch PSTH: baseline vs stimulation ({base}) [{window_suffix}]",
                    )

                # Stimulation-only time course
                if tmax_stim > 0:
                    plot_stim_timecourse(
                        style=plot_style,
                        stim_trials=stim_trials_full,
                        tmax=tmax_stim,
                        bin_s=FULL_EPOCH_BIN_S,
                        title=f"Stimulation time course ({base}) [{window_suffix}]",
                        out_png=os.path.join(qc_kind_root, f"stim_timecourse_{base}.png"),
                    )

                # Temporal summary metrics
                spikes_base  = restrict_spikes_to_windows(spike_times_s, baseline_windows)
                spikes_early = restrict_spikes_to_windows(spike_times_s, early_stim_windows)
                spikes_late  = restrict_spikes_to_windows(spike_times_s, late_stim_windows)
                spikes_full  = restrict_spikes_to_windows(spike_times_s, stim_windows)

                fr_base  = rate_in_windows(spikes_base,  baseline_windows)
                fr_early = rate_in_windows(spikes_early, early_stim_windows)
                fr_late  = rate_in_windows(spikes_late,  late_stim_windows)
                fr_full  = rate_in_windows(spikes_full,  stim_windows)

                dfr_early = fr_early - fr_base \
                    if (np.isfinite(fr_base) and np.isfinite(fr_early)) else float("nan")
                dfr_late  = fr_late  - fr_base \
                    if (np.isfinite(fr_base) and np.isfinite(fr_late))  else float("nan")
                dfr_full  = fr_full  - fr_base \
                    if (np.isfinite(fr_base) and np.isfinite(fr_full))  else float("nan")

                if kind not in summary_rows_by_kind:
                    summary_rows_by_kind[kind] = []

                summary_rows_by_kind[kind].append({
                    "folder":                  folder,
                    "region":                  region_name,
                    "signal":                  base,
                    "signal_kind":             kind,
                    "channel":                 obj.get("channel"),
                    "reference":               obj.get("reference"),
                    "fs":                      fs,
                    "window_mode":             window_suffix,
                    "n_spikes_total":          int(spike_times_s.size),
                    "baseline_rate_hz":        fr_base,
                    "early_stim_rate_hz":      fr_early,
                    "late_stim_rate_hz":       fr_late,
                    "full_stim_rate_hz":       fr_full,
                    "delta_rate_early_hz":     dfr_early,
                    "delta_rate_late_hz":      dfr_late,
                    "delta_rate_full_hz":      dfr_full,
                    "baseline_spike_count":    int(spikes_base.size),
                    "early_stim_spike_count":  int(spikes_early.size),
                    "late_stim_spike_count":   int(spikes_late.size),
                    "full_stim_spike_count":   int(spikes_full.size),
                    "n_stim_windows":          int(len(stim_windows)),
                    "early_stim_s":            EARLY_STIM_S,
                    "late_stim_s":             LATE_STIM_S,
                    "align_tmin_s":            ALIGN_WINDOW_S[0],
                    "align_tmax_s":            ALIGN_WINDOW_S[1],
                    "psth_bin_s":              PSTH_BIN_S,
                    "full_epoch_bin_s":        FULL_EPOCH_BIN_S,
                    "sigma_mad":               det_metrics["sigma_mad"],
                    "threshold":               det_metrics["threshold"],
                })

            for kind, rows in summary_rows_by_kind.items():
                if not rows:
                    continue

                df           = pd.DataFrame(rows)
                out_kind_root = os.path.join(qc_root, kind)
                ensure_dir(out_kind_root)

                out_csv = os.path.join(out_kind_root, "summary.csv")
                df.to_csv(out_csv, index=False)
                print(f"  \u2713 Wrote summary ({kind}): {out_csv}")

                plot_temporal_rate_summary(
                    style=plot_style, df=df,
                    out_png=os.path.join(out_kind_root, "temporal_rate_summary.png"),
                    title=f"Temporal rate summary | {folder} | {region_name} | {kind}",
                )
                plot_temporal_delta_summary(
                    style=plot_style, df=df,
                    out_png=os.path.join(out_kind_root, "temporal_delta_summary.png"),
                    title=f"Temporal delta summary | {folder} | {region_name} | {kind}",
                )

    dt = time.perf_counter() - t0
    print(f"\nDone. Total processing time: {dt:.1f}s")


if __name__ == "__main__":
    main()