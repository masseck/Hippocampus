# -*- coding: utf-8 -*-
"""
postprocess_spikes_isi_supplement.py

Postprocessing script for paper-style ISI plots averaged across animals.

Workflow
--------
- Load detected spike times from process_spikes_isi.py output
- Rebuild baseline/stimulation windows from call log / pTrain
- Compute ISIs per channel
- Convert each channel to a normalized ISI histogram
- Average channel histograms within each animal
- Average animal histograms across animals
- Plot mean ± SEM for baseline vs stimulation

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

Notes
-----
This script uses the channel-wise outputs from process_spikes_isi.py and
does not redetect spikes.
"""

from __future__ import annotations

import os
import time
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Functions_processing_spikes import (
    ensure_dir,
    load_json,
    list_manifests,
    load_epochs_from_pynapse_csv,
    epochs_to_windows,
    load_ptrain_windows,
    merge_windows,
    summarize_window_durations,
    restrict_spikes_to_windows,
    compute_isi,
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
only_regions: Optional[List[str]] = ["CA1_R"]

PROCESS_UNREFERENCED = False  # set True to also process single-wire channels
PROCESS_LMR          = True

BASELINE_LABEL = "In baseline state"
STIM_LABEL     = "In stimulation state"
IGNORE_LABEL   = "In start delay"

# ISI histogram settings
MAX_ISI_S          = 0.15
BIN_S              = 0.001
NORMALIZE_MODE     = "fraction"   # "fraction" or "density"
MIN_CHANNELS_PER_ANIMAL = 1

# Plot settings
PLOT_THEME          = "light"
STIM_COLOR_OVERRIDE = None

FIG_HEIGHT_MM = 50
FIG_ASPECT    = 1.4

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
    f"Tool: {tool_name} | "
    f"Adjusted windows: {USE_ADJUSTED_WINDOWS}"
    + (
        f" (baseline={ADJUSTED_BASELINE_S:.0f} s, stim={ADJUSTED_STIM_S:.0f} s)"
        if USE_ADJUSTED_WINDOWS else ""
    )
)


# -----------------------------
# Helpers
# -----------------------------
def load_detected_pkls(detected_kind_root: str) -> List[str]:
    if not os.path.isdir(detected_kind_root):
        return []
    files = sorted(f for f in os.listdir(detected_kind_root) if f.endswith(".pkl"))
    return [os.path.join(detected_kind_root, f) for f in files]


def normalize_histogram(
    values: np.ndarray,
    bin_edges: np.ndarray,
    mode: str = "fraction",
) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bin_edges)
    if mode == "fraction":
        total = counts.sum()
        return counts / total if total > 0 \
            else np.zeros_like(counts, dtype=float)
    if mode == "density":
        widths = np.diff(bin_edges)
        total  = counts.sum()
        return counts / (total * widths) if total > 0 \
            else np.zeros_like(counts, dtype=float)
    raise ValueError("mode must be 'fraction' or 'density'")


def mean_sem(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    n    = np.sum(np.isfinite(x), axis=0)
    sem  = np.nanstd(x, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    sem[n < 2] = np.nan
    return mean, sem


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
    stim_windows_calllog = epochs_to_windows(epochs, STIM_LABEL)
    stim_duration_hint_s = None
    if stim_windows_calllog:
        stim_duration_hint_s = float(
            np.median([b - a for a, b in stim_windows_calllog])
        )
    stim_windows: List[Tuple[float, float]] = []
    if block_path is not None:
        try:
            stim_windows = merge_windows(
                load_ptrain_windows(
                    block_path, stim_duration_hint_s=stim_duration_hint_s
                ),
                min_gap_s=0.0,
            )
            if stim_windows:
                print(f" - Using pTrain: {len(stim_windows)} epoch(s).")
            else:
                print(" - pTrain empty; falling back to call log.")
        except Exception as e:
            print(f" - Could not read pTrain ({e}); falling back to call log.")
    return stim_windows if stim_windows else stim_windows_calllog


def plot_group_isi(
    bin_edges: np.ndarray,
    mean_base: np.ndarray,
    sem_base: np.ndarray,
    mean_stim: np.ndarray,
    sem_stim: np.ndarray,
    out_png: str,
    out_svg: str,
    title: str,
    style: dict,
    normalize_mode: str,
    n_animals: int,
) -> None:
    ensure_dir(os.path.dirname(out_png))
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    apply_figure_style(fig, style)
    apply_axes_style(ax, style)

    centers_ms = (bin_edges[:-1] + np.diff(bin_edges) / 2.0) * 1000.0

    ax.plot(centers_ms, mean_base,
            color=style["baseline_color"], linewidth=1.5, label="Baseline")
    ax.fill_between(centers_ms, mean_base - sem_base, mean_base + sem_base,
                    color=style["baseline_color"], alpha=0.25)
    ax.plot(centers_ms, mean_stim,
            color=style["stim_color"], linewidth=1.5, label="Stimulation")
    ax.fill_between(centers_ms, mean_stim - sem_stim, mean_stim + sem_stim,
                    color=style["stim_color"], alpha=0.25)

    ax.set_xlabel("Inter-spike interval (ms)")
    ax.set_ylabel("Fraction of ISIs" if normalize_mode == "fraction"
                  else "Probability density")
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
    print("postprocess_spikes_isi_supplement.py — path configuration")
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

    print(f"Postprocessing {len(manifests)} folder(s) for full ISI plots...")

    window_suffix = (
        f"adj_base{int(ADJUSTED_BASELINE_S)}s_stim{int(ADJUSTED_STIM_S)}s"
        if USE_ADJUSTED_WINDOWS else "full"
    )

    grouped: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = {}
    bin_edges = np.arange(0.0, MAX_ISI_S + BIN_S, BIN_S)

    for manifest_path in manifests:
        manifest = load_json(manifest_path)

        folder                = manifest["folder"]
        processed_folder_path = manifest["processed_folder_path"]
        block_path            = manifest.get("block_path", None)

        print("\n==============================")
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
            print(" - Missing windows, skipping.")
            continue

        regions = manifest.get("regions", [])
        if only_regions is not None:
            regions = [r for r in regions if r.get("region_name") in only_regions]

        for region in regions:
            region_name   = region["region_name"]
            print(f"\nRegion: {region_name}")

            detected_root = os.path.join(
                processed_folder_path, region_name, "Spikes", "detected"
            )

            for kind in ["unreferenced", "lmr"]:
                if kind == "unreferenced" and not PROCESS_UNREFERENCED:
                    continue
                if kind == "lmr" and not PROCESS_LMR:
                    continue

                detected_files = load_detected_pkls(os.path.join(detected_root, kind))
                if not detected_files:
                    print(f" - No detected pkls for {kind}")
                    continue

                channel_hists_base: List[np.ndarray] = []
                channel_hists_stim: List[np.ndarray] = []

                for det_path in detected_files:
                    with open(det_path, "rb") as f:
                        det = pickle.load(f)

                    spike_times_s = np.asarray(det.get("spike_times_s", []), dtype=float)
                    if spike_times_s.size < 2:
                        continue

                    spikes_base = restrict_spikes_to_windows(spike_times_s, baseline_windows)
                    spikes_stim = restrict_spikes_to_windows(spike_times_s, stim_windows)

                    isi_base = compute_isi(spikes_base)
                    isi_stim = compute_isi(spikes_stim)

                    # Clip ISIs to the display range before histogramming.
                    # Note: normalization (fraction) is therefore relative to
                    # ISIs within [0, MAX_ISI_S], not the full ISI distribution.
                    isi_base = isi_base[(isi_base >= 0) & (isi_base <= MAX_ISI_S)]
                    isi_stim = isi_stim[(isi_stim >= 0) & (isi_stim <= MAX_ISI_S)]

                    if isi_base.size < 1 or isi_stim.size < 1:
                        continue

                    channel_hists_base.append(
                        normalize_histogram(isi_base, bin_edges, mode=NORMALIZE_MODE)
                    )
                    channel_hists_stim.append(
                        normalize_histogram(isi_stim, bin_edges, mode=NORMALIZE_MODE)
                    )

                if len(channel_hists_base) < MIN_CHANNELS_PER_ANIMAL:
                    print(f" - Not enough channels for {folder}/{region_name}/{kind}")
                    continue

                key = (region_name, kind)
                grouped.setdefault(key, {"baseline": [], "stim": [], "animals": []})
                grouped[key]["baseline"].append(
                    np.mean(np.vstack(channel_hists_base), axis=0)
                )
                grouped[key]["stim"].append(
                    np.mean(np.vstack(channel_hists_stim), axis=0)
                )
                grouped[key]["animals"].append(folder)

                print(
                    f" - Added {folder}/{region_name}/{kind} "
                    f"(n_channels={len(channel_hists_base)})"
                )

    # -----------------------------
    # Export
    # -----------------------------
    out_root = os.path.join(
        export_path_base, relative_data_path,
        "Postprocessing", "Spike_ISI_Supplement",
    )
    ensure_dir(out_root)

    for (region_name, kind), d in grouped.items():
        if not d["baseline"] or not d["stim"]:
            continue

        arr_base = np.vstack(d["baseline"])
        arr_stim = np.vstack(d["stim"])
        mean_base, sem_base = mean_sem(arr_base)
        mean_stim, sem_stim = mean_sem(arr_stim)
        n_animals = arr_base.shape[0]

        region_kind_root = os.path.join(out_root, region_name, kind, window_suffix)
        ensure_dir(region_kind_root)

        centers_ms = (bin_edges[:-1] + np.diff(bin_edges) / 2.0) * 1000.0
        pd.DataFrame({
            "isi_bin_center_ms": centers_ms,
            "baseline_mean":     mean_base,
            "baseline_sem":      sem_base,
            "stim_mean":         mean_stim,
            "stim_sem":          sem_stim,
        }).to_csv(
            os.path.join(region_kind_root, "full_isi_distribution.csv"),
            index=False,
        )

        pd.DataFrame({"folder": d["animals"]}).to_csv(
            os.path.join(region_kind_root, "included_animals.csv"),
            index=False,
        )

        plot_group_isi(
            bin_edges=bin_edges,
            mean_base=mean_base, sem_base=sem_base,
            mean_stim=mean_stim, sem_stim=sem_stim,
            out_png=os.path.join(
                region_kind_root,
                f"ISI_full_{region_name}_{kind}_{window_suffix}.png",
            ),
            out_svg=os.path.join(
                region_kind_root,
                f"ISI_full_{region_name}_{kind}_{window_suffix}.svg",
            ),
            title=(
                f"ISI distribution | {region_name} | {kind} | {window_suffix}"
            ),
            style=plot_style,
            normalize_mode=NORMALIZE_MODE,
            n_animals=n_animals,
        )
        print(
            f"\u2713 Saved full ISI plot: {region_name}/{kind}/{window_suffix} "
            f"(n={n_animals})"
        )

    dt = time.perf_counter() - t0
    print(f"\nDone. Total processing time: {dt:.1f}s")


if __name__ == "__main__":
    main()