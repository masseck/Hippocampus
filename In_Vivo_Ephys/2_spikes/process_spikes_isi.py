# -*- coding: utf-8 -*-
"""
process_spikes_isi.py

Spike processing focused on:
- spike detection
- ISI baseline vs stimulation
- summary export

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
    restrict_spikes_to_windows,
    compute_isi,
    refractory_violations_fraction,
    plot_isi_hist,
    save_pickle,
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
USE_ADJUSTED_WINDOWS = True   # True = adjusted | False = full epochs

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
def rate_in_windows(spk: np.ndarray, windows: List[Tuple[float, float]]) -> float:
    if not windows:
        return float("nan")
    total_t = sum(max(0.0, b - a) for a, b in windows)
    if total_t <= 0:
        return float("nan")
    n = restrict_spikes_to_windows(spk, windows).size
    return float(n / total_t)


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

    For ChR2, pTrain contains only short trigger pulses and cannot reliably
    recover epoch boundaries. The call log is used directly.
    For vSWO and vLWO, pTrain carries the continuous stimulation signal and
    is the more precise source, with the call log as fallback.
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


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    t0 = time.perf_counter()

    print("=" * 60)
    print("process_spikes_isi.py — path configuration")
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

    print(f"Processing {len(manifests)} folder(s) for spike ISI analysis...")

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
            print(f" - Missing Pynapse_call_log.csv at {pynapse_csv}, skipping folder.")
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

        regions = manifest.get("regions", [])
        if only_regions is not None:
            regions = [r for r in regions if r.get("region_name") in only_regions]

        for region in regions:
            region_name = region["region_name"]
            print(f"\nRegion: {region_name}")

            spikes_root   = os.path.join(processed_folder_path, region_name, "Spikes")
            spikes_cont   = os.path.join(spikes_root, "continuous")
            detected_root = os.path.join(spikes_root, "detected")

            # QC output subfolder reflects window mode
            window_suffix = (
                f"adj_base{int(ADJUSTED_BASELINE_S)}s_stim{int(ADJUSTED_STIM_S)}s"
                if USE_ADJUSTED_WINDOWS else "full"
            )
            qc_root = os.path.join(
                processed_folder_path, region_name, "QC",
                "Spikes_ISI", window_suffix,
            )

            ensure_dir(detected_root)
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

                detected_kind_root = os.path.join(detected_root, kind)
                qc_kind_root       = os.path.join(qc_root, kind)
                ensure_dir(detected_kind_root)
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

                # Detected pkls are always saved with full windows so they
                # can be reused by postprocessing scripts in both modes.
                detected = {
                    "spike_times_s":     spike_times_s,
                    "spike_indices":     peaks_samp,
                    "fs":                fs,
                    "detector": {
                        "method":       "mad_threshold_crossing",
                        "thresh_mult":  THRESH_MULT,
                        "polarity":     POLARITY,
                        **det_metrics,
                    },
                    "source_continuous_pkl": pkl_path,
                    "channel":    obj.get("channel", None),
                    "reference":  obj.get("reference", None),
                    "signal_kind": kind,
                }
                out_detected = os.path.join(detected_kind_root, f"detected_{base}.pkl")
                save_pickle(detected, out_detected)

                spikes_baseline = restrict_spikes_to_windows(spike_times_s, baseline_windows)
                spikes_stim     = restrict_spikes_to_windows(spike_times_s, stim_windows)

                isi_base = compute_isi(spikes_baseline)
                isi_stim = compute_isi(spikes_stim)

                plot_isi_hist(
                    style=plot_style,
                    isi_baseline=isi_base,
                    isi_stim=isi_stim,
                    out_png=os.path.join(qc_kind_root, f"isi_{base}.png"),
                    title=f"ISI: baseline vs stimulation ({base}) [{window_suffix}]",
                    max_isi_s=0.15,
                    bin_s=0.001,
                )

                rv_all  = refractory_violations_fraction(spike_times_s,  refr_ms=REFRACTORY_MS)
                rv_base = refractory_violations_fraction(spikes_baseline, refr_ms=REFRACTORY_MS)
                rv_stim = refractory_violations_fraction(spikes_stim,     refr_ms=REFRACTORY_MS)

                fr_base = rate_in_windows(spike_times_s, baseline_windows)
                fr_stim = rate_in_windows(spike_times_s, stim_windows)
                dfr     = fr_stim - fr_base \
                    if (np.isfinite(fr_base) and np.isfinite(fr_stim)) \
                    else float("nan")

                if kind not in summary_rows_by_kind:
                    summary_rows_by_kind[kind] = []

                summary_rows_by_kind[kind].append({
                    "folder":                        folder,
                    "region":                        region_name,
                    "signal":                        base,
                    "signal_kind":                   kind,
                    "channel":                       obj.get("channel"),
                    "reference":                     obj.get("reference"),
                    "fs":                            fs,
                    "window_mode":                   window_suffix,
                    "n_spikes_total":                int(spike_times_s.size),
                    "firing_rate_baseline_hz":       fr_base,
                    "firing_rate_stim_hz":           fr_stim,
                    "delta_rate_hz":                 dfr,
                    "sigma_mad":                     det_metrics["sigma_mad"],
                    "threshold":                     det_metrics["threshold"],
                    "refractory_viol_frac_all":      rv_all,
                    "refractory_viol_frac_baseline": rv_base,
                    "refractory_viol_frac_stim":     rv_stim,
                    "n_isi_baseline":                int(isi_base.size),
                    "n_isi_stim":                    int(isi_stim.size),
                })

            for kind, rows in summary_rows_by_kind.items():
                if not rows:
                    continue
                df      = pd.DataFrame(rows)
                out_csv = os.path.join(qc_root, kind, "summary.csv")
                ensure_dir(os.path.dirname(out_csv))
                df.to_csv(out_csv, index=False)
                print(f"  ✓ Wrote summary ({kind}): {out_csv}")

    dt = time.perf_counter() - t0
    print(f"\nDone. Total processing time: {dt:.1f}s")


if __name__ == "__main__":
    main()