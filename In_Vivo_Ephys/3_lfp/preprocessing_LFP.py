# -*- coding: utf-8 -*-
# %% Header
"""
preprocessing_lfp.py

Manifest-driven LFP preprocessing:
- Load raw channels (high sampling rate)
- Lowpass filter (Butterworth or FIR)
- Unreferenced export per channel
- Differential referencing per channel pair (ref_pairs from manifest)
- Downsample to target LFP rate
- QC plots (raw vs filtered)
- Noise reference export (folder-level)

Outputs:
  <folder>/<region>/LFP/continuous/
      preprocessed_lfp_unreferenced_ch{ch}_ref-1.pkl
      preprocessed_lfp_referenced_ch{a}_ref{b}.pkl
  <folder>/<region>/LFP/Preprocessed_plots/
      raw_vs_filtered_*.png
  <folder>/Noise_reference/LFP/
      preprocessed_lfp_noise_reference_ch{x}.pkl

@author: Juliana Groß (pipeline)
"""

from __future__ import annotations

import os
import json
import time
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Functions_preprocessing_lfp import (
    load_block,
    extract_channel,
    filter_data,
    plot_raw_and_filtered_data,
    downsample_data,
    normalize_noise_reference,
)

# -----------------------------
# USER SETTINGS — adjust before running
# -----------------------------
# Paths: set your raw data root and processed output root here.
# When running interactively, you will be prompted to confirm or
# override these defaults at startup.
_DEFAULT_DATA_PATH_BASE    = r"C:\Users\Juliana\Documents\_PhD\Data\_Raw"
_DEFAULT_EXPORT_PATH_BASE  = r"C:\Users\Juliana\Documents\_PhD\Data\_Processed"
_DEFAULT_RELATIVE_DATA_PATH = r"Jills_paper\ChR2_cohort\OFT_experimental"
# Other cohorts:
# _DEFAULT_RELATIVE_DATA_PATH = r"Jills_paper\vLWO_cohort\OFT_experimental"
# _DEFAULT_RELATIVE_DATA_PATH = r"Jills_paper\vSWO_cohort\OFT_experimental"

only_folder: Optional[str] = None
only_regions: Optional[List[str]] = ["CA1_R"]  # ["CA1_L"] or ["CA1_R"] or None for both

# Animals to exclude from this script.
# Early ChR2 animals (0714, 0895, 0896) had a disconnected headstage jumper
# wire, resulting in separate ground and reference potentials (floating
# reference). This caused elevated common-mode noise that was not amenable
# to standard differential referencing (LMR). These animals are instead
# preprocessed with CAR referencing via preprocessing_lfp_car.py.
# Leave as an empty set for vLWO and vSWO cohorts (no exclusions needed).
EXCLUDE_ANIMAL_IDS: set = {"0714", "0895", "0896"}

# Set one to False and run twice to save memory on large datasets
run_unreferenced = False
run_referenced   = True


def _prompt_path(label: str, default: str) -> str:
    """Prompt the user to confirm or override a path at startup."""
    user_input = input(f"{label}\n  [{default}]: ").strip()
    return user_input if user_input else default


# -----------------------------
# Helper Functions
# -----------------------------
def print_timed(task_name: str, func, *args, **kwargs):
    print(f" - {task_name}", end="")
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start
    print(f" finished after {duration:.1f} seconds")
    return result


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_manifests(export_path_base: str, relative_data_path: str) -> List[str]:
    """Find all run_manifest.json files under the processed experiment path."""
    root = os.path.join(export_path_base, relative_data_path)
    manifests: List[str] = []
    if not os.path.exists(root):
        return manifests
    for folder in os.listdir(root):
        m = os.path.join(root, folder, "metadata", "run_manifest.json")
        if os.path.isfile(m):
            manifests.append(m)
    manifests.sort()
    return manifests


def filter_kwargs_from_settings(filter_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Rename 'type' -> 'filter_type' to match filter_data() signature."""
    return {("filter_type" if k == "type" else k): v for k, v in filter_settings.items()}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_filter_settings(lfp_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate filter settings from lfp_settings dict."""
    fset = lfp_settings.get("filter")
    if fset is None:
        raise ValueError(
            "Missing LFP filter settings in manifest "
            "(region['lfp_settings']['filter'])."
        )
    return fset


def get_downsampling_hz(lfp_settings: Dict[str, Any]) -> float:
    """Extract and validate downsampling rate from lfp_settings dict."""
    ds_hz = lfp_settings.get("downsampling_Hz")
    if ds_hz is None:
        raise ValueError(
            "Missing downsampling_Hz in manifest "
            "(region['lfp_settings']['downsampling_Hz'])."
        )
    return float(ds_hz)


def filter_downsample_export(
    *,
    time_s: np.ndarray,
    data: np.ndarray,
    fs: float,
    channel: int,
    reference: int,
    suffix: str,
    lfp_settings: Dict[str, Any],
    save_path: str,
    plots_path: str,
) -> None:
    """
    Filter → plot → downsample → export a single channel or differential pair.

    Parameters
    ----------
    time_s : np.ndarray
        Time vector in seconds.
    data : np.ndarray
        Signal to process (raw or differential).
    fs : float
        Sampling rate of input data in Hz.
    channel : int
        Channel number (for filename and metadata).
    reference : int
        Reference channel number, or -1 if unreferenced.
    suffix : str
        Label for filename (e.g. 'unreferenced', 'referenced').
    lfp_settings : dict
        Filter and downsampling settings from manifest.
    save_path : str
        Output directory for pkl files.
    plots_path : str
        Output directory for QC plots.
    """
    fset = get_filter_settings(lfp_settings)
    ds_hz = get_downsampling_hz(lfp_settings)
    fkwargs = filter_kwargs_from_settings(fset)

    filtered = print_timed(
        f"Filter ch{channel} ({suffix})", filter_data, data, fs, **fkwargs
    )

    print_timed(
        f"Plot ch{channel} ({suffix})",
        plot_raw_and_filtered_data,
        time_s,
        data,
        filtered,
        str(channel),
        fset.get("type", "unknown"),
        suffix,
        plots_path,
    )

    downsampled = print_timed(
        f"Downsample ch{channel} ({suffix})", downsample_data, filtered, fs, ds_hz
    )

    export_data = {
        "downsampled_data": downsampled,
        "channel": channel,
        "reference": reference,
        "settings": lfp_settings,
        "fs_original": float(fs),
        "suffix": suffix,
    }

    out_file = os.path.join(
        save_path, f"preprocessed_lfp_{suffix}_ch{channel}_ref{reference}.pkl"
    )
    with open(out_file, "wb") as f:
        pickle.dump(export_data, f)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    start_time_script = time.perf_counter()

    print("=" * 60)
    print("preprocessing_lfp.py — path configuration")
    print("Press Enter to accept the default path shown in brackets.")
    print("=" * 60)
    data_path_base    = _prompt_path("Raw data root:",      _DEFAULT_DATA_PATH_BASE)
    export_path_base  = _prompt_path("Processed data root:", _DEFAULT_EXPORT_PATH_BASE)
    relative_data_path = _prompt_path("Relative cohort path:", _DEFAULT_RELATIVE_DATA_PATH)
    print()

    manifests = list_manifests(export_path_base, relative_data_path)
    if not manifests:
        raise FileNotFoundError(
            "No manifests found. Run common_ingest.py first.\n"
            f"Expected under: "
            f"{os.path.join(export_path_base, relative_data_path, '<folder>', 'metadata', 'run_manifest.json')}"
        )

    # Exclude floating-reference animals — these are handled by
    # preprocessing_lfp_car.py which applies Common Average Referencing.
    if EXCLUDE_ANIMAL_IDS:
        manifests_before = len(manifests)
        manifests = [
            m for m in manifests
            if not any(
                aid in os.path.basename(os.path.dirname(os.path.dirname(m)))
                for aid in EXCLUDE_ANIMAL_IDS
            )
        ]
        n_excluded = manifests_before - len(manifests)
        if n_excluded > 0:
            print(
                f"Excluded {n_excluded} folder(s) matching EXCLUDE_ANIMAL_IDS "
                f"{EXCLUDE_ANIMAL_IDS} — use preprocessing_lfp_car.py for these."
            )

    if only_folder is not None:
        manifests = [
            m for m in manifests
            if os.path.basename(os.path.dirname(os.path.dirname(m))) == only_folder
        ]
        if not manifests:
            raise FileNotFoundError(f"No manifest found for only_folder='{only_folder}'.")

    print(f"Processing {len(manifests)} folder(s) using manifests...")

    for manifest_path in manifests:
        manifest = load_json(manifest_path)

        folder = manifest["folder"]
        block_path = manifest["block_path"]
        processed_folder_path = manifest["processed_folder_path"]

        print(f"\n==============================")
        print(f"Folder: {folder}")
        print(f"Manifest: {manifest_path}")
        print(f"Block path: {block_path}")

        if not os.path.isdir(block_path):
            print(f" - Raw block path does not exist, skipping: {block_path}")
            continue

        regions = manifest.get("regions", [])
        if only_regions is not None:
            regions = [r for r in regions if r.get("region_name") in only_regions]

        if not regions:
            print(" - No regions to process (check manifest or only_regions).")
            continue

        # Load block once per folder — reused across all regions and channels
        print(f" - Loading TDT block...")
        block_data = print_timed("Load block", load_block, block_path)

        for region in regions:
            region_name = region["region_name"]
            lfp_settings = region.get("lfp_settings", {})
            ref_pairs: List[List[int]] = region.get("ref_pairs", [])
            channels_unref: List[int] = region.get("channels_unreferenced", [])

            print(f"\nProcessing Region: {region_name}")

            lfp_root = os.path.join(processed_folder_path, region_name, "LFP")
            save_path = os.path.join(lfp_root, "continuous")
            plots_path = os.path.join(lfp_root, "Preprocessed_plots")
            ensure_dir(save_path)
            ensure_dir(plots_path)

            # ---------------------------
            # Unreferenced channels
            # ---------------------------
            if run_unreferenced:
                for ch in channels_unref:
                    print(f"\nUnreferenced channel {ch}")
                    time_s, data, fs = extract_channel(block_data, ch)
                    filter_downsample_export(
                        time_s=time_s,
                        data=data,
                        fs=fs,
                        channel=ch,
                        reference=-1,
                        suffix="unreferenced",
                        lfp_settings=lfp_settings,
                        save_path=save_path,
                        plots_path=plots_path,
                    )

            # ---------------------------
            # Differential reference pairs
            # ---------------------------
            if run_referenced:
                for pair in ref_pairs:
                    ref_a, ref_b = pair
                    print(f"\nReferenced pair: ch{ref_a} - ch{ref_b}")

                    time_a, data_a, fs_a = extract_channel(block_data, ref_a)
                    time_b, data_b, fs_b = extract_channel(block_data, ref_b)

                    if fs_a != fs_b:
                        raise ValueError(
                            f"Sampling rate mismatch: ch{ref_a}={fs_a}Hz, ch{ref_b}={fs_b}Hz"
                        )
                    if not np.array_equal(time_a, time_b):
                        raise ValueError(
                            f"Time vector mismatch between ch{ref_a} and ch{ref_b}."
                        )

                    data_diff = data_a - data_b
                    del data_a, data_b  # free memory

                    filter_downsample_export(
                        time_s=time_a,
                        data=data_diff,
                        fs=fs_a,
                        channel=ref_a,
                        reference=ref_b,
                        suffix="referenced",
                        lfp_settings=lfp_settings,
                        save_path=save_path,
                        plots_path=plots_path,
                    )

        # ---------------------------
        # Folder-level noise reference
        # ---------------------------
        noise_ref = manifest.get("noise_reference_channel", None)
        if noise_ref is not None:
            print(f"\nNoise Reference Channel: {noise_ref}")

            # Use filter/downsampling settings from the first processed region
            first_region = regions[0]
            fset = first_region.get("lfp_settings", {}).get("filter")
            ds_hz = first_region.get("lfp_settings", {}).get("downsampling_Hz")

            if fset is None or ds_hz is None:
                print(
                    " - Noise ref requested but filter/downsampling settings missing "
                    "in first region; skipping."
                )
            else:
                noise_root = os.path.join(processed_folder_path, "Noise_reference")
                noise_lfp_root = os.path.join(noise_root, "LFP")
                ensure_dir(noise_lfp_root)

                time_s, noise_data, fs = extract_channel(block_data, int(noise_ref))

                normalized = print_timed(
                    "Normalize noise ref", normalize_noise_reference, noise_data
                )

                fkwargs = filter_kwargs_from_settings(fset)
                filtered = print_timed(
                    "Filter noise ref", filter_data, normalized, fs, **fkwargs
                )

                print_timed(
                    "Plot noise ref",
                    plot_raw_and_filtered_data,
                    time_s,
                    normalized,
                    filtered,
                    str(noise_ref),
                    fset.get("type", "unknown"),
                    "noise_reference",
                    noise_lfp_root,
                )

                downsampled = print_timed(
                    "Downsample noise ref", downsample_data, filtered, fs, float(ds_hz)
                )

                export_data = {
                    "downsampled_data": downsampled,
                    "channel": int(noise_ref),
                    "reference": None,
                    "settings": {
                        "filter": fset,
                        "downsampling_Hz": ds_hz,
                    },
                    "fs_original": float(fs),
                    "suffix": "noise_reference",
                }

                out_file = os.path.join(
                    noise_lfp_root,
                    f"preprocessed_lfp_noise_reference_ch{int(noise_ref)}.pkl",
                )
                with open(out_file, "wb") as f:
                    pickle.dump(export_data, f)

                print(f" - Noise reference saved: {out_file}")
        else:
            print("\n - No noise reference channel in manifest; skipping.")

    total_duration = time.perf_counter() - start_time_script
    print(f"\nScript finished after {total_duration:.1f} seconds")


if __name__ == "__main__":
    main()