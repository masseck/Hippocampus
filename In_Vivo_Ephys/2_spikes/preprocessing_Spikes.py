# -*- coding: utf-8 -*-
"""
preprocessing_spikes.py

Manifest-driven SPIKE preprocessing (minimal, like LFP preprocessing):
- Load raw channels (high sampling rate)
- Spike-band bandpass filter (Butterworth + filtfilt)
- Optional differential referencing using ref_pairs
- NO spike detection, NO waveform extraction (that comes later)
- Export filtered continuous traces (original fs) for later detection/sorting

Outputs:
  <folder>/<region>/Spikes/continuous/
      preprocessed_spikes_unreferenced_chXX.pkl
      preprocessed_spikes_referenced_chA_refB.pkl
  <folder>/<region>/Spikes/Preprocessed_plots/
      raw_vs_filtered_*.png

Optional:
  <folder>/Noise_reference/Spikes/
      preprocessed_spikes_noise_reference_chX.pkl

@author: Juliana Groß (pipeline)
"""

from __future__ import annotations

import os
import json
import time
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from Functions_preprocessing_spikes import load_and_extract_data  # reuse your loader



def _prompt_path(label: str, default: str) -> str:
    """Prompt the user to confirm or override a path at startup."""
    user_input = input(f"{label}\n  [{default}]: ").strip()
    return user_input if user_input else default


# -----------------------------
# Constants and Paths
# -----------------------------
data_path_base = r"C:\Users\Juliana\Documents\_PhD\Data\_Raw"
_DEFAULT_EXPORT_PATH_BASE    = r"C:\Users\Juliana\Documents\_PhD\Data\_Processed"
# _DEFAULT_RELATIVE_DATA_PATH = r"Jills_paper\vSWO_cohort\OFT_experimental"
# _DEFAULT_RELATIVE_DATA_PATH = r"Jills_paper\vLWO_cohort\OFT_experimental"
_DEFAULT_RELATIVE_DATA_PATH  = r"Jills_paper\ChR2_cohort\OFT_experimental"

only_folder: Optional[str] = None
only_regions: Optional[List[str]] = ["CA1_R"]  # ["CA1_L"] or ["CA1_R"]

# Set False and run twice for saving memory (LMR is always run):
run_unreferenced = True  # True or False

# Noise reference
process_noise_reference_spikes = True  # set True if you want spike-band noise ref saved too

# -----------------------------
# Spike filter settings (minimal)
# -----------------------------
spike_filter = {
    "type": "butter_bandpass_filtfilt",
    "lowcut_hz": 300.0,
    "highcut_hz": 6000.0, # 3000 Hz or 6000 Hz
    "order": 4,
}


# -----------------------------
# Helper Functions
# -----------------------------
def print_timed(task_name, func, *args, **kwargs):
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def butter_bandpass_filtfilt(
    x: np.ndarray,
    fs: float,
    lowcut_hz: float,
    highcut_hz: float,
    order: int,
) -> np.ndarray:
    b, a = signal.butter(order, [lowcut_hz, highcut_hz], btype="bandpass", fs=fs)
    # float32 saves a lot of disk/memory; keep raw in float64 if needed elsewhere
    return signal.filtfilt(b, a, x).astype(np.float32)


def unique_channels_from_ref_pairs(ref_pairs: List[List[int]]) -> List[int]:
    return sorted({ch for pair in ref_pairs for ch in pair})


def local_median_reference(
    X: np.ndarray,
    leave_one_out: bool = True,
) -> np.ndarray:
    """
    X: shape (n_channels, n_samples)
    Returns X referenced by median across channels at each sample.
    leave_one_out=True uses median of all channels except itself (safest).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D: (n_channels, n_samples)")

    if not leave_one_out:
        ref = np.median(X, axis=0)
        return X - ref

    X_lmr = np.empty_like(X)
    for i in range(X.shape[0]):
        ref = np.median(np.delete(X, i, axis=0), axis=0)
        X_lmr[i] = X[i] - ref
    return X_lmr


def plot_raw_and_filtered(
    time_s: np.ndarray,
    raw: np.ndarray,
    filtered: np.ndarray,
    title: str,
    out_png: str,
    max_seconds: float = 2.0,
) -> None:
    """
    Keep plots light: only show first max_seconds to avoid huge files.
    """
    fs = 1.0 / np.median(np.diff(time_s))
    n = min(raw.size, int(max_seconds * fs))

    plt.figure()
    plt.plot(time_s[:n], raw[:n], label="raw", linewidth=0.8)
    plt.plot(time_s[:n], filtered[:n], label="filtered", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def process_and_export_continuous(
    *,
    label: str,
    time_s: np.ndarray,
    data: np.ndarray,
    fs: float,
    channel: int,
    reference: Optional[int],
    save_path: str,
    plots_path: str,
    filter_settings: Dict[str, Any],
) -> None:
    filtered = print_timed(
        f"Filter ({label})",
        butter_bandpass_filtfilt,
        data,
        fs,
        filter_settings["lowcut_hz"],
        filter_settings["highcut_hz"],
        filter_settings["order"],
    )

    # QC plot (short snippet)
    out_png = os.path.join(plots_path, f"raw_vs_filtered_{label}.png")
    plot_raw_and_filtered(
        time_s,
        data,
        filtered,
        title=f"{label} ({filter_settings['lowcut_hz']}-{filter_settings['highcut_hz']} Hz)",
        out_png=out_png,
        max_seconds=2.0,
    )

    export_data = {
        "filtered_data": filtered,
        "channel": int(channel),
        "reference": int(reference) if reference is not None else None,
        "fs_original": float(fs),
        "filter_settings": dict(filter_settings),
        "label": label,
    }

    out_pkl = os.path.join(save_path, f"preprocessed_spikes_{label}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(export_data, f)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    start_time_script = time.perf_counter()

    print("=" * 60)
    print("preprocessing_spikes.py — path configuration")
    print("Press Enter to accept the default path shown in brackets.")
    print("=" * 60)
    export_path_base   = _prompt_path("Processed data root:", _DEFAULT_EXPORT_PATH_BASE)
    relative_data_path = _prompt_path("Relative cohort path:", _DEFAULT_RELATIVE_DATA_PATH)
    print()

    manifests = list_manifests(export_path_base, relative_data_path)
    if not manifests:
        raise FileNotFoundError(
            "No manifests found. Run common_ingest.py first.\n"
            f"Expected under: {os.path.join(export_path_base, relative_data_path, '<folder>', 'metadata', 'run_manifest.json')}"
        )

    # Optionally filter to one folder
    if only_folder is not None:
        manifests = [m for m in manifests if os.path.basename(os.path.dirname(os.path.dirname(m))) == only_folder]
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

        # Region processing
        for region in regions:
            region_name = region["region_name"]
            ref_pairs: List[List[int]] = region.get("ref_pairs", [])
            channels_unref: List[int] = region.get("channels_unreferenced", [])

            # Derive the "bundle" channel list from ref_pairs (e.g. 6-wire set)
            channels_group = unique_channels_from_ref_pairs(ref_pairs)

            print(f"\nProcessing Region: {region_name}")

            spikes_root = os.path.join(processed_folder_path, region_name, "Spikes")
            save_path = os.path.join(spikes_root, "continuous")
            plots_path = os.path.join(spikes_root, "Preprocessed_plots")
            ensure_dir(save_path)
            ensure_dir(plots_path)

            # ---------------------------
            # Compute LMR once per region
            # ---------------------------
            X_lmr = None
            t0 = None
            fs0 = None

            if channels_group:
                print(f" - Computing LMR from channels: {channels_group}")

                # Load all group channels into a matrix
                t0, x0, fs0 = print_timed(
                    f"Load LMR group ch{channels_group[0]}",
                    load_and_extract_data,
                    block_path,
                    channels_group[0],
                )

                X = np.empty((len(channels_group), x0.size), dtype=np.float32)
                X[0] = x0.astype(np.float32)

                for i, ch in enumerate(channels_group[1:], start=1):
                    t, x, fs = print_timed(f"Load LMR group ch{ch}", load_and_extract_data, block_path, ch)
                    if fs != fs0:
                        raise ValueError("Sampling rates mismatch in LMR group.")
                    if not np.array_equal(t, t0):
                        raise ValueError("Time vectors mismatch in LMR group.")
                    X[i] = x.astype(np.float32)

                # Compute local median reference (leave-one-out = safest)
                X_lmr = local_median_reference(X, leave_one_out=True)

            # ---------------------------
            # Export unreferenced
            # ---------------------------
            if run_unreferenced:
                for ch in channels_unref:
                    print(f"\nUnreferenced channel {ch}")
                    time_s, data, fs = print_timed(
                        f"Load Channel {ch}", load_and_extract_data, block_path, ch
                    )

                    label = f"unreferenced_ch{ch}"
                    process_and_export_continuous(
                        label=label,
                        time_s=time_s,
                        data=data,
                        fs=fs,
                        channel=int(ch),
                        reference=None,
                        save_path=save_path,
                        plots_path=plots_path,
                        filter_settings=spike_filter,
                    )

            # ---------------------------
            # Export LMR (no pairwise subtraction)
            # ---------------------------
            if X_lmr is not None:
                # Export LMR for the channels you actually care about.
                channels_to_export = [ch for ch in channels_unref if ch in channels_group]

                for ch in channels_to_export:
                    idx = channels_group.index(ch)
                    data_lmr = X_lmr[idx]

                    print(f"\nLMR channel {ch} (from group median)")
                    label = f"lmr_ch{ch}"
                    process_and_export_continuous(
                        label=label,
                        time_s=t0,
                        data=data_lmr,
                        fs=fs0,
                        channel=int(ch),
                        reference=None,
                        save_path=save_path,
                        plots_path=plots_path,
                        filter_settings=spike_filter,
                    )
            else:
                print(" - No LMR group channels found from ref_pairs; skipping LMR export.")

        # Optional: folder-level noise reference (spike-band)
        noise_ref = manifest.get("noise_reference_channel", None)
        if process_noise_reference_spikes and (noise_ref is not None):
            print(f"\nSpike-band Noise Reference Channel: {noise_ref}")

            noise_spikes_root = os.path.join(processed_folder_path, "Noise_reference", "Spikes")
            ensure_dir(noise_spikes_root)

            time_s, data, fs = print_timed(
                f"Load Noise Ref {noise_ref}", load_and_extract_data, block_path, int(noise_ref)
            )

            filtered = print_timed(
                "Filter Noise Ref (spike band)",
                butter_bandpass_filtfilt,
                data,
                fs,
                spike_filter["lowcut_hz"],
                spike_filter["highcut_hz"],
                spike_filter["order"],
            )

            export_data = {
                "filtered_data": filtered,
                "channel": int(noise_ref),
                "reference": None,
                "fs_original": float(fs),
                "filter_settings": dict(spike_filter),
                "label": "noise_reference_spikes",
            }

            out_pkl = os.path.join(
                noise_spikes_root,
                f"preprocessed_spikes_noise_reference_ch{int(noise_ref)}.pkl"
            )
            with open(out_pkl, "wb") as f:
                pickle.dump(export_data, f)

        elif noise_ref is None:
            print("\n- No noise reference channel in manifest; skipping spike-band noise ref.")

    total_duration = time.perf_counter() - start_time_script
    print(f"\nScript finished after {total_duration:.1f} seconds")


if __name__ == "__main__":
    main()