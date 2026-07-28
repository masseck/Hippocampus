# -*- coding: utf-8 -*-
# %% Header
"""
preprocessing_lfp_car.py

Manifest-driven LFP preprocessing with Common Average Reference (CAR)
for animals recorded with a floating (separate) ground/reference configuration.

Use this script for: ChR2 IDs 0714, 0895, 0896
Use standard preprocessing_lfp.py for: all other animals

Why CAR?
--------
In the early ChR2 animals, the headstage jumper wire was disconnected,
meaning ground and reference had separate floating potentials. This caused
common-mode noise to appear on every channel without a stable common basis
for subtraction. Standard differential referencing (LMR) did not reduce
noise as expected.

CAR is a re-referencing method: the new reference is the instantaneous mean
across all channels of one hemisphere, representing the shared noise floor.
Subtracting it from each channel removes the common-mode noise that LMR
failed to capture.

What this script exports
------------------------
For each unreferenced channel, two versions are exported:

  suffix "unreferenced"
      Raw channel → [noise filter] → lowpass filter → downsample.
      No re-referencing. Used as baseline to assess CAR effect.

  suffix "referenced_car"
      (raw channel - CAR) → [noise filter] → lowpass filter → downsample.
      CAR computed from all unreferenced channels of the same hemisphere.
      This is the corrected signal for use in process_lfp_psd.py.

Noise filtering
---------------
Two options are available, controlled by NOISE_FILTER_MODE:

  "nlms"
      Adaptive NLMS filter using the noise reference channel from the
      manifest. Removes broadband common-mode environmental noise.
      Applied after CAR subtraction, before lowpass filtering.
      Requires a noise_reference_channel in the manifest.

  "notch"
      IIR notch filter at 50 Hz and harmonics (100 Hz, 150 Hz, ...).
      Removes narrowband mains hum that NLMS cannot capture.
      Applied after CAR subtraction, before lowpass filtering.
      Does not require a noise reference channel.

  None
      No noise filtering. Only CAR + lowpass + downsample.

The noise reference channel is always exported unchanged regardless of
NOISE_FILTER_MODE, for compatibility with process_lfp_psd.py.

Output format
-------------
Output pkl files are identical in structure to preprocessing_lfp.py.
All downstream scripts work without modification.

Outputs:
  <folder>/<region>/LFP/continuous/
      preprocessed_lfp_unreferenced_ch{ch}_ref-1.pkl
      preprocessed_lfp_referenced_car_ch{ch}_ref-1.pkl
  <folder>/<region>/LFP/Preprocessed_plots_CAR/
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
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import iirnotch, filtfilt

from Functions_preprocessing_lfp import (
    load_block,
    extract_channel,
    filter_data,
    plot_raw_and_filtered_data,
    downsample_data,
    normalize_noise_reference,
)


def _prompt_path(label: str, default: str) -> str:
    """Prompt the user to confirm or override a path at startup."""
    user_input = input(f"{label}\n  [{default}]: ").strip()
    return user_input if user_input else default


# -----------------------------
# Settings
# -----------------------------
# USER SETTINGS — adjust before running
_DEFAULT_DATA_PATH_BASE     = r"C:\Users\Juliana\Documents\_PhD\Data\_Raw"
_DEFAULT_EXPORT_PATH_BASE   = r"C:\Users\Juliana\Documents\_PhD\Data\_Processed"
_DEFAULT_RELATIVE_DATA_PATH = r"Jills_paper\ChR2_cohort\OFT_experimental"

only_folder: Optional[str] = None
FLOATING_REF_ANIMAL_IDS = {"0714", "0895", "0896"}

only_regions: Optional[List[str]] = ["CA1_L", "CA1_R"]

# What to export per channel
run_unreferenced   = True   # raw channel without CAR (baseline comparison)
run_referenced_car = True   # CAR-corrected channel (the actual correction)

# -----------------------------
# Noise filter settings
# -----------------------------
# NOISE_FILTER_MODE controls how environmental noise is removed
# BEFORE the lowpass filter and downsampling:
#
#   "nlms"  — adaptive NLMS filter using the noise reference channel.
#              Best for broadband common-mode noise.
#              Requires noise_reference_channel in the manifest.
#
#   "notch" — IIR notch filter at 50 Hz and harmonics.
#              Best for narrowband mains hum (50 Hz and overtones).
#              Does not require a noise reference channel.
#              Adjust NOTCH_FREQS_HZ and NOTCH_Q as needed.
#
#   None    — no noise filtering (CAR + lowpass + downsample only).
#
NOISE_FILTER_MODE = "notch"   # "nlms" | "notch" | None

# NLMS settings (used when NOISE_FILTER_MODE = "nlms")
NLMS_MU            = 0.015
NLMS_N_TAPS_FACTOR = 3   # n_taps = fs / 50 * factor

# Notch filter settings (used when NOISE_FILTER_MODE = "notch")
# List all frequencies to suppress in Hz.
# For European mains hum: 50 Hz + harmonics up to Nyquist of raw signal.
# Q controls notch width: higher Q = narrower notch.
# Q = 30 gives a bandwidth of ~1.7 Hz at 50 Hz — narrow enough to preserve
# neighbouring LFP content while fully suppressing the line noise peak.
NOTCH_FREQS_HZ: List[float] = [50.0, 100.0, 150.0, 200.0]
NOTCH_Q: float = 30.0


# -----------------------------
# Helpers
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
    return {("filter_type" if k == "type" else k): v for k, v in filter_settings.items()}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_filter_settings(lfp_settings: Dict[str, Any]) -> Dict[str, Any]:
    fset = lfp_settings.get("filter")
    if fset is None:
        raise ValueError("Missing LFP filter settings in manifest.")
    return fset


def get_downsampling_hz(lfp_settings: Dict[str, Any]) -> float:
    ds_hz = lfp_settings.get("downsampling_Hz")
    if ds_hz is None:
        raise ValueError("Missing downsampling_Hz in manifest.")
    return float(ds_hz)


# -----------------------------
# CAR
# -----------------------------
def compute_car(raw_signals: List[np.ndarray]) -> np.ndarray:
    """
    Compute Common Average Reference from a list of raw signals.

    Parameters
    ----------
    raw_signals : list of np.ndarray
        Raw LFP traces per channel, each shape (n_samples,).

    Returns
    -------
    car : np.ndarray
        Common average reference, shape (n_samples,).
    """
    return np.mean(np.vstack(raw_signals), axis=0)


# -----------------------------
# Noise filtering
# -----------------------------
def apply_notch_filter(
    data: np.ndarray,
    fs: float,
    freqs_hz: List[float],
    q: float,
) -> np.ndarray:
    """
    Apply IIR notch filters sequentially at each frequency in freqs_hz.

    Each notch is applied with zero-phase filtering (filtfilt) to avoid
    phase distortion. Frequencies above Nyquist (fs/2) are skipped.

    Parameters
    ----------
    data : np.ndarray
        Input signal, shape (n_samples,).
    fs : float
        Sampling rate in Hz.
    freqs_hz : list of float
        Frequencies to suppress in Hz.
    q : float
        Quality factor controlling notch width.
        Bandwidth = f0 / Q (e.g. Q=30 → ~1.7 Hz wide at 50 Hz).

    Returns
    -------
    np.ndarray : filtered signal, same shape as data.
    """
    nyquist = fs / 2.0
    result  = data.copy()

    for f0 in freqs_hz:
        if f0 >= nyquist:
            print(f"   ! Notch {f0:.0f} Hz >= Nyquist ({nyquist:.0f} Hz), skipping.")
            continue
        b, a = iirnotch(w0=f0, Q=q, fs=fs)
        result = filtfilt(b, a, result)
        print(f"   Notch filter applied: {f0:.0f} Hz (Q={q:.0f})")

    return result


def apply_nlms_filter(
    data: np.ndarray,
    reference: np.ndarray,
    fs: float,
    mu: float,
    n_taps_factor: int,
) -> np.ndarray:
    """
    Apply NLMS adaptive filter to remove broadband noise using a reference channel.

    Parameters
    ----------
    data : np.ndarray
        Signal to filter, shape (n_samples,).
    reference : np.ndarray
        Noise reference signal, shape (n_samples,) or different length
        (will be resampled to match data length).
    fs : float
        Sampling rate of data in Hz.
    mu : float
        NLMS step size.
    n_taps_factor : int
        Number of filter taps = int(fs / 50) * n_taps_factor.

    Returns
    -------
    np.ndarray : filtered signal.
    """
    if len(reference) != len(data):
        reference = np.interp(
            np.linspace(0, 1, len(data)),
            np.linspace(0, 1, len(reference)),
            reference,
        )

    n_taps        = int(fs / 50) * n_taps_factor
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


def apply_noise_filter(
    data: np.ndarray,
    fs: float,
    noise_ref: Optional[np.ndarray],
    mode: Optional[str],
) -> np.ndarray:
    """
    Dispatch to the selected noise filter.

    Parameters
    ----------
    data : np.ndarray
        Signal to filter.
    fs : float
        Sampling rate in Hz.
    noise_ref : np.ndarray or None
        Noise reference channel (required for "nlms", ignored for "notch").
    mode : str or None
        "nlms", "notch", or None.

    Returns
    -------
    np.ndarray : filtered signal.
    """
    if mode is None:
        return data

    if mode == "notch":
        return apply_notch_filter(data, fs, NOTCH_FREQS_HZ, NOTCH_Q)

    if mode == "nlms":
        if noise_ref is None:
            print("   ! NLMS requested but no noise reference available; skipping.")
            return data
        return apply_nlms_filter(data, noise_ref, fs, NLMS_MU, NLMS_N_TAPS_FACTOR)

    raise ValueError(f"Unknown NOISE_FILTER_MODE: {mode!r}. Use 'nlms', 'notch', or None.")


# -----------------------------
# Export helper
# -----------------------------
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
    noise_ref: Optional[np.ndarray] = None,
) -> None:
    """
    [Noise filter] → lowpass filter → plot → downsample → export.

    The noise filter (NLMS or notch) is applied first on the raw-rate signal,
    before the lowpass filter and downsampling.
    """
    fset    = get_filter_settings(lfp_settings)
    ds_hz   = get_downsampling_hz(lfp_settings)
    fkwargs = filter_kwargs_from_settings(fset)

    # Step 1: Noise filter at raw sampling rate
    if NOISE_FILTER_MODE is not None:
        print(f"   Noise filter ({NOISE_FILTER_MODE}): ch{channel} ({suffix})")
        data = apply_noise_filter(data, fs, noise_ref, NOISE_FILTER_MODE)

    # Step 2: Lowpass filter
    filtered = print_timed(
        f"Lowpass ch{channel} ({suffix})", filter_data, data, fs, **fkwargs
    )

    # Step 3: QC plot
    print_timed(
        f"Plot ch{channel} ({suffix})",
        plot_raw_and_filtered_data,
        time_s, data, filtered,
        str(channel), fset.get("type", "unknown"), suffix, plots_path,
    )

    # Step 4: Downsample
    downsampled = print_timed(
        f"Downsample ch{channel} ({suffix})", downsample_data, filtered, fs, ds_hz
    )

    export_data = {
        "downsampled_data": downsampled,
        "channel":          channel,
        "reference":        reference,
        "settings":         lfp_settings,
        "fs_original":      float(fs),
        "suffix":           suffix,
        "noise_filter":     NOISE_FILTER_MODE,
    }

    out_file = os.path.join(
        save_path, f"preprocessed_lfp_{suffix}_ch{channel}_ref{reference}.pkl"
    )
    with open(out_file, "wb") as f:
        pickle.dump(export_data, f)
    print(f"   → Saved: {os.path.basename(out_file)}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    start_time_script = time.perf_counter()

    print("=" * 60)
    print("preprocessing_lfp_car.py — path configuration")
    print("Press Enter to accept the default path shown in brackets.")
    print("=" * 60)
    data_path_base     = _prompt_path("Raw data root:",       _DEFAULT_DATA_PATH_BASE)
    export_path_base   = _prompt_path("Processed data root:", _DEFAULT_EXPORT_PATH_BASE)
    relative_data_path = _prompt_path("Relative cohort path:", _DEFAULT_RELATIVE_DATA_PATH)
    print()

    print(f"Noise filter mode: {NOISE_FILTER_MODE}")
    if NOISE_FILTER_MODE == "notch":
        print(f"  Notch freqs: {NOTCH_FREQS_HZ} Hz | Q={NOTCH_Q}")
    elif NOISE_FILTER_MODE == "nlms":
        print(f"  NLMS mu={NLMS_MU} | n_taps_factor={NLMS_N_TAPS_FACTOR}")

    manifests = list_manifests(export_path_base, relative_data_path)
    if not manifests:
        raise FileNotFoundError(
            "No manifests found. Run common_ingest.py first.\n"
            f"Expected under: "
            f"{os.path.join(export_path_base, relative_data_path)}"
        )

    manifests = [
        m for m in manifests
        if any(
            aid in os.path.basename(os.path.dirname(os.path.dirname(m)))
            for aid in FLOATING_REF_ANIMAL_IDS
        )
    ]
    if not manifests:
        raise FileNotFoundError(
            f"No manifests found for floating-reference animals "
            f"{FLOATING_REF_ANIMAL_IDS}."
        )

    if only_folder is not None:
        manifests = [
            m for m in manifests
            if os.path.basename(os.path.dirname(os.path.dirname(m))) == only_folder
        ]
        if not manifests:
            raise FileNotFoundError(f"No manifest found for only_folder='{only_folder}'.")

    print(f"\nProcessing {len(manifests)} floating-reference folder(s) with CAR...")
    print(f"Animal IDs: {FLOATING_REF_ANIMAL_IDS}\n")

    for manifest_path in manifests:
        manifest = load_json(manifest_path)

        folder                = manifest["folder"]
        block_path            = manifest["block_path"]
        processed_folder_path = manifest["processed_folder_path"]

        print(f"\n==============================")
        print(f"Folder: {folder}")
        print(f"Block path: {block_path}")

        if not os.path.isdir(block_path):
            print(f" - Raw block path does not exist, skipping: {block_path}")
            continue

        regions = manifest.get("regions", [])
        if only_regions is not None:
            regions = [r for r in regions if r.get("region_name") in only_regions]

        if not regions:
            print(" - No regions to process.")
            continue

        print(f" - Loading TDT block...")
        block_data = print_timed("Load block", load_block, block_path)

        # Load noise reference once per folder (needed for NLMS, ignored for notch)
        noise_ref_data: Optional[np.ndarray] = None
        if NOISE_FILTER_MODE == "nlms":
            noise_ref_ch = manifest.get("noise_reference_channel", None)
            if noise_ref_ch is not None:
                _, noise_raw, noise_fs = extract_channel(block_data, int(noise_ref_ch))
                noise_ref_data = normalize_noise_reference(noise_raw)
                print(f" - Noise reference loaded: ch{noise_ref_ch}")
            else:
                print(" - NLMS requested but no noise_reference_channel in manifest; skipping NLMS.")

        for region in regions:
            region_name               = region["region_name"]
            lfp_settings              = region.get("lfp_settings", {})
            channels_unref: List[int] = region.get("channels_unreferenced", [])

            print(f"\nProcessing Region: {region_name}")

            if not channels_unref:
                print(f" - No unreferenced channels in manifest for {region_name}, skipping.")
                continue

            lfp_root   = os.path.join(processed_folder_path, region_name, "LFP")
            save_path  = os.path.join(lfp_root, "continuous")
            plots_path = os.path.join(lfp_root, "Preprocessed_plots_CAR")
            ensure_dir(save_path)
            ensure_dir(plots_path)

            # Step 1: Load all raw channels
            print(f" - Loading {len(channels_unref)} raw channels: {channels_unref}")

            raw_signals: Dict[int, np.ndarray] = {}
            time_s_ref: Optional[np.ndarray]   = None
            fs_ref:     Optional[float]         = None

            for ch in channels_unref:
                t, d, fs = extract_channel(block_data, ch)
                raw_signals[ch] = d
                if fs_ref is None:
                    fs_ref     = fs
                    time_s_ref = t
                elif fs != fs_ref:
                    raise ValueError(
                        f"Sampling rate mismatch: ch{ch}={fs}Hz vs {fs_ref}Hz"
                    )

            # Step 2: Compute CAR
            print(
                f" - Computing CAR from {len(channels_unref)} channels "
                f"({region_name} = one hemisphere)"
            )
            car = print_timed(
                "Compute CAR",
                compute_car,
                [raw_signals[ch] for ch in channels_unref],
            )
            print(
                f"   CAR stats: mean={car.mean():.4f} µV | "
                f"std={car.std():.4f} µV | "
                f"p2p={np.ptp(car):.4f} µV"
            )

            # Step 3: Export both versions per channel
            for ch in channels_unref:

                if run_unreferenced:
                    print(f"\nUnreferenced (no CAR): channel {ch}")
                    filter_downsample_export(
                        time_s=time_s_ref,
                        data=raw_signals[ch].copy(),
                        fs=fs_ref,
                        channel=ch,
                        reference=-1,
                        suffix="unreferenced",
                        lfp_settings=lfp_settings,
                        save_path=save_path,
                        plots_path=plots_path,
                        noise_ref=noise_ref_data,
                    )

                if run_referenced_car:
                    print(f"\nReferenced CAR: channel {ch}")
                    data_car = raw_signals[ch] - car
                    filter_downsample_export(
                        time_s=time_s_ref,
                        data=data_car,
                        fs=fs_ref,
                        channel=ch,
                        reference=-1,
                        suffix="referenced_car",
                        lfp_settings=lfp_settings,
                        save_path=save_path,
                        plots_path=plots_path,
                        noise_ref=noise_ref_data,
                    )

        # Export noise reference channel unchanged (for downstream compatibility)
        noise_ref_ch = manifest.get("noise_reference_channel", None)
        if noise_ref_ch is not None:
            print(f"\nNoise Reference Channel: {noise_ref_ch}")

            first_region = regions[0]
            fset  = first_region.get("lfp_settings", {}).get("filter")
            ds_hz = first_region.get("lfp_settings", {}).get("downsampling_Hz")

            if fset is None or ds_hz is None:
                print(" - Noise ref settings missing; skipping.")
            else:
                noise_root     = os.path.join(processed_folder_path, "Noise_reference")
                noise_lfp_root = os.path.join(noise_root, "LFP")
                ensure_dir(noise_lfp_root)

                time_s, noise_data, fs = extract_channel(block_data, int(noise_ref_ch))
                normalized = print_timed(
                    "Normalize noise ref", normalize_noise_reference, noise_data
                )
                fkwargs  = filter_kwargs_from_settings(fset)
                filtered = print_timed(
                    "Filter noise ref", filter_data, normalized, fs, **fkwargs
                )
                print_timed(
                    "Plot noise ref",
                    plot_raw_and_filtered_data,
                    time_s, normalized, filtered,
                    str(noise_ref_ch), fset.get("type", "unknown"),
                    "noise_reference", noise_lfp_root,
                )
                downsampled = print_timed(
                    "Downsample noise ref", downsample_data, filtered, fs, float(ds_hz)
                )

                export_data = {
                    "downsampled_data": downsampled,
                    "channel":          int(noise_ref_ch),
                    "reference":        None,
                    "settings": {
                        "filter":          fset,
                        "downsampling_Hz": ds_hz,
                    },
                    "fs_original": float(fs),
                    "suffix":      "noise_reference",
                }

                out_file = os.path.join(
                    noise_lfp_root,
                    f"preprocessed_lfp_noise_reference_ch{int(noise_ref_ch)}.pkl",
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