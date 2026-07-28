# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:06:53 2026

Shared ingest step for ephys analysis pipeline (LFP + spikes).

Steps:
- Discover folders (recordings) inside a raw experiment path
- Validate against experiments dict
- Create processed output folder structure
- Copy shared metadata (Pynapse log, Notes, StoresListing, Video)
- Build + save a run_manifest.json per folder (single source of truth)

Downstream scripts:
- preprocessing_lfp.py reads manifest and produces LFP outputs
- preprocessing_spikes.py reads manifest and produces spike outputs

@author: Juliana
"""

from __future__ import annotations

import os
import json
import time
import shutil
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Your project imports
from experiments_differential_ref_pairs import experiment_data

# %%
# Config / constants

REGION_ORDER = ["CA1_L", "CA1_R"]

METADATA_FILES = [
    "Pynapse_call_log.csv",
    "Notes.txt",
    "StoresListing.txt",
]

# %% 
# Helpers

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_recording_folders(full_data_path: str) -> List[str]:
    """List recording folders inside the raw experiment directory."""
    folders = []
    for name in os.listdir(full_data_path):
        p = os.path.join(full_data_path, name)
        if os.path.isdir(p) and name not in ("ignore", "processed"):
            folders.append(name)
    folders.sort()
    return folders

def copy_metadata_files(src_block_path: str, dst_metadata_path: str, video_file: Optional[str]) -> None:
    """Copy metadata files + optional video into dst/metadata/."""
    safe_makedirs(dst_metadata_path)

    for fname in METADATA_FILES:
        src = os.path.join(src_block_path, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst_metadata_path, fname))
        else:
            print(f"  ! Missing metadata file: {src}")

    if video_file:
        src_vid = os.path.join(src_block_path, video_file)
        if os.path.exists(src_vid):
            shutil.copy(src_vid, os.path.join(dst_metadata_path, "Recording.mp4"))
        else:
            print(f"  ! Missing video file: {src_vid}")

def pick_noise_ref_channel(experiments: Dict[str, Any], folder: str) -> Optional[int]:
    """Try to find a noise reference channel for this recording folder."""
    # Your current logic: stored per region sometimes.
    for region in REGION_ORDER:
        if region in experiments.get(folder, {}) and "noise_ref" in experiments[folder][region]:
            return experiments[folder][region]["noise_ref"]
    return None

def collect_region_spec(experiments: Dict[str, Any], folder: str, region: str) -> Optional[Dict[str, Any]]:
    """Return a minimal region spec used by downstream pipelines."""
    if region not in experiments.get(folder, {}):
        return None

    settings = experiments[folder][region]

    # Flatten channels from ref_pairs (same as you do now)
    ref_pairs = settings.get("ref_pairs", [])
    channels = sorted(set(sum(ref_pairs, []))) if ref_pairs else []

    region_spec = {
        "region_name": region,
        "ref_pairs": ref_pairs,
        "channels_unreferenced": channels,
        # Keep pipeline-specific settings in the manifest so downstream is deterministic
        "lfp_settings": {
            "filter": settings.get("filter"),
            "downsampling_Hz": settings.get("downsampling_Hz"),
        },
        # Spikes settings can be filled later; keep a placeholder here
        "spike_settings": settings.get("spike_settings", None),
    }
    return region_spec


def create_output_skeleton(processed_folder_path: str, regions: List[Dict[str, Any]]) -> None:
    """
    Create folder skeleton so downstream scripts have consistent targets.

    Layout:
      <folder>/metadata/
      <folder>/<region>/LFP/
      <folder>/<region>/Spikes/
      <folder>/<region>/QC/
      <folder>/Noise_reference/LFP/
      <folder>/Noise_reference/Spikes/
    """
    safe_makedirs(processed_folder_path)

    # Region-level structure
    for region in regions:
        region_root = os.path.join(processed_folder_path, region["region_name"])
        safe_makedirs(os.path.join(region_root, "LFP"))
        safe_makedirs(os.path.join(region_root, "Spikes"))
        safe_makedirs(os.path.join(region_root, "QC"))

    # NEW: folder-level noise reference structure
    noise_root = os.path.join(processed_folder_path, "Noise_reference")
    safe_makedirs(os.path.join(noise_root, "LFP"))
    safe_makedirs(os.path.join(noise_root, "Spikes"))


def write_json(obj: Any, path: str) -> None:
    safe_makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# %%
# Manifest schema (lightweight)

@dataclass
class Manifest:
    schema_version: str
    created_at: str

    data_path_base: str
    export_path_base: str
    relative_data_path: str  # e.g. 'Jills_paper\\vSWO_cohort\\OFT_experimental'

    folder: str
    block_path: str
    processed_folder_path: str

    metadata_path: str
    video_file_expected: Optional[str]

    regions: List[Dict[str, Any]]
    noise_reference_channel: Optional[int]

    # Put global knobs here if you like (good for reproducibility)
    notes: str = ""

# %%
# Main ingest

def build_manifest(
    *,
    experiments: Dict[str, Any],
    data_path_base: str,
    export_path_base: str,
    relative_data_path: str,
    folder: str,
) -> Manifest:
    full_data_path = os.path.join(data_path_base, relative_data_path)
    block_path = os.path.join(full_data_path, folder)

    processed_folder_path = os.path.join(export_path_base, relative_data_path, folder)
    metadata_path = os.path.join(processed_folder_path, "metadata")

    experiment_folder = os.path.basename(relative_data_path.rstrip("\\/"))
    video_file_expected = f"{experiment_folder}_{folder}_CamTOP.mp4"

    regions: List[Dict[str, Any]] = []
    for region in REGION_ORDER:
        spec = collect_region_spec(experiments, folder, region)
        if spec is not None:
            regions.append(spec)

    noise_ref = pick_noise_ref_channel(experiments, folder)

    return Manifest(
        schema_version="1.0",
        created_at=_now_iso(),
        data_path_base=data_path_base,
        export_path_base=export_path_base,
        relative_data_path=relative_data_path,
        folder=folder,
        block_path=block_path,
        processed_folder_path=processed_folder_path,
        metadata_path=metadata_path,
        video_file_expected=video_file_expected,
        regions=regions,
        noise_reference_channel=noise_ref,
        notes="Created by common_ingest.py",
    )

def run_ingest(
    *,
    data_path_base: str,
    export_path_base: str,
    relative_data_path: str,
) -> None:
    experiments = experiment_data()

    full_data_path = os.path.join(data_path_base, relative_data_path)
    folders = list_recording_folders(full_data_path)

    print(f"Found {len(folders)} recording folders in:\n  {full_data_path}\n")

    for folder in folders:
        print(f"\n=== {folder} ===")

        if folder not in experiments:
            print(f"  ! ERROR: No experiment info for '{folder}'. Update experiments file.")
            continue

        manifest = build_manifest(
            experiments=experiments,
            data_path_base=data_path_base,
            export_path_base=export_path_base,
            relative_data_path=relative_data_path,
            folder=folder,
        )

        # Create base processed dirs
        safe_makedirs(manifest.processed_folder_path)
        safe_makedirs(manifest.metadata_path)

        # Copy metadata (once per recording folder)
        copy_metadata_files(
            src_block_path=manifest.block_path,
            dst_metadata_path=manifest.metadata_path,
            video_file=manifest.video_file_expected,
        )

        create_output_skeleton(manifest.processed_folder_path, manifest.regions)

        # Save manifest
        manifest_path = os.path.join(manifest.metadata_path, "run_manifest.json")
        write_json(asdict(manifest), manifest_path)

        print(f"  ✓ Manifest: {manifest_path}")
        print(f"  ✓ Regions: {[r['region_name'] for r in manifest.regions]}")
        if manifest.noise_reference_channel is not None:
            print(f"  ✓ Noise ref: {manifest.noise_reference_channel}")
        else:
            print("  - Noise ref: none found")

# %%
# Path configuration helper

def _prompt_path(label: str, default: str) -> str:
    """
    Prompt the user to confirm or override a path.
    Press Enter to accept the default, or type a new path.
    """
    user_input = input(f"{label}\n  [{default}]: ").strip()
    return user_input if user_input else default


# %%
# CLI-like usage

if __name__ == "__main__":
    print("=" * 60)
    print("common_ingest.py — path configuration")
    print("Press Enter to accept the default path shown in brackets.")
    print("=" * 60)

    data_path_base = _prompt_path(
        "Raw data root (contains recording folders):",
        r"C:\Users\Juliana\Documents\_PhD\Data\_Raw",
    )
    export_path_base = _prompt_path(
        "Processed data root (outputs will be written here):",
        r"C:\Users\Juliana\Documents\_PhD\Data\_Processed",
    )
    relative_data_path = _prompt_path(
        "Relative path to cohort (e.g. Jills_paper\\ChR2_cohort\\OFT_experimental):",
        r"Jills_paper\ChR2_cohort\OFT_experimental",
    )

    print()
    run_ingest(
        data_path_base=data_path_base,
        export_path_base=export_path_base,
        relative_data_path=relative_data_path,
    )