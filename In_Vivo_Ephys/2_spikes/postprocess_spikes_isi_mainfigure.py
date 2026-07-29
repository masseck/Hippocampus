# -*- coding: utf-8 -*-
"""
postprocess_spikes_isi_mainfigure.py

Postprocessing script for main-figure ISI analysis.

Outputs
-------
1) Group ISI distribution plot (0-25 ms), averaged across animals
2) Paired burst-fraction Gardner-Altman plot per animal
   + companion CSV with dabest estimation statistics (mean diff, BCa CI)
3) Per-animal ISI overview plot (QC for outlier detection)
   - All channels overlaid per animal subplot
   - All animals side by side in one figure
   - Refractory violation fraction annotated per animal

CI export
---------
For the burst-fraction GA plot, a companion CSV is saved with:
  - mean_diff              : mean difference (stimulation - baseline)
  - ci                     : confidence interval level (default 95%)
  - bca_low, bca_high      : bias-corrected and accelerated bootstrap CI
  - pct_low, pct_high      : percentile bootstrap CI
  - n_animals              : number of animals
  - mean_baseline, mean_stim : group means

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

Notes on normalization
----------------------
Histograms are normalized over the FULL ISI distribution (0-1 s), not the
zoom window. Zoom to MAX_ISI_S is applied only at plot time via zoom_mask.
This ensures fractions are comparable across conditions and tools.
"""

from __future__ import annotations

import os
import time
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import dabest

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
    refractory_violations_fraction,
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
only_regions: Optional[List[str]] = ["CA1_L"]

PROCESS_UNREFERENCED = False
PROCESS_LMR          = True

BASELINE_LABEL = "In baseline state"
STIM_LABEL     = "In stimulation state"
IGNORE_LABEL   = "In start delay"

# ISI settings
MAX_ISI_S      = 0.025   # zoom window in seconds (display only)
BIN_S          = 0.0005  # 0.5 ms bins
FULL_ISI_MAX_S = 1.0     # upper bound for full-range normalization
NORMALIZE_MODE = "fraction"

ISI_YLIM = (0.0, 0.16)   # fixed y-axis across tools for direct comparison

BURST_THRESHOLD_MS      = 6.0
MIN_CHANNELS_PER_ANIMAL = 1

# Plot settings
PLOT_THEME          = "light"
STIM_COLOR_OVERRIDE = None

FIG_HEIGHT_MM       = 45
FIG_ASPECT          = 1.35
FIG_ASPECT_OVERVIEW = 1.0

# -----------------------------
# Adjusted window settings
# -----------------------------
USE_ADJUSTED_WINDOWS = True   # True = adjusted | False = full epochs
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

FIG_SIZE          = get_figure_size(height_mm=FIG_HEIGHT_MM, aspect_ratio=FIG_ASPECT)
FIG_SIZE_OVERVIEW = get_figure_size(height_mm=FIG_HEIGHT_MM, aspect_ratio=FIG_ASPECT_OVERVIEW)

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
    if np.any(n < 2):
        print(f"   ! mean_sem: {int(np.sum(n < 2))} bin(s) have n<2 — SEM=nan.")
    return mean, sem


def burst_fraction_val(isi_s: np.ndarray, threshold_ms: float = 6.0) -> float:
    if isi_s.size == 0:
        return np.nan
    return float(np.mean(isi_s < threshold_ms / 1000.0))


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


def build_stim_windows(epochs, block_path: Optional[str]) -> List[Tuple[float, float]]:
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


def extract_dabest_ci(mean_diff_obj, title: str, n_animals: int,
                      mean_base: float, mean_stim: float) -> dict:
    """
    Extract estimation statistics from a dabest mean_diff object.

    Returns a dict with mean_diff_hz, bca_low, bca_high, pct_low, pct_high,
    ci, n_animals, mean_baseline, mean_stim.
    Handles column name differences across dabest versions gracefully.
    """
    ci_row = {
        "title":        title,
        "n_animals":    n_animals,
        "mean_baseline": float(mean_base),
        "mean_stim":     float(mean_stim),
        "mean_diff":    float("nan"),
        "ci":           float("nan"),
        "bca_low":      float("nan"),
        "bca_high":     float("nan"),
        "pct_low":      float("nan"),
        "pct_high":     float("nan"),
    }

    try:
        stats = mean_diff_obj.statistical_tests

        def _get(col_candidates):
            for c in col_candidates:
                if c in stats.columns:
                    return float(stats[c].iloc[0])
            # fallback: partial match
            for c in stats.columns:
                for cand in col_candidates:
                    if cand.lower() in c.lower():
                        return float(stats[c].iloc[0])
            return float("nan")

        ci_row["mean_diff"] = _get(["difference", "mean_diff", "effect_size"])
        ci_row["ci"]        = _get(["ci", "confidence_interval"])
        ci_row["bca_low"]   = _get(["bca_low", "BCa_low", "bca_lower"])
        ci_row["bca_high"]  = _get(["bca_high", "BCa_high", "bca_upper"])
        ci_row["pct_low"]   = _get(["pct_low", "percentile_low"])
        ci_row["pct_high"]  = _get(["pct_high", "percentile_high"])

        print(
            f"  CI: mean diff={ci_row['mean_diff']:.4f} | "
            f"95% BCa [{ci_row['bca_low']:.4f}, {ci_row['bca_high']:.4f}]"
        )

    except Exception as e:
        print(f"  ! Could not extract CI statistics: {e}")
        try:
            print(f"  Available columns: {list(mean_diff_obj.statistical_tests.columns)}")
        except Exception:
            pass

    return ci_row


# -----------------------------
# Plotting
# -----------------------------
def plot_group_isi_zoom(
    bin_edges, mean_base, sem_base, mean_stim, sem_stim,
    out_png, out_svg, title, style, normalize_mode, n_animals,
) -> None:
    ensure_dir(os.path.dirname(out_png))
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    apply_figure_style(fig, style)
    apply_axes_style(ax, style)

    centers_ms = (bin_edges[:-1] + np.diff(bin_edges) / 2.0) * 1000.0
    zoom_mask  = centers_ms <= MAX_ISI_S * 1000.0

    ax.plot(centers_ms[zoom_mask], mean_base[zoom_mask],
            color=style["baseline_color"], linewidth=1.8, label="Baseline")
    ax.fill_between(centers_ms[zoom_mask],
                    (mean_base - sem_base)[zoom_mask],
                    (mean_base + sem_base)[zoom_mask],
                    color=style["baseline_color"], alpha=0.25)
    ax.plot(centers_ms[zoom_mask], mean_stim[zoom_mask],
            color=style["stim_color"], linewidth=1.8, label="Stimulation")
    ax.fill_between(centers_ms[zoom_mask],
                    (mean_stim - sem_stim)[zoom_mask],
                    (mean_stim + sem_stim)[zoom_mask],
                    color=style["stim_color"], alpha=0.25)

    ax.set_xlim(0, MAX_ISI_S * 1000.0)
    ax.set_ylim(*ISI_YLIM)
    # x-ticks: always include 0 and MAX, plus intermediate steps of 5 ms
    xticks = list(range(0, int(MAX_ISI_S * 1000) + 1, 5))
    ax.set_xticks(xticks)
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


def plot_per_animal_overview(
    animal_data: List[Dict],
    out_png: str,
    out_svg: str,
    region_name: str,
    kind: str,
    style: dict,
    bin_edges: np.ndarray,
    window_suffix: str,
) -> None:
    if not animal_data:
        return
    ensure_dir(os.path.dirname(out_png))

    n_animals = len(animal_data)
    fig, axes = plt.subplots(
        1, n_animals,
        figsize=(FIG_SIZE_OVERVIEW[0] * n_animals, FIG_SIZE_OVERVIEW[1]),
        sharey=True,
    )
    if n_animals == 1:
        axes = [axes]

    centers_ms = (bin_edges[:-1] + np.diff(bin_edges) / 2.0) * 1000.0
    zoom_mask  = centers_ms <= MAX_ISI_S * 1000.0

    for ax, animal in zip(axes, animal_data):
        apply_figure_style(fig, style)
        apply_axes_style(ax, style)

        for h_b, h_s in zip(animal["hists_base"], animal["hists_stim"]):
            ax.plot(centers_ms[zoom_mask], h_b[zoom_mask],
                    color=style["baseline_color"], linewidth=0.7, alpha=0.5)
            ax.plot(centers_ms[zoom_mask], h_s[zoom_mask],
                    color=style["stim_color"], linewidth=0.7, alpha=0.5)

        rv_b = [v for v in animal["rv_base_channels"] if np.isfinite(v)]
        rv_s = [v for v in animal["rv_stim_channels"] if np.isfinite(v)]
        rv_b_mean = float(np.mean(rv_b)) if rv_b else float("nan")
        rv_s_mean = float(np.mean(rv_s)) if rv_s else float("nan")

        ax.text(
            0.97, 0.97,
            f"RV base: {rv_b_mean:.3f}\nRV stim: {rv_s_mean:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=4.5,
            color=style["text_color"],
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor=style["axes_facecolor"],
                      edgecolor=style["neutral_color"], alpha=0.8),
        )

        n_ch     = len(animal["hists_base"])
        short_id = animal["folder"].split("-")[0] if "-" in animal["folder"] \
            else animal["folder"][:12]
        ax.set_title(f"{short_id}\n(n={n_ch} ch)", fontsize=5.5)
        ax.set_xlabel("ISI (ms)", fontsize=5.5)
        ax.set_xlim(0, MAX_ISI_S * 1000.0)
        ax.set_ylim(*ISI_YLIM)
        xticks = list(range(0, int(MAX_ISI_S * 1000) + 1, 5))
        ax.set_xticks(xticks)
        ax.grid(False)

    axes[0].set_ylabel("Fraction of ISIs", fontsize=5.5)
    fig.legend(
        handles=[
            Line2D([0], [0], color=style["baseline_color"],
                   linewidth=1.2, label="Baseline"),
            Line2D([0], [0], color=style["stim_color"],
                   linewidth=1.2, label="Stimulation"),
        ],
        loc="upper right", fontsize=5.5, frameon=False,
        bbox_to_anchor=(1.0, 1.0),
    )
    fig.suptitle(
        f"Per-animal ISI overview | {region_name} | {kind} | {window_suffix}",
        fontsize=7, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    plt.close(fig)


def apply_dabest_style_burst(fig, style: dict) -> None:
    fig.patch.set_facecolor(style["figure_facecolor"])
    for i, ax in enumerate(fig.axes):
        # Only style the left (raw data) panel.
        # The right (difference) panel is left entirely to dabest —
        # any post-hoc color or position changes cause the axis to float.
        if i != 0:
            continue
        ax.set_facecolor(style["axes_facecolor"])
        ax.tick_params(colors=style["text_color"])
        ax.xaxis.label.set_color(style["text_color"])
        ax.yaxis.label.set_color(style["text_color"])
        ax.title.set_color(style["text_color"])
        for side in ["left", "bottom", "right", "top"]:
            if side in ax.spines:
                ax.spines[side].set_position(("outward", 0))
                ax.spines[side].set_edgecolor(style["text_color"])
        for txt in ax.texts:
            txt.set_color(style["text_color"])
        for line in ax.lines:
            try:
                line.set_color(style["text_color"])
                line.set_alpha(0.7)
            except Exception as e:
                print(f"  ! Dabest style warning (line): {e}")
        for coll in ax.collections:
            try:
                coll.set_edgecolor(style["text_color"])
            except Exception as e:
                print(f"  ! Dabest style warning (edgecolor): {e}")
    if len(fig.axes) > 1:
        ax_right = fig.axes[1]
        ax_right.spines["left"].set_visible(False)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["bottom"].set_visible(True)
        ax_right.spines["right"].set_visible(True)
        ax_right.tick_params(axis="both", colors=style["text_color"])
        ax_right.xaxis.label.set_color(style["text_color"])
        ax_right.yaxis.label.set_color(style["text_color"])


def plot_burst_fraction_ga(
    df_burst, out_png, out_svg, out_ci_csv, title, style,
    burst_threshold_ms,
) -> None:
    ensure_dir(os.path.dirname(out_png))
    df_plot = df_burst.dropna(
        subset=["burst_fraction_baseline", "burst_fraction_stim"]
    ).reset_index(drop=True)
    if df_plot.empty:
        print("  ! plot_burst_fraction_ga: no valid rows, skipping.")
        return

    df_est = pd.DataFrame({
        "id": list(range(len(df_plot))) * 2,
        "condition": ["Baseline"] * len(df_plot) + ["Stimulation"] * len(df_plot),
        "burst_fraction": (
            list(df_plot["burst_fraction_baseline"].values)
            + list(df_plot["burst_fraction_stim"].values)
        ),
    })
    dabest_data = dabest.load(
        df_est, idx=("Baseline", "Stimulation"),
        paired="sequential", id_col="id",
        x="condition", y="burst_fraction",
    )
    mean_diff = dabest_data.mean_diff

    # ---------------------------
    # Export CI statistics
    # ---------------------------
    ci_row = extract_dabest_ci(
        mean_diff_obj=mean_diff,
        title=title,
        n_animals=len(df_plot),
        mean_base=float(df_plot["burst_fraction_baseline"].mean()),
        mean_stim=float(df_plot["burst_fraction_stim"].mean()),
    )
    ensure_dir(os.path.dirname(out_ci_csv))
    pd.DataFrame([ci_row]).to_csv(out_ci_csv, index=False)
    print(f"\u2713 Saved burst fraction CI: {os.path.basename(out_ci_csv)}")

    # ---------------------------
    # Plot
    # ---------------------------
    est_plot = mean_diff.plot(
        custom_palette={"Baseline": style["baseline_color"],
                        "Stimulation":     style["stim_color"]},
        show_pairs=True,
    )

    # Note: post-hoc y-limit adjustment on the contrast axis causes
    # axis detachment in this dabest version and is therefore not applied.
    # dabest scales the contrast axis automatically.

    fig     = est_plot.figure
    ax_left = fig.axes[0]

    # ── Font size ──────────────────────────────────────────────────────────
    for ax in fig.axes:
        ax.tick_params(labelsize=5)
        ax.xaxis.label.set_fontsize(5)
        ax.yaxis.label.set_fontsize(5)
        for txt in ax.texts:
            txt.set_fontsize(5)

    # ── Mean ± SEM overlay on the raw data axis (left panel) ───────────────
    for x_pos, col, face_color in [
        (0, "burst_fraction_baseline", style["baseline_color"]),
        (1, "burst_fraction_stim", style["stim_color"]),
    ]:
        vals = df_plot[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            ax_left.errorbar(
                x=x_pos, y=np.mean(vals),
                yerr=np.std(vals, ddof=1) / np.sqrt(len(vals)),
                fmt="o", color=style["text_color"],
                markerfacecolor=face_color, markeredgecolor=style["text_color"],
                markersize=5, elinewidth=1.0, capsize=2, zorder=6,
            )
    # ── Tool color on contrast axis (right panel) ─────────────────────────
    # Only the bootstrap distribution (collections) is recoloured.
    # No position, spine, or limit changes are made to avoid axis detachment.
    for coll in fig.axes[1].collections:
        try:
            coll.set_facecolor(style["stim_color"])
            coll.set_edgecolor(style["stim_color"])
        except Exception:
            pass
    for line in fig.axes[1].lines:
        try:
            line.set_color(style["stim_color"])
        except Exception:
            pass

    fig.suptitle(title, color=style["text_color"], fontsize=5, y=1.01)
    apply_dabest_style_burst(fig, style)
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
    print("postprocess_spikes_isi_mainfigure.py — path configuration")
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

    print(f"Postprocessing {len(manifests)} folder(s) for main-figure ISI plots...")

    window_suffix = (
        f"adj_base{int(ADJUSTED_BASELINE_S)}s_stim{int(ADJUSTED_STIM_S)}s"
        if USE_ADJUSTED_WINDOWS else "full"
    )

    grouped_hists: Dict[Tuple[str, str], Dict[str, List]] = {}
    grouped_burst: Dict[Tuple[str, str], List[Dict]]      = {}
    overview_data: Dict[Tuple[str, str], List[Dict]]      = {}

    bin_edges = np.arange(0.0, FULL_ISI_MAX_S + BIN_S, BIN_S)

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
                burst_base_vals:    List[float]       = []
                burst_stim_vals:    List[float]       = []
                rv_base_channels:   List[float]       = []
                rv_stim_channels:   List[float]       = []

                for det_path in detected_files:
                    with open(det_path, "rb") as f:
                        det = pickle.load(f)

                    spike_times_s = np.asarray(
                        det.get("spike_times_s", []), dtype=float
                    )
                    if spike_times_s.size < 2:
                        continue

                    spikes_base = restrict_spikes_to_windows(spike_times_s, baseline_windows)
                    spikes_stim = restrict_spikes_to_windows(spike_times_s, stim_windows)

                    isi_base = compute_isi(spikes_base)
                    isi_stim = compute_isi(spikes_stim)

                    if isi_base.size == 0 or isi_stim.size == 0:
                        continue

                    burst_base_vals.append(burst_fraction_val(isi_base, BURST_THRESHOLD_MS))
                    burst_stim_vals.append(burst_fraction_val(isi_stim, BURST_THRESHOLD_MS))

                    rv_base_channels.append(
                        refractory_violations_fraction(spikes_base, refr_ms=1.0)
                    )
                    rv_stim_channels.append(
                        refractory_violations_fraction(spikes_stim, refr_ms=1.0)
                    )

                    h_base = normalize_histogram(isi_base, bin_edges, mode=NORMALIZE_MODE)
                    h_stim = normalize_histogram(isi_stim, bin_edges, mode=NORMALIZE_MODE)
                    channel_hists_base.append(h_base)
                    channel_hists_stim.append(h_stim)

                if len(channel_hists_base) < MIN_CHANNELS_PER_ANIMAL:
                    print(f" - Not enough channels for {folder}/{region_name}/{kind}")
                    continue

                animal_hist_base = np.mean(np.vstack(channel_hists_base), axis=0)
                animal_hist_stim = np.mean(np.vstack(channel_hists_stim), axis=0)

                key = (region_name, kind)

                grouped_hists.setdefault(key, {"baseline": [], "stim": [], "animals": []})
                grouped_hists[key]["baseline"].append(animal_hist_base)
                grouped_hists[key]["stim"].append(animal_hist_stim)
                grouped_hists[key]["animals"].append(folder)

                if burst_base_vals and burst_stim_vals:
                    grouped_burst.setdefault(key, []).append({
                        "folder":                  folder,
                        "burst_fraction_baseline": float(np.nanmean(burst_base_vals)),
                        "burst_fraction_stim":     float(np.nanmean(burst_stim_vals)),
                    })

                overview_data.setdefault(key, []).append({
                    "folder":           folder,
                    "hists_base":       channel_hists_base,
                    "hists_stim":       channel_hists_stim,
                    "rv_base_channels": rv_base_channels,
                    "rv_stim_channels": rv_stim_channels,
                })

                print(
                    f" - Added {folder}/{region_name}/{kind} "
                    f"(n_channels={len(channel_hists_base)})"
                )

    # -----------------------------
    # Export
    # -----------------------------
    out_root = os.path.join(
        export_path_base, relative_data_path,
        "Postprocessing", "Spike_ISI_MainFigure",
    )
    ensure_dir(out_root)

    for (region_name, kind), d in grouped_hists.items():
        if not d["baseline"]:
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
            os.path.join(region_kind_root, "group_isi_distribution.csv"),
            index=False,
        )

        # 1) Group ISI zoom plot
        plot_group_isi_zoom(
            bin_edges=bin_edges,
            mean_base=mean_base, sem_base=sem_base,
            mean_stim=mean_stim, sem_stim=sem_stim,
            out_png=os.path.join(
                region_kind_root,
                f"ISI_zoom_{region_name}_{kind}_{window_suffix}.png",
            ),
            out_svg=os.path.join(
                region_kind_root,
                f"ISI_zoom_{region_name}_{kind}_{window_suffix}.svg",
            ),
            title=f"ISI 0\u2013{int(MAX_ISI_S * 1000)} ms | {region_name} | {kind} | {window_suffix}",
            style=plot_style,
            normalize_mode=NORMALIZE_MODE,
            n_animals=n_animals,
        )
        print(f"\u2713 Saved group ISI plot: {region_name}/{kind}/{window_suffix} (n={n_animals})")

        # 2) Per-animal overview plot
        animal_list = overview_data.get((region_name, kind), [])
        if animal_list:
            plot_per_animal_overview(
                animal_data=animal_list,
                out_png=os.path.join(
                    region_kind_root,
                    f"PerAnimal_ISI_{region_name}_{kind}_{window_suffix}.png",
                ),
                out_svg=os.path.join(
                    region_kind_root,
                    f"PerAnimal_ISI_{region_name}_{kind}_{window_suffix}.svg",
                ),
                region_name=region_name, kind=kind,
                style=plot_style, bin_edges=bin_edges,
                window_suffix=window_suffix,
            )
            print(
                f"\u2713 Saved per-animal overview: {region_name}/{kind}/{window_suffix} "
                f"(n={len(animal_list)} animals)"
            )

        # 3) Burst fraction GA plot + CI CSV
        burst_rows = grouped_burst.get((region_name, kind), [])
        if burst_rows:
            df_burst = pd.DataFrame(burst_rows)
            df_burst.to_csv(
                os.path.join(region_kind_root, "burst_fraction_per_animal.csv"),
                index=False,
            )
            plot_burst_fraction_ga(
                df_burst=df_burst,
                out_png=os.path.join(
                    region_kind_root,
                    f"BurstFraction_{region_name}_{kind}_{window_suffix}.png",
                ),
                out_svg=os.path.join(
                    region_kind_root,
                    f"BurstFraction_{region_name}_{kind}_{window_suffix}.svg",
                ),
                out_ci_csv=os.path.join(
                    region_kind_root,
                    f"BurstFractionStats_{region_name}_{kind}_{window_suffix}.csv",
                ),
                title=f"Burst fraction | {region_name} | {kind} | {window_suffix}",
                style=plot_style,
                burst_threshold_ms=BURST_THRESHOLD_MS,
            )
            print(f"\u2713 Saved burst fraction plot: {region_name}/{kind}/{window_suffix}")

    dt = time.perf_counter() - t0
    print(f"\nDone. Total processing time: {dt:.1f}s")


if __name__ == "__main__":
    main()