# -*- coding: utf-8 -*-
"""
postprocess_spikes_mua_firing_rate.py

Postprocessing script for channel-level Gardner-Altman plots based on the
summary output of process_spikes_psth.py.

Creates paired mean-difference plots for:
- Baseline vs Early stimulation
- Baseline vs Late stimulation
- Baseline vs Full stimulation

Also exports responder classifications and responder bar plots for each
comparison separately.

CI export
---------
For each GA plot, a companion CSV is saved with the dabest estimation
statistics, including:
  - mean_diff           : mean difference (stimulation - baseline)
  - ci                  : confidence interval level (default 95%)
  - bca_low, bca_high   : bias-corrected and accelerated bootstrap CI
  - pct_low, pct_high   : percentile bootstrap CI
  - n_baseline, n_test  : number of channels per condition

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

Note: adjusted window mode requires the Pynapse call log to be available
in the processed folder, as window boundaries are re-derived at runtime
from the epoch timestamps rather than the pre-computed summary CSVs.
When USE_ADJUSTED_WINDOWS = False, the script reads directly from the
summary CSVs produced by process_spikes_psth.py (faster).
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import dabest
import matplotlib.pyplot as plt

from Functions_processing_spikes import (
    ensure_dir,
    load_json,
    list_manifests,
    load_epochs_from_pynapse_csv,
    epochs_to_windows,
    restrict_spikes_to_windows,
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
# relative_data_path = r"Jills_paper\ChR2_cohort\OFT_experimental"
# relative_data_path = r"Jills_paper\vSWO_cohort\OFT_experimental"
_DEFAULT_RELATIVE_DATA_PATH = r"Jills_paper\vLWO_cohort\OFT_experimental"
# Alias for module-level calls (e.g. infer_tool_name).
relative_data_path = _DEFAULT_RELATIVE_DATA_PATH

only_folder: Optional[str] = None
only_regions: Optional[List[str]] = ["CA1_L"]

PROCESS_UNREFERENCED = False
PROCESS_LMR = True

MIN_ABS_CHANGE_HZ = 1.0
MIN_REL_CHANGE    = 0.20

BASELINE_LABEL = "In baseline state"
STIM_LABEL     = "In stimulation state"
IGNORE_LABEL   = "In start delay"

# -----------------------------
# Adjusted window settings
# -----------------------------
# Set USE_ADJUSTED_WINDOWS = True to use a common window length across all
# animals regardless of their original recording protocol.
# Set to False to use the full epoch durations (default).
USE_ADJUSTED_WINDOWS = False    # True = adjusted | False = full epochs

ADJUSTED_BASELINE_S  = 120.0  # last N seconds before stimulation onset
ADJUSTED_STIM_S      = 300.0  # first N seconds after stimulation onset

# Plot settings
PLOT_THEME          = "light"
STIM_COLOR_OVERRIDE = None

FIG_HEIGHT_MM = 40
FIG_ASPECT    = 1.5

tool_name  = infer_tool_name(relative_data_path)
stim_color = STIM_COLOR_OVERRIDE if STIM_COLOR_OVERRIDE is not None \
    else get_tool_color(tool_name)
plot_style = get_plot_style(theme=PLOT_THEME, stim_color=stim_color)

set_global_plot_style(
    theme=PLOT_THEME,
    font_family="Arial",
    base_font_size=5,
    axes_title_size=7,
    axes_label_size=5,
    tick_label_size=5,
    legend_font_size=5,
    axes_linewidth=1.2,
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
# Window adjustment helpers
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


def rate_in_windows(
    spk: np.ndarray,
    windows: List[Tuple[float, float]],
) -> float:
    if not windows:
        return float("nan")
    total_t = sum(max(0.0, b - a) for a, b in windows)
    if total_t <= 0:
        return float("nan")
    return float(restrict_spikes_to_windows(spk, windows).size / total_t)


def load_detected_pkls(detected_kind_root: str) -> List[str]:
    if not os.path.isdir(detected_kind_root):
        return []
    files = sorted(f for f in os.listdir(detected_kind_root) if f.endswith(".pkl"))
    return [os.path.join(detected_kind_root, f) for f in files]


# -----------------------------
# Response classification
# -----------------------------
def classify_channel_response(
    baseline_rate_hz: float,
    test_rate_hz: float,
    min_abs_change_hz: float = 1.0,
    min_rel_change: float = 0.20,
) -> Dict:
    if not np.isfinite(baseline_rate_hz) or not np.isfinite(test_rate_hz):
        return {"delta_rate_hz": np.nan, "delta_rel": np.nan,
                "response_class": "invalid"}

    delta     = test_rate_hz - baseline_rate_hz
    delta_rel = delta / baseline_rate_hz if baseline_rate_hz > 0 else np.nan

    if baseline_rate_hz > 0:
        if delta >= min_abs_change_hz and delta_rel >= min_rel_change:
            cls = "increase"
        elif delta <= -min_abs_change_hz and delta_rel <= -min_rel_change:
            cls = "decrease"
        else:
            cls = "no_change"
    else:
        cls = "increase" if delta >= min_abs_change_hz \
            else "decrease" if delta <= -min_abs_change_hz \
            else "no_change"

    return {
        "delta_rate_hz":   float(delta),
        "delta_rel":       float(delta_rel) if np.isfinite(delta_rel) else np.nan,
        "response_class":  cls,
    }


# -----------------------------
# Dabest styling
# -----------------------------
def apply_dabest_style(fig, style: dict) -> None:
    fig.patch.set_facecolor(style["figure_facecolor"])
    for i, ax in enumerate(fig.axes):
        # Only style the left (raw data) panel — i == 0.
        # Any post-hoc changes to the right (difference) panel cause it to
        # float or detach, so we leave it entirely to dabest defaults.
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


# -----------------------------
# GA plot + CI export
# -----------------------------
def make_ga_plot(
    df: pd.DataFrame,
    baseline_col: str,
    test_col: str,
    out_png: str,
    out_svg: str,
    out_ci_csv: str,
    title: str,
    style: dict,
) -> None:
    df_plot = df.dropna(subset=[baseline_col, test_col]).reset_index(drop=True)
    if df_plot.empty:
        print(f"  ! make_ga_plot: no valid rows for {title}, skipping.")
        return

    df_est = pd.DataFrame({
        "id":        list(range(len(df_plot))) * 2,
        "condition": ["Baseline"] * len(df_plot) + ["Test"] * len(df_plot),
        "spike_rate": (
            list(df_plot[baseline_col].values)
            + list(df_plot[test_col].values)
        ),
    })

    dabest_data = dabest.load(
        df_est, idx=("Baseline", "Test"),
        paired="sequential", id_col="id",
        x="condition", y="spike_rate",
    )
    mean_diff = dabest_data.mean_diff

    # ---------------------------
    # Export CI statistics
    # ---------------------------
    try:
        stats = mean_diff.statistical_tests

        # dabest may name columns differently across versions — handle both
        ci_row = {
            "title":        title,
            "n_baseline":   int(df_plot[baseline_col].notna().sum()),
            "n_test":       int(df_plot[test_col].notna().sum()),
            "mean_baseline_hz": float(df_plot[baseline_col].mean()),
            "mean_test_hz":     float(df_plot[test_col].mean()),
            "mean_diff_hz": float(stats["difference"].iloc[0])
                            if "difference" in stats.columns
                            else float(np.nan),
            "ci":           float(stats["ci"].iloc[0])
                            if "ci" in stats.columns
                            else float(np.nan),
            "bca_low":      float(stats["bca_low"].iloc[0])
                            if "bca_low" in stats.columns
                            else float(np.nan),
            "bca_high":     float(stats["bca_high"].iloc[0])
                            if "bca_high" in stats.columns
                            else float(np.nan),
            "pct_low":      float(stats["pct_low"].iloc[0])
                            if "pct_low" in stats.columns
                            else float(np.nan),
            "pct_high":     float(stats["pct_high"].iloc[0])
                            if "pct_high" in stats.columns
                            else float(np.nan),
        }

        # Fallback: some dabest versions use different column names
        if np.isnan(ci_row["bca_low"]):
            for col in stats.columns:
                if "low" in col.lower() and "bca" in col.lower():
                    ci_row["bca_low"] = float(stats[col].iloc[0])
                if "high" in col.lower() and "bca" in col.lower():
                    ci_row["bca_high"] = float(stats[col].iloc[0])

        ensure_dir(os.path.dirname(out_ci_csv))
        pd.DataFrame([ci_row]).to_csv(out_ci_csv, index=False)
        print(
            f"  CI: mean diff={ci_row['mean_diff_hz']:.2f} Hz | "
            f"95% BCa [{ci_row['bca_low']:.2f}, {ci_row['bca_high']:.2f}]"
        )
        print(f"✓ Saved CI: {os.path.basename(out_ci_csv)}")

    except Exception as e:
        print(f"  ! Could not extract CI statistics: {e}")
        print(f"  Available columns: {list(mean_diff.statistical_tests.columns)}")

    # ---------------------------
    # Plot
    # ---------------------------
    est_plot = mean_diff.plot(
        custom_palette={"Baseline": style["baseline_color"],
                        "Test":     style["stim_color"]},
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
        (0, baseline_col, style["baseline_color"]),
        (1, test_col,     style["stim_color"]),
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
    apply_dabest_style(fig, style)
    ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, dpi=600, bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                transparent=(style["theme"] == "dark"))
    plt.close(fig)
    print(f"✓ Saved: {os.path.basename(out_png)}")


# -----------------------------
# Responder outputs
# -----------------------------
def save_responder_outputs(
    df: pd.DataFrame,
    response_col: str,
    region_kind_root: str,
    suffix: str,
    region_name: str,
    kind: str,
    style: dict,
) -> None:
    counts = (
        df[response_col]
        .value_counts(dropna=False)
        .rename_axis("ResponseClass")
        .reset_index(name="N_Channels")
    )
    counts["Fraction"] = counts["N_Channels"] / counts["N_Channels"].sum()
    counts.to_csv(
        os.path.join(region_kind_root, f"response_class_counts_{suffix}.csv"),
        index=False,
    )
    (
        df.groupby(["folder", response_col])
        .size().reset_index(name="N_Channels")
        .rename(columns={response_col: "ResponseClass"})
    ).to_csv(
        os.path.join(region_kind_root,
                     f"response_class_counts_by_subject_{suffix}.csv"),
        index=False,
    )

    order = ["increase", "decrease", "no_change"]
    counts_plot = (
        counts.set_index("ResponseClass").reindex(order).fillna(0).reset_index()
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    apply_figure_style(fig, style); apply_axes_style(ax, style)
    ax.bar(counts_plot["ResponseClass"], counts_plot["N_Channels"],
           color=[style["increase_color"], style["decrease_color"],
                  style["neutral_color"]],
           edgecolor=style["spine_color"])
    ax.set_title(f"Responder classes ({suffix}) | {region_name} | {kind}")
    ax.set_ylabel("N channels"); ax.set_xlabel("Response class")
    ax.grid(True, linestyle="--", alpha=0.3, color=style["grid_color"], axis="y")
    plt.tight_layout()
    for ext, fmt in [(".png", None), (".svg", "svg")]:
        path = os.path.join(
            region_kind_root,
            f"ResponderBarPlot_{region_name}_{kind}_{suffix}{ext}",
        )
        kwargs = {"dpi": 600, "bbox_inches": "tight",
                  "transparent": (style["theme"] == "dark")}
        if fmt:
            kwargs["format"] = fmt
        fig.savefig(path, **kwargs)
    plt.close(fig)
    print(f"✓ Saved responder outputs ({suffix})")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    t0 = time.perf_counter()

    print("=" * 60)
    print("postprocess_spikes_mua_firing_rate.py — path configuration")
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

    print(f"Postprocessing {len(manifests)} folder(s) for Gardner-Altman channel plots...")

    all_rows_by_region_kind: Dict[Tuple[str, str], List[pd.DataFrame]] = {}

    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        folder                = manifest["folder"]
        processed_folder_path = manifest["processed_folder_path"]

        print(f"\n==============================")
        print(f"Folder: {folder}")

        regions = manifest.get("regions", [])
        if only_regions is not None:
            regions = [r for r in regions if r.get("region_name") in only_regions]

        # ------------------------------------------------------------------
        # Mode A: adjusted windows — recompute firing rates from detected pkls
        # ------------------------------------------------------------------
        if USE_ADJUSTED_WINDOWS:
            pynapse_csv = os.path.join(
                processed_folder_path, "metadata", "Pynapse_call_log.csv"
            )
            if not os.path.isfile(pynapse_csv):
                print(f" - Missing Pynapse_call_log.csv, skipping.")
                continue

            epochs           = load_epochs_from_pynapse_csv(pynapse_csv, ignore_label=IGNORE_LABEL)
            baseline_windows = epochs_to_windows(epochs, BASELINE_LABEL)
            stim_windows     = epochs_to_windows(epochs, STIM_LABEL)

            if not baseline_windows or not stim_windows:
                print(" - Missing baseline or stim windows, skipping.")
                continue

            adj_base = adjust_baseline_window(baseline_windows, ADJUSTED_BASELINE_S)
            adj_stim = adjust_stim_window(stim_windows, ADJUSTED_STIM_S)

            base_dur = sum(b - a for a, b in adj_base)
            stim_dur = sum(b - a for a, b in adj_stim)
            print(f" - Adjusted windows: baseline={base_dur:.1f} s | stim={stim_dur:.1f} s")

            for region in regions:
                region_name   = region["region_name"]
                detected_root = os.path.join(
                    processed_folder_path, region_name, "Spikes", "detected"
                )

                for kind in ["unreferenced", "lmr"]:
                    if kind == "unreferenced" and not PROCESS_UNREFERENCED:
                        continue
                    if kind == "lmr" and not PROCESS_LMR:
                        continue

                    det_files = load_detected_pkls(os.path.join(detected_root, kind))
                    if not det_files:
                        print(f" - No detected pkls for {kind}")
                        continue

                    for det_path in det_files:
                        with open(det_path, "rb") as f:
                            det = pickle.load(f)
                        spike_times_s = np.asarray(
                            det.get("spike_times_s", []), dtype=float
                        )
                        if spike_times_s.size < 2:
                            continue

                        fr_base = rate_in_windows(spike_times_s, adj_base)
                        fr_stim = rate_in_windows(spike_times_s, adj_stim)

                        if not np.isfinite(fr_base) or not np.isfinite(fr_stim):
                            continue

                        key = (region_name, kind)
                        all_rows_by_region_kind.setdefault(key, []).append(
                            pd.DataFrame([{
                                "folder":              folder,
                                "region":              region_name,
                                "signal_kind":         kind,
                                "channel":             det.get("channel", "?"),
                                "baseline_rate_hz":    fr_base,
                                "full_stim_rate_hz":   fr_stim,
                                "early_stim_rate_hz":  fr_stim,
                                "late_stim_rate_hz":   fr_stim,
                            }])
                        )

        # ------------------------------------------------------------------
        # Mode B: full epochs — read directly from summary CSVs
        # ------------------------------------------------------------------
        else:
            for region in regions:
                region_name = region["region_name"]
                qc_root     = os.path.join(
                    processed_folder_path, region_name, "QC", "Spikes_PSTH", "full"
                )
                if not os.path.isdir(qc_root):
                    # Fallback: try old path without window_suffix subfolder
                    qc_root_legacy = os.path.join(
                        processed_folder_path, region_name, "QC", "Spikes_PSTH"
                    )
                    if os.path.isdir(qc_root_legacy):
                        qc_root = qc_root_legacy
                        print(f" - Using legacy QC path (no window_suffix): {qc_root}")
                    else:
                        print(f" - Missing Spikes_PSTH folder: {qc_root}")
                        continue

                for kind in ["unreferenced", "lmr"]:
                    if kind == "unreferenced" and not PROCESS_UNREFERENCED:
                        continue
                    if kind == "lmr" and not PROCESS_LMR:
                        continue

                    summary_csv = os.path.join(qc_root, kind, "summary.csv")
                    if not os.path.isfile(summary_csv):
                        print(f" - Missing summary CSV: {summary_csv}")
                        continue

                    df = pd.read_csv(summary_csv)
                    if df.empty:
                        continue

                    df["folder"]      = folder
                    df["region"]      = region_name
                    df["signal_kind"] = kind

                    all_rows_by_region_kind.setdefault(
                        (region_name, kind), []
                    ).append(df)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    window_suffix = (
        f"adj_base{int(ADJUSTED_BASELINE_S)}s_stim{int(ADJUSTED_STIM_S)}s"
        if USE_ADJUSTED_WINDOWS else "full"
    )
    out_root = os.path.join(
        export_path_base, relative_data_path,
        "Postprocessing", "Spike_Gardner_Altmann_Channels",
    )
    ensure_dir(out_root)

    for (region_name, kind), df_list in all_rows_by_region_kind.items():
        if not df_list:
            continue

        df               = pd.concat(df_list, ignore_index=True)
        region_kind_root = os.path.join(out_root, region_name, kind, window_suffix)
        ensure_dir(region_kind_root)

        for suffix, baseline_col, test_col, class_col in [
            ("early", "baseline_rate_hz", "early_stim_rate_hz", "response_class_early"),
            ("late",  "baseline_rate_hz", "late_stim_rate_hz",  "response_class_late"),
            ("full",  "baseline_rate_hz", "full_stim_rate_hz",  "response_class_full"),
        ]:
            resp = df.apply(
                lambda r, bc=baseline_col, tc=test_col: classify_channel_response(
                    r[bc], r[tc],
                    min_abs_change_hz=MIN_ABS_CHANGE_HZ,
                    min_rel_change=MIN_REL_CHANGE,
                ),
                axis=1,
            )
            df[f"delta_rate_{suffix}_hz"] = [d["delta_rate_hz"] for d in resp]
            df[f"delta_rel_{suffix}"]     = [d["delta_rel"]     for d in resp]
            df[class_col]                 = [d["response_class"] for d in resp]

        df.to_csv(os.path.join(region_kind_root, "channel_rates.csv"), index=False)

        for suffix, class_col in [
            ("early", "response_class_early"),
            ("late",  "response_class_late"),
            ("full",  "response_class_full"),
        ]:
            save_responder_outputs(
                df=df, response_col=class_col,
                region_kind_root=region_kind_root,
                suffix=suffix, region_name=region_name,
                kind=kind, style=plot_style,
            )

        for suffix, baseline_col, test_col in [
            ("early", "baseline_rate_hz", "early_stim_rate_hz"),
            ("late",  "baseline_rate_hz", "late_stim_rate_hz"),
            ("full",  "baseline_rate_hz", "full_stim_rate_hz"),
        ]:
            if USE_ADJUSTED_WINDOWS and suffix != "full":
                continue

            make_ga_plot(
                df=df,
                baseline_col=baseline_col,
                test_col=test_col,
                out_png=os.path.join(
                    region_kind_root,
                    f"EstimationPlot_{region_name}_{kind}_{suffix}_{window_suffix}.png",
                ),
                out_svg=os.path.join(
                    region_kind_root,
                    f"EstimationPlot_{region_name}_{kind}_{suffix}_{window_suffix}.svg",
                ),
                out_ci_csv=os.path.join(
                    region_kind_root,
                    f"EstimationStats_{region_name}_{kind}_{suffix}_{window_suffix}.csv",
                ),
                title=(
                    f"Baseline ({ADJUSTED_BASELINE_S:.0f} s) vs "
                    f"Stimulation ({ADJUSTED_STIM_S:.0f} s) | "
                    f"{region_name} | {kind}"
                    if USE_ADJUSTED_WINDOWS else
                    f"Baseline vs {suffix.capitalize()} stimulation | "
                    f"{region_name} | {kind}"
                ),
                style=plot_style,
            )

        n_animals = df["folder"].nunique()
        pd.DataFrame([{
            "Region":                   region_name,
            "SignalKind":               kind,
            "WindowMode":               window_suffix,
            "N_Animals":                n_animals,
            "N_Channels":               len(df),
            "Baseline_Rate_Hz_Mean":    df["baseline_rate_hz"].mean(),
            "Full_Stim_Rate_Hz_Mean":   df["full_stim_rate_hz"].mean(),
            "Delta_Rate_Full_Hz_Mean":  df["delta_rate_full_hz"].mean(),
        }]).to_csv(
            os.path.join(region_kind_root, "average_rates_summary.csv"),
            index=False,
        )
        print(f"✓ Done: {region_name}/{kind}/{window_suffix} (n={n_animals} animals)")

    dt = time.perf_counter() - t0
    print(f"\nScript finished after {dt:.1f} seconds")


if __name__ == "__main__":
    main()