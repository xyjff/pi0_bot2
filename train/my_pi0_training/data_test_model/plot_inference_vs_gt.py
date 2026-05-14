#!/usr/bin/env python3
"""Plot offline inference output against ground truth actions.

cd /share/0xyj/model3_openpi0.5/my_pi0_training
source /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/activate
python data_test_model/plot_inference_vs_gt.py \
    --input-json data_test_model/results/ep1_step20000.json \
    --horizon-step 0
    
cd /share/0xyj/model3_openpi0.5/my_pi0_training
source /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/activate
python data_test_model/plot_inference_vs_gt.py \
    --input-json data_test_model/results/ep1_step26000.json \
    --horizon-step 0

cd /share/0xyj/model3_openpi0.5/my_pi0_training
source /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/activate
python data_test_model/plot_inference_vs_gt.py \
    --input-json data_test_model/results/ep1_step32000.json \
    --horizon-step 0

cd /share/0xyj/model3_openpi0.5/my_pi0_training
source /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/activate
python data_test_model/plot_inference_vs_gt.py \
    --input-json data_test_model/results/ep1_step38000.json \
    --horizon-step 0

cd /share/0xyj/model3_openpi0.5/my_pi0_training
source /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/activate
python data_test_model/plot_inference_vs_gt.py \
    --input-json data_test_model/results/ep2_step20000.json \
    --horizon-step 0
    
cd /share/0xyj/model3_openpi0.5/my_pi0_training
source /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/activate
python data_test_model/plot_inference_vs_gt.py \
    --input-json data_test_model/results/ep2_step26000.json \
    --horizon-step 0

cd /share/0xyj/model3_openpi0.5/my_pi0_training
source /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/activate
python data_test_model/plot_inference_vs_gt.py \
    --input-json data_test_model/results/ep2_step32000.json \
    --horizon-step 0

cd /share/0xyj/model3_openpi0.5/my_pi0_training
source /share/0xyj/model3_openpi0.5/openpi-main/.venv/bin/activate
python data_test_model/plot_inference_vs_gt.py \
    --input-json data_test_model/results/ep2_step38000.json \
    --horizon-step 0






"""

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np


JOINT_NAMES_8D = [
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_dexterous_hand",
]


def _to_display_units(values: np.ndarray) -> np.ndarray:
    """Convert joints from radians to degrees while keeping gripper raw."""
    out = np.asarray(values, dtype=np.float32).copy()
    out[..., :7] = out[..., :7] * 180.0 / math.pi
    return out


def _load_series(records: list[dict], horizon_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.array([r["frame_index"] for r in records], dtype=np.int32)

    preds = []
    targets = []
    values_are_display_units = False
    for r in records:
        if "model_output_horizon_display" in r and "action_target_horizon_display" in r:
            values_are_display_units = True
            valid_horizon = int(r.get("valid_horizon", len(r["model_output_horizon_display"])))
            if horizon_step >= valid_horizon:
                preds.append([np.nan] * 8)
                targets.append([np.nan] * 8)
            else:
                preds.append(r["model_output_horizon_display"][horizon_step])
                targets.append(r["action_target_horizon_display"][horizon_step])
        elif "model_output_horizon_rad" in r and "action_target_horizon_rad" in r:
            valid_horizon = int(r.get("valid_horizon", len(r["model_output_horizon_rad"])))
            if horizon_step >= valid_horizon:
                preds.append([np.nan] * 8)
                targets.append([np.nan] * 8)
            else:
                preds.append(r["model_output_horizon_rad"][horizon_step])
                targets.append(r["action_target_horizon_rad"][horizon_step])
        else:
            if horizon_step != 0:
                raise ValueError("This result file only contains horizon step 0 fields.")
            if "model_output_display" in r and "action_target_display" in r:
                values_are_display_units = True
                preds.append(r["model_output_display"])
                targets.append(r["action_target_display"])
            else:
                preds.append(r["model_output_rad"])
                targets.append(r["action_target_rad"])

    preds = np.asarray(preds, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if not values_are_display_units:
        preds = _to_display_units(preds)
        targets = _to_display_units(targets)
    return frames, preds, targets


def _points_to_svg(points: np.ndarray) -> str:
    valid = ~np.isnan(points).any(axis=1)
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points[valid])


def _load_rolling_series(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset_state, rolling_state, model_output, action_target for rolling inference."""
    frames = np.array([r["frame_index"] for r in records], dtype=np.int32)

    ds_list, rs_list, mo_list, at_list = [], [], [], []
    for r in records:
        if "rolling_state_display" in r:
            ds_list.append(r["dataset_state_display"])
            rs_list.append(r["rolling_state_display"])
            mo_list.append(r["model_output_display"])
            at_list.append(r["action_target_display"])
        elif "rolling_state_rad" in r:
            ds = np.asarray(r["dataset_state_rad"], dtype=np.float32).copy()
            ds[:7] = ds[:7] * 180.0 / math.pi
            rs = np.asarray(r["rolling_state_rad"], dtype=np.float32).copy()
            rs[:7] = rs[:7] * 180.0 / math.pi
            mo = np.asarray(r["model_output_rad"], dtype=np.float32).copy()
            mo[:7] = mo[:7] * 180.0 / math.pi
            at = np.asarray(r["action_target_rad"], dtype=np.float32).copy()
            at[:7] = at[:7] * 180.0 / math.pi
            ds_list.append(ds)
            rs_list.append(rs)
            mo_list.append(mo)
            at_list.append(at)
        else:
            raise ValueError("Result file does not contain rolling inference fields.")

    ds = np.asarray(ds_list, dtype=np.float32)
    rs = np.asarray(rs_list, dtype=np.float32)
    mo = np.asarray(mo_list, dtype=np.float32)
    at = np.asarray(at_list, dtype=np.float32)
    return frames, ds, rs, mo, at


def _map_xy(
    frames: np.ndarray,
    values: np.ndarray,
    x0: float,
    y0: float,
    width: float,
    height: float,
    ymin: float,
    ymax: float,
) -> np.ndarray:
    xmin = float(np.nanmin(frames))
    xmax = float(np.nanmax(frames))
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    x = x0 + (frames - xmin) / (xmax - xmin) * width
    y = y0 + height - (values - ymin) / (ymax - ymin) * height
    return np.stack([x, y], axis=1)


def plot_rolling_result(input_json: Path, output_svg: Path) -> None:
    """Plot rolling inference: Dataset State vs Rolling State vs Model Output for each joint."""
    records = json.loads(input_json.read_text())
    if not records:
        raise ValueError(f"No records found in {input_json}")

    frames, ds_display, rs_display, mo_display, at_display = _load_rolling_series(records)

    canvas_w = 1800
    canvas_h = 1300
    margin_l = 80
    margin_t = 95
    panel_gap_x = 70
    panel_gap_y = 70
    panel_w = (canvas_w - 2 * margin_l - panel_gap_x) / 2
    panel_h = (canvas_h - margin_t - 70 - 3 * panel_gap_y) / 4

    stem = input_json.stem

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:24px;font-weight:700}.panel-title{font-size:15px;font-weight:700}.label{font-size:12px;fill:#333}.tick{font-size:11px;fill:#666}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{canvas_w / 2}" y="38" text-anchor="middle" class="title">Rolling Inference: Dataset State vs Model Output - {html.escape(stem)}</text>',
        '<line x1="80" y1="62" x2="110" y2="62" stroke="#2ca02c" stroke-width="4"/>',
        '<text x="118" y="67" class="label">Dataset State (GT)</text>',
        '<line x1="270" y1="62" x2="300" y2="62" stroke="#ff7f0e" stroke-width="4"/>',
        '<text x="308" y="67" class="label">Rolling State (Model Input)</text>',
        '<line x1="500" y1="62" x2="530" y2="62" stroke="#d62728" stroke-width="4"/>',
        '<text x="538" y="67" class="label">Model Output</text>',
    ]

    for i, name in enumerate(JOINT_NAMES_8D):
        row = i // 2
        col = i % 2
        x0 = margin_l + col * (panel_w + panel_gap_x)
        y0 = margin_t + row * (panel_h + panel_gap_y)

        ds = ds_display[:, i]
        rs = rs_display[:, i]
        mo = mo_display[:, i]
        at = at_display[:, i]
        unit = "raw" if i == 7 else "deg"

        mae_vs_gt = np.nanmean(np.abs(mo - at))
        mae_vs_rs = np.nanmean(np.abs(mo - rs))
        drift = np.nanmean(np.abs(ds - rs))

        all_vals = np.concatenate([ds, rs, mo, at])
        ymin = float(np.nanmin(all_vals))
        ymax = float(np.nanmax(all_vals))
        pad = max((ymax - ymin) * 0.08, 1e-3)
        ymin -= pad
        ymax += pad

        ds_pts = _map_xy(frames, ds, x0, y0, panel_w, panel_h, ymin, ymax)
        rs_pts = _map_xy(frames, rs, x0, y0, panel_w, panel_h, ymin, ymax)
        mo_pts = _map_xy(frames, mo, x0, y0, panel_w, panel_h, ymin, ymax)

        parts.append(
            f'<text x="{x0}" y="{y0 - 12}" class="panel-title">'
            f'{html.escape(name)}  MAE={mae_vs_gt:.3f}{unit}  Drift={drift:.3f}{unit}</text>'
        )
        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#bbb"/>')
        for g in range(1, 4):
            gy = y0 + panel_h * g / 4
            parts.append(f'<line x1="{x0}" y1="{gy}" x2="{x0 + panel_w}" y2="{gy}" stroke="#ddd" stroke-width="1"/>')

        parts.append(
            f'<polyline points="{_points_to_svg(ds_pts)}" fill="none" stroke="#2ca02c" stroke-width="2.2"/>'
        )
        parts.append(
            f'<polyline points="{_points_to_svg(rs_pts)}" fill="none" stroke="#ff7f0e" stroke-width="2.0" opacity="0.85"/>'
        )
        parts.append(
            f'<polyline points="{_points_to_svg(mo_pts)}" fill="none" stroke="#d62728" stroke-width="1.9" opacity="0.88"/>'
        )

        parts.append(f'<text x="{x0 - 8}" y="{y0 + 4}" text-anchor="end" class="tick">{ymax:.2f}</text>')
        parts.append(f'<text x="{x0 - 8}" y="{y0 + panel_h}" text-anchor="end" class="tick">{ymin:.2f}</text>')
        parts.append(f'<text x="{x0 + panel_w / 2}" y="{y0 + panel_h + 34}" text-anchor="middle" class="label">frame index</text>')

    parts.append("</svg>")
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(parts))


def plot_result(input_json: Path, output_svg: Path, horizon_step: int) -> None:
    records = json.loads(input_json.read_text())
    if not records:
        raise ValueError(f"No records found in {input_json}")

    frames, preds_display, targets_display = _load_series(records, horizon_step)

    canvas_w = 1800
    canvas_h = 1300
    margin_l = 80
    margin_t = 95
    panel_gap_x = 70
    panel_gap_y = 70
    panel_w = (canvas_w - 2 * margin_l - panel_gap_x) / 2
    panel_h = (canvas_h - margin_t - 70 - 3 * panel_gap_y) / 4

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:24px;font-weight:700}.panel-title{font-size:15px;font-weight:700}.label{font-size:12px;fill:#333}.tick{font-size:11px;fill:#666}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{canvas_w / 2}" y="38" text-anchor="middle" class="title">Model Output vs Ground Truth - {html.escape(input_json.name)} - horizon step {horizon_step}</text>',
        '<line x1="80" y1="62" x2="110" y2="62" stroke="#1f77b4" stroke-width="4"/>',
        '<text x="118" y="67" class="label">Ground truth</text>',
        '<line x1="250" y1="62" x2="280" y2="62" stroke="#d62728" stroke-width="4"/>',
        '<text x="288" y="67" class="label">Model output</text>',
    ]

    for i, name in enumerate(JOINT_NAMES_8D):
        row = i // 2
        col = i % 2
        x0 = margin_l + col * (panel_w + panel_gap_x)
        y0 = margin_t + row * (panel_h + panel_gap_y)

        pred = preds_display[:, i]
        target = targets_display[:, i]
        mae = np.nanmean(np.abs(pred - target))
        unit = "raw" if i == 7 else "deg"

        ymin = float(np.nanmin([np.nanmin(pred), np.nanmin(target)]))
        ymax = float(np.nanmax([np.nanmax(pred), np.nanmax(target)]))
        pad = max((ymax - ymin) * 0.08, 1e-3)
        ymin -= pad
        ymax += pad

        target_points = _map_xy(frames, target, x0, y0, panel_w, panel_h, ymin, ymax)
        pred_points = _map_xy(frames, pred, x0, y0, panel_w, panel_h, ymin, ymax)

        parts.append(f'<text x="{x0}" y="{y0 - 12}" class="panel-title">{html.escape(name)}  MAE={mae:.3f} {unit}</text>')
        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#bbb"/>')
        for g in range(1, 4):
            gy = y0 + panel_h * g / 4
            parts.append(f'<line x1="{x0}" y1="{gy}" x2="{x0 + panel_w}" y2="{gy}" stroke="#ddd" stroke-width="1"/>')
        parts.append(
            f'<polyline points="{_points_to_svg(target_points)}" fill="none" stroke="#1f77b4" stroke-width="2.2"/>'
        )
        parts.append(
            f'<polyline points="{_points_to_svg(pred_points)}" fill="none" stroke="#d62728" stroke-width="1.9" opacity="0.88"/>'
        )
        parts.append(f'<text x="{x0 - 8}" y="{y0 + 4}" text-anchor="end" class="tick">{ymax:.2f}</text>')
        parts.append(f'<text x="{x0 - 8}" y="{y0 + panel_h}" text-anchor="end" class="tick">{ymin:.2f}</text>')
        parts.append(f'<text x="{x0 + panel_w / 2}" y="{y0 + panel_h + 34}" text-anchor="middle" class="label">frame index</text>')

    parts.append("</svg>")
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot inference results from datatest_right_arm_headcam.py.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "ep1_step20000.json",
        help="Path to inference JSON produced by datatest_right_arm_headcam.py",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=None,
        help="Output SVG path.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rolling",
        choices=["rolling", "horizon"],
        help=(
            "'rolling': Dataset State vs Rolling State vs Model Output (for rolling inference). "
            "'horizon': Ground Truth vs Model Output per horizon step (legacy/original)."
        ),
    )
    parser.add_argument(
        "--horizon-step",
        type=int,
        default=0,
        help="Which predicted horizon step to plot in 'horizon' mode. Use 0 for the first action in the chunk.",
    )
    args = parser.parse_args()

    if args.output_svg is None:
        if args.mode == "rolling":
            args.output_svg = args.input_json.with_name(f"{args.input_json.stem}_rolling_compare.svg")
        else:
            args.output_svg = args.input_json.with_name(f"{args.input_json.stem}_h{args.horizon_step}_compare.svg")

    if args.mode == "rolling":
        plot_rolling_result(args.input_json, args.output_svg)
    else:
        plot_result(args.input_json, args.output_svg, args.horizon_step)

    print(f"Saved plot: {args.output_svg}")


if __name__ == "__main__":
    main()
