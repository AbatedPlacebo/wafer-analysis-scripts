#!/usr/bin/env python3
"""
Wafer edge video analysis with automatic Flat/Jump block removal.
Outputs: video_{phase}_t{trial}.json, edge_{phase}_t{trial}.csv, etc.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import os
from datetime import datetime

# Актуализированное разрешение на основе калибровки
UM_PER_PIXEL_X_DEFAULT = 2.707
UM_PER_PIXEL_Y_DEFAULT = 2.707

def parse_args():
    parser = argparse.ArgumentParser(
        description="Wafer edge TIR analysis from video (pre/post) with stability filtering")
    parser.add_argument("--video", required=True)
    parser.add_argument("--trial_id", type=int, default=None,
                        help="Link to controller trial number")
    parser.add_argument("--phase", choices=["pre", "post"], required=True,
                        help="Measurement phase: pre or post alignment")
    parser.add_argument("--odir", default=None,
                        help="Output directory (auto-names files by trial/phase)")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_video", default=None)
    parser.add_argument("--out_plot", default=None)
    parser.add_argument("--out_summary", default=None)
    parser.add_argument("--out_json", default=None,
                        help="Standardized JSON (auto if --odir)")
    parser.add_argument("--um_per_px_x", type=float, default=UM_PER_PIXEL_X_DEFAULT)
    parser.add_argument("--um_per_px_y", type=float, default=UM_PER_PIXEL_Y_DEFAULT)
    parser.add_argument("--side", choices=["left", "right"], default="right")
    parser.add_argument("--roi", type=int, nargs=4, default=None,
                        metavar=("X", "Y", "W", "H"))
    parser.add_argument("--select_roi", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--smooth", type=int, default=11)
    parser.add_argument("--n_measurement_points", type=int, default=5)
    parser.add_argument("--min_inliers", type=int, default=80)
    parser.add_argument("--min_phase_response", type=float, default=0.02)
    parser.add_argument("--trend_window", type=int, default=31)
    parser.add_argument("--mad_thresh", type=float, default=7.0)
    parser.add_argument("--flat_guard", type=int, default=2)
    
    # Параметры автомата стабильности (удаление флэта)
    parser.add_argument("--max_frame_diff", type=float, default=25.0,
                        help="Threshold for consecutive frame jump detection")
    parser.add_argument("--required_stable_frames", type=int, default=15,
                        help="Number of consecutive stable frames required to exit quarantine")
    parser.add_argument("--min_search_x", type=int, default=300,
                        help="Start pixel for edge search to bypass lens distortions")
    return parser.parse_args()


# ── Вспомогательные математические функции ─────────────────────────────

def ensure_odd(k):
    k = max(1, int(k))
    return k if k % 2 else k + 1


def moving_average_nan(arr, win):
    win = ensure_odd(win)
    arr = np.asarray(arr, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    half = win // 2
    for i in range(len(arr)):
        chunk = arr[max(0, i - half):min(len(arr), i + half + 1)]
        valid = np.isfinite(chunk)
        if np.any(valid):
            out[i] = np.mean(chunk[valid])
    return out


def interpolate_nans(arr):
    arr = np.asarray(arr, dtype=np.float64)
    out = arr.copy()
    idx = np.arange(len(arr))
    good = np.isfinite(arr)
    if good.sum() == 0:
        return np.zeros_like(arr)
    if good.sum() == 1:
        out[:] = arr[good][0]
        return out
    out[~good] = np.interp(idx[~good], idx[good], arr[good])
    return out


def preprocess(gray):
    return cv2.GaussianBlur(gray, (5, 5), 0)


def select_roi_interactive(frame):
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    if roi[2] <= 0 or roi[3] <= 0:
        raise RuntimeError("ROI не выбрана")
    return tuple(roi)


def default_roi(frame):
    h, w = frame.shape[:2]
    return (int(w * 0.25), int(h * 0.05), int(w * 0.45), int(h * 0.90))


def estimate_global_horizontal_shift(ref_gray, curr_gray, roi):
    x, y, w, h = roi
    ref_roi = ref_gray[y:y + h, x:x + w].astype(np.float32)
    cur_roi = curr_gray[y:y + h, x:x + w].astype(np.float32)
    ref_roi = (ref_roi - ref_roi.mean()) / (ref_roi.std() + 1e-6)
    cur_roi = (cur_roi - cur_roi.mean()) / (cur_roi.std() + 1e-6)
    shift, response = cv2.phaseCorrelate(ref_roi, cur_roi)
    return shift[0], shift[1], response


def find_dark_band_edges(gray_roi):
    profile = gray_roi.mean(axis=0).astype(np.float32)
    k = max(11, (len(profile) // 40) | 1)
    ps = cv2.GaussianBlur(profile.reshape(1, -1), (k, 1), 0).ravel()
    mi = int(np.argmin(ps))
    thresh = ps[mi] + 0.45 * (np.median(ps) - ps[mi])
    left = mi
    while left > 1 and ps[left] < thresh:
        left -= 1
    right = mi
    while right < len(ps) - 2 and ps[right] < thresh:
        right += 1
    return left, right, ps


def subpixel_peak_1d(values, idx):
    if idx <= 0 or idx >= len(values) - 1:
        return 0.0
    y1, y2, y3 = float(values[idx - 1]), float(values[idx]), float(values[idx + 1])
    denom = y1 - 2 * y2 + y3
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y1 - y3) / denom, -1.0, 1.0))


def detect_edge_per_row_subpixel(gray_roi, side="right", band_left=None, band_right=None, min_x=300):
    h, w = gray_roi.shape
    gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    center = band_right if side == "right" else band_left
    if center is None:
        center = int(w * 0.65 if side == "right" else w * 0.35)
    shw = max(20, w // 18)
    x1, x2 = max(1, center - shw), min(w - 2, center + shw)
    
    # Ограничение диапазона сканирования для исключения краевых артефактов
    if x1 < min_x:
        x1 = min_x
    if x2 <= x1:
        x2 = x1 + 1

    xs, ys, scores = [], [], []
    for y in range(h):
        row = gx[y, x1:x2]
        if len(row) == 0:
            xs.append(float(x1))
            ys.append(float(y))
            scores.append(0.0)
            continue
        if side == "right":
            idx = int(np.argmax(row))
            delta = subpixel_peak_1d(row, idx)
            score = float(row[idx])
        else:
            idx = int(np.argmin(row))
            delta = subpixel_peak_1d(-row, idx)
            score = float(-row[idx])
        xs.append(x1 + idx + delta)
        ys.append(float(y))
        scores.append(score)
    return (np.array(xs, np.float32), np.array(ys, np.float32), np.array(scores, np.float32))


def robust_line_fit(xs, ys, weights=None, iterations=4, sigma_thresh=2.5):
    mask = np.ones_like(xs, dtype=bool)
    if weights is None:
        weights = np.ones_like(xs, dtype=np.float32)
    a, b = 0.0, float(np.median(xs))
    for _ in range(iterations):
        if mask.sum() < 10:
            break
        p = np.polyfit(ys[mask], xs[mask], 1, w=weights[mask])
        a, b = p
        err = xs - (a * ys + b)
        s = np.std(err[mask]) + 1e-6
        mask = np.abs(err) < sigma_thresh * s
    return a, b, mask


def draw_line(frame, roi, a, b, color=(0, 0, 255), thickness=2):
    x0, y0, w, h = roi
    p1 = (x0 + int(round(b)), y0)
    p2 = (x0 + int(round(a * (h - 1) + b)), y0 + h - 1)
    cv2.line(frame, p1, p2, color, thickness)


def draw_measurement_points(frame, roi, a, b, y_positions, color=(0, 255, 255), radius=3):
    x0, y0 = roi[0], roi[1]
    for yp in y_positions:
        pt = (x0 + int(round(a * yp + b)), y0 + int(round(yp)))
        cv2.circle(frame, pt, radius, color, -1)


def compute_stats(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {k: np.nan for k in ('mean', 'min', 'max', 'ptp', 'std')}
    return dict(mean=float(np.mean(arr)), min=float(np.min(arr)),
                max=float(np.max(arr)), ptp=float(np.ptp(arr)), std=float(np.std(arr)))


def mad_based_outliers(signal, trend_window=31, mad_thresh=4.0):
    signal = np.asarray(signal, dtype=np.float64)
    trend = moving_average_nan(signal, ensure_odd(trend_window))
    residual = signal - trend
    valid = np.isfinite(residual)
    med = np.median(residual[valid]) if np.any(valid) else 0.0
    mad = np.median(np.abs(residual[valid] - med)) if np.any(valid) else 0.0
    sigma = 1.4826 * mad + 1e-12
    outliers = np.zeros_like(signal, dtype=bool)
    if np.any(valid):
        outliers[valid] = np.abs(residual[valid] - med) > mad_thresh * sigma
    return trend, residual, outliers


def dilate_mask(mask, radius=2):
    out = mask.copy()
    for i in np.where(mask)[0]:
        out[max(0, i - radius):min(len(mask), i + radius + 1)] = True
    return out


def measure_multipoint_runout(df, roi, n_points, um_x, um_y):
    h = roi[3]
    y_positions = np.linspace(h * 0.15, h * 0.85, n_points)
    measurements = []
    for _, row in df.iterrows():
        if row['is_outlier']:
            continue
        a, b = row['line_a_px_per_y'], row['line_b_px']
        dx = row['horizontal_shift_px']
        for i, yp in enumerate(y_positions):
            ex = a * yp + b
            measurements.append({
                'frame': row['frame'], 'point_id': i,
                'y_pos_px': yp, 'y_pos_um': yp * um_y,
                'edge_x_px': ex, 'edge_x_um': ex * um_x,
                'edge_x_comp_px': ex - dx,
                'edge_x_comp_um': (ex - dx) * um_x,
            })
    if not measurements:
        return None, y_positions
    df_m = pd.DataFrame(measurements)
    tir_total = df_m['edge_x_comp_um'].max() - df_m['edge_x_comp_um'].min()
    tir_per_pos = df_m.groupby('point_id')['edge_x_comp_um'].apply(lambda x: x.max() - x.min())
    tir_per_frame = df_m.groupby('frame')['edge_x_comp_um'].apply(lambda x: x.max() - x.min())
    mean_per_frame = df_m.groupby('frame')['edge_x_comp_um'].mean()
    return {
        'tir_total_um': float(tir_total),
        'tir_mean_per_position_um': float(tir_per_pos.mean()),
        'tir_max_per_position_um': float(tir_per_pos.max()),
        'tir_min_per_position_um': float(tir_per_pos.min()),
        'tir_mean_per_frame_um': float(tir_per_frame.mean()),
        'repeatability_um': float(mean_per_frame.std()),
        'measurements': df_m,
        'tir_per_position': tir_per_pos,
        'tir_per_frame': tir_per_frame,
    }, y_positions


def calculate_wobble(df):
    valid = ~df['is_outlier']
    if valid.sum() < 2:
        return {k: np.nan for k in ('wobble_deg', 'tilt_mean_deg', 'tilt_std_deg', 'tilt_min_deg', 'tilt_max_deg')}
    ta = df.loc[valid, 'tilt_angle_deg'].values
    return dict(wobble_deg=float(np.ptp(ta)), tilt_mean_deg=float(np.mean(ta)),
                tilt_std_deg=float(np.std(ta)), tilt_min_deg=float(np.min(ta)), tilt_max_deg=float(np.max(ta)))


def resolve_outputs(args):
    tid = args.trial_id
    ph = args.phase
    tag = f"{ph}_t{tid}" if tid is not None else ph
    base = args.odir if args.odir else "."
    if args.odir:
        os.makedirs(args.odir, exist_ok=True)

    def resolve(explicit, default_name):
        return explicit if explicit else os.path.join(base, default_name)

    args.out_csv     = resolve(args.out_csv,     f"edge_{tag}.csv")
    args.out_video   = resolve(args.out_video,   f"edge_{tag}.mp4")
    args.out_plot    = resolve(args.out_plot,    f"plot_{tag}.png")
    args.out_summary = resolve(args.out_summary, f"summary_{tag}.txt")
    args.out_json    = resolve(args.out_json,    f"video_{tag}.json")


def save_standardized_json(df, multipoint_results, wobble_stats, stats_comp, stats_rx, stats_ry, args, out_path):
    data = {
        'data_type': 'video_edge_analysis',
        'version': 2,
        'timestamp': datetime.now().isoformat(),
        'source_video': args.video,
        'trial_id': args.trial_id,
        'phase': args.phase,
        'calibration': {'um_per_px_x': args.um_per_px_x, 'um_per_px_y': args.um_per_px_y},
        'roi': list(args.roi) if args.roi else None,
        'n_frames': len(df),
        'n_outliers': int(df['is_outlier'].sum()),
        'n_measurement_points': args.n_measurement_points,
        'tir': {
            'single_point_um': round(stats_comp['ptp'], 3),
            'single_point_std_um': round(stats_comp['std'], 3),
            'single_point_mean_um': round(stats_comp['mean'], 3),
        },
        'runout': {
            'horizontal_ptp_um': round(stats_rx['ptp'], 3),
            'horizontal_std_um': round(stats_rx['std'], 3),
            'vertical_ptp_um': round(stats_ry['ptp'], 3),
            'vertical_std_um': round(stats_ry['std'], 3),
        },
        'wobble': {
            'wobble_deg': round(wobble_stats['wobble_deg'], 6),
            'tilt_mean_deg': round(wobble_stats['tilt_mean_deg'], 6),
            'tilt_std_deg': round(wobble_stats['tilt_std_deg'], 6),
        },
        'timeseries': {
            'frames': df['frame'].tolist(),
            'edge_x_um': df['edge_x_um'].tolist(),
            'edge_x_comp_um': df['edge_x_compensated_um'].tolist(),
            'edge_x_comp_clean_um': df['edge_x_compensated_um_clean_interp'].tolist(),
            'runout_x_um': df['horizontal_shift_um'].tolist(),
            'runout_x_smooth_um': df['horizontal_shift_um_smooth'].tolist(),
            'runout_y_um': df['vertical_shift_um'].tolist(),
            'tilt_deg': df['tilt_angle_deg'].tolist(),
            'is_outlier': df['is_outlier'].tolist(),
        },
    }
    if multipoint_results is not None:
        data['multipoint_tir'] = {
            'tir_total_um': round(multipoint_results['tir_total_um'], 3),
            'tir_mean_per_position_um': round(multipoint_results['tir_mean_per_position_um'], 3),
            'tir_max_per_position_um': round(multipoint_results['tir_max_per_position_um'], 3),
            'tir_mean_per_frame_um': round(multipoint_results['tir_mean_per_frame_um'], 3),
            'repeatability_um': round(multipoint_results['repeatability_um'], 3),
            'tir_per_position': multipoint_results['tir_per_position'].tolist(),
        }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {out_path}")


def save_plot(df, out_plot, multipoint_results, wobble_stats, args):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bad = df["is_outlier"]
    phase_label = args.phase.upper()

    ax = axes[0, 0]
    ax.plot(df["frame"], df["edge_x_um"], alpha=0.5, lw=0.8)
    ax.scatter(df.loc[bad, "frame"], df.loc[bad, "edge_x_um"], color="red", s=12, zorder=5)
    ax.plot(df["frame"], df["edge_x_um_clean_interp"], lw=1.5)
    ax.set_title(f"[{phase_label}] Edge position")
    ax.set_xlabel("Frame"); ax.set_ylabel("μm"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df["frame"], df["horizontal_shift_um"], lw=0.8)
    ax.plot(df["frame"], df["horizontal_shift_um_smooth"], lw=1.5)
    s_rx = compute_stats(df["horizontal_shift_um_clean"].values)
    ax.set_title(f"[{phase_label}] Runout = {s_rx['ptp']:.2f} μm")
    ax.set_xlabel("Frame"); ax.set_ylabel("μm"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df["frame"], df["edge_x_compensated_um_clean_interp"], lw=1.5)
    s_c = compute_stats(df["edge_x_compensated_um_clean"].values)
    ax.set_title(f"[{phase_label}] TIR = {s_c['ptp']:.2f} μm")
    ax.set_xlabel("Frame"); ax.set_ylabel("μm"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(df["frame"], df["tilt_angle_deg"], lw=1.0)
    ws = calculate_wobble(df)
    ax.set_title(f"[{phase_label}] Wobble = {ws['wobble_deg']:.4f}°")
    ax.set_xlabel("Frame"); ax.set_ylabel("deg"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Trial {args.trial_id or '?'} — {phase_label} (Flat Omitted)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    resolve_outputs(args)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    height, width = first_frame.shape[:2]

    if args.select_roi:
        roi = select_roi_interactive(first_frame)
    elif args.roi:
        roi = tuple(args.roi)
    else:
        roi = default_roi(first_frame)
    args.roi = roi

    x, y, w, h = roi
    y_positions = np.linspace(h * 0.15, h * 0.85, args.n_measurement_points)

    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    gray_ref = preprocess(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY))

    phase_tag = args.phase.upper()
    print(f"\n[{phase_tag}] Trial {args.trial_id or '?'}")
    print(f"  Video: {args.video}")
    print(f"  ROI: {roi}, Optical Cal: {args.um_per_px_x:.4f} μm/px")

    current_frame = first_frame
    frame_idx = args.start_frame
    records = []

    # Переменные State Machine для вырезания нестабильных зон (флэта)
    is_stable = False
    stability_counter = 0
    stable_buffer = []
    prev_gray_for_jump = None
    saved_count = 0

    while True:
        if frame_idx > args.start_frame:
            ret, current_frame = cap.read()
            if not ret:
                break

        gray = preprocess(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY))
        
        # 1. Вычисление покадрового JUMP
        is_jump = False
        if prev_gray_for_jump is not None:
            frame_diff = np.mean(cv2.absdiff(gray, prev_gray_for_jump))
            if frame_diff > args.max_frame_diff:
                is_jump = True
        prev_gray_for_jump = gray.copy()

        # 2. Попиксельный анализ края геометрического профиля
        gray_roi = gray[y:y + h, x:x + w]
        bl, br, _ = find_dark_band_edges(gray_roi)
        
        # Передаем min_x (300) для защиты от аббераций линзы по краям кадра
        xs_, ys_, sc_ = detect_edge_per_row_subpixel(gray_roi, args.side, bl, br, min_x=args.min_search_x)
        a, b, mask = robust_line_fit(xs_, ys_, weights=np.clip(sc_, 1e-3, None))

        y_mid = (h - 1) / 2.0
        edge_x = x + a * y_mid + b
        dx, dy, pr = estimate_global_horizontal_shift(gray_ref, gray, roi)
        edge_x_comp = edge_x - dx
        tilt_deg = np.degrees(np.arctan(a * args.um_per_px_y / args.um_per_px_x))
        
        # Внутренние признаки падения качества (флэт или срыв трекинга)
        qbad = (mask.sum() < args.min_inliers) or (pr < args.min_phase_response)
        
        # Комплексный триггер нестабильности кадра
        is_unstable_frame = is_jump or qbad

        # Отрисовка разметки (подготовка кадра к выводу)
        vis = current_frame.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw_line(vis, roi, a, b)
        draw_measurement_points(vis, roi, a, b, y_positions)
        color = (0, 0, 255) if is_unstable_frame else (0, 255, 0)
        cv2.putText(vis, f"[{phase_tag}] F={frame_idx} Edge={edge_x:.1f}px", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)

        rec = dict(
            frame=frame_idx,
            edge_x_px=float(edge_x),
            edge_x_um=float(edge_x * args.um_per_px_x),
            horizontal_shift_px=float(dx),
            horizontal_shift_um=float(dx * args.um_per_px_x),
            vertical_shift_px=float(dy),
            vertical_shift_um=float(dy * args.um_per_px_y),
            edge_x_compensated_px=float(edge_x_comp),
            edge_x_compensated_um=float(edge_x_comp * args.um_per_px_x),
            line_a_px_per_y=float(a), line_b_px=float(b),
            tilt_angle_deg=float(tilt_deg),
            phase_response=float(pr),
            inliers=int(mask.sum()),
            quality_bad=bool(qbad),
        )

        # 3. Логика автомата состояний карантина
        if is_unstable_frame:
            is_stable = False
            stability_counter = 0
            stable_buffer.clear()  # Аннулируем подозрительные накопления
        else:
            if is_stable:
                writer.write(vis)
                records.append(rec)
                saved_count += 1
            else:
                stable_buffer.append((vis, rec))
                stability_counter += 1
                if stability_counter >= args.required_stable_frames:
                    is_stable = True
                    # Фиксируем выход из карантина — переносим накопленный буфер в результат
                    for b_vis, b_rec in stable_buffer:
                        writer.write(b_vis)
                        records.append(b_rec)
                        saved_count += 1
                    stable_buffer.clear()

        if args.display:
            cv2.imshow(f"Video [{phase_tag}]", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        frame_idx += 1
        if args.max_frames > 0 and (frame_idx - args.start_frame) >= args.max_frames:
            break
        if (frame_idx - args.start_frame) % 100 == 0:
            print(f"  {frame_idx - args.start_frame} frames read from source...")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if not records:
        raise RuntimeError("No frames survived after stability filtration (Flat/Jump blocks cut off everything)")

    # 4. Пост-аналитика по чистым непрерывным данным края пластины
    df = pd.DataFrame(records)
    win = ensure_odd(args.smooth)

    # Дополнительная фильтрация единичных импульсных выбросов (шум трека) через MAD
    _, _, out1 = mad_based_outliers(df["edge_x_um"].values, args.trend_window, args.mad_thresh)
    _, _, out2 = mad_based_outliers(df["edge_x_compensated_um"].values, args.trend_window, args.mad_thresh)
    is_outlier = df["quality_bad"].values | out1 | out2
    is_outlier = dilate_mask(is_outlier, args.flat_guard)
    df["is_outlier"] = is_outlier

    for col in ("edge_x_um", "edge_x_compensated_um", "horizontal_shift_um", "vertical_shift_um"):
        clean = df[col].values.astype(np.float64).copy()
        clean[is_outlier] = np.nan
        df[f"{col}_clean"] = clean
        df[f"{col}_clean_interp"] = interpolate_nans(clean)

    df["horizontal_shift_um_smooth"] = moving_average_nan(df["horizontal_shift_um_clean_interp"].values, win)
    df["vertical_shift_um_smooth"] = moving_average_nan(df["vertical_shift_um_clean_interp"].values, win)

    valid = ~df["is_outlier"]
    stats_comp = compute_stats(df.loc[valid, "edge_x_compensated_um_clean"].values)
    stats_rx = compute_stats(df.loc[valid, "horizontal_shift_um_clean"].values)
    stats_ry = compute_stats(df.loc[valid, "vertical_shift_um_clean"].values)
    wobble = calculate_wobble(df)
    mp_results, _ = measure_multipoint_runout(df, roi, args.n_measurement_points, args.um_per_px_x, args.um_per_px_y)

    # Сохранение результатов
    df.to_csv(args.out_csv, index=False)
    if mp_results:
        mp_csv = args.out_csv.replace('.csv', '_multipoint.csv')
        mp_results['measurements'].to_csv(mp_csv, index=False)

    save_plot(df, args.out_plot, mp_results, wobble, args)
    save_standardized_json(df, mp_results, wobble, stats_comp, stats_rx, stats_ry, args, args.out_json)

    print(f"\n  [{phase_tag}] Done: {len(df)} frames analyzed (Flat removed successfully).")
    print(f"  TIR = {stats_comp['ptp']:.3f} μm, Runout = {stats_rx['ptp']:.3f} μm, Wobble = {wobble['wobble_deg']:.4f}°")


if __name__ == "__main__":
    main()
