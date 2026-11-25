"""Shared robust Hough lane segment selection for race and calibration dashboard."""
import math
import cv2
import numpy as np


def weighted_median(values, weights):
    order = np.argsort(values)
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    return float(values[np.searchsorted(np.cumsum(weights), sum(weights) / 2)])


def detect_side(gray, bounds, side, p):
    top, bottom, left, right = (int(v) for v in bounds)
    h, w = gray.shape[:2]
    top, bottom = max(0, top), min(h, bottom)
    left, right = max(0, left), min(w, right)
    out = {'center_x': None, 'slope_dx_dy': None, 'kept': [], 'rejected': [], 'edges': np.zeros((8, 8), np.uint8), 'longest': 0}
    if bottom - top < 4 or right - left < 4:
        return out
    roi = gray[top:bottom, left:right]
    edges = cv2.Canny(roi, int(p['CANNY_LOW']), int(p['CANNY_HIGH']))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    out['edges'] = edges
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, int(p['HOUGH_THRESHOLD']),
                            minLineLength=int(p['MIN_LINE_LENGTH']), maxLineGap=int(p['MAX_LINE_GAP']))
    candidates = []
    min_slope = float(p['RIGHT_MIN_ABS_SLOPE'] if side == 'right' else p['LEFT_MIN_ABS_SLOPE'])
    min_angle = float(p.get('LINE_MIN_ANGLE_DEG', 24.0))
    max_angle = float(p.get('LINE_MAX_ANGLE_DEG', 88.0))
    if lines is None:
        return out
    for line in lines:
        x1, y1, x2, y2 = map(int, line.flatten())
        segment = (x1 + left, y1 + top, x2 + left, y2 + top)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(abs(dy), abs(dx))) if length > 0 else 0
        if abs(dx) < 1 or abs(dy) < 1 or not min_angle <= angle <= max_angle or abs(dy / dx) <= min_slope:
            out['rejected'].append(segment)
            continue
        # Extrapolate to y=bottom (exclusive) without clamping the result into the ROI.
        x_bottom = left + x1 + ((bottom - top - 1 - y1) * dx / dy)
        # Permit a small extrapolation margin, but reject implausible distant intersections.
        margin = float(p.get('LINE_EXTRAPOLATION_MARGIN_PX', 25))
        if not left - margin <= x_bottom <= right + margin:
            out['rejected'].append(segment)
            continue
        if (side == 'right' and x_bottom < w / 2) or (side == 'left' and x_bottom > w / 2):
            out['rejected'].append(segment)
            continue
        candidates.append((segment, x_bottom, angle, length, 1 if dx * dy > 0 else -1, dx / dy))
    if not candidates:
        return out
    # Dominant inclination sign avoids mixing two competing line families.
    groups = {sign: [v for v in candidates if v[4] == sign] for sign in (-1, 1)}
    dominant = max(groups.values(), key=lambda g: sum(v[3] for v in g))
    xs = [v[1] for v in dominant]
    lengths = [v[3] for v in dominant]
    angles = [v[2] for v in dominant]
    median_x = weighted_median(xs, lengths)
    median_angle = weighted_median(angles, lengths)
    x_tol = float(p.get('LINE_BOTTOM_TOLERANCE_PX', 22))
    angle_tol = float(p.get('LINE_ANGLE_TOLERANCE_DEG', 18))
    kept = [v for v in dominant if abs(v[1] - median_x) <= x_tol and abs(v[2] - median_angle) <= angle_tol]
    kept_segs = {v[0] for v in kept}
    out['kept'] = [v[0] for v in kept]
    out['rejected'].extend(v[0] for v in candidates if v[0] not in kept_segs)
    if kept:
        out['center_x'] = float(sum(v[1] * v[3] for v in kept) / sum(v[3] for v in kept))
        out['longest'] = int(max(v[3] for v in kept))
        out['slope_dx_dy'] = weighted_median([v[5] for v in kept], [v[3] for v in kept])
    return out
