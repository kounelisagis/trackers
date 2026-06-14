# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Exact pairwise IoU for oriented (rotated) boxes.

Overlap is computed by exact convex-polygon intersection
(:func:`cv2.intersectConvexConvex`), pruned by a vectorised axis-aligned
envelope gate: any pair whose upright bounding rectangles do not overlap has
zero oriented overlap and skips the polygon step. This avoids the
rasterisation error (and empty-input pitfalls) of mask-based implementations,
which is important when the IoU drives data-association decisions.
"""

from __future__ import annotations

import cv2
import numpy as np


def _polygon_areas(polygons: np.ndarray) -> np.ndarray:
    """Shoelace area of every ``(4, 2)`` polygon in an ``(N, 4, 2)`` batch."""
    x, y = polygons[:, :, 0], polygons[:, :, 1]
    return 0.5 * np.abs((x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1))


def _envelopes(polygons: np.ndarray) -> np.ndarray:
    """Axis-aligned ``(x_min, y_min, x_max, y_max)`` of each ``(4, 2)`` polygon."""
    xs, ys = polygons[:, :, 0], polygons[:, :, 1]
    return np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1)


def oriented_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of oriented boxes.

    Args:
        boxes_a: ``(N, 4, 2)`` oriented-box corners.
        boxes_b: ``(M, 4, 2)`` oriented-box corners.

    Returns:
        ``(N, M)`` IoU matrix in ``[0, 1]``; entry ``(i, j)`` is the IoU of
        ``boxes_a[i]`` and ``boxes_b[j]``.
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4, 2)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4, 2)
    n, m = len(boxes_a), len(boxes_b)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    areas_a, areas_b = _polygon_areas(boxes_a), _polygon_areas(boxes_b)
    env_a, env_b = _envelopes(boxes_a), _envelopes(boxes_b)

    x1 = np.maximum(env_a[:, None, 0], env_b[None, :, 0])
    y1 = np.maximum(env_a[:, None, 1], env_b[None, :, 1])
    x2 = np.minimum(env_a[:, None, 2], env_b[None, :, 2])
    y2 = np.minimum(env_a[:, None, 3], env_b[None, :, 3])
    rows, cols = np.where((x2 > x1) & (y2 > y1))

    polys_a = [b.astype(np.float32) for b in boxes_a]
    polys_b = [b.astype(np.float32) for b in boxes_b]

    iou = np.zeros((n, m), dtype=np.float64)
    for i, j in zip(rows, cols):
        intersection, _ = cv2.intersectConvexConvex(polys_a[i], polys_b[j])
        if intersection <= 0:
            continue
        union = areas_a[i] + areas_b[j] - intersection
        if union > 0:
            iou[i, j] = intersection / union
    return np.clip(iou, 0.0, 1.0)
