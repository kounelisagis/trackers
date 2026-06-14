# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv
from supervision.config import ORIENTED_BOX_COORDINATES

from trackers import OrientedSORTTracker
from trackers.core.base import BaseTracker


def _rotated_rect(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    """Four corners (4, 2) of a rectangle centred at (cx, cy), rotated by angle_deg."""
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (corners @ rot.T + [cx, cy]).astype(np.float32)


def _detections(obbs: list[np.ndarray], confidence: float = 0.9) -> sv.Detections:
    """Build sv.Detections carrying oriented boxes (+ their axis-aligned envelope)."""
    obb = np.asarray(obbs, dtype=np.float32).reshape(-1, 4, 2)
    xyxy = np.stack(
        [obb[:, :, 0].min(1), obb[:, :, 1].min(1), obb[:, :, 0].max(1), obb[:, :, 1].max(1)],
        axis=1,
    ).astype(np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.full(len(obb), confidence, dtype=np.float32),
        data={ORIENTED_BOX_COORDINATES: obb},
    )


class TestOrientedSORTTracker:
    """Tests for `OrientedSORTTracker`."""

    def test_is_registered(self) -> None:
        """The tracker auto-registers under the `oriented-sort` id."""
        assert "oriented-sort" in BaseTracker._registered_trackers()

    def test_requires_oriented_boxes(self) -> None:
        """Detections without oriented-box data raise a helpful error."""
        tracker = OrientedSORTTracker()
        axis_aligned = sv.Detections(
            xyxy=np.array([[10, 10, 50, 30]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
        )

        with pytest.raises(ValueError, match="requires oriented boxes"):
            tracker.update(axis_aligned)

    def test_single_object_keeps_stable_id(self) -> None:
        """One object moving in a straight line keeps a single confirmed id."""
        tracker = OrientedSORTTracker(minimum_consecutive_frames=2, minimum_iou_threshold=0.1)

        ids: list[int] = []
        for step in range(10):
            obb = _rotated_rect(30 + step * 12, 100, 60, 16, 25)
            result = tracker.update(_detections([obb]))
            ids.append(int(result.tracker_id[0]))

        confirmed = [i for i in ids if i != -1]
        assert len(confirmed) >= 6  # confirmed shortly after warmup
        assert len(set(confirmed)) == 1  # never switches

    def test_crossing_perpendicular_bars_have_no_id_swap(self) -> None:
        """Two perpendicular bars whose envelopes overlap but bodies do not keep
        distinct, stable ids through the crossing — the core oriented-IoU win."""
        tracker = OrientedSORTTracker(
            minimum_consecutive_frames=2,
            minimum_iou_threshold=0.1,
            lost_track_buffer=30,
        )

        steps = 21
        slot_a_ids: list[int] = []
        slot_b_ids: list[int] = []
        for step in range(steps):
            t = step / (steps - 1)
            cx_a, cx_b = 60 + 80 * t, 140 - 80 * t  # cross at the midpoint
            bar_a = _rotated_rect(cx_a, 100, 90, 10, 45)  # "/" bar
            bar_b = _rotated_rect(cx_b, 100, 90, 10, -45)  # "\" bar

            result = tracker.update(_detections([bar_a, bar_b]))
            slot_a_ids.append(int(result.tracker_id[0]))
            slot_b_ids.append(int(result.tracker_id[1]))

        confirmed_a = {i for i in slot_a_ids if i != -1}
        confirmed_b = {i for i in slot_b_ids if i != -1}

        # Each object holds exactly one id for its whole life...
        assert len(confirmed_a) == 1
        assert len(confirmed_b) == 1
        # ...and the two objects are never confused with each other.
        assert confirmed_a.isdisjoint(confirmed_b)

    def test_reset_restarts_id_allocation(self) -> None:
        """`reset()` clears tracks and restarts ids from zero."""
        tracker = OrientedSORTTracker(minimum_consecutive_frames=1, minimum_iou_threshold=0.1)
        tracker.reset()  # zero the shared id counter for a deterministic check
        for step in range(3):
            tracker.update(_detections([_rotated_rect(30 + step * 8, 100, 60, 16, 25)]))
        assert tracker.tracks[0].tracker_id == 0  # ids started at zero

        tracker.reset()
        assert tracker.tracks == []

        # A freshly spawned track is unconfirmed on its first frame (-1) and is
        # assigned its id once it matches on the next frame — restarting from 0.
        tracker.update(_detections([_rotated_rect(30, 100, 60, 16, 25)]))
        result = tracker.update(_detections([_rotated_rect(38, 100, 60, 16, 25)]))
        assert int(result.tracker_id[0]) == 0
