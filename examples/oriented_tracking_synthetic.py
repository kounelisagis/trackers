# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Synthetic, dependency-free demo of OrientedSORTTracker.

Renders a scene of rotating, translating oriented boxes (some crossing
diagonally so their axis-aligned envelopes overlap heavily) and writes an
annotated video with stable track ids. No detector or external data needed —
the "detections" are the ground-truth oriented boxes of the scripted scene.

    uv run python examples/oriented_tracking_synthetic.py --output out.mp4
"""

from __future__ import annotations

import argparse

import numpy as np
import supervision as sv
from supervision.config import ORIENTED_BOX_COORDINATES

from trackers import OrientedSORTTracker

WIDTH, HEIGHT, FPS, FRAMES = 720, 440, 20, 70


def rotated_rect(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    """Four corners (4, 2) of a rectangle centred at (cx, cy), rotated by angle_deg."""
    angle = np.deg2rad(angle_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (corners @ rot.T + [cx, cy]).astype(np.float32)


# (start_xy, end_xy, start_angle, end_angle, w, h) per scripted object.
TRAJECTORIES = [
    ((80, 110), (640, 150), 8, 18, 78, 28),  # lane, gentle turn
    ((80, 330), (640, 300), -6, -14, 78, 28),  # lane, opposite gentle turn
    ((150, 60), (560, 380), 45, 60, 70, 26),  # diagonal "\" descending
    ((560, 60), (150, 380), -45, -60, 70, 26),  # diagonal "/" descending (crosses #2)
    ((360, 40), (360, 400), 88, 92, 64, 24),  # vertical, near-upright
]


def scene_detections(step: int) -> sv.Detections:
    """Ground-truth oriented boxes for one frame, as sv.Detections."""
    t = step / (FRAMES - 1)
    obbs = []
    for (x0, y0), (x1, y1), a0, a1, w, h in TRAJECTORIES:
        cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        angle = a0 + (a1 - a0) * t
        obbs.append(rotated_rect(cx, cy, w, h, angle))
    obb = np.asarray(obbs, dtype=np.float32)
    xyxy = np.stack(
        [obb[:, :, 0].min(1), obb[:, :, 1].min(1), obb[:, :, 0].max(1), obb[:, :, 1].max(1)],
        axis=1,
    ).astype(np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.full(len(obb), 0.9, dtype=np.float32),
        data={ORIENTED_BOX_COORDINATES: obb},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="oriented_tracking_synthetic.mp4")
    args = parser.parse_args()

    tracker = OrientedSORTTracker(minimum_consecutive_frames=2, minimum_iou_threshold=0.1)
    obb_annotator = sv.OrientedBoxAnnotator(thickness=3, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2, color_lookup=sv.ColorLookup.TRACK)

    video_info = sv.VideoInfo(width=WIDTH, height=HEIGHT, fps=FPS, total_frames=FRAMES)
    with sv.VideoSink(args.output, video_info) as sink:
        for step in range(FRAMES):
            detections = scene_detections(step)
            tracked = tracker.update(detections)
            # Draw only confirmed tracks (a track is unconfirmed -> id -1 on its
            # first frames), coloured consistently by track id.
            confirmed = tracked[tracked.tracker_id != -1]

            scene = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
            scene = obb_annotator.annotate(scene, confirmed)
            labels = [f"#{tid}" for tid in confirmed.tracker_id]
            scene = label_annotator.annotate(scene, confirmed, labels=labels)
            sink.write_frame(scene)

    print(f"wrote {FRAMES} frames -> {args.output}")


if __name__ == "__main__":
    main()
