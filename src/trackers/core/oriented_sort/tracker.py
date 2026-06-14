# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import ClassVar

import numpy as np
import supervision as sv
from supervision.config import ORIENTED_BOX_COORDINATES

from trackers.core.oriented_sort.tracklet import OrientedSORTTracklet
from trackers.core.sort.tracker import SORTTracker
from trackers.core.sort.utils import _get_alive_tracklets
from trackers.utils.detections import default_confidences
from trackers.utils.oriented_iou import oriented_iou_batch
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCWHStateEstimator,
)


class OrientedSORTTracker(SORTTracker):
    """SORT for oriented (rotated) bounding boxes.

    Identical in spirit to :class:`SORTTracker` — predict, associate, update,
    manage lifecycle — but association uses **oriented IoU** instead of
    axis-aligned IoU, so detections whose envelopes overlap heavily while their
    rotated bodies do not (diagonally-parked vehicles, ships in a harbour) are
    no longer confused. Each track carries the orientation of its most recent
    detection; position is smoothed by the same Kalman motion model as SORT.

    Consumes and produces ``supervision.Detections`` carrying oriented boxes in
    ``detections.data["xyxyxyxy"]`` (shape ``(N, 4, 2)``), exactly as produced
    by ``sv.Detections.from_*`` for OBB models.

    Args:
        lost_track_buffer: Number of frames to keep a lost track alive while
            coasting on prediction.
        frame_rate: Video frame rate, used to scale ``lost_track_buffer``.
        track_activation_threshold: Minimum detection confidence to start a
            new track.
        minimum_consecutive_frames: Updates required before a track is
            confirmed (emits a real ``tracker_id`` instead of ``-1``).
        minimum_iou_threshold: Minimum **oriented** IoU to associate a
            detection with a track.
        state_estimator_class: Kalman state estimator for the tracked envelope.
            Defaults to :class:`XCYCWHStateEstimator`.
    """

    tracker_id = "oriented-sort"

    search_space: ClassVar[dict[str, dict]] = {
        "lost_track_buffer": {"type": "randint", "range": [10, 91]},
        "track_activation_threshold": {"type": "uniform", "range": [0.1, 0.9]},
        "minimum_consecutive_frames": {"type": "randint", "range": [1, 4]},
        "minimum_iou_threshold": {"type": "uniform", "range": [0.05, 0.7]},
    }

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
    ) -> None:
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold
        self.state_estimator_class = state_estimator_class
        self.tracks: list[OrientedSORTTracklet] = []
        self._reset_id_allocator()

    @staticmethod
    def _extract_oriented_boxes(detections: sv.Detections) -> np.ndarray:
        """Pull oriented boxes from ``detections`` as an ``(N, 4, 2)`` array.

        Raises:
            ValueError: If the detections carry no oriented-box data.
        """
        if len(detections) == 0:
            return np.empty((0, 4, 2), dtype=np.float64)
        obb = detections.data.get(ORIENTED_BOX_COORDINATES)
        if obb is None:
            raise ValueError(
                "OrientedSORTTracker requires oriented boxes in "
                f"detections.data['{ORIENTED_BOX_COORDINATES}'] (shape (N, 4, 2)). "
                "Use an OBB detection model / the corresponding sv.Detections.from_* "
                "connector, or the standard SORTTracker for axis-aligned boxes."
            )
        return np.asarray(obb, dtype=np.float64).reshape(-1, 4, 2)

    def _spawn_new_tracklets(
        self,
        confidences: np.ndarray,
        detection_boxes: np.ndarray,
        unmatched_detections: list[int],
    ) -> None:
        for detection_idx in unmatched_detections:
            if confidences[detection_idx] >= self.track_activation_threshold:
                self.tracks.append(
                    OrientedSORTTracklet(
                        detection_boxes[detection_idx],
                        state_estimator_class=self.state_estimator_class,
                    )
                )

    def update(self, detections: sv.Detections, frame: np.ndarray | None = None) -> sv.Detections:
        """Assign track IDs to oriented detections.

        Args:
            detections: Current-frame detections carrying oriented boxes in
                ``detections.data["xyxyxyxy"]``.
            frame: Ignored by SORT; a warning is emitted if provided.

        Returns:
            The input detections enriched with ``tracker_id``. Unmatched or
            not-yet-confirmed tracks are labelled ``-1``.
        """
        self._warn_if_frame_unused(frame)
        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        detection_obbs = self._extract_oriented_boxes(detections)

        for tracklet in self.tracks:
            tracklet.predict()

        predicted_obbs = np.array([t.get_state_obb() for t in self.tracks]) if self.tracks else np.empty((0, 4, 2))
        iou_matrix = oriented_iou_batch(predicted_obbs, detection_obbs)

        matched_indices, _unmatched_tracklets, unmatched_detections = self._get_associated_indices(
            iou_matrix, detection_obbs
        )

        matched_tracklet_for_det: dict[int, OrientedSORTTracklet] = {}
        for row, col in matched_indices:
            self.tracks[row].update(detection_obbs[col])
            matched_tracklet_for_det[col] = self.tracks[row]

        confidences = default_confidences(detections)
        self._spawn_new_tracklets(confidences, detection_obbs, unmatched_detections)

        self.tracks = _get_alive_tracklets(
            self.tracks,
            self.minimum_consecutive_frames,
            self.maximum_frames_without_update,
        )

        tracker_ids = np.full(len(detection_obbs), -1, dtype=int)
        for det_idx, tracklet in matched_tracklet_for_det.items():
            if tracklet.number_of_successful_updates >= self.minimum_consecutive_frames:
                if tracklet.tracker_id == -1:
                    tracklet.tracker_id = self._allocate_tracker_id()
                tracker_ids[det_idx] = tracklet.tracker_id

        result = sv.Detections.empty() if len(detections) == 0 else detections[np.arange(len(detections))]
        result.tracker_id = tracker_ids
        return result

    @property
    def tracked_objects(self) -> sv.Detections:
        """Confirmed alive tracks with Kalman-predicted **oriented** boxes.

        Like :attr:`SORTTracker.tracked_objects` but additionally attaches the
        predicted oriented box of each track in ``data["xyxyxyxy"]`` so the
        result can be drawn with ``sv.OrientedBoxAnnotator`` (e.g. to visualise
        tracks coasting through occlusion).
        """
        result = super().tracked_objects
        alive = [t for t in self.tracks if t.tracker_id != -1]
        if alive:
            result.data[ORIENTED_BOX_COORDINATES] = np.array([t.get_state_obb() for t in alive])
        return result

    def reset(self) -> None:
        """Clear all tracks and restart ID allocation."""
        self.tracks = []
        self._reset_id_allocator()
