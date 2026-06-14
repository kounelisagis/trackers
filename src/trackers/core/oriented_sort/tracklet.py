# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from trackers.core.sort.tracklet import SORTTracklet
from trackers.utils.state_representations import (
    BaseStateEstimator,
    XCYCWHStateEstimator,
)


def obb_envelope(obb: np.ndarray) -> np.ndarray:
    """Axis-aligned envelope ``[x1, y1, x2, y2]`` of an oriented box.

    Args:
        obb: Oriented box as ``(4, 2)`` corner coordinates.

    Returns:
        The tight axis-aligned bounding box of the four corners.
    """
    xs, ys = obb[:, 0], obb[:, 1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


class OrientedSORTTracklet(SORTTracklet):
    """SORT tracklet for oriented (rotated) boxes.

    The Kalman filter tracks the object's **axis-aligned envelope**, whose
    centre coincides with the oriented-box centre (the centroid of a rectangle
    equals the centre of its axis-aligned bounding box). This reuses the entire
    SORT motion model and lifecycle for smoothed, occlusion-robust position.

    The oriented **shape** — size *and* angle together — is carried verbatim
    from the most recent detection as per-corner offsets from the centre. The
    tracked oriented box is therefore the Kalman-predicted centre plus that
    carried shape. No angle filtering is performed (that is a later refinement);
    this is the "carry the last detection's orientation" model, which sidesteps
    the periodicity / 180-degree ambiguity entirely.

    Attributes:
        shape_offsets: ``(4, 2)`` per-corner offsets from the box centre,
            describing the carried oriented shape.
    """

    def __init__(
        self,
        initial_obb: np.ndarray,
        state_estimator_class: type[BaseStateEstimator] = XCYCWHStateEstimator,
    ) -> None:
        """Initialise from the first oriented detection.

        Args:
            initial_obb: First detection as ``(4, 2)`` oriented-box corners.
            state_estimator_class: Kalman state estimator for the envelope.
                Defaults to :class:`XCYCWHStateEstimator`, whose state holds the
                centre directly.
        """
        initial_obb = np.asarray(initial_obb, dtype=np.float64)
        self.shape_offsets = initial_obb - initial_obb.mean(axis=0)
        super().__init__(obb_envelope(initial_obb), state_estimator_class)

    def update(self, obb: np.ndarray) -> None:
        """Update with a new oriented-box observation.

        Refreshes the carried shape from ``obb`` and feeds the box's envelope to
        the Kalman filter.

        Args:
            obb: Oriented box as ``(4, 2)`` corner coordinates.
        """
        obb = np.asarray(obb, dtype=np.float64)
        self.shape_offsets = obb - obb.mean(axis=0)
        super().update(obb_envelope(obb))

    def _predicted_center(self) -> np.ndarray:
        """Centre of the current Kalman-estimated envelope, as ``(x, y)``."""
        x1, y1, x2, y2 = self.get_state_bbox()
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)

    def get_state_obb(self) -> np.ndarray:
        """Current oriented-box estimate as ``(4, 2)`` corners.

        Returns:
            The Kalman-predicted centre plus the carried oriented shape.
        """
        return self._predicted_center() + self.shape_offsets
