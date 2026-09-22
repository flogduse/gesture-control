"""Pure gesture mathematics — no OpenCV, no camera, no OS access.

Everything here is deterministic and unit-testable: `Landmarks` is a plain data
container, `HandFrame` classifies one frame's gesture, and the two stateful
wrappers (PinchDebouncer, SwipeDetector) turn frame-wise booleans into events.

Coordinate system: MediaPipe normalized coords, x/y in [0, 1] (y grows DOWN),
z relative to the wrist. All comparisons use these conventions explicitly.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

# MediaPipe Hands landmark indices
WRIST, THUMB_TIP, INDEX_TIP, INDEX_PIP = 0, 4, 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18


class Gesture(str, Enum):
    PINCH = "pinch"          # thumb + index tips together
    OPEN = "open"            # all four fingers extended
    FIST = "fist"            # all four fingers curled
    POINT = "point"          # index extended, others curled
    IDLE = "idle"            # anything else


@dataclass(frozen=True)
class Landmarks:
    """21 (x, y, z) triplets, normalized exactly as MediaPipe emits them."""
    pts: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        if len(self.pts) != 21:
            raise ValueError(f"expected 21 landmarks, got {len(self.pts)}")

    def d(self, a: int, b: int) -> float:
        """Euclidean distance between two landmarks (x, y, z)."""
        ax, ay, az = self.pts[a]
        bx, by, bz = self.pts[b]
        return ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5

    def d2(self, a: int, b: int) -> float:
        """Planar distance, ignoring z — z is noisy at webcam range."""
        ax, ay, _ = self.pts[a]
        bx, by, _ = self.pts[b]
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    @property
    def palm(self) -> float:
        """Palm scale reference: wrist -> middle-finger MCP distance.

        All thresholds are ratios of this, making gestures invariant to how
        close the hand is to the camera.
        """
        return max(self.d2(WRIST, MIDDLE_PIP), 1e-6)


class HandFrame:
    """Classifies a single frame of one hand."""

    def __init__(self, lm: Landmarks, handedness: str = "Right",
                 pinch_close: float = 0.40, pinch_open: float = 0.55):
        self.lm = lm
        self.handedness = handedness
        self.pinch_close = pinch_close
        self.pinch_open = pinch_open

    # -- finger extension: tip clearly beyond its PIP joint (planar, y-axis) --
    def finger_extended(self, tip: int, pip: int) -> bool:
        scale = self.lm.palm
        return (self.lm.pts[pip][1] - self.lm.pts[tip][1]) > 0.35 * scale

    @property
    def index_up(self) -> bool:
        return self.finger_extended(INDEX_TIP, INDEX_PIP)

    @property
    def middle_up(self) -> bool:
        return self.finger_extended(MIDDLE_TIP, MIDDLE_PIP)

    @property
    def ring_up(self) -> bool:
        return self.finger_extended(RING_TIP, RING_PIP)

    @property
    def pinky_up(self) -> bool:
        return self.finger_extended(PINKY_TIP, PINKY_PIP)

    @property
    def fingers_up(self) -> int:
        return sum([self.index_up, self.middle_up, self.ring_up, self.pinky_up])

    @property
    def pinch_ratio(self) -> float:
        """Thumb-to-index distance as a fraction of palm scale. Smaller = closer."""
        return self.lm.d2(THUMB_TIP, INDEX_TIP) / self.lm.palm

    @property
    def gesture(self) -> Gesture:
        if self.pinch_ratio < self.pinch_close:
            return Gesture.PINCH
        if self.fingers_up == 4:
            return Gesture.OPEN
        if self.fingers_up == 0:
            return Gesture.FIST
        if self.index_up and self.fingers_up == 1:
            return Gesture.POINT
        return Gesture.IDLE


class PinchDebouncer:
    """Pinch with hysteresis + a minimum hold time, so a wobbling thumb
    doesn't machine-gun the action. Yields True exactly once per press."""

    def __init__(self, close: float = 0.40, open_: float = 0.55,
                 hold_seconds: float = 0.12):
        self.close = close
        self.open_ = open_
        self.hold = hold_seconds
        self._closed_at: float | None = None
        self._is_down = False

    def update(self, ratio: float, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        fired = False
        if not self._is_down:
            if ratio < self.close:
                if self._closed_at is None:
                    self._closed_at = now
                elif now - self._closed_at >= self.hold:
                    self._is_down = True
                    fired = True
            else:
                self._closed_at = None
        else:
            if ratio > self.open_:  # must clearly reopen before re-firing
                self._is_down = False
                self._closed_at = None
        return fired


class SwipeDetector:
    """Horizontal two-finger swipe state machine over index+middle fingertips.

    Phases: OPEN (both fingers up, hand quiet) -> ARMED (fast movement) ->
    FIRED (edge crossed) -> cooldown back to OPEN. Direction locks at fire time.
    """

    LEFT, RIGHT = "left", "right"

    def __init__(self, enter: float = 0.30, exit_: float = 0.12,
                 edge: float = 0.25, cooldown: float = 0.7):
        self.enter, self.exit_, self.edge = enter, exit_, edge
        self.cooldown = cooldown
        self._phase = "OPEN"
        self._start_x = 0.5
        self._armed_at = 0.0

    def update(self, index_x: float, two_fingers: bool,
               now: float | None = None) -> str | None:
        now = time.monotonic() if now is None else now
        if self._phase == "COOLDOWN":
            if now - self._armed_at >= self.cooldown:
                self._phase = "OPEN"
            return None
        if not two_fingers:
            self._phase = "OPEN"
            return None
        if self._phase == "OPEN":
            # start tracking from a quiet hand near the middle band
            if self.exit_ + self.edge < index_x < 1.0 - self.exit_ - self.edge:
                self._phase = "TRACKING"
                self._start_x = index_x
            return None
        # TRACKING
        dx = index_x - self._start_x
        if abs(dx) < self.enter:
            return None
        direction = self.RIGHT if dx > 0 else self.LEFT
        self._phase = "COOLDOWN"
        self._armed_at = now
        return direction


@dataclass
class GestureEngine:
    """Convenience facade: feed raw MediaPipe output, get events out."""
    pinch: PinchDebouncer = field(default_factory=PinchDebouncer)
    swipe: SwipeDetector = field(default_factory=SwipeDetector)

    def events(self, lm: Landmarks | None, handedness: str = "Right",
               now: float | None = None) -> Iterator[tuple[str, object]]:
        """Yields ('gesture', Gesture), ('pinch', True) and/or ('swipe', 'left'|'right')."""
        now = time.monotonic() if now is None else now
        if lm is None:
            return
        frame = HandFrame(lm, handedness)
        g = frame.gesture
        yield "gesture", g
        if g == Gesture.PINCH:
            yield "pinch", self.pinch.update(frame.pinch_ratio, now)
        else:
            self.pinch.update(1.0, now)  # feed open ratio to reset debounce
        if g in (Gesture.OPEN, Gesture.POINT):
            two = frame.index_up and frame.middle_up
            if two:
                s = self.swipe.update(frame.lm.pts[INDEX_TIP][0], True, now)
                if s:
                    yield "swipe", s
            else:
                self.swipe.update(0.5, False, now)


def smooth(prev: tuple[float, float] | None, new: tuple[float, float],
           alpha: float = 0.35) -> tuple[float, float]:
    """Exponential smoothing for cursor-like values (kills jitter)."""
    if prev is None or not (0.0 <= alpha <= 1.0):
        return new
    return (alpha * new[0] + (1 - alpha) * prev[0],
            alpha * new[1] + (1 - alpha) * prev[1])


__all__ = [
    "Gesture", "Landmarks", "HandFrame", "PinchDebouncer", "SwipeDetector",
    "GestureEngine", "smooth",
]
