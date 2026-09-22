"""Tests for the pure gesture engine — deterministic, no camera, no OS calls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.gestures import (  # noqa: E402
    Gesture, GestureEngine, HandFrame, Landmarks, PinchDebouncer,
    SwipeDetector, smooth,
)
from app.synth import fist, open_palm, pinched, pointing  # noqa: E402


class TestLandmarks:
    def test_rejects_wrong_count(self):
        import pytest
        with pytest.raises(ValueError):
            Landmarks(tuple([(0.0, 0.0, 0.0)] * 20))

    def test_distance_metric(self):
        lm = open_palm(0.5, 0.5)
        d_down = lm.d2(0, 8)          # wrist -> index tip (straight up)
        assert d_down > 0.35
        assert lm.d(0, 8) >= lm.d2(0, 8)  # 3D >= planar

    def test_palm_scale_positive(self):
        assert open_palm().palm > 0


class TestClassification:
    def test_open_palm(self):
        assert HandFrame(open_palm()).gesture == Gesture.OPEN

    def test_fist(self):
        assert HandFrame(fist()).gesture == Gesture.FIST

    def test_pinch(self):
        assert HandFrame(pinched()).gesture == Gesture.PINCH

    def test_point(self):
        assert HandFrame(pointing()).gesture == Gesture.POINT

    def test_scale_invariance(self):
        """Same hand shape at a different distance -> same gesture."""
        small = open_palm(0.5, 0.5)
        big = Landmarks(tuple((0.5 + (p[0] - 0.5) * 2,
                               0.5 + (p[1] - 0.5) * 2, p[2])
                              for p in open_palm(0.5, 0.5).pts))
        assert HandFrame(small).gesture == HandFrame(big).gesture == Gesture.OPEN


class TestPinchDebouncer:
    def test_requires_hold(self):
        p = PinchDebouncer(hold_seconds=0.12)
        assert p.update(0.2, now=1.0) is False
        assert p.update(0.2, now=1.05) is False      # inside hold window
        assert p.update(0.2, now=1.2) is True        # held long enough

    def test_fires_once_per_press(self):
        p = PinchDebouncer(hold_seconds=0.0)
        assert p.update(0.2, now=1.0) is False   # first sight only starts the clock
        assert p.update(0.2, now=1.05) is True   # hold satisfied -> fires
        assert p.update(0.2, now=1.1) is False   # still closed
        assert p.update(0.2, now=2.0) is False   # STILL closed (hysteresis)

    def test_hysteresis_then_refire(self):
        p = PinchDebouncer(hold_seconds=0.0)
        assert p.update(0.2, now=1.0) is False
        assert p.update(0.2, now=1.05) is True   # fired, is_down = True
        p.update(0.6, now=1.5)                   # clearly reopens (> open_=0.55)
        assert p._is_down is False
        assert p.update(0.2, now=2.0) is False   # fresh press: clock restarts
        assert p.update(0.2, now=2.05) is True

    def test_jitter_between_thresholds_does_not_refire(self):
        p = PinchDebouncer(hold_seconds=0.0)
        p.update(0.2, now=1.0)                       # fires
        for i, r in enumerate([0.45, 0.50, 0.45, 0.48]):
            assert p.update(r, now=1.1 + i * 0.05) is False


class TestSwipeDetector:
    def test_fires_right(self):
        s = SwipeDetector()
        assert s.update(0.5, True, now=1.0) is None
        assert s.update(0.9, True, now=1.2) == SwipeDetector.RIGHT

    def test_fires_left(self):
        s = SwipeDetector()
        assert s.update(0.5, True, now=1.0) is None
        assert s.update(0.1, True, now=1.2) == SwipeDetector.LEFT

    def test_cooldown_blocks_immediate_refire(self):
        s = SwipeDetector()
        s.update(0.5, True, now=1.0)
        assert s.update(0.9, True, now=1.2) == SwipeDetector.RIGHT
        assert s.update(0.2, True, now=1.3) is None      # inside cooldown
        assert s.update(0.1, True, now=1.85) is None     # just before expiry
        assert s.update(0.5, True, now=2.0) is None      # expiry call: back to OPEN
        assert s.update(0.5, True, now=2.1) is None      # center band: re-arms tracking
        assert s.update(0.1, True, now=2.3) == SwipeDetector.LEFT

    def test_losing_fingers_resets(self):
        s = SwipeDetector()
        s.update(0.5, True, now=1.0)
        s.update(0.6, False, now=1.1)                # fingers gone
        assert s.update(0.9, True, now=1.2) is None  # re-arms from center, no fire


class TestEngineFacade:
    def test_open_palm_yields_gesture_event(self):
        eng = GestureEngine()
        events = list(eng.events(open_palm(), "Right", now=1.0))
        assert ("gesture", Gesture.OPEN) in events

    def test_pinch_path_yields_pinch(self):
        eng = GestureEngine()
        events = list(eng.events(pinched(), "Right", now=1.0))
        assert events[0] == ("gesture", Gesture.PINCH)
        assert events[1][0] == "pinch"

    def test_no_hand_yields_nothing(self):
        assert list(GestureEngine().events(None)) == []


class TestSmoothing:
    def test_none_returns_new(self):
        assert smooth(None, (0.3, 0.7)) == (0.3, 0.7)

    def test_interpolation(self):
        assert smooth((0.0, 0.0), (1.0, 1.0), alpha=0.5) == (0.5, 0.5)

    def test_converges(self):
        x = (0.0, 0.0)
        for _ in range(50):
            x = smooth(x, (1.0, 1.0), alpha=0.35)
        assert abs(x[0] - 1.0) < 0.01
