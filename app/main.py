"""Gesture-Control Desktop — main loop (MediaPipe Tasks API).

Modes (switch with keys 1-4):
  1 VOLUME — open palm: hand height sets master volume; PINCH toggles mute
  2 MEDIA  — PINCH = play/pause, swipe right = next track, left = previous
  3 SCROLL — two fingers up: hand high = scroll up, low = scroll down
  4 SLIDES — swipe right/left = next/prev slide, PINCH = start presentation (F5)

Keys: 1/2/3/4 mode · h help · q quit
Run:  python -m app.main            (camera, VOLUME mode)
      python -m app.main --mode slides
      python -m app.main --selftest (no camera; scripted-event assertions)

Heavy imports (cv2, mediapipe) are lazy so `--selftest` runs anywhere.
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

from . import synth
from .controllers import NullController, best_controller
from .gestures import (
    Gesture, GestureEngine, Landmarks, smooth,
)

MODES = ["VOLUME", "MEDIA", "SCROLL", "SLIDES"]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

# fingertip connections for a light HUD skeleton
SKELETON = [(4, 8), (8, 6), (12, 10), (16, 14), (20, 18),
            (0, 4), (0, 8), (0, 12), (0, 16), (0, 20)]


def now_stale(t0: float, seconds: float = 1.2) -> bool:
    return time.monotonic() - t0 > seconds


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Fetch the HandLandmarker model once (~7.8 MB)."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading hand-landmarker model -> {path.name} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("model ready")
    return path


class App:
    def __init__(self, mode: int, cam_index: int = 0):
        import cv2  # lazy: keeps --selftest dependency-free
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        self._cv2 = cv2
        self._mp = mp
        self.mode = mode
        self.engine = GestureEngine()

        model = ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = 0

        vol, media, scroll, slides = best_controller()
        self.vol = vol or NullController()
        self.media = media or NullController()
        self.scroll = scroll or NullController()
        self.slides = slides or NullController()

        self.cam = cv2.VideoCapture(cam_index)
        if not self.cam.isOpened():
            raise RuntimeError(f"camera {cam_index} not available")
        self.prev_xy: tuple[float, float] | None = None
        self._last_scroll_burst = 0.0
        self.help = False
        self.status = ""
        self.status_time = 0.0
        self._raw_volume = hasattr(self.vol, "_vol")  # real pycaw controller

    # ---------- mode actions ----------
    def flash(self, text: str, now: float):
        self.status = text
        self.status_time = now

    def on_gesture(self, g: Gesture, lm: Landmarks, now: float):
        if now - self.status_time > 1.2:
            self.status = ""

        if self.mode == 0 and g == Gesture.PINCH:            # VOLUME
            self.vol.mute_toggle()
            self.flash("MUTE TOGGLE", now)
        elif self.mode == 1 and g == Gesture.PINCH:          # MEDIA
            self.media.play_pause()
            self.flash("PLAY/PAUSE", now)
        elif self.mode == 2 and g == Gesture.POINT:          # SCROLL (positional)
            y = lm.pts[8][1]
            if y < 0.30:
                self._burst(-1, now)
            elif y > 0.70:
                self._burst(+1, now)
        elif self.mode == 3 and g == Gesture.PINCH:          # SLIDES
            self.slides.start()
            self.flash("START PRESENTATION (F5)", now)

        # VOLUME mode: continuous hand-height control with an open-ish hand
        if self.mode == 0 and self._raw_volume and g in (Gesture.OPEN, Gesture.POINT, Gesture.IDLE):
            self.prev_xy = smooth(self.prev_xy, (0.5, lm.pts[8][1]), alpha=0.25)
            level = 1.0 - min(max((self.prev_xy[1] - 0.25) / 0.5, 0.0), 1.0)
            try:
                self.vol._vol.SetMasterVolumeLevelScalar(level, None)
                self.flash(f"VOLUME {int(level * 100)}%", now)
            except Exception:
                pass

    def _burst(self, direction: int, now: float, gap: float = 0.25):
        if now - self._last_scroll_burst >= gap:
            self._last_scroll_burst = now
            self.scroll.scroll(direction)
            self.flash("SCROLL " + ("DOWN" if direction > 0 else "UP"), now)

    def on_swipe(self, direction: str):
        now = time.monotonic()
        if self.mode == 1:                                   # MEDIA
            if direction == "right":
                self.media.next_track(); self.flash("NEXT TRACK", now)
            else:
                self.media.prev_track(); self.flash("PREV TRACK", now)
        elif self.mode == 3:                                 # SLIDES
            if direction == "right":
                self.slides.next(); self.flash("SLIDE ->", now)
            else:
                self.slides.prev(); self.flash("<- SLIDE", now)

    # ---------- detection ----------
    def detect(self, img_rgb) -> tuple[list | None, Landmarks | None]:
        """Returns (pixel keypoints, normalized Landmarks) for the first hand."""
        cv2, mp = self._cv2, self._mp
        ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
        self._last_ts = ts
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = self.landmarker.detect_for_video(mp_img, ts)

        if not res.hand_landmarks:
            return None, None
        raw = res.hand_landmarks[0]
        h, w = img_rgb.shape[:2]
        lm_px = [(int(p.x * w), int(p.y * h), p.z) for p in raw]
        lm_norm = Landmarks(tuple((p.x, p.y, p.z) for p in raw))
        return lm_px, lm_norm

    # ---------- HUD ----------
    def draw(self, img, lm_px, gesture_label: str):
        cv2 = self._cv2
        h, w = img.shape[:2]
        if lm_px:
            for a, b in SKELETON:
                cv2.line(img, lm_px[a][:2], lm_px[b][:2], (40, 220, 40), 2)
            for x, y, _ in lm_px:
                cv2.circle(img, (x, y), 3, (0, 60, 255), -1)
        cv2.rectangle(img, (0, 0), (w, 34), (20, 20, 20), -1)
        label = "—" if gesture_label in ("", "-") else gesture_label
        cv2.putText(img, f"[{self.mode + 1}] {MODES[self.mode]}  gesture: {label}  {self.status}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (240, 240, 240), 1, cv2.LINE_AA)
        if self.help:
            lines = ["1 volume  2 media  3 scroll  4 slides",
                     "pinch: mute / play-pause / F5    swipe: tracks or slides",
                     "volume: open-palm hand height   scroll: point up/down",
                     "h help   q quit"]
            for i, t in enumerate(lines):
                cv2.putText(img, t, (10, 58 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (200, 255, 200), 1, cv2.LINE_AA)

    # ---------- loop ----------
    def run(self):
        cv2 = self._cv2
        print("Gesture-Control running — keys: 1-4 modes, h help, q quit")
        prev_t = time.monotonic()
        fps = 0.0
        while True:
            ok, img = self.cam.read()
            if not ok:
                continue
            img = cv2.flip(img, 1)  # selfie view
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            lm_px, lm_norm = self.detect(rgb)

            g = None
            if lm_norm is not None:
                for kind, val in self.engine.events(lm_norm, "Right"):
                    if kind == "gesture":
                        g = val
                    elif kind == "swipe":
                        self.on_swipe(val)
                if g is not None:
                    self.on_gesture(g, lm_norm, time.monotonic())

            t = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(t - prev_t, 1e-6))
            prev_t = t
            if now_stale(self.status_time):
                self.status = ""
            self.draw(img, lm_px, g.value if g else "")
            cv2.setWindowTitle("Gesture Control", f"Gesture Control — {fps:.0f} fps")
            cv2.imshow("Gesture Control", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                self.help = not self.help
            if ord("1") <= key <= ord("4"):
                self.mode = key - ord("1")
                self.flash(f"MODE {MODES[self.mode]}", t)
        self.cam.release()
        cv2.destroyAllWindows()


def selftest() -> int:
    """Scripted landmark sequences -> expected events. No camera required."""
    from .gestures import HandFrame, SwipeDetector
    open_palm, fist, pinched = synth.open_palm, synth.fist, synth.pinched

    # classification
    assert HandFrame(open_palm()).gesture == Gesture.OPEN
    assert HandFrame(fist()).gesture == Gesture.FIST
    assert HandFrame(pinched()).gesture == Gesture.PINCH

    # pinch debounce: hold required, fires once, hysteresis before re-arm
    eng = GestureEngine()
    ratio = HandFrame(pinched()).pinch_ratio
    assert eng.pinch.update(ratio, now=10.0) is False        # first sight starts clock
    assert eng.pinch.update(ratio, now=10.2) is True         # hold satisfied -> fire
    assert eng.pinch.update(ratio, now=10.3) is False        # still held, no refire
    assert eng.pinch.update(1.0, now=10.5) is False          # reopened (> open_)
    assert eng.pinch.update(ratio, now=10.6) is False        # fresh press: clock restarts
    assert eng.pinch.update(ratio, now=10.8) is True         # fires again

    # swipe FSM: arm from center band, fire on crossing enter threshold
    s = SwipeDetector()
    assert s.update(0.5, True, now=20.0) is None             # starts tracking
    assert s.update(0.90, True, now=20.2) == SwipeDetector.RIGHT
    assert s.update(0.95, True, now=20.3) is None            # cooldown
    assert s.update(0.95, True, now=21.2) is None            # cooldown over, hand still outside band
    assert s.update(0.5, True, now=21.3) is None             # re-enter center band -> re-arm
    assert s.update(0.20, True, now=21.5) == SwipeDetector.LEFT

    # smoothing
    assert smooth(None, (0.5, 0.5)) == (0.5, 0.5)
    assert smooth((0.0, 0.0), (1.0, 1.0), alpha=0.5) == (0.5, 0.5)

    print("selftest OK — classification, pinch debounce, swipe FSM, smoothing all pass")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Gesture-Control Desktop")
    ap.add_argument("--mode", default="volume", choices=[m.lower() for m in MODES])
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                    help="run scripted-event assertions without a camera")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    App(MODES.index(args.mode.upper()), args.cam).run()


if __name__ == "__main__":
    main()
