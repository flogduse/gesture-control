# Gesture-Control Desktop

Control your PC with bare hands, through your webcam. MediaPipe hand tracking
drives a pure-Python gesture engine that types media keys, moves the system
volume and drives presentations — no extra hardware.

Flagship Project #2 on the portfolio roadmap (upgrades the earlier
Synth-Vision experiment into a real desktop app).

## Gestures

| Gesture | VOLUME mode | MEDIA mode | SCROLL mode | SLIDES mode |
|---|---|---|---|---|
| **Open palm** (4 fingers) | hand height = volume level | — | — | — |
| **Pinch** (thumb+index) | mute toggle | play/pause | — | start presentation (F5) |
| **Two-finger swipe** L/R | — | next/prev track | — | next/prev slide |
| **Point** (index only) | — | — | hand high/low = scroll up/down | — |

Keys: `1-4` switch modes · `h` help overlay · `q` quit.
All thresholds are ratios of the palm size, so distance to the camera doesn't matter.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m app.main --selftest   # verify the engine without a camera
python -m app.main              # run (VOLUME mode) — needs a webcam
python -m app.main --mode slides
```

The HandLandmarker model (~8 MB) auto-downloads to `models/` on first run.
Uses the current **MediaPipe Tasks API** (0.10.30+ / 1.x compatible — the legacy
`mp.solutions` API no longer exists in Py3.13 builds).

Press `h` inside the app for the on-screen cheat sheet.

Windows notes: pycaw controls the master volume via Core Audio; if pycaw or
pynput fail to load, the app still runs in HUD-only mode (gestures detected,
no OS actions). MediaPipe handles one hand at a time by design.

## Architecture

```
webcam ─► OpenCV frames ─► MediaPipe Hands (21 landmarks + handedness)
                               │
                               ▼
                   app/gestures.py  (PURE math, fully unit-tested)
                     HandFrame ─ classifies one frame: pinch/open/fist/point/idle
                     PinchDebouncer ─ hysteresis + hold time -> one event per press
                     SwipeDetector ─ FSM: OPEN → TRACKING → COOLDOWN
                               │
                               ▼
                   app/main.py  (mode dispatch, status HUD, fps)
                               │
                               ▼
                   app/controllers.py  (pycaw volume · pynput keys/scroll · Null for tests)
```

**Design decisions worth defending in an interview:**

- **Pure engine / impure shell.** All gesture math is free of I/O, so 30+
  pytest cases run in milliseconds without a camera or Windows APIs. The
  camera loop is a thin adapter.
- **Palm-scale normalization.** Every threshold is a ratio of
  wrist→middle-MCP distance, making gestures work at 30 cm and 1.5 m alike.
- **Hysteresis everywhere.** Pinch has separate close/open thresholds plus a
  minimum hold; swipes have an edge-enter band and a cooldown — physical
  hands wobble, and naive thresholding machine-guns actions.
- **Graceful degradation.** Missing OS libs disable features, not the app.

## Testing

```bash
python -m pytest tests/ -v      # pure engine: classification, debounce, FSM
python -m app.main --selftest   # quick scripted assertions, no pytest needed
```

## Roadmap

- [ ] Cursor mode: pinch-drag to move the mouse
- [ ] Configurable gesture→action bindings (TOML)
- [ ] Demo GIF for the portfolio Work card
