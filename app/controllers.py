"""OS action layer. Each controller is a tiny class with one method per action
so the main loop stays thin and tests can inject the NullController.

Windows: pycaw for volume (Core Audio), pynput for media keys/scroll.
All imports are lazy/optional — the app runs in HUD-only mode if missing.
"""
from __future__ import annotations

from ctypes import POINTER, cast

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from pynput.keyboard import Controller as KbController, Key


class VolumeController:
    """System master volume via the Windows Core Audio endpoint."""

    def __init__(self):
        device = AudioUtilities.GetSpeakers()
        interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self._vol = cast(interface, POINTER(IAudioEndpointVolume))
        self.step = 0.04  # 4% per burst tick

    def nudge(self, direction: int):
        current = self._vol.GetMasterVolumeLevelScalar()
        target = min(1.0, max(0.0, current + direction * self.step))
        self._vol.SetMasterVolumeLevelScalar(target, None)

    def mute_toggle(self):
        self._vol.SetMute(0 if self._vol.GetMute() else 1, None)

    @property
    def level(self) -> float:
        return self._vol.GetMasterVolumeLevelScalar()


class MediaController:
    def __init__(self):
        self._kb = KbController()

    def play_pause(self):
        self._kb.tap(Key.media_play_pause)

    def next_track(self):
        self._kb.tap(Key.media_next)

    def prev_track(self):
        self._kb.tap(Key.media_previous)


class ScrollController:
    def __init__(self):
        self._kb = KbController()
        self._mouse = None  # created lazily; mouse scroll needs pynput.mouse

    def scroll(self, direction: int, ticks: int = 3):
        try:
            from pynput.mouse import Controller as MouseController
            if self._mouse is None:
                self._mouse = MouseController()
            self._mouse.scroll(0, direction * ticks)
        except Exception:
            key = Key.page_down if direction > 0 else Key.page_up
            self._kb.tap(key)


class SlideController:
    """Presentation control: F5 start/stop, arrows to navigate."""

    def __init__(self):
        self._kb = KbController()

    def start(self):
        self._kb.tap(Key.f5)

    def next(self):
        self._kb.press(Key.right)
        self._kb.release(Key.right)

    def prev(self):
        self._kb.press(Key.left)
        self._kb.release(Key.left)

    def black(self):
        self._kb.tap("b")


class NullController:
    """Records actions instead of touching the OS — used in tests and
    as a graceful fallback when pycaw/pynput are unavailable."""

    def __init__(self):
        self.calls: list[tuple] = []

    def nudge(self, direction: int):
        self.calls.append(("nudge", direction))

    def mute_toggle(self):
        self.calls.append(("mute_toggle",))

    def play_pause(self):
        self.calls.append(("play_pause",))

    def next_track(self):
        self.calls.append(("next_track",))

    def prev_track(self):
        self.calls.append(("prev_track",))

    def scroll(self, direction: int, ticks: int = 3):
        self.calls.append(("scroll", direction, ticks))

    def start(self):
        self.calls.append(("start",))

    def next(self):
        self.calls.append(("next",))

    def prev(self):
        self.calls.append(("prev",))

    def black(self):
        self.calls.append(("black",))


def best_controller():
    """Return (VolumeController|None, MediaController|None, ...tuple) —
    whatever loads; callers degrade feature-by-feature."""
    try:
        vol = VolumeController()
    except Exception:
        vol = None
    try:
        media = MediaController()
        scroll = ScrollController()
        slides = SlideController()
    except Exception:
        media = scroll = slides = None
    return vol, media, scroll, slides
