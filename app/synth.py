"""Synthetic hand landmark builders — test fixtures AND --selftest input.

MediaPipe Hands: 21 landmarks,
  0 wrist · 1-4 thumb (CMC,MCP,IP,TIP) · 5-8 index (MCP,PIP,DIP,TIP)
  9-12 middle · 13-16 ring · 17-20 pinky.
y grows DOWN (image convention), so "finger up" means tip y < pip y.
"""
from __future__ import annotations

from .gestures import Landmarks

BASES = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19]  # 11 non-tip/pip joints


def open_palm(x: float = 0.5, y: float = 0.5) -> Landmarks:
    """All four fingers extended, thumb away from the index."""
    pts = [None] * 21
    pts[0] = (x, y, 0.0)                                  # wrist
    pts[1] = pts[2] = pts[3] = (x - 0.06, y - 0.05, 0.0)  # thumb base joints
    pts[4] = (x - 0.12, y - 0.20, 0.0)                    # thumb tip (far from index)
    for tip, pip, ext in [(8, 6, 0.40), (12, 10, 0.38), (16, 14, 0.35), (20, 18, 0.30)]:
        pts[tip] = (x, y - ext, 0.0)
        pts[pip] = (x, y - ext + 0.18, 0.0)
    for b in BASES:
        pts[b] = (x, y - 0.02, 0.0)
    return Landmarks(tuple(pts))


def fist(x: float = 0.5, y: float = 0.5) -> Landmarks:
    """All four fingertips curled below their PIP joints."""
    pts = [None] * 21
    pts[0] = (x, y, 0.0)
    pts[1] = pts[2] = pts[3] = (x - 0.05, y, 0.0)
    pts[4] = (x - 0.10, y - 0.02, 0.0)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        pts[tip] = (x, y + 0.05, 0.0)
        pts[pip] = (x, y + 0.02, 0.0)
    for b in BASES:
        pts[b] = (x, y - 0.02, 0.0)
    return Landmarks(tuple(pts))


def pinched(x: float = 0.5, y: float = 0.5) -> Landmarks:
    """Open palm with the thumb tip moved onto the index tip."""
    pts = list(open_palm(x, y).pts)
    pts[4] = pts[8]
    return Landmarks(tuple(pts))


def pointing(x: float = 0.5, y: float = 0.5) -> Landmarks:
    """Index extended, middle/ring/pinky curled."""
    pts = list(open_palm(x, y).pts)
    for tip, pip in [(12, 10), (16, 14), (20, 18)]:
        pts[tip] = (pts[pip][0], pts[pip][1] + 0.03, 0.0)
    return Landmarks(tuple(pts))


def two_fingers(x: float = 0.5, y: float = 0.5) -> Landmarks:
    """Index + middle extended (the swipe pose)."""
    pts = list(open_palm(x, y).pts)
    for tip, pip in [(16, 14), (20, 18)]:
        pts[tip] = (pts[pip][0], pts[pip][1] + 0.03, 0.0)
    return Landmarks(tuple(pts))
