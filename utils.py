import numpy as np
import math

# ============================================================
#  utils.py  —  Shared helpers
# ============================================================

def smooth_point(px, py, nx, ny, alpha=0.55):
    """
    Exponential moving average cursor smoother.
    alpha=1.0 → raw (no smoothing)
    alpha=0.3 → heavy smoothing
    """
    if px == 0 and py == 0:
        return float(nx), float(ny)
    return alpha * nx + (1-alpha) * px, alpha * ny + (1-alpha) * py


def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def map_range(val, in_lo, in_hi, out_lo, out_hi):
    ratio = (val - in_lo) / max(1e-9, in_hi - in_lo)
    return out_lo + clamp(ratio, 0.0, 1.0) * (out_hi - out_lo)


def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    """Draw a filled rounded rectangle."""
    import cv2
    x1,y1 = pt1; x2,y2 = pt2
    r = min(radius, (x2-x1)//2, (y2-y1)//2)
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)
    cv2.circle(img, (x1+r, y1+r), r, color, thickness)
    cv2.circle(img, (x2-r, y1+r), r, color, thickness)
    cv2.circle(img, (x1+r, y2-r), r, color, thickness)
    cv2.circle(img, (x2-r, y2-r), r, color, thickness)
