"""
extract_rgb.py — 核心函数
"""
import cv2
import numpy as np

COLOR_NAMES = ["glucose", "leuko", "blood", "protein", "buffering", "ph"]
CLS_BLUE = 0
CLS_COLOR = 1

def mask_info(polygon):
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    w = float(np.max(xs) - np.min(xs))
    h = float(np.max(ys) - np.min(ys))
    return {"cx": cx, "cy": cy, "w": w, "h": h, "bbox": {"w": w, "h": h}, "centroid": {"x": cx, "y": cy}}

def extract_rgb_from_region(img, cx, cy, w, h):
    H, W = img.shape[:2]
    x1 = int(max(0, cx - w / 2))
    y1 = int(max(0, cy - h / 2))
    x2 = int(min(W, cx + w / 2))
    y2 = int(min(H, cy + h / 2))
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return {"r": 0.0, "g": 0.0, "b": 0.0}
    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    mean_rgb = region_rgb.reshape(-1, 3).mean(axis=0)
    return {
        "r": round(float(mean_rgb[0]), 1),
        "g": round(float(mean_rgb[1]), 1),
        "b": round(float(mean_rgb[2]), 1),
    }
