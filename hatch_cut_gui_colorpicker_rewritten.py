import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd
import io
import datetime
import xlsxwriter

# =============================
# App Config & Styling
# =============================
st.set_page_config(page_title="Hatch Cut Adhesion Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .small-muted { color: #6c757d; font-size: 0.9rem; }
    .metric-box { padding: 0.6rem 0.8rem; border: 1px solid #e9ecef; border-radius: 0.5rem; background: #fcfcfc; }
    .section-title { margin-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Hatch Cut Adhesion Analyzer+NICHOLAS CHU")
st.caption("Batch-process hatch cut images, visualize failure cells, and export a clean Excel report.")

# =============================
# Helpers
# =============================
def astm_grade(failure_pct: float) -> str:
    if failure_pct < 5: return "5B"
    elif failure_pct < 15: return "4B"
    elif failure_pct < 35: return "3B"
    elif failure_pct < 65: return "2B"
    elif failure_pct < 85: return "1B"
    else: return "0B"

def iso_grade(failure_pct: float) -> str:
    if failure_pct < 5: return "0"
    elif failure_pct < 15: return "1"
    elif failure_pct < 35: return "2"
    elif failure_pct < 65: return "3"
    elif failure_pct < 85: return "4"
    else: return "5"

@st.cache_data(show_spinner=False)
def kmeans_colors(img_np: np.ndarray, n_clusters: int = 2, sample_px: int = 40000, seed: int = 42):
    """Return KMeans cluster centers (RGB) from a random pixel sample for speed."""
    flat = img_np.reshape(-1, 3)
    if sample_px and sample_px < flat.shape[0]:
        idx = np.random.RandomState(seed).choice(flat.shape[0], size=sample_px, replace=False)
        data = flat[idx]
    else:
        data = flat
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(data)
    centers = np.uint8(kmeans.cluster_centers_)
    return centers

def luminance(rgb):
    r, g, b = rgb
    return 0.2126*r + 0.7152*g + 0.0722*b

def build_mask(img_np: np.ndarray, color_rgb: np.ndarray, sensitivity: int) -> np.ndarray:
    lower = np.clip(color_rgb - sensitivity, 0, 255).astype(np.uint8)
    upper = np.clip(color_rgb + sensitivity, 0, 255).astype(np.uint8)
    mask = cv2.inRange(img_np, lower, upper)
    return mask

def analyze_grid(mask: np.ndarray, grid_size: int, fail_threshold: float = 0.5):
    H, W = mask.shape
    cell_h = H // grid_size
    cell_w = W // grid_size
    failure_cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i*cell_h, (i+1)*cell_h if i < grid_size-1 else H
            x1, x2 = j*cell_w, (j+1)*cell_w if j < grid_size-1 else W
            cell = mask[y1:y2, x1:x2]
            fill_ratio = cv2.countNonZero(cell) / float(cell.size)
            failed = (fill_ratio < (1.0 - fail_threshold))  # low coating coverage -> fail
            if failed:
                failure_cells.append((x1, y1, x2, y2))
    total_cells = grid_size * grid_size
    return failure_cells, total_cells

def draw_overlay(img_np: np.ndarray, failure_cells, grid_size: int, alpha: float = 0.35, show_grid: bool = True):
    overlay = img_np.copy()
    # Red rectangles for failed cells (filled)
    for (x1, y1, x2, y2) in failure_cells:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), thickness=-1)
    blended = cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0)

    if show_grid:
        H, W, _ = img_np.shape
        cell_h = H // grid_size
        cell_w = W // grid_size
        grid_img = blended.copy()
        # vertical lines
        for j in range(1, grid_size):
            x = j * cell_w
            cv2.line(grid_img, (x, 0), (x, H), (0, 0, 0), 1)
        # horizontal lines
        for i in range(1, grid_size):
            y = i * cell_h
            cv2.line(grid_img, (0, y), (W, y), (0, 0, 0), 1)
        return grid_img
    return blended

def to_thumbnail(pil_img: Image.Image, max_side=220) -> bytes:
    th = pil_img.copy()
    th.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    th.save(buf, format="PNG")
    return buf.getvalue()

def create_excel_report(df: pd.DataFrame, thumb_map: dict, overlay_map: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Results", startrow=4)
        wb = writer.book
        ws = writer.sheets["Results"]

        # Title & metadata
        title_fmt = wb.add_format({"bold": True, "font_size": 14})
        meta_fmt = wb.add_format({"font_size": 10, "italic": True, "font_color": "#666666"})
        ws.write(0, 0, "Hatch Cut Adhesion Analyzer — Pro", title_fmt)
        ws.write(1, 0, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_fmt)

        # Header styling
        header_fmt = wb.add_format({"bold": True, "bg_color": "#F0F0F0", "border": 1})
        for col, col_name in enum_
