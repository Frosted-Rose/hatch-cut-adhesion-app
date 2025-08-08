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
st.set_page_config(page_title="Hatch Cut Adhesion Analyzer (Pro)", layout="wide")
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
st.title("Hatch Cut Adhesion Analyzer — Pro")
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
        for col, col_name in enumerate(df.columns):
            ws.write(4, col, col_name, header_fmt)
            ws.set_column(col, col, 20)

        # Thumbnails (Original, Overlay)
        base_col = len(df.columns)
        ws.write(4, base_col, "Original", header_fmt)
        ws.write(4, base_col + 1, "Overlay", header_fmt)
        ws.set_column(base_col, base_col + 1, 18)

        for i, row in df.iterrows():
            fn = row["Filename"]
            r = 5 + i
            if fn in thumb_map:
                ws.insert_image(r, base_col, fn, {"image_data": io.BytesIO(thumb_map[fn]), "x_scale": 0.7, "y_scale": 0.7})
            if fn in overlay_map:
                ws.insert_image(r, base_col + 1, f"{fn}_overlay", {"image_data": io.BytesIO(overlay_map[fn]), "x_scale": 0.7, "y_scale": 0.7})
    return output.getvalue()

# =============================
# Sidebar Controls
# =============================
st.sidebar.header("Settings")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more hatch cut images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)
grid_size = st.sidebar.slider("Grid Size", 2, 20, 10)
sensitivity = st.sidebar.slider("Color Sensitivity (±RGB)", 5, 100, 40)
fail_threshold = st.sidebar.slider(
    "Coverage Pass Threshold",
    0.5, 0.95, 0.5,
    help="Minimum coating coverage fraction per cell to pass."
)
alpha = st.sidebar.slider("Overlay Opacity", 0.1, 0.9, 0.35)
show_grid = st.sidebar.checkbox("Show Grid Lines", True)

st.sidebar.subheader("Coating Color Selection")
color_mode = st.sidebar.radio("Mode", ["Auto (darker cluster)", "Manual"], index=0)
manual_color = st.sidebar.color_picker("Manual Color (RGB)", value="#555555")

st.sidebar.caption("Tip: Use a consistent lighting setup for best KMeans color separation.")

# =============================
# Main Pipeline
# =============================
if uploaded_files:
    results = []
    thumb_map = {}
    overlay_map = {}

    # Determine coating color from the first image (or manual)
    first_img = Image.open(uploaded_files[0]).convert("RGB")
    first_np = np.array(first_img)

    centers = kmeans_colors(first_np, n_clusters=2)
    if color_mode == "Auto (darker cluster)":
        lum0, lum1 = luminance(centers[0]), luminance(centers[1])
        coating_rgb = centers[0] if lum0 < lum1 else centers[1]
    else:
        hexval = manual_color.lstrip("#")
        coating_rgb = np.array([int(hexval[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)

    with st.expander("Detected Clusters (from first image)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            sw1 = np.full((50, 50, 3), centers[0], dtype=np.uint8)
            st.image(sw1, caption=f"Cluster 1 RGB: {tuple(int(x) for x in centers[0])}", width=60)
        with c2:
            sw2 = np.full((50, 50, 3), centers[1], dtype=np.uint8)
            st.image(sw2, caption=f"Cluster 2 RGB: {tuple(int(x) for x in centers[1])}", width=60)
        st.markdown(f"**Using coating color:** {tuple(int(x) for x in coating_rgb)}")

    # Process each image
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)

        mask = build_mask(img_np, coating_rgb, sensitivity)
        failure_cells, total_cells = analyze_grid(mask, grid_size, fail_threshold=fail_threshold)
        fail_pct = (len(failure_cells) / total_cells) * 100.0

        overlay_img = draw_overlay(img_np, failure_cells, grid_size, alpha=alpha, show_grid=show_grid)

        # Show visuals (no use_container_width to support Streamlit 1.35)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img_np, caption=f"{file.name} — Original")
        with col2:
            st.image(overlay_img, caption=f"{file.name} — Failure Overlay")

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Failure %", f"{fail_pct:.2f}%")
        m2.metric("ASTM Grade", astm_grade(fail_pct))
        m3.metric("ISO Grade", iso_grade(fail_pct))
        m4.metric("Failed Cells", f"{len(failure_cells)}/{total_cells}")

        # Save thumbnails for Excel
        thumb_map[file.name] = to_thumbnail(img)
        overlay_pil = Image.fromarray(overlay_img)
        overlay_map[file.name] = to_thumbnail(overlay_pil)

        # Aggregate results
        results.append({
            "Filename": file.name,
            "Failure %": round(fail_pct, 2),
            "ASTM Grade": astm_grade(fail_pct),
            "ISO Grade": iso_grade(fail_pct),
            "Grid Size": grid_size,
            "Sensitivity (±RGB)": sensitivity,
            "Coverage Threshold": fail_threshold,
        })

        st.divider()

    # Summary Table & Export
    df = pd.DataFrame(results)
    st.subheader("Summary")
    st.dataframe(df)  # compatible with Streamlit 1.35.0

    excel_bytes = create_excel_report(df, thumb_map, overlay_map)
    st.download_button(
        label="💾 Download Excel Report",
        data=excel_bytes,
        file_name="hatch_cut_results_pro.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.caption("Done. Review the overlay to ensure the selected coating color and sensitivity isolate coating regions correctly.")
else:
    st.info("👆 Upload one or more images to begin. PNG or JPEG are supported.")
