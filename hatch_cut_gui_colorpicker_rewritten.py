
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("Hatch Cut Adhesion Detector (Dual Color Auto-Detect)")

uploaded_file = st.file_uploader("Upload hatch cut image", type=["png", "jpg", "jpeg"])

def get_two_main_colors(image_np):
    img_small = cv2.resize(image_np, (64, 64))
    data = img_small.reshape((-1, 3))
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
    centers = kmeans.cluster_centers_.astype(int)
    return centers

def get_astm_grade(pct):
    if pct <= 0:
        return "5B"
    elif pct <= 5:
        return "4B"
    elif pct <= 15:
        return "3B"
    elif pct <= 35:
        return "2B"
    elif pct <= 65:
        return "1B"
    else:
        return "0B"

def get_iso_grade(pct):
    if pct <= 0:
        return "0"
    elif pct <= 5:
        return "1"
    elif pct <= 15:
        return "2"
    elif pct <= 30:
        return "3"
    elif pct <= 65:
        return "4"
    else:
        return "5"

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    colors = get_two_main_colors(img_np)

    st.sidebar.header("Choose Coating Color")
    for i, color in enumerate(colors):
        st.sidebar.markdown(f"**Color {i + 1}:**")
        st.sidebar.markdown(f'<div style="width:100%%;height:30px;background-color:rgb({color[0]},{color[1]},{color[2]});border:1px solid #000;"></div>', unsafe_allow_html=True)

    selected_index = st.sidebar.radio("Select the color that represents the **coating**", [0, 1], index=0)
    coating_color = colors[selected_index]

    st.sidebar.subheader("Grid Size")
    cols = st.sidebar.slider("Grid columns", 2, 20, 6)
    rows = st.sidebar.slider("Grid rows", 2, 20, 6)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(img, caption="Original Image", width=400)
    with col_img2:
        st.image(overlay_img, caption="Detected Failures Overlay", channels="RGB", width=400)

    color_lower = np.clip(coating_color - 30, 0, 255)
    color_upper = np.clip(coating_color + 30, 0, 255)
    mask = cv2.inRange(img_np, color_lower, color_upper)

    h, w = img_np.shape[:2]
    cell_h, cell_w = h // rows, w // cols

    failure_count = 0
    total_cells = rows * cols
    overlay = img_np.copy()

    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = mask[y1:y2, x1:x2]
            ratio = cv2.countNonZero(cell) / cell.size
            if ratio < 0.5:
                failure_count += 1
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)

    fail_pct = (failure_count / total_cells) * 100
    astm = get_astm_grade(fail_pct)
    iso = get_iso_grade(fail_pct)

    
    st.write(f"### **Adhesion Failure: {fail_pct:.2f}%**")
    st.write(f"**ASTM D3359 Grade:** {astm}   **ISO 2409:2020 Class:** {iso}")

    with st.expander("Show Grading Table"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ASTM D3359")
            st.markdown("""
            | Rating | Description |
            |--------|-------------|
            | 5B     | 0% area removed |
            | 4B     | < 5% area removed |
            | 3B     | 5–15% area removed |
            | 2B     | 15–35% area removed |
            | 1B     | 35–65% area removed |
            | 0B     | > 65% area removed |
            """)
        with col2:
            st.markdown("### ISO 2409:2020")
            st.markdown("""
            | Class | Description |
            |-------|-------------|
            | 0     | 0% detached |
            | 1     | < 5% detached |
            | 2     | 5–15% detached |
            | 3     | 15–30% detached |
            | 4     | 30–65% detached |
            | 5     | > 65% detached |
            """)
else:
    st.info("Upload a hatch cut image to begin.")
