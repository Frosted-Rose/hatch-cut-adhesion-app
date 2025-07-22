
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("Hatch Cut Adhesion Analyzer")

uploaded_file = st.file_uploader("Upload Hatch Cut Test Image", type=["png", "jpg", "jpeg"])
grid_size = st.sidebar.slider("Grid Size Selector", min_value=2, max_value=15, value=10)
sensitivity = st.sidebar.slider("Color Sensitivity (*Do Not Touch*)", min_value=10, max_value=100, value=40)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    reshaped = img_np.reshape(-1, 3)

    # KMeans to find 2 dominant colors
    kmeans = KMeans(n_clusters=2, random_state=42).fit(reshaped)
    colors = np.uint8(kmeans.cluster_centers_)

    # Sidebar color selection with swatches
    st.sidebar.subheader("Detected Colors")
    color_index = st.sidebar.radio(
        "Select Coating Color",
        options=[0, 1],
        format_func=lambda i: f"Color {i+1} - RGB: {tuple(colors[i])}"
    )
    selected_color = colors[color_index]

    # Show swatches
    for i in range(2):
        swatch = np.full((50, 50, 3), colors[i], dtype=np.uint8)
        st.sidebar.image(swatch, caption=f"Color {i+1}")

    # Color thresholding
    lower = np.clip(selected_color - sensitivity, 0, 255)
    upper = np.clip(selected_color + sensitivity, 0, 255)
    mask = cv2.inRange(img_np, lower, upper)

    # Analyze grid
    height, width = mask.shape
    cell_h = height // grid_size
    cell_w = width // grid_size
    failure_count = 0
    total_cells = grid_size * grid_size
    overlay = img_np.copy()

    for i in range(grid_size):
        for j in range(grid_size):
            x1, y1 = j * cell_w, i * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            cell = mask[y1:y2, x1:x2]
            cell_area = cell.size
            coated = cv2.countNonZero(cell)
            if coated / cell_area < 0.5:
                failure_count += 1
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    fail_percent = (failure_count / total_cells) * 100

    # ASTM D3359 grading
    def astm_grade(failure):
        if failure < 5: return "5B"
        elif failure < 15: return "4B"
        elif failure < 35: return "3B"
        elif failure < 65: return "2B"
        elif failure < 85: return "1B"
        else: return "0B"

    # ISO 2409 grading
    def iso_grade(failure):
        if failure < 5: return "0"
        elif failure < 15: return "1"
        elif failure < 35: return "2"
        elif failure < 65: return "3"
        elif failure < 85: return "4"
        else: return "5"

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(img, caption="Original Image", width=400)
    with col_img2:
        st.image(overlay, caption="Detected Failures", channels="RGB", width=400)

    st.subheader("Results")
    st.write(f"**Failure Area:** {fail_percent:.2f}%")
    st.write(f"**ASTM Grade (D3359):** {astm_grade(fail_percent)}")
    st.write(f"**ISO Grade (2409):** {iso_grade(fail_percent)}")
    st.write(":pink[BOB was here]")
