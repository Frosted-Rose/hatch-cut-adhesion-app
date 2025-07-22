
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("Hatch Cut Adhesion Failure Detector with Auto Color Detection")

uploaded_file = st.file_uploader("Upload hatch cut image", type=["png", "jpg", "jpeg"])

def get_dominant_color(image, k=3):
    img_small = cv2.resize(image, (64, 64))  # Reduce size for speed
    img_flat = img_small.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(img_flat)
    cluster_centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    dominant_color = cluster_centers[np.argmax(counts)]
    return dominant_color

standard_colors = {
    "Green": np.array([0, 128, 0]),
    "Red": np.array([200, 0, 0]),
    "Blue": np.array([0, 0, 200]),
    "Gray": np.array([128, 128, 128]),
    "White": np.array([255, 255, 255]),
    "Yellow": np.array([200, 200, 0]),
}

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    dominant_color = get_dominant_color(img_np)
    st.sidebar.subheader("Detected coating color")
    st.sidebar.color_picker("Auto Detected (shown only)", value="#{:02x}{:02x}{:02x}".format(*dominant_color), label_visibility="collapsed")

    use_manual = st.sidebar.checkbox("Override with standard color")

    if use_manual:
        selected_color_name = st.sidebar.selectbox("Choose standard color", list(standard_colors.keys()))
        coating_color = standard_colors[selected_color_name]
    else:
        coating_color = dominant_color

    st.sidebar.header("Grid Settings")
    cols = st.sidebar.slider("Grid columns", min_value=2, max_value=20, value=6)
    rows = st.sidebar.slider("Grid rows", min_value=2, max_value=20, value=6)

    st.image(img, caption="Original Image", use_column_width=True)

    color_lower = np.clip(coating_color - 30, 0, 255)
    color_upper = np.clip(coating_color + 30, 0, 255)
    mask = cv2.inRange(img_np, color_lower, color_upper)

    h, w = img_np.shape[:2]
    cell_h, cell_w = h // rows, w // cols

    failure_count = 0
    total_cells = rows * cols
    overlay_img = img_np.copy()

    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell_mask = mask[y1:y2, x1:x2]
            cell_area = cell_mask.size
            coated_area = cv2.countNonZero(cell_mask)
            if coated_area / cell_area < 0.5:
                failure_count += 1
                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    failure_pct = (failure_count / total_cells) * 100
    st.image(overlay_img, channels="RGB", caption="Detected Failure Overlay")
    st.write(f"**Adhesion Failure:** {failure_pct:.2f}% ({failure_count} of {total_cells} cells failed)")

    st.subheader("Grading")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ASTM D3359 Grading")
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
        st.markdown("### ISO 2409:2020 Grading")
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
    st.info("Upload an image to begin.")
