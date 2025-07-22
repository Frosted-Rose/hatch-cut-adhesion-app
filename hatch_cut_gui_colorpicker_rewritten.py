
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2

st.set_page_config(layout="wide")
st.title("Hatch Cut Adhesion Failure Detector")

uploaded_file = st.file_uploader("Upload hatch cut image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.image(img, caption="Original Image", use_column_width=True)

    with st.sidebar:
        st.header("Grid Settings")
        cols = st.slider("Grid columns", min_value=2, max_value=20, value=6)
        rows = st.slider("Grid rows", min_value=2, max_value=20, value=6)

    st.subheader("Step 1: Click on the coating area to select coating color")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=10,
        background_image=img,
        update_streamlit=True,
        height=img_np.shape[0],
        width=img_np.shape[1],
        drawing_mode="point",
        key="canvas",
    )

    if canvas_result.json_data and canvas_result.json_data["objects"]:
        last_point = canvas_result.json_data["objects"][-1]
        x, y = int(last_point["left"]), int(last_point["top"])
        if 0 <= y < img_np.shape[0] and 0 <= x < img_np.shape[1]:
            picked_color = img_np[y, x]
            st.success(f"Selected coating color: {picked_color.tolist()}")

            st.subheader("Step 2: Detection Result")
            color_lower = np.clip(picked_color - 30, 0, 255)
            color_upper = np.clip(picked_color + 30, 0, 255)

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

            st.write(f"Adhesion Failure by Grid Cell Count: **{failure_pct:.2f}%**")
            col_right.image(overlay_img, channels="RGB", caption="Detected Failure Overlay")

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
            st.error("Clicked point is outside image bounds.")
    else:
        st.warning("Please click on the coating area in the image above to begin analysis.")
else:
    st.info("Upload an image to begin.")
