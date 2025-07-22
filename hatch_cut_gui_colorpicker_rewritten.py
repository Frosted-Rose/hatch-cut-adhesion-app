
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import io

st.set_page_config(layout="wide")
st.title("Hatch Cut Adhesion Failure Detector with Custom Color and Grid Cell Analysis")

uploaded_file = st.file_uploader("Upload hatch cut image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

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

            st.subheader("Step 2: Set Grid Size")
            cols = st.number_input("Number of columns", min_value=2, max_value=20, value=6)
            rows = st.number_input("Number of rows", min_value=2, max_value=20, value=6)

            st.subheader("Step 3: Detection Result")
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
            st.write(f"Adhesion Failure by Grid Cell Count: {failure_pct:.2f}% ({failure_count} of {total_cells} cells failed)")

            st.image(overlay_img, channels="RGB", caption="Grid Cell Analysis")
        else:
            st.error("Clicked point is outside image bounds.")
    else:
        st.warning("Please click on the coating area in the image above to begin analysis.")
else:
    st.info("Upload an image to begin.")
