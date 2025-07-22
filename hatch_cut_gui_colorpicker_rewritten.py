
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2

st.set_page_config(layout="wide")
st.title("Hatch Cut Adhesion Failure Detector")

# Upload image
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

            # Process the image to detect non-coating areas
            st.subheader("Step 2: Detection Result")
            color_lower = np.clip(picked_color - 30, 0, 255)
            color_upper = np.clip(picked_color + 30, 0, 255)

            mask = cv2.inRange(img_np, color_lower, color_upper)
            coating_area = cv2.countNonZero(mask)
            total_area = img_np.shape[0] * img_np.shape[1]
            failure_area = total_area - coating_area
            failure_pct = (failure_area / total_area) * 100

            st.write(f"Adhesion Failure Area: {failure_pct:.2f}%")

            result_img = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)
            result_img[mask == 0] = [0, 0, 255]  # Highlight failure in red
            st.image(result_img, channels="BGR", caption="Detected Failures Highlighted in Red")
        else:
            st.error("Clicked point is outside image bounds.")
    else:
        st.warning("Please click on the coating area in the image above to begin analysis.")
else:
    st.info("Upload an image to begin.")
