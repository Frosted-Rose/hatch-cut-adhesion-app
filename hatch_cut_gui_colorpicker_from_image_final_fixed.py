
import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Hatch Cut Adhesion Analyzer", layout="wide")
st.title("🎯 Hatch Cut Adhesion Analyzer (Pre-cropped Image Only)")

# Sidebar controls
st.sidebar.header("🛠 Settings")
grid_rows = st.sidebar.slider("Number of Rows", 2, 20, 6)
grid_cols = st.sidebar.slider("Number of Columns", 2, 20, 6)
failure_threshold = st.sidebar.slider("Cell Failure Threshold (%)", 0, 100, 40) / 100.0

coating_rgb = st.sidebar.selectbox(
    "Coating Color",
    options=["Green", "Red", "Blue"],
    index=0
)

# ASTM classification function
# ISO 2409:2020 classification
def get_iso_grade(failure_percent):
    if failure_percent == 0:
        return "0"
    elif failure_percent <= 5:
        return "1"
    elif failure_percent <= 15:
        return "2"
    elif failure_percent <= 35:
        return "3"
    elif failure_percent <= 65:
        return "4"
    else:
        return "5"

def get_astm_grade(failure_percent):
    if failure_percent == 0:
        return "5B"
    elif failure_percent <= 5:
        return "4B"
    elif failure_percent <= 15:
        return "3B"
    elif failure_percent <= 35:
        return "2B"
    elif failure_percent <= 65:
        return "1B"
    else:
        return "0B"

# HSV ranges
color_hsv_ranges = {
    "Green": [(35, 40, 40), (85, 255, 255)],
    "Red": [(0, 50, 50), (10, 255, 255)],
    "Blue": [(90, 50, 50), (130, 255, 255)],
}

uploaded_file = st.file_uploader("Upload a cropped grid image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
img = Image.open(uploaded_file).convert("RGB")
    img_rgb = img

    st.subheader("🔍 Select Coating Color from Image")

    # Convert image to displayable format and allow clicking
    img_array = np.array(img_rgb)

    st.write("Click on the image to pick coating color:")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        background_image=img_rgb,
        update_streamlit=True,
        height=img.height,
        width=img.width,
        drawing_mode="point",
        point_display_radius=5,
        key="color_picker_canvas"
    )

    if canvas_result.json_data and canvas_result.json_data["objects"]:
        point = canvas_result.json_data["objects"][-1]
        cx, cy = int(point["left"]), int(point["top"])
        picked_color = tuple(img_array[cy, cx])
        st.success(f"Selected Coating Color: {picked_color}")
        coating_rgb = picked_color
    else:
        coating_rgb = (0, 255, 0)  # Default green fallback
        st.info("No color selected. Using default green: (0, 255, 0)")
    img_array = np.array(img)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    lower, upper = color_hsv_ranges[coating_rgb]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    failure_mask = cv2.bitwise_not(mask)

    h, w = failure_mask.shape
    cell_h, cell_w = h // grid_rows, w // grid_cols
    failed_cells = 0
    cell_map = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    for row in range(grid_rows):
        for col in range(grid_cols):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell = failure_mask[y1:y2, x1:x2]
            if np.sum(cell > 0) / (cell.shape[0] * cell.shape[1]) > failure_threshold:
                failed_cells += 1
                cell_map[row, col] = 1

    failure_pct = failed_cells / (grid_rows * grid_cols) * 100
    astm_grade = get_astm_grade(failure_pct)
    overlay = img_array.copy()

    for row in range(grid_rows):
        for col in range(grid_cols):
            if cell_map[row, col]:
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    st.subheader(f"📊 Adhesion Failure: {failure_pct:.2f}%")
    st.markdown(f"**🧪 ASTM Classification: {astm_grade}**")
    st.markdown(f"**📘 ISO 2409 Classification: {get_iso_grade(failure_pct)}**")

    col1, col2, col3 = st.columns(3)
    col1.image(img, caption="Uploaded Image")
    col2.image(failure_mask, caption="Failure Mask", channels="GRAY")
    col3.image(overlay, caption="Failed Cells Overlay")

    st.markdown("""
    ---

    ### 🧪 ASTM D3359 & ISO 2409:2020 Classification Table

    | % Area Affected | ISO Grade | ASTM Grade | Description |
    |------------------|------------|--------------|-------------|
    | 0%               | 0          | 5B           | Edges are smooth; no coating detached |
    | ≤ 5%             | 1          | 4B           | Slight flaking at cuts/intersections |
    | ≤ 15%            | 2          | 3B           | Moderate flaking at edges/intersections |
    | ≤ 35%            | 3          | 2B           | Flaking in ribbons or partial square detachment |
    | ≤ 65%            | 4          | 1B           | Large flaking or full square detachment |
    | > 65%            | 5          | 0B           | Severe flaking beyond classification 1B/4 |

    """)