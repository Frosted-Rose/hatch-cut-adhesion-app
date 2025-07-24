import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import base64
import pandas as pd
import io
import xlsxwriter



# === Page config ===
st.set_page_config(layout="wide")
st.title("Hatch Cut Adhesion Analyzer")
st.divider()

# === Sidebar Settings ===
st.sidebar.title("Settings")
st.sidebar.divider()
uploaded_files = st.file_uploader("Upload one or more Hatch Cut Test Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
grid_size = st.sidebar.slider("Grid Size Selector", 2, 10, 10)
with st.sidebar.expander("Advanced Settings"):
    sensitivity = st.slider("Color Sensitivity", 10, 100, 40)

results = []
image_thumbnails = {}

# === ASTM & ISO grading functions ===
def astm_grade(failure):
    if failure < 5: return "5B"
    elif failure < 15: return "4B"
    elif failure < 35: return "3B"
    elif failure < 65: return "2B"
    elif failure < 85: return "1B"
    else: return "0B"

def iso_grade(failure):
    if failure < 5: return "0"
    elif failure < 15: return "1"
    elif failure < 35: return "2"
    elif failure < 65: return "3"
    elif failure < 85: return "4"
    else: return "5"

# === Image Processing ===
chosen_color = None

if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)
        reshaped = img_np.reshape(-1, 3)

        # KMeans color detection
        kmeans = KMeans(n_clusters=2, random_state=42).fit(reshaped)
        colors = np.uint8(kmeans.cluster_centers_)

        # Only for first image, allow user to pick coating color
        if idx == 0:
            st.sidebar.subheader(f"{file.name} - Detected Colors")
            color_index = st.sidebar.radio(
                "Select Coating Color",
                options=[0, 1],
                index=0,
                format_func=lambda i: f"Color {i+1} - RGB: {tuple(colors[i])}",
                key="coating_color_selector"
            )
            selected_color = colors[color_index]
            chosen_color = selected_color
            for i in range(2):
                swatch = np.full((50, 50, 3), colors[i], dtype=np.uint8)
                st.sidebar.image(swatch, caption=f"Color {i+1}", width=50)
        else:
            selected_color = chosen_color

        # Threshold and mask
        lower = np.clip(selected_color - sensitivity, 0, 255)
        upper = np.clip(selected_color + sensitivity, 0, 255)
        mask = cv2.inRange(img_np, lower, upper)

        # Grid analysis
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
                if cv2.countNonZero(cell) / cell.size < 0.5:
                    failure_count += 1
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

        fail_percent = (failure_count / total_cells) * 100

        # Save result
        results.append({
            "Filename": file.name,
            "Failure %": round(fail_percent, 2),
            "ASTM Grade": astm_grade(fail_percent),
            "ISO Grade": iso_grade(fail_percent)
        })

        # Save image preview for Excel
        thumbnail = img.copy()
        thumbnail.thumbnail((100, 100))
        buf = io.BytesIO()
        thumbnail.save(buf, format="PNG")
        image_thumbnails[file.name] = buf.getvalue()

        # Show UI
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption=f"{file.name} (Original)", width=300)
        with col2:
            st.image(overlay, caption="Detected Failures", channels="RGB", width=300)

    # === Summary Table ===
    df = pd.DataFrame(results)
    st.subheader("Summary Table")
    st.dataframe(df)

    # === Excel Export ===
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Results", startrow=1)
        workbook = writer.book
        worksheet = writer.sheets["Results"]

        # Headers & column format
        header_format = workbook.add_format({'bold': True, 'bg_color': '#F0F0F0'})
        for col_num, value in enumerate(df.columns):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 20)

        # Insert images
        for row_num, file in enumerate(uploaded_files, start=1):
            image_data = image_thumbnails[file.name]
            image_io = io.BytesIO(image_data)
            worksheet.insert_image(row_num, len(df.columns), file.name, {
                'image_data': image_io,
                'x_scale': 0.5,
                'y_scale': 0.5
            })

    st.download_button(
        label=":floppy_disk: Download Results (Excel)",
        data=output.getvalue(),
        file_name="hatch_cut_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success("Analysis complete.")
    st.divider()
    st.caption("Bob Was Here")
