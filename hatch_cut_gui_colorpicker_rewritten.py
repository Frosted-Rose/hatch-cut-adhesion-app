import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import base64
import pandas as pd
import io
import xlsxwriter

# Set layout and title
st.set_page_config(layout="wide")
st.title("Hatch Cut Adhesion Analyzer")
st.divider()

# === Background Image ===
## def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)),
                    url("data:image/png;base64,{encoded}");
        background-size: 50%;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_bg_from_local("background.png")

# === Upload and settings ===
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

# === Process each image ===
if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        img_np = np.array(img)
        reshaped = img_np.reshape(-1, 3)

        # Get 2 dominant colors
        kmeans = KMeans(n_clusters=2, random_state=42).fit(reshaped)
        colors = np.uint8(kmeans.cluster_centers_)

        # Let user pick dominant coating color (default 0)
        selected_color = colors[0]
        lower = np.clip(selected_color - sensitivity, 0, 255)
        upper = np.clip(selected_color + sensitivity, 0, 255)
        mask = cv2.inRange(img_np, lower, upper)

        # Grid analysis
        height, width = mask.shape
        cell_h, cell_w = height // grid_size, width // grid_size
        failure_count, total_cells = 0, grid_size * grid_size
        overlay = img_np.copy()

        for i in range(grid_size):
            for j in range(grid_size):
                x1, y1 = j * cell_w, i * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                cell = mask[y1:y2, x1:x2]
                coated = cv2.countNonZero(cell)
                if coated / cell.size < 0.5:
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

        # Save image for Excel export
        thumbnail = img.copy()
        thumbnail.thumbnail((100, 100))
        buf = io.BytesIO()
        thumbnail.save(buf, format="PNG")
        image_thumbnails[file.name] = buf.getvalue()

        # Show in UI
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption=f"{file.name} (Original)", width=300)
        with col2:
            st.image(overlay, caption="Detected Failures", channels="RGB", width=300)

    # === Show Results Table ===
    df = pd.DataFrame(results)
    st.subheader("Summary Table")
    st.dataframe(df)

    # === Excel Export with Images ===
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Results", startrow=1)

        workbook = writer.book
        worksheet = writer.sheets["Results"]

        # Add header and formatting
        header_format = workbook.add_format({'bold': True, 'bg_color': '#F0F0F0'})
        for col_num, value in enumerate(df.columns):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 20)

        # Add images
        for row_num, file in enumerate(uploaded_files, start=1):
            image_data = image_thumbnails[file.name]
            image_io = io.BytesIO(image_data)
            worksheet.insert_image(row_num, len(df.columns), file.name, {'image_data': image_io, 'x_scale': 0.5, 'y_scale': 0.5})

    st.download_button(
        label=":ramen: Download Results (Excel)",
        data=output.getvalue(),
        file_name="hatch_cut_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success("Analysis complete.")
    st.divider()
    st.write("Bob was here")
