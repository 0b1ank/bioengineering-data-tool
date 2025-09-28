# Below is a single Python script that merges the functionality of both codes,
# including the image feature extraction from the first script.
#
# You can run this file (e.g., python or streamlit run scripts.py) and test it.
# The script will allow you to upload CSV or Image files, then extract features
# from images or handle CSV data as requested.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from PIL import Image
import cv2
import pytesseract

st.set_page_config(page_title="Bioengineering Data Analysis", layout="centered")

# Title and intro
st.title("Tyler and Larry's Bioengineering Data Analysis Tool")
st.write("Welcome! This web app allows you to upload a CSV file or an image, perform cleaning, visualization, and exploration.")

############################################
# 1. File Upload
############################################
file_option = st.selectbox("Select file type to upload", ["CSV", "Image"])


def extract_features_from_image(image):
    """
    Extract cell-related features from an image, such as positions, sizes, etc.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply thresholding to segment the cells
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours of cells
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract features for each cell
    features = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        features.append({
            "X_cell_position": x + w // 2,
            "Y_cell_position": y + h // 2,
            "Cell_size": area,
        })

    return pd.DataFrame(features)


# Let the user either upload a CSV or an Image
if file_option == "CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file, index_col="ID")
            st.success("File uploaded and loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

elif file_option == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Attempt to extract features
        try:
            extracted_data = extract_features_from_image(image)
            st.success("Features extracted successfully from the image!")
            st.dataframe(extracted_data)

            # Allow user to download extracted features as CSV
            csv_buffer = StringIO()
            extracted_data.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Extracted Data as CSV",
                data=csv_buffer.getvalue(),
                file_name="extracted_data.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error extracting features: {e}")

# If CSV data was successfully uploaded, proceed
if 'data' in locals():
    st.info("Images have been standardized to 800x800 (simulation)")

    ############################################
    # 2. Show Raw Data
    ############################################
    st.subheader("Raw Data Preview")
    st.dataframe(data.head())

    # Display Missing Values Before Cleaning
    st.subheader("Missing Values Before Cleaning")
    missing_values = data.isnull().sum()
    st.write(missing_values)

    # Drop Rows Missing 'Experiment ID'
    data = data.dropna(subset=["Experiment ID"])
    st.write("Rows with no 'Experiment ID' have been removed.")

    missing_values_after = data.isnull().sum()
    st.subheader("Missing Values After Cleaning")
    st.write(missing_values_after)

    # Copy Data for Imputation
    data_copy = data.copy()

    # Impute Missing Values
    st.subheader("Impute Missing Values")
    if "Cell_size" in data_copy.columns:
        median_cell_size = data_copy["Cell_size"].median()
        missing_count = data_copy["Cell_size"].isna().sum()
        data_copy["Cell_size"] = data_copy["Cell_size"].fillna(median_cell_size)
        st.write(f"Filled {missing_count} missing values in 'Cell_size' using median: {median_cell_size:.2f}")
    else:
        st.warning("Column 'Cell_size' not found.")

    if "X_cell_position" in data_copy.columns:
        mean_x_cell_pos = data_copy["X_cell_position"].mean()
        missing_count = data_copy["X_cell_position"].isna().sum()
        data_copy["X_cell_position"] = data_copy["X_cell_position"].fillna(mean_x_cell_pos)
        st.write(f"Filled {missing_count} missing values in 'X_cell_position' using mean: {mean_x_cell_pos:.2f}")
    else:
        st.warning("Column 'X_cell_position' not found.")

    # One-Hot Encoding
    st.subheader("One-Hot Encoding for Categorical Columns")
    cat_cols = [col for col in ["Apparatus used", "Cell type"] if col in data_copy.columns]
    if cat_cols:
        data_copy = pd.get_dummies(data_copy, columns=cat_cols, prefix=[col.replace(" ", "_") for col in cat_cols])
        st.write("One-hot encoding applied. Here's a preview:")
        st.dataframe(data_copy.head())
    else:
        st.warning("No categorical columns found for one-hot encoding.")

    # Filtering Examples
    st.subheader("Filtering Examples")
    if {"Cell_size", "Cell type"}.issubset(data_copy.columns):
        filtered_data = data_copy[(data_copy["Cell_size"] > 14) & (data_copy["Cell type"] == "White Blood Cell")]
        st.write("Filtered Data: Cell_size > 14 AND 'White Blood Cell'")
        st.dataframe(filtered_data)
    else:
        st.warning("'Cell_size' or 'Cell type' column not found for filtering.")

    # Visualizations
    st.subheader("Visualizations")
    if "Cell_size" in data_copy.columns:
        fig, ax = plt.subplots()
        ax.hist(data_copy["Cell_size"].dropna(), bins=20, color="skyblue", edgecolor="black")
        ax.set_title("Distribution of Cell Sizes")
        ax.set_xlabel("Cell Size")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        if "Cell type" in data_copy.columns:
            fig, ax = plt.subplots()
            data_copy.boxplot(column="Cell_size", by="Cell type", grid=False, ax=ax)
            ax.set_title("Cell Size by Cell Type")
            plt.suptitle("")
            st.pyplot(fig)
    else:
        st.warning("'Cell_size' column not found for visualizations.")

    if {"X_cell_position", "Y_cell_position"}.issubset(data_copy.columns):
        valid_positions = data_copy.dropna(subset=["X_cell_position", "Y_cell_position"])
        fig, ax = plt.subplots()
        ax.scatter(valid_positions["X_cell_position"], valid_positions["Y_cell_position"], alpha=0.6)
        ax.set_title("Cell Positions")
        ax.set_xlabel("X_cell_position")
        ax.set_ylabel("Y_cell_position")
        st.pyplot(fig)
    else:
        st.warning("'X_cell_position' or 'Y_cell_position' column not found for scatter plot.")

    # Interactive Query (Conceptual)
    st.subheader("Interactive Query (Conceptual)")
    user_query = st.text_input("Enter a query (e.g., 'average cell size for Electron Microscope')")
    if user_query:
        if "electron microscope" in user_query.lower() and "Apparatus used" in data_copy.columns:
            em_data = data_copy[data_copy["Apparatus used"] == "Electron Microscope"]
            if not em_data.empty:
                avg_size_em = em_data["Cell_size"].mean()
                st.write(f"Average Cell Size for Electron Microscope: {avg_size_em:.2f}")
            else:
                st.warning("No rows match the filter (Electron Microscope).")
        else:
            st.warning("Query not recognized or 'Apparatus used' column not found.")
else:
    st.warning("Please upload a file to proceed.")