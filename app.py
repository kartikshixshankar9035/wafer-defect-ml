import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Page configuration
st.set_page_config(
    page_title="Wafer Defect Classifier",
    page_icon="🔬",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.loadresults/wafer_random_forest_best_compressed.pkl

model = load_model()

# Feature extraction (same as training)
def extract_features(wafer):

    wafer = np.array(wafer)

    total_dies = np.count_nonzero(wafer)
    good_dies = np.count_nonzero(wafer == 1)
    defective_dies = np.count_nonzero(wafer == 2)

    defect_ratio = (
        defective_dies / total_dies
        if total_dies else 0
    )

    features = [
        good_dies,
        defective_dies,
        defect_ratio
    ]

    row_parts = np.array_split(wafer, 3, axis=0)

    for row in row_parts:
        col_parts = np.array_split(row, 3, axis=1)

        for region in col_parts:
            features.append(
                np.count_nonzero(region == 2)
            )

    return features


# Application title
st.title("🔬 Wafer Defect Classification")

st.write(
    "Upload a wafer map to predict its semiconductor "
    "defect category using Random Forest."
)

st.info(
    "Supported formats: CSV and NPY. "
    "Values: 0 = Background, 1 = Good die, 2 = Defective die."
)

# File upload
uploaded_file = st.file_uploader(
    "Upload Wafer Map",
    type=["csv", "npy"]
)

if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".npy"):
            wafer = np.load(uploaded_file, allow_pickle=False)
        else:
            wafer = np.loadtxt(
                uploaded_file,
                delimiter=","
            )

        if wafer.ndim != 2 or min(wafer.shape) < 3:
            st.error("Please upload a valid 2D wafer map.")
            st.stop()

        if not np.all(np.isin(wafer, [0, 1, 2])):
            st.error("Wafer map must contain only 0, 1 and 2.")
            st.stop()

        st.subheader("Uploaded Wafer Map")

        fig, ax = plt.subplots()
        ax.imshow(wafer, cmap="viridis")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

        # Extract features
        features = extract_features(wafer)

        # Predict defect class
        prediction = model.predict([features])[0]

        st.subheader("Prediction Result")

        if prediction == "none":
            st.success("No Defect Pattern Detected")
        else:
            st.warning(f"Predicted Defect: {prediction}")

        # Prediction confidence
        probabilities = model.predict_proba([features])[0]
        confidence = np.max(probabilities) * 100

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.caption(
            "Confidence is the model's predicted class probability, "
            "not a guarantee of correctness."
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")

st.divider()

st.caption(
    "Explainable Machine Learning for Semiconductor "
    "Wafer Defect Pattern Classification"
)