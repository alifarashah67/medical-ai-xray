import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Medical AI X-ray Detection",
    page_icon="🩻",
    layout="wide"
)

st.title("🩻 Medical AI: Chest X-ray Pneumonia Detection")

st.markdown("""
This demo is designed to classify chest X-ray images as **Normal** or **Pneumonia**
and later show explainable Grad-CAM heatmaps.
""")

st.warning(
    "Disclaimer: This project is for educational and demonstration purposes only. "
    "It is not a medical device and must not be used for clinical diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded X-ray")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction Result")
        st.info("Model prediction will be added in the next step.")
        st.metric("Prediction", "Coming soon")
        st.metric("Confidence", "Coming soon")

else:
    st.info("Please upload a chest X-ray image to start.")
