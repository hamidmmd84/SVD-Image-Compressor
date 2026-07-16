import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="SVD Image compressor", layout="centered")
st.title("SVD Image Compressor")

uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    img_array = np.array(img, dtype="float32")

    m, n = img_array.shape
    max_k = min(m, n)

    st.image(img, caption="SVD Image Compressor", use_column_width=True)
    st.sidebar.header("SVD Image Compressor")

    mode = st.sidebar.radio(
        "Selection Strategy:",
        ("Manual Components (k)", "Adaptive Energy Preservation")
    )

    U, S, V = np.linalg.svd(img_array, full_matrices=False)

    if mode == "Manual Components (k)":
        k = st.sidebar.slider("Number of Singular Values", min_value=1, max_value=max_k, value=int(max_k * 0.1))
    else:
        total_energy =  np.sum(S ** 2)
        cumulative_energy = np.cumsum(S ** 2) / total_energy

        target_energy = st.sidebar.slider(
            "Target Information/Energy Preservation (%):",
            min_value=90.0,
            max_value=100.0,
            value=99.0,
            step=0.1,
        )

        k = int(np.argmax(cumulative_energy >= (target_energy / 100.0))) + 1
        st.sidebar.info(f"Keeping {k} out of {max_k} components preserves {target_energy}% of the image energy.")

    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_k = V[:k, :]

    compressed_array = np.dot(U_k, np.dot(S_k, V_k))

    compressed_array = np.clip(compressed_array, 0, 255).astype(np.uint8)
    compressed_img = Image.fromarray(compressed_array)


    original_entries = m * n
    compressed_entries = (m * k) + k + (k * n)
    space_saved = (1 - (compressed_entries / original_entries)) * 100

    st.markdown("compression output")
    st.image(compressed_img, caption=f"Reconstruction Image using k = {k}", use_column_width=True)


    col1, col2 = st.columns(2)
    col1.metric("Retained Components (k)", f"{k} / {max_k}")
    col2.metric("Theoretical Date Reduction", f"{space_saved:.2f}%")


    buffer = BytesIO()
    compressed_img.save(buffer, format="JPEG")
    st.download_button(
        label="Download the compressed image",
        data=buffer.getvalue(),
        file_name="SVD_compressed.jpg",
        mime="image/jpeg",
    )

    st.success("low-rank approximation calculated successfully using SVD")
else:
    st.info("Please upload an image")