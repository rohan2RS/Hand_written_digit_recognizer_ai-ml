import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
try:
    model = tf.keras.models.load_model('/content/Hand_written_digit_recog_model.keras')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

st.header('Hand Digit Recognition Model')

uploaded_file = st.file_uploader("Upload a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        st.error("Failed to load the image. Please ensure it's a valid image file.")
    else:
        st.image(image, caption='Uploaded Image', width=200)

        # --- Preprocess the image ---

        # Resize to 28x28
        image_resized = cv2.resize(image, (28, 28))

        # Invert the image *before* normalization
        # np.invert works on integer types like uint8
        image_inverted = np.invert(image_resized)

        # Normalize the inverted image to [0, 1]
        # This matches the normalization step in your training data preparation
        image_normalized = image_inverted / 255.0

        # Add batch dimension and channel dimension
        image_processed = image_normalized.reshape(1, 28, 28, 1)

        # --- Make a prediction ---
        try:
            output = model.predict(image_processed)
            predicted_digit = np.argmax(output)

            st.success(f'Predicted Digit: **{predicted_digit}**')

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")