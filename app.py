import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("/Users/ronitgandhi/Desktop/DL_for_classification/land_classification_model.h5")  # Ensure model is in the same directory

model = load_model()

# Define the class labels (Same as used during training)
class_labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
                'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Remove alpha channel if present (convert RGBA -> RGB)
    image = image.resize((256, 256))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image.astype(np.float32)  # Convert to float32

# Streamlit App Layout
st.title("🌍 Satellite Land Classification App 🚀")
st.write("Upload a satellite image, and the model will predict the land category.")

# File uploader
uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and Predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    # Get highest confidence prediction
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display Prediction
    st.subheader("🔍 Prediction:")
    st.write(f"🏷️ **Category:** {predicted_class}")
    st.write(f"📊 **Confidence:** {confidence:.2f}")

    # Show confidence distribution
    st.bar_chart(dict(zip(class_labels, predictions[0])))

