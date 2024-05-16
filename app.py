import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Function to load the model
@st.cache_resource
def load_model():
    model = create_model()  # Assuming create_model() is defined earlier
    model.load_weights("BestModel.h5")  # Load the trained weights
    return model

# Function to make predictions
def predict(image):
    # Preprocess the image
    image = image.resize((192, 192))  # Resize the image
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=3)  # Add batch dimension
    
    # Load the model
    model = load_model()
    
    # Make prediction
    prediction = model.predict(image)
    
    return prediction

# Streamlit app
def main():
    st.title("Vehicle Classification App")
    st.text("Upload an image of a vehicle to classify it.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        prediction = predict(image)

        # Display the prediction
        classes = ['bus', 'car', 'motorcycle', 'train', 'truck']  # Define your class names
        st.subheader("Prediction:")
        st.write(classes[np.argmax(prediction)], f"({np.max(prediction)*100:.2f}% certain)")

if __name__ == "__main__":
    main()
