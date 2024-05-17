import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("BestModel.h5")
    return model


def predict(image, model):
    image = image.resize((192, 192))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    
    return prediction


def main():
    st.title("Vehicle Type Classification")
    st.text("Upload an image of a vehicle to classify it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        model = load_model()
        prediction = predict(image, model)

        classes = ['bus', 'car', 'motorcycle', 'train', 'truck']  
        st.subheader("Prediction:")
        string="OUTPUT : "+classes[np.argmax(prediction)]
        st.success(string)

if __name__ == "__main__":
    main()
