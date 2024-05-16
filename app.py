import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, SeparableConv2D, GlobalAveragePooling2D, Dense
from keras.optimizers import Adam


def create_model():
    inputs = tf.keras.Input(shape=(192, 192, 3))
    
    x = Conv2D(256, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    previous_block_activation = x  # Set aside residual

    for size in [64, 128, 256]:
        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(size, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.Add()([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = SeparableConv2D(256, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    outputs = Dense(5, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = create_model()
    model.load_weights("BestModel.h5")  # Ensure that the path is correct
    return model

# Function to make predictions
def predict(image):
    # Preprocess the image
    image = image.resize((192, 192))  # Resize the image
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    model = load_model()
    prediction = model.predict(image)
    
    return prediction

# Streamlit app
def main():
    st.title("Vehicle Type Classification")
    st.text("Upload an image of a vehicle to classify it.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        model = load_model()
        prediction = predict(image)

        # Display the prediction
        classes = ['bus', 'car', 'motorcycle', 'train', 'truck']  # Define your class names
        st.subheader("Prediction:")
        st.write(classes[np.argmax(prediction)], f"({np.max(prediction)*100:.2f}% certain)")

if __name__ == "__main__":
    main()
