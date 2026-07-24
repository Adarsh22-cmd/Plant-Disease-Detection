import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load Model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Load Class Names
train_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    "Dataset/PlantVillage/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())

# Streamlit UI
st.title("🌿 Plant Disease Detection")
st.write("Upload a tomato leaf image to predict the disease.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    index = np.argmax(prediction)

    confidence = float(prediction[0][index]) * 100

    st.success(f"Prediction: {class_names[index]}")
    st.write(f"Confidence: {confidence:.2f}%")