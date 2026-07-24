import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load Model
model = tf.keras.models.load_model("plant_disease_model.keras")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    "Dataset/PlantVillage/train",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print(class_names)


# Test Image
img = image.load_img("test.jpg", target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

index = np.argmax(prediction)

print("Prediction Shape:", prediction.shape)
print("Predicted Index:", index)

print("Predicted Disease:", class_names[index])
print("Confidence:", round(float(prediction[0][index]) * 100,2), "%")
                            