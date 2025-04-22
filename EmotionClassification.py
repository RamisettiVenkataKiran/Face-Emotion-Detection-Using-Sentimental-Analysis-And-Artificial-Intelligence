import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import numpy as np
import cv2
import os

# ✅ 1. Load Dataset
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 32
DATASET_PATH = os.path.abspath("D:/Sentimental Analysis Face emotion/Emotions/train")  # Ensure correct path

# Data Augmentation & Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,    # Normalize pixel values
    validation_split=0.2  # 20% images for validation
)

# Training data
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)

# Validation data
val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# Class labels
class_labels = list(train_generator.class_indices.keys())
print("Class Labels:", class_labels)

# ✅ 2. Build CNN Model with Batch Normalization & Dropout
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')  # Output layer
])

# ✅ 3. Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ 4. Set Up Callbacks (Early Stopping & Model Checkpoint)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("best_emotion_model.h5", monitor='val_accuracy', save_best_only=True)
]

# ✅ 5. Train the Model
EPOCHS = 25
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ✅ 6. Save Trained Model
model.save("emotions.h5")
print("🎉 Model trained and saved as emotions.h5")

# ✅ 7. Load Model & Test on New Image
def predict_emotion(image_path):
    model = load_model("emotions.h5")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    image = cv2.resize(image, (48, 48))  # Resize to match model input
    image = image.reshape(1, 48, 48, 1) / 255.0  # Normalize

    # Predict emotion
    prediction = model.predict(image)
    emotion_label = class_labels[np.argmax(prediction)]
    return emotion_label

# Example Test
test_image = "test_face.jpg"  # Replace with an actual image path
if os.path.exists(test_image):
    print(f"Predicted Emotion: {predict_emotion(test_image)}")
else:
    print("⚠️ Test image not found. Please provide a valid image path.")
