import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix

# Define data directories
train_dir = (
    "D:/College study Material/Projects/Research_Project/ultrasound breast classification/train"
)
validation_dir = (
    "D:/College study Material/Projects/Research_Project/ultrasound breast classification/val"
)


# Define image size and batch size
img_size = (224, 224)
batch_size = 4  # Adjust the batch size as needed

# Create data generators with custom data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1.0 / 255.0,
)

validation_datagen = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1.0 / 255.0,
)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    class_mode="binary",
)

model = Sequential()
model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        input_shape=(img_size[0], img_size[1], 3),
        kernel_regularizer=l2(0.0001),
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Add more convolutional layers with increased complexity
model.add(Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Add a few more convolutional layers

model.add(Flatten())
model.add(Dense(256, activation="relu", kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# Compile the model with the updated learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])


# Train the model with callbacks
history = model.fit(
    train_generator,
    steps_per_epoch=1000,
    epochs=25,  # Increase the number of epochs as needed
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
)

# Load the best model
model.load_weights("C:/Users/Raghav/Desktop/Research_Project/Model 1/best_model.h5")

# Evaluate the model
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {validation_loss:.4f}")
print(f"Validation Accuracy: {validation_accuracy:.4f}")

# Generate predictions
validation_generator.reset()
predictions = model.predict(validation_generator)
y_pred = (predictions > 0.5).astype(int)

# Generate classification report and confusion matrix
class_labels = ["benign", "malignant"]
print(
    classification_report(
        validation_generator.classes, y_pred, target_names=class_labels
    )
)
conf_matrix = confusion_matrix(validation_generator.classes, y_pred)
print(conf_matrix)

# Save the model for future use at the specified location
model.save(
    "C:/Users/Raghav/Desktop/Research_Project/Model 1/breast_cancer_cnn_model.h5"
)

# Plot accuracy and loss over epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
