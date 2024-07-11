import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Define data directories
train_dir = 'D:/College study Material/Projects/Research_Project/ultrasound breast classification/train'
validation_dir = 'D:/College study Material/Projects/Research_Project/ultrasound breast classification/val'

# Define image size and batch size
img_size = (224, 224)
batch_size = 32  # Adjust the batch size as needed

# Create data generators with custom data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1.0/255.0
)

validation_datagen = ImageDataGenerator(
    rescale=1.0/255.0  # Only rescale for validation
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary'
)

# Load ResNet-50 pre-trained model without top (fully connected) layers
weights_path = '/Users/piyushgoyal/Desktop/Project-Research Paper/Model - 2(ResNet 18)/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights=None,  # Load pre-trained ImageNet weights
    input_shape=(img_size[0], img_size[1], 3)
)

# Load the weights from the local file
base_model.load_weights(weights_path)

# Add custom top layers for binary classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# Set the pre-trained layers to non-trainable (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Set up callbacks for early stopping, model checkpointing, and learning rate reduction
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('/Users/piyushgoyal/Desktop/Project-Research Paper/Model - 2(ResNet 18)/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)

# Train the model without class weights
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,  # Increase the number of epochs as needed
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stop, model_checkpoint, reduce_lr]
)

# Load the best model
model.load_weights('/Users/piyushgoyal/Desktop/Project-Research Paper/Model - 2(ResNet 18)/best_model.h5')

# Evaluate the model
validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {validation_loss:.4f}")
print(f"Validation Accuracy: {validation_accuracy:.4f}")

# Generate predictions
validation_generator.reset()
predictions = model.predict(validation_generator)
y_pred = (predictions > 0.5).astype(int)

# Generate classification report and confusion matrix
class_labels = ['benign', 'malignant']
print(classification_report(validation_generator.classes, y_pred, target_names=class_labels))
conf_matrix = confusion_matrix(validation_generator.classes, y_pred)
print(conf_matrix)

# Save the model for future use at the specified location
model.save('/Users/piyushgoyal/Desktop/Project-Research Paper/Model - 2(ResNet 18)/breast_cancer_cnn_model.h5')

# Plot accuracy and loss over epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
