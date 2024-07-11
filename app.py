import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    return img_array


# Set page title and icon
st.set_page_config(page_title="Breast Cancer Detector", page_icon=":microscope:")

# Create a dictionary to store session state
session_state = st.session_state

# Initialize session state variables
if "ground_truth_label" not in session_state:
    session_state.ground_truth_label = "Benign"

# Set app title
st.title("Breast Cancer Detector")

# Dictionary of model paths
model_paths = {
    "Custom CNN": "D:/College study Material/Projects\Research_Project/Model 1/breast_cancer_cnn_model.h5",
    "DenseNet 121": "D:/College study Material/Projects/Research_Project/Model 3/breast_cancer_densenet121_model.h5",
    "Inception": "D:/College study Material/Projects/Research_Project/Model 4/breast_cancer_inception_v3_model.h5",
}

# Load models
model_names = list(model_paths.keys())
selected_model_name = st.selectbox("Choose a model", model_names)
selected_model_path = model_paths[selected_model_name]
selected_model = load_model(selected_model_path)

# Get ground truth label from user
ground_truth_label = st.radio("Select ground truth label:", ["Benign", "Malignant"])
session_state.ground_truth_label = ground_truth_label

# Upload image through Streamlit
image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Display selected image
    st.image(image_file, caption="Uploaded Image.", use_column_width=True)

    # Make predictions
    img_array = load_and_preprocess_image(image_file, target_size=(224, 224))
    predicted_class_prob = selected_model.predict(img_array)[0, 0]
    predicted_class = "Benign" if predicted_class_prob < 0.5 else "Malignant"

    # Display predictions
    st.write(f"Class Prediction: {predicted_class}")

    # Calculate and display accuracy
    ground_truth_mapping = {"Benign": 0, "Malignant": 1}
    ground_truth = ground_truth_mapping.get(session_state.ground_truth_label, -1)
    if ground_truth != -1:
        predicted_label = 0 if predicted_class == "Benign" else 1
        accuracy = 100 * int(predicted_label == ground_truth)
        st.write(f"Accuracy: {accuracy}%")
    else:
        st.write("Invalid ground truth label. Please select Benign or Malignant.")
