# real_time_classification.py

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import json

def load_model():
    model = tf.keras.models.load_model('staff_mobilenet_v2_model.h5')
    return model

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match the model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def run():
    st.title("Real-Time Staff Classification")

    # Load pre-trained classification model
    model = load_model()

    # Define class names
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    
    # Invert the dictionary for lookup
    class_names_inv = {v: k for k, v in class_names.items()}
    
    # Print statements for debugging
    print("Class names:", class_names)
    print("Inverted class names:", class_names_inv)

    # Start video capture
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.write("Failed to access camera. Please check the camera connection or deployment environment.")
        # Optionally, provide fallback content or instructions here
        return

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL image for preprocessing
        pil_image = Image.fromarray(frame_rgb)
        processed_image = preprocess_image(pil_image)

        # Predict class
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        
        # Check if the index exists in the inverted dictionary
        predicted_class = class_names_inv.get(predicted_class_index, "Unknown")
        probability = np.max(predictions)

        # Draw the prediction on the frame
        label = f"{predicted_class}: {probability:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the result
        stframe.image(frame, channels='BGR', use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
