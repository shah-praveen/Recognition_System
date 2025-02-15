import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import csv
import pandas as pd

# Create directory for images
image_directory = "uploaded_images"
os.makedirs(image_directory, exist_ok=True)

# Load Model
model = ResNet50(weights='imagenet')

st.title("📦 Object Recognition")

# Upload Image
uploaded_file = st.file_uploader("Upload an object image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_path = os.path.join(image_directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = image.load_img(file_path, target_size=(224, 224))
    st.image(img, caption="📷 Uploaded Image.", use_column_width=True)

    if st.button("🔍 Recognize Object"):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=1)[0]
        label = decoded_preds[0][1].replace('_', ' ')
        accuracy = decoded_preds[0][2]

        st.write(f"✅ **Object:** {label}")
        st.write(f"📊 **Accuracy:** {accuracy:.2f}")

      

        # Save Results
        with open("results.csv", "a+", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Date", "Time", "Label", "Accuracy", "Image Path"])
            writer.writerow([pd.Timestamp.now().strftime("%Y-%m-%d"), pd.Timestamp.now().strftime("%H:%M:%S"), label, accuracy, file_path])

        st.success("📁 Results saved successfully!")
