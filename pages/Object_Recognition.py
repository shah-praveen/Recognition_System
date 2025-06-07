
import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv
import pandas as pd
import wikipedia

# Page config
st.set_page_config(page_title="Object Recognition", layout="wide")

# Directory for image uploads
image_directory = "uploaded_images"
os.makedirs(image_directory, exist_ok=True)

# Load ResNet50 model
model = ResNet50(weights='imagenet')

# Styling
st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1506748686218-900aefebf056');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(2px);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #ff7e5f, #feb47b);
        padding: 2rem;
        color: black;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3, h4 {
        color: #1e3d59;
    }
    .stTextInput > label, .stFileUploader > label {
        font-weight: bold;
        color: #1b3e2b;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    st.markdown("#### 🛠️ Object Recognition")
    st.markdown("Upload an object image and predict what it is!")

# Title
st.title("🛠️ Object Recognition System")

# Wikipedia summary fetcher
def get_object_summary(object_name):
    try:
        return wikipedia.summary(object_name, sentences=2)
    except wikipedia.exceptions.DisambiguationError:
        return "Multiple results found. Please refine your search."
    except wikipedia.exceptions.PageError:
        return "No summary found."

# Display previous results
def display_previous_results_object():
    try:
        df = pd.read_csv("object_results.csv", on_bad_lines="skip")
        if df.empty:
            st.warning("No previous results found.")
            return
        st.write("### 📜 Previous Object Recognition Results:")
        for _, row in df.iterrows():
            st.write(f"📅 Date: {row['Date']} | ⏰ Time: {row['Time']}")
            st.write(f"🔍 Label: {row['Label']} | 🎯 Accuracy: {row['Accuracy']:.2f}")
            if os.path.exists(row['Image Path']):
                st.image(row['Image Path'], caption=row['Label'], use_column_width=True)
    except Exception:
        st.warning("No previous results found or file error.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an object image...", type=["jpg", "jpeg", "png"])

# Show results button
if st.button("📜 Show Object Recognition Results"):
    display_previous_results_object()

# Predict if image is uploaded
if uploaded_file:
    file_path = os.path.join(image_directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = image.load_img(file_path, target_size=(224, 224))
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=1)[0]
        synset = decoded_preds[0][0]        # e.g., 'n02504458'
        label = decoded_preds[0][1].replace('_', ' ')
        accuracy = decoded_preds[0][2]

        # Block animals based on WordNet synset prefix
        if synset.startswith("n02"):  # n02* = animals (e.g., n02084071 = dog)
            st.error(f"🚫 Detected: **{label}** — This appears to be an animal. Please upload an object.")
        else:
            st.write(f"✅ **Label:** {label}")
            st.write(f"📊 **Accuracy:** {accuracy:.2f}")
            st.write("📖 **About this Object:**")
            st.info(get_object_summary(label))

            # Save results
            with open("object_results.csv", "a+", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["Date", "Time", "Label", "Accuracy", "Image Path"])
                writer.writerow([
                    pd.Timestamp.now().strftime("%Y-%m-%d"),
                    pd.Timestamp.now().strftime("%H:%M:%S"),
                    label,
                    accuracy,
                    file_path
                ])
            st.success("📁 Results saved successfully!")
