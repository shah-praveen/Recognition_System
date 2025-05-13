import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import csv
import pandas as pd
import wikipedia

# Create directory for images
image_directory = "uploaded_images"
os.makedirs(image_directory, exist_ok=True)

# Load Pre-trained Model
model = ResNet50(weights='imagenet')

st.set_page_config(page_title="Object Recognition", layout="wide")

# Custom Background Image and Overlay (with two images)
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1526364782041-d9f9fa6bcf4e');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        height: 100vh;
        margin: 0;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.75);  /* Lighter white overlay */
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(2px);  /* Slight blur effect for better readability */
    }

    h1, h2, h3, h4, h5, h6 {
        color: #1e3d59;
    }

    .stTextInput > label, .stFileUploader > label, .stSelectbox > label {
        font-weight: bold;
        color: #1b3e2b;
    }

    /* Background image for the second image */
    .stApp::before {
        content: '';
        background-image: url('https://images.unsplash.com/photo-1506748686218-900aefebf056');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        opacity: 0.4; /* Make the second image less prominent */
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛠️ Object Recognition System")

# Wikipedia Summary Function
def get_object_summary(object_name):
    try:
        summary = wikipedia.summary(object_name, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError:
        return "Multiple results found. Please refine your search."
    except wikipedia.exceptions.PageError:
        return "No summary found."

# Display Previous Results for Object Recognition
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
    except:
        st.warning("No previous results found.")

# Upload Image
uploaded_file = st.file_uploader("Upload an object image...", type=["jpg", "jpeg", "png"])

if st.button("📜 Show Object Recognition Results"):
    display_previous_results_object()

if uploaded_file:
    file_path = os.path.join(image_directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = image.load_img(file_path, target_size=(224, 224))
    st.image(img, caption="📷 Uploaded Image.", use_column_width=True)

    if st.button("🔍 Predict"):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=1)[0]
        label = decoded_preds[0][1].replace('_', ' ')
        accuracy = decoded_preds[0][2]

        st.write(f"✅ **Label:** {label}")
        st.write(f"📊 **Accuracy:** {accuracy:.2f}")

        # Fetch Wikipedia Summary
        st.write("📖 **About this Object:**")
        st.info(get_object_summary(label))

        # Save Results for Object Recognition
        with open("object_results.csv", "a+", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Date", "Time", "Label", "Accuracy", "Image Path"])
            writer.writerow([pd.Timestamp.now().strftime("%Y-%m-%d"), pd.Timestamp.now().strftime("%H:%M:%S"), label, accuracy, file_path])

        st.success("📁 Results saved successfully!")
