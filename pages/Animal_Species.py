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

st.set_page_config(page_title="Animal Species Recognition", layout="wide")

# Custom Background Image and Overlay
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
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
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🐾 Animal Species Recognition")

# Wikipedia Summary Function
def get_animal_summary(animal_name):
    try:
        summary = wikipedia.summary(animal_name, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError:
        return "Multiple results found. Please refine your search."
    except wikipedia.exceptions.PageError:
        return "No summary found."

# Display Previous Results
def display_previous_results():
    try:
        df = pd.read_csv("results.csv", on_bad_lines="skip")
        if df.empty:
            st.warning("No previous results found.")
            return
        st.write("### 📜 Previous Results:")
        for _, row in df.iterrows():
            st.write(f"📅 Date: {row['Date']} | ⏰ Time: {row['Time']}")
            st.write(f"🔍 Label: {row['Label']} | 🎯 Accuracy: {row['Accuracy']:.2f}")
            if os.path.exists(row['Image Path']):
                st.image(row['Image Path'], caption=row['Label'], use_column_width=True)
    except:
        st.warning("No previous results found.")

# Upload Image
uploaded_file = st.file_uploader("Upload an animal image...", type=["jpg", "jpeg", "png"])

if st.button("📜 Show Previous Results"):
    display_previous_results()

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
        st.write("📖 **About this Animal:**")
        st.info(get_animal_summary(label))

        # Save Results
        with open("results.csv", "a+", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Date", "Time", "Label", "Accuracy", "Image Path"])
            writer.writerow([pd.Timestamp.now().strftime("%Y-%m-%d"), pd.Timestamp.now().strftime("%H:%M:%S"), label, accuracy, file_path])

        st.success("📁 Results saved successfully!")
