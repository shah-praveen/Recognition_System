import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import csv
import pandas as pd
import wikipedia
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Animal Species Recognition", layout="wide")

# Create image save directory
image_directory = "uploaded_images"
os.makedirs(image_directory, exist_ok=True)

# Load model
model = ResNet50(weights='imagenet')

# Styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.75);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
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
    """,
    unsafe_allow_html=True
)

# Static species population data
species_population_data = {
    "tiger": {
        "2006": 1411,
        "2010": 1706,
        "2014": 2226,
        "2018": 2967
    },
    "elephant": {
        "2007": 27600,
        "2012": 27312,
        "2017": 27312
    },
    "lion": {
        "2010": 411,
        "2015": 523,
        "2020": 674
    }
}

# Sidebar content
with st.sidebar:
    st.markdown("### 🧽 Navigation")
    st.markdown("#### 🐾 Animal Species")
    st.markdown("Upload an image and get species info 📸")
    show_graph = st.checkbox("Show Species Distribution Graph")

# Title
st.title("🐾 Animal Species Recognition")

# Wikipedia summary fetcher
def get_animal_summary(animal_name):
    try:
        return wikipedia.summary(animal_name, sentences=2)
    except wikipedia.exceptions.DisambiguationError:
        return "Multiple results found. Please refine your search."
    except wikipedia.exceptions.PageError:
        return "No summary found."

# Display previous results
def display_previous_results():
    try:
        df = pd.read_csv("results.csv", on_bad_lines="skip")
        if df.empty:
            st.warning("No previous results found.")
            return
        st.write("### 📜 Previous Results:")
        for _, row in df.iterrows():
            st.write(f"🗕️ Date: {row['Date']} | ⏰ Time: {row['Time']}")
            st.write(f"🔍 Label: {row['Label']} | 🎯 Accuracy: {row['Accuracy']:.2f}")
            if os.path.exists(row['Image Path']):
                st.image(row['Image Path'], caption=row['Label'], use_column_width=True)
    except Exception:
        st.warning("No previous results found.")

# Check if label is an animal
def is_animal_label(label):
    animal_keywords = ['dog', 'cat', 'lion', 'tiger', 'elephant', 'zebra', 'giraffe', 'horse',
                       'cow', 'bear', 'monkey', 'chimpanzee', 'leopard', 'cheetah', 'wolf',
                       'fox', 'panda', 'koala', 'kangaroo', 'raccoon', 'buffalo', 'bull',
                       'deer', 'ox', 'goat', 'sheep', 'rabbit', 'squirrel', 'boar', 'pig',
                       'mouse', 'rat', 'bat', 'jaguar', 'panther', 'coyote', 'hyena']
    return any(animal in label.lower() for animal in animal_keywords)

# Plot species frequency graph
def plot_species_distribution():
    try:
        df = pd.read_csv("results.csv", on_bad_lines="skip")
        species_counts = df['Label'].value_counts()
        plt.figure(figsize=(8,5))
        species_counts.plot(kind='bar', color='coral')
        plt.title("Species Recognition Frequency")
        plt.xlabel("Species")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)
    except Exception:
        st.warning("No data available to display the graph.")

# Plot species trend for India
def plot_species_trend(species_name):
    name = species_name.lower()
    if name in species_population_data:
        data = species_population_data[name]
        years = list(data.keys())
        counts = list(data.values())
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(years, counts, marker='o', linestyle='-', color='darkgreen')
        ax.set_title(f"{species_name.title()} Population Over Years in India")
        ax.set_xlabel("Year")
        ax.set_ylabel("Population")
        ax.grid(True)
        st.sidebar.pyplot(fig)
    else:
        st.sidebar.warning(f"No historical data available for '{species_name}'.")

# Upload section
uploaded_file = st.file_uploader("📄 Upload an animal image...", type=["jpg", "jpeg", "png"])

if st.button("📜 Show Previous Results"):
    display_previous_results()

# Show graph if checkbox checked
if show_graph:
    st.sidebar.markdown("### Species Distribution")
    plot_species_distribution()

# Prediction logic
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
        label = decoded_preds[0][1].replace('_', ' ')
        accuracy = decoded_preds[0][2]

        if not is_animal_label(label):
            st.error(f"🚫 Detected: **{label}** — This is not an animal. Please upload an animal image.")
        else:
            st.write(f"✅ **Label:** {label}")
            st.write(f"📊 **Accuracy:** {accuracy:.2f}")
            st.write("📖 **About this Animal:**")
            st.info(get_animal_summary(label))

            with open("results.csv", "a+", newline="") as f:
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
            plot_species_trend(label)
