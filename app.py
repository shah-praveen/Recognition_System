import streamlit as st
from datetime import datetime
import random

# Page config
st.set_page_config(page_title="Recognition System", layout="wide")

# Add custom background image and translucent overlay
st.markdown(
    """
    <style>
    /* Set background image for the entire page */
    body {
        background-image: url('https://images.unsplash.com/photo-1508672019048-805c876b67e2');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Add overlay to improve readability */
    .stApp {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dynamic greeting
hour = datetime.now().hour
if hour < 12:
    greeting = "Good morning"
elif 12 <= hour < 17:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"

# Welcome Title
st.title("👋 Welcome to the  Recognition System")

# Dynamic subtitle
usernames = ["Explorer", "Wildlife Enthusiast", "Biologist", "Nature Lover", "Researcher"]
st.subheader(f"{greeting}, {random.choice(usernames)}! 👨‍🔬🐾")

# Description
st.markdown("### 🔍 Use the sidebar to navigate between:")
st.markdown("- 🐶 **Upload Image for Recognition**")
st.markdown("- 📈 **View Recognition History**")
st.markdown("- ⚙️ **Settings and Help**")

# Optional name input
name = st.text_input("Enter your name (optional):")
if name:
    st.success(f"Great to have you here, {name}! Let's get started! 🚀")

# Balloons animation
st.balloons()
