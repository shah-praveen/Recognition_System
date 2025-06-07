import streamlit as st
from datetime import datetime
import random

# Page config
st.set_page_config(page_title="Recognition System", layout="wide")

# Background and layout styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1508672019048-805c876b67e2');
        background-size: cover;
        background-attachment: fixed;
    }

    .stApp {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 12px;
        animation: fadeIn 1s ease-in;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #ff7e5f, #feb47b);  /* Attractive gradient */
        padding: 2rem;
        color: Black;  /* Text color in sidebar */
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);  /* Subtle shadow effect */
    }

    .typewriter h2 {
        overflow: hidden;
        border-right: .15em solid orange;
        white-space: nowrap;
        margin: 0 auto;
        animation: typing 3.5s steps(40, end), blink-caret 0.75s step-end infinite;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: orange; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar content
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    st.markdown("#### 🏠 Home")
    st.markdown("Welcome to the Recognition System!")
    st.markdown("Use this panel to get started 🚀")

# Greeting
hour = datetime.now().hour
if hour < 12:
    greeting = "Good morning"
elif 12 <= hour < 17:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"

usernames = ["Code Ninja", "Debugging Guru", "Algorithm Ace", "Pythonista", "Future Developer"]
chosen_name = random.choice(usernames)

# Title with fade + typewriter animation
st.markdown("<h1 style='text-align:center;'>👋 Welcome to the Recognition System</h1>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="typewriter">
        <h2 style='text-align:center;'>{greeting}, {chosen_name}! 💻🔧</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Description
st.markdown("### 🔍 Use the sidebar to navigate between:")
st.markdown("- 🐶 **Upload Image for Recognition**")
st.markdown("- 📈 **View Recognition History**")
st.markdown("- ⚙️ **Settings and Help**")

# Name input
name = st.text_input("Enter your name (optional):")
if name:
    st.success(f"Welcome aboard, {name}! Let's explore the system 🚀")
