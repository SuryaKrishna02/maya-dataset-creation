import streamlit as st
import time
import os

def is_valid_file(file):
    if file is not None:
        file_extension = os.path.splitext(file.name)[1].lower()
        return file_extension in ['.csv', '.json']
    return False

st.set_page_config(page_title="Maya Multilingual Dataset Generator", layout="wide")

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .main {
        background-color: #2d2d2d;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4a86e8;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a76d8;
    }
    .stSelectbox {
        background-color: #3d3d3d;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #4a86e8;
    }
    .stAlert {
        background-color: #4a1c24;
        color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #15572f;
        color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .stDateInput>div>div>input {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.title("Maya Multilingual Dataset Generator")

    st.markdown("Generate multilingual datasets with ease! 🌍✨")

    # File uploader for English dataset
    english_dataset = st.file_uploader("📁 English Dataset File", type=['csv', 'json'])

    # Language selection
    language = st.selectbox("🗣️ Target Language", ["", "Russian", "Spanish", "French", "German", "Chinese", "Japanese"])

    # Hardware selection
    hardware = st.selectbox("💻 Hardware", ["", "GPU", "CPU"])

    if st.button("🚀 Generate Dataset"):
        if not english_dataset or not language or not hardware:
            st.error("⚠️ Please select all required fields: English dataset file, language, and hardware.")
        elif not is_valid_file(english_dataset):
            st.error("⚠️ Only CSV and JSON files are allowed for the English dataset.")
        else:
            with st.spinner("Generating your multilingual dataset... 🔄"):
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.05)  # Simulating work being done
            
            st.success("🎉 Dataset generated successfully!")
            
            st.download_button(
                label="📥 Download Dataset",
                data=b"Your dataset content here",  # Replace with actual generated dataset
                file_name="multilingual_dataset.csv",
                mime="text/csv"
            )

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Upload your English dataset (CSV or JSON)
    2. Select the target language for translation
    3. Choose your preferred hardware for processing
    4. Click 'Generate Dataset' and wait for the magic to happen!
    """)

st.sidebar.header("🔍 About")
st.sidebar.info(
    "This app uses advanced AI techniques to generate high-quality multilingual datasets. "
    "Perfect for machine learning projects, localization efforts, and cross-lingual studies."
)

st.sidebar.header("💡 Tips")
st.sidebar.markdown("""
- Ensure your English dataset is clean and well-formatted
- Larger datasets may take longer to process
- For best results, use GPU hardware for faster generation
""")

st.sidebar.header("📊 Stats")
st.sidebar.metric("Supported Languages", "7")
st.sidebar.metric("Avg. Processing Time", "2.5 mins")