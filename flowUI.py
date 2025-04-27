import streamlit as st
from PIL import Image
import torch
import os
import time
import json
from io import BytesIO
import base64
import pandas as pd
import numpy as np
from transformers import BitsAndBytesConfig, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from sentence_transformers import SentenceTransformer
import chromadb
import matplotlib.pyplot as plt
from flowsearch import FlowMatchingSearch
from yt_dlp import YoutubeDL
import requests
from pydub import AudioSegment
import tempfile
from streamlit_player import st_player

import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="MoodSync - Image to Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Apply custom CSS with horizontal navigation
# Apply custom CSS with horizontal navigation
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    
    /* Hide default sidebar */
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Horizontal Navigation Bar */
    .horizontal-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #191414;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    .nav-logo {
        display: flex;
        align-items: center;
    }
    
    .nav-logo img {
        height: 40px;
        margin-right: 10px;
    }
    
    .nav-logo-text {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .nav-logo-text span {
        color: #1DB954;
    }
    
    .nav-links {
        display: flex;
        gap: 0.3rem; /* Giảm khoảng cách giữa các nút điều hướng */
        justify-content: flex-start; /* Căn trái */
    }
    
    .nav-link {
        color: white;
        background-color: rgba(29, 185, 84, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 500;
        text-decoration: none;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .nav-link:hover {
        background-color: rgba(29, 185, 84, 0.2);
    }
    
    .nav-link.active {
        background-color: #1DB954;
        color: white;
    }
    
    .nav-link-icon {
        margin-right: 5px;
    }
    
    /* Settings panel floating button */
    .settings-button {
        position: fixed;
        right: 20px;
        bottom: 20px;
        background-color: #1DB954;
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        cursor: pointer;
        z-index: 1000;
        transition: transform 0.2s ease;
    }
    
    .settings-button:hover {
        transform: scale(1.1);
    }
    
    /* Typography */
    .title-text {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #1DB954, #191414);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 1rem !important;
        text-align: center;
    }
    
    .subtitle-text {
        font-size: 1.2rem !important;
        font-weight: 400 !important;
        color: #64748b !important;
        margin-bottom: 2rem !important;
        text-align: center;
    }
    
    # h1, h2, h3 {
    #     color: #191414 !important;
    # }
    
    /* Đảm bảo h3 trong dashboard-card có màu trắng */
    .dashboard-card h3 {
        color: #ffffff !important;
    }
    
    /* Cards */
    .card-container {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        background-color: white;
        box-shadow: 0 4px 15px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    
    .card-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    /* Dashboard cards */
    .dashboard-card {
        background: #2c2c2c;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        padding: 2rem;
        margin-bottom: 1.5rem;
        color: white;
        transition: transform 0.3s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
    }
    
    /* Music Cards */
    .music-card {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        height: 100%;
        background: white;
        display: flex;
        flex-direction: column;
    }
    
    .music-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        transform: translateY(-5px);
    }
    
    .music-card-header {
        padding: 1rem;
        border-bottom: 1px solid #f0f0f0;
        background: linear-gradient(135deg, #1DB954, #191414);
        color: white;
    }
    
    .music-card-body {
        padding: 1rem;
        flex-grow: 1;
    }
    
    .music-card-footer {
        padding: 1rem;
        border-top: 1px solid #f0f0f0;
        background: #fafafa;
    }
    
    /* Tags */
    .genre-tag {
        display: inline-block;
        padding: 5px 10px;
        margin: 5px 5px 5px 0;
        border-radius: 50px;
        background-color: #1DB954;
        color: white;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .emotion-tag {
        display: inline-block;
        padding: 5px 10px;
        margin: 5px 5px 5px 0;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Buttons */
    div.stButton > button {
        background-color: #1DB954;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        background-color: #18a449;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Inputs */
    div.stTextInput > div > div > input {
        border-radius: 50px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
    }
    
    div.stTextInput > div > div > input:focus {
        border-color: #1DB954;
        box-shadow: 0 0 0 2px rgba(29, 185, 84, 0.3);
    }
    
    /* File uploader */
    .stFileUploader > div > button {
        background-color: #1DB954;
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #1DB954;
    }
    
    /* History cards */
    .history-item {
        display: flex;
        margin-bottom: 1rem;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .history-image {
        width: 150px;
        height: 150px;
        object-fit: cover;
    }
    
    .history-content {
        padding: 1rem;
        flex: 1;
    }
    
    /* Settings panel */
    .settings-panel {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e0e0e0;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        margin-right: 2px;
        color: #191414 !important; /* Đảm bảo văn bản luôn hiển thị */
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #d0d0d0; /* Hiệu ứng hover nhẹ */
        color: #1DB954 !important; /* Đổi màu chữ khi hover */
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1DB954 !important; /* Tab đang hoạt động */
        color: white !important; /* Màu chữ trắng cho tab đang hoạt động */
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #1DB954;
        height: 3px;
    }
    
    /* Make dataframe headers match theme */
    .dataframe th {
        background-color: #1DB954 !important;
        color: white !important;
    }
    
    /* Container borders */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    [data-testid="stMetricValue"] {
        color: #1DB954 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .nav-links {
            gap: 0.2rem;
        }
        .nav-link {
            padding: 0.4rem 0.8rem;
            font-size: 0.8rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 12px;
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Set device based on availability of GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CHROMA_DB_PATH = "./chroma_db"
SONG_COLLECTION_NAME = "spotify_tracks_collection_all_features"
FAVORITES_FILE = "user_favorites.json"

# Create a session state object to store user favorites
if 'favorites' not in st.session_state:
    # Load favorites from file if exists
    if os.path.exists(FAVORITES_FILE):
        with open(FAVORITES_FILE, 'r') as f:
            st.session_state.favorites = json.load(f)
    else:
        st.session_state.favorites = []

if 'history' not in st.session_state:
    st.session_state.history = []

if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'preferred_genres': [],
        'mood_weight': 0.7,  # Default weight for mood vs genre
        'energy_level': 'medium'
    }

if 'show_settings' not in st.session_state:
    st.session_state.show_settings = False

# Enhanced YouTube URL finder with video extraction
@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def get_audio_stream(query):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'noplaylist': True,
        'geo_bypass': True,
        'extractor_args': {'youtube': {'skip': ['dash', 'hls']}},
        'default_search': 'ytsearch'
    }
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(f"ytsearch:{query} official video", download=False)['entries'][0]
            video_url = f"https://www.youtube.com/watch?v={info.get('id')}"
            audio_url = info.get('url')
            thumbnail = info.get('thumbnail')
            return video_url, audio_url, thumbnail
        except Exception as e:
            try:  # Try again with simpler query
                info = ydl.extract_info(f"ytsearch:{query}", download=False)['entries'][0]
                video_url = f"https://www.youtube.com/watch?v={info.get('id')}"
                audio_url = info.get('url')
                thumbnail = info.get('thumbnail')
                return video_url, audio_url, thumbnail
            except:
                return None, None, None

# Load models with caching
@st.cache_resource
def load_embedding_model():
    # Create or load the chromadb collection
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.create_collection(name=SONG_COLLECTION_NAME)
    except:
        collection = client.get_collection(name=SONG_COLLECTION_NAME)
    
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Sentence embedding model
    return embed_model, collection

@st.cache_resource
def load_LLM_model():
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            quantization_config=bnb_config,
            use_cache=True
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, use_cache=True)
    
    processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    try:
        model.load_adapter("./checkpoint_qwen")  # Load adapter if available
    except:
        st.warning("Could not load adapter. Using base model.")
    return model, processor

# Initialize flow matching search system
@st.cache_resource
def load_flow_matching_search(_embed_model, _collection):
    return FlowMatchingSearch(
        embedding_model=_embed_model,
        collection=_collection,
        device=device
    )

# Simple sentiment analysis function
def analyze_sentiment(text):
    # Simple keyword-based emotion detection
    emotions = {
        'joy': 0,
        'sadness': 0,
        'anger': 0,
        'fear': 0,
        'surprise': 0,
        'love': 0,
        'excitement': 0
    }
    
    # Simple keyword matching for emotions
    emotion_keywords = {
        'joy': ['happy', 'joy', 'joyful', 'glad', 'delight', 'pleased', 'cheerful', 'content'],
        'sadness': ['sad', 'unhappy', 'somber', 'depressed', 'gloomy', 'sorrow', 'melancholy'],
        'anger': ['angry', 'mad', 'furious', 'rage', 'annoyed', 'irritated', 'indignant'],
        'fear': ['fear', 'afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried'],
        'surprise': ['surprised', 'astonished', 'amazed', 'shocked', 'startled', 'unexpected'],
        'love': ['love', 'affection', 'fondness', 'adore', 'caring', 'compassion', 'warmth'],
        'excitement': ['excited', 'thrilled', 'eager', 'enthusiastic', 'exhilarated', 'animated']
    }
    
    # Check for emotion keywords
    text_lower = text.lower()
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                emotions[emotion] += 1
    
    # Normalize emotions
    total = sum(emotions.values())
    if total > 0:
        for emotion in emotions:
            emotions[emotion] /= total
    else:
        # If no emotions detected, provide a balanced default
        emotions = {e: 1/len(emotions) for e in emotions}
    
    # Get dominant emotion
    dominant_emotion = max(emotions, key=emotions.get)
    
    return {
        'polarity': 0,  # Simplified
        'emotions': emotions,
        'dominant_emotion': dominant_emotion
    }

# Function to get a color for an emotion
def get_emotion_color(emotion):
    colors = {
        'joy': '#FFD700',       # Gold
        'sadness': '#4169E1',   # Royal Blue
        'anger': '#FF4500',     # Red Orange
        'fear': '#800080',      # Purple
        'surprise': '#00FFFF',  # Cyan
        'love': '#FF1493',      # Deep Pink
        'excitement': '#FF8C00' # Dark Orange
    }
    return colors.get(emotion, '#808080')  # Default gray

# Enhanced image to base64 function (for caching images)
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load the models
try:
    with st.spinner("Loading models... Please wait..."):
        embed_model, collection = load_embedding_model()
        model, processor = load_LLM_model() 
        flow_search = load_flow_matching_search(embed_model, collection)
        models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

############################################################

# Main app with horizontal navigation
def main():
    # Create horizontal navigation bar using Streamlit components
    st.markdown("""
    <div class="horizontal-nav">
        <div class="nav-logo">
            <img src="https://i1.sndcdn.com/avatars-000545467674-x65c1f-t240x240.jpg">
            <div class="nav-logo-text">Mood<span>Sync</span></div>
        </div>
        <div class="nav-links">
    """, unsafe_allow_html=True)

    # Navigation links using st.button
    current_page = st.query_params.get("page", "home")
    
    # Use a single container with smaller columns to place buttons closer
    with st.container():
        col1, col2, col3, col4, _ = st.columns([1, 1, 1, 1, 4])  # Giảm chiều rộng của các cột, để lại không gian trống bên phải
        with col1:
            if st.button("🏠 Home", key="nav_home", type="primary" if current_page == "home" else "secondary"):
                st.query_params["page"] = "home"
                st.rerun()
        with col2:
            if st.button("🔍 Discover", key="nav_discover", type="primary" if current_page == "discover" else "secondary"):
                st.query_params["page"] = "discover"
                st.rerun()
        with col3:
            if st.button("📜 History", key="nav_history", type="primary" if current_page == "history" else "secondary"):
                st.query_params["page"] = "history"
                st.rerun()
        with col4:
            if st.button("ℹ️ About", key="nav_about", type="primary" if current_page == "about" else "secondary"):
                st.query_params["page"] = "about"
                st.rerun()

    st.markdown("""
        </div>
    </div>
    
    <div class="settings-button" onclick="document.getElementById('settings-toggle').click();">
        ⚙️
    </div>
    """, unsafe_allow_html=True)
    
    # Hidden button to toggle settings panel
    if st.button("Toggle Settings", key="settings-toggle", help="Toggle settings panel"):
        st.session_state.show_settings = not st.session_state.show_settings
    
    # Display settings panel if enabled
    if st.session_state.show_settings:
        show_settings_panel()
    
    # Get current page from URL query parameter
    current_page = st.query_params.get("page", "home")
    
    # Show the selected page
    if current_page == "discover":
        show_discover()
    elif current_page == "history":
        show_history()
    elif current_page == "about":
        show_about()
    else:  # Default to home
        show_home()

def show_settings_panel():
    # Khởi tạo search_settings nếu chưa tồn tại
    if 'search_settings' not in st.session_state:
        st.session_state.search_settings = {
            'method': "Traditional Similarity",
            'flow_steps': 50,
            'visualize_flow': False,
            'n_results': 12,  # Giá trị mặc định ban đầu
            'apply_preferences': True
        }

    # Lấy giá trị hiện tại từ session_state
    current_settings = st.session_state.search_settings

    st.markdown("""
    <div class="settings-panel">
        <h3 style="margin-top: 0; color: #1DB954;">Search Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Search method radio buttons
    search_method = st.radio(
        "Search Method",
        ["Traditional Similarity", "Flow Matching Search"],
        index=0 if current_settings['method'] == "Traditional Similarity" else 1,
        horizontal=True,
        help="Choose how MoodSync will find matching music"
    )
    
    # Flow matching specific settings
    col1, col2 = st.columns(2)
    
    with col1:
        if search_method == "Flow Matching Search":
            flow_steps = st.slider("Flow Steps", 10, 100, current_settings['flow_steps'], 
                                help="Higher values give more accurate but slower results")
            visualize_flow = st.checkbox("Visualize Flow", value=current_settings['visualize_flow'], 
                                       help="Show the trajectory in embedding space")
        else:
            flow_steps = 50
            visualize_flow = False
    
    with col2:
        # Sử dụng giá trị từ session_state làm giá trị khởi tạo cho slider
        n_results = st.slider("Number of Results", 5, 30, current_settings['n_results'], 
                            help="How many music recommendations to show")
        
        apply_preferences = st.checkbox("Apply Genre Preferences", value=current_settings['apply_preferences'],
                                      help="Factor in your preferred genres")
    
    # Tạm thời lưu các giá trị mới vào một dictionary
    temp_settings = {
        'method': search_method,
        'flow_steps': flow_steps,
        'visualize_flow': visualize_flow,
        'n_results': n_results,
        'apply_preferences': apply_preferences
    }

    # Thêm nút "OK" và "Cancel" để xác nhận
    col_confirm, col_cancel = st.columns(2)
    
    with col_confirm:
        if st.button("OK", key="confirm_settings", help="Save your settings"):
            # Khi người dùng nhấn "OK", lưu các giá trị vào session_state
            st.session_state.search_settings = temp_settings
            st.success("Settings saved successfully!")
    
    with col_cancel:
        if st.button("Cancel", key="cancel_settings", help="Discard changes"):
            # Đặt lại các widget về giá trị đã lưu
            st.rerun()  # Thay st.experimental_rerun() bằng st.rerun()
    
    # Display current runtime info
    st.markdown(f"""
    <div style="background-color: rgba(29, 185, 84, 0.1); padding: 10px; border-radius: 5px; border-left: 3px solid #1DB954;">
        <p style="margin: 0;"><b>System Info:</b> Running on {device.upper()}</p>
        {f'<p style="margin: 0; color: #1DB954;">{torch.cuda.get_device_name(0)}</p>' if device == "cuda" else ''}
    </div>
    """, unsafe_allow_html=True)

def show_home():
    st.markdown("<h1 class='title-text' style='text-align: left;'>Welcome to MoodSync</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text' style='text-align: left;'>Discover music that matches your visual emotions</p>", unsafe_allow_html=True)

    
    # Main sections in cards
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class='dashboard-card' style='background-color: #2c2c2c; color: white; padding: 20px; border-radius: 10px;'>
        <h3 style='color: #ffffff;'>How It Works</h3>
        
        <ol>
            <li><b>Upload an image</b> that represents your mood</li>
            <li>Our AI will <b>analyze the emotional content</b> of your image</li>
            <li>Get <b>personalized music recommendations</b> based on the detected emotions</li>
            <li><b>Save your favorites</b> for later listening</li>
        </ol>

        <h3 style='color: #ffffff;'>Key Features</h3>
        """, unsafe_allow_html=True)

        # Use Streamlit's native columns for the features grid
        feat_col1, feat_col2 = st.columns(2)

        with feat_col1:
            st.markdown("""
            <div style="background: rgba(29, 185, 84, 0.1); padding: 10px; border-radius: 5px;">
                <h4 style="color: #1DB954; margin-top: 0;">🧠 AI Emotion Analysis</h4>
                <p style="color: white;">Advanced image emotion recognition</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background: rgba(29, 185, 84, 0.1); padding: 10px; border-radius: 5px; margin-top: 10px;">
                <h4 style="color: #1DB954; margin-top: 0;">⚙️ Personalization</h4>
                <p style="color: white;">Customized to your preferences</p>
            </div>
            """, unsafe_allow_html=True)

        with feat_col2:
            st.markdown("""
            <div style="background: rgba(29, 185, 84, 0.1); padding: 10px; border-radius: 5px;">
                <h4 style="color: #1DB954; margin-top: 0;">🎵 Smart Matching</h4>
                <p style="color: white;">Two powerful search algorithms</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background: rgba(29, 185, 84, 0.1); padding: 10px; border-radius: 5px; margin-top: 10px;">
                <h4 style="color: #1DB954; margin-top: 0;">📊 Analytics</h4>
                <p style="color: white;">Visualize your music journey</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)



def show_about():
    # CSS tùy chỉnh
    st.markdown("""
        <style>
            h2 {
                color: #1DB954;
            }
            .step-number {
                background: #1DB954;
                color: white;
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
            }
            .faq-question {
                font-weight: 600;
                margin-top: 10px;
            }
            .about-image {
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .footer {
                text-align: center;
                font-size: 14px;
                margin-top: 30px;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)

    # Tiêu đề chính
    st.markdown("""
                <h2 style='color: #00e0ff;'>🎧 About MoodSync</h2>
                """, unsafe_allow_html=True)
    st.markdown("<i>The technology behind our image-to-music recommendations</i>", unsafe_allow_html=True)

    # Nội dung chính
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### How MoodSync Works")
        st.write("MoodSync uses advanced AI to analyze images and find music that matches their emotional content:")

        # Các bước xử lý
        for i, (title, desc) in enumerate([
            ("Image Analysis", "We use Qwen2.5-VL to analyze your images and extract emotional content."),
            ("Emotion Detection", "Our system identifies the dominant emotions and their intensity."),
            ("Music Matching", "We use two sophisticated search methods to find music that matches the emotional profile."),
            ("Personalization", "Your preferences for genres are factored into the recommendations.")
        ], 1):
            step_col, desc_col = st.columns([1, 5])
            with step_col:
                st.markdown(f'<div class="step-number">{i}</div>', unsafe_allow_html=True)
            with desc_col:
                st.markdown(f"**{title}**")
                st.write(desc)

        # Technology Stack
        st.markdown("### Technology Stack")
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.markdown("**Computer Vision**")
            st.write("Qwen2.5-VL for image analysis")
            st.markdown("**Vector Search**")
            st.write("ChromaDB for similarity search")
        with tech_col2:
            st.markdown("**NLP**")
            st.write("Sentence embedding models")
            st.markdown("**UI**")
            st.write("Streamlit for interactive interface")

    with col2:

        st.markdown("### Frequently Asked Questions")
        st.markdown('<div class="faq-question">How accurate is the emotion detection?</div>', unsafe_allow_html=True)
        st.write("Our system combines vision-language models with keyword-based emotion detection. While no system is perfect, our approach works well for a wide range of images.")

        st.markdown('<div class="faq-question">What\'s the difference between search methods?</div>', unsafe_allow_html=True)
        st.write("**Traditional Similarity Search** finds direct matches to your image's emotions.")
        st.write("**Flow Matching Search** models trajectories between your image's emotions and music, discovering more nuanced connections.")

        st.markdown('<div class="faq-question">Can I use my own music library?</div>', unsafe_allow_html=True)
        st.write("Currently, MoodSync uses a curated database of songs. Support for custom libraries is coming in a future update.")

        st.markdown('<div class="faq-question">Version Information</div>', unsafe_allow_html=True)
        st.write("**MoodSync:** v1.2.0")
        st.write("**Model Version:** Qwen2.5-VL-3B-Instruct")
        st.write("**Last Updated:** April 2025")

    # Footer section
    st.markdown("### 🚀 Get Started Today")
    st.write("Discover the perfect soundtrack for any visual emotion")
    if st.button("🎵 Try MoodSync Now"):
        st.query_params["page"] = "discover"
        st.rerun()
        
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://i1.sndcdn.com/avatars-000545467674-x65c1f-t240x240.jpg' width='80'>

        </div>
        """, unsafe_allow_html=True)


    st.markdown('<div class="footer">© 2025 MoodSync - Connecting Images to Music<br>Made with ❤️ for music lovers everywhere</div>', unsafe_allow_html=True)


# skk

def show_discover():
    st.markdown("<h1 class='title-text'>Discover Music From Images</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Upload an image to find matching music</p>", unsafe_allow_html=True)
    
    # Get search settings
    search_settings = st.session_state.get('search_settings', {
        'method': "Traditional Similarity",
        'flow_steps': 50,
        'visualize_flow': False,
        'n_results': 12,
        'apply_preferences': True
    })
    
    # Image upload with better styling
    st.markdown("""
    <div class="dashboard-card">
        <h3>Step 1: Upload Your Image</h3>
        <p>Select an image that represents your current mood or the emotion you want to explore</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Image upload
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    # Camera input option
    use_camera = st.checkbox("Or use camera to take a photo", False)
    if use_camera:
        camera_file = st.camera_input("")
        if camera_file is not None:
            uploaded_file = camera_file
    if uploaded_file is not None:
        # Process the image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Save image for history
        img_path = f"temp_img_{int(time.time())}.jpg"
        image.save(img_path)
        
        st.markdown("""
        <div class="dashboard-card">
            <h3>Step 2: Image Analysis</h3>
            <p>Our AI analyzes the emotional content of your image</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: white; padding: 10px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            """, unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if models_loaded:
                with st.spinner("Analyzing image emotions..."):
                    # Process image with model
                    image_inputs = image.resize((224, 224))
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": image}, 
                        {"type": "text", "text": "Describe this image with deep emotional detail. What emotions and mood does it convey?"}
                    ]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)
                    
                    # Generate caption
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    
                    caption = output_text[0]
                    
                    # Analyze sentiment
                    sentiment_analysis = analyze_sentiment(caption)
                    dominant_emotion = sentiment_analysis['dominant_emotion']
                    
                    # Display results with better styling
                    st.markdown(f"""
                    <div class="dashboard-card">
                        <h4>Emotional Analysis</h4>
                        <div style="margin: 15px 0;">
                            <span class="emotion-tag" style="background-color: {get_emotion_color(dominant_emotion)}; font-size: 1rem; padding: 8px 15px;">
                                {dominant_emotion.title()}
                            </span>
                        </div>
                        <h4>Image Caption</h4>
                        <div style="background-color: black; padding: 15px; border-radius: 8px; border-left: 4px solid #1DB954; margin-top: 10px;">
                            <p style="margin: 0; font-style: italic;">{caption}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display emotions as horizontal bars
                    st.markdown("""
                    <div class="dashboard-card" style="margin-top: 15px;">
                        <h4>Emotion Breakdown</h4>
                    """, unsafe_allow_html=True)
                    
                    emotions = sentiment_analysis['emotions']
                    for emotion, value in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                        if value > 0:
                            st.markdown(f"""
                            <div style="margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span>{emotion.title()}</span>
                                    <span>{int(value * 100)}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(value)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Add to history
                    history_item = {
                        'image_path': img_path,
                        'caption': caption,
                        'emotion': dominant_emotion,
                        'sentiment': sentiment_analysis,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.history.append(history_item)
            else:
                st.error("Models failed to load. Please check your environment and dependencies.")
                return
        
        # Music recommendations with improved styling
        st.markdown("""
        <div class="dashboard-card">
            <h3>Step 3: Music Recommendations</h3>
            <p>Discover music that matches the emotional profile of your image</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Finding your perfect music matches..."):
            try:
                # Search using selected method
                if search_settings['method'] == "Traditional Similarity":
                    # Create query embedding
                    query_embedding = embed_model.encode([caption])[0].tolist()
                    # Pad if needed
                    query_embedding = query_embedding + [0] * 15
                    
                    # Search
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=search_settings['n_results']
                    )
                else:
                    # Flow matching search
                    results = flow_search.search(
                        query_text=caption,
                        n_results=search_settings['n_results'],
                        flow_steps=search_settings['flow_steps'],
                        visualize=search_settings['visualize_flow']
                    )
                    
                    # Show visualization if available
                    if search_settings['visualize_flow'] and os.path.exists('flow_trajectory.png'):
                        st.markdown("<h4>Flow Trajectory Visualization</h4>", unsafe_allow_html=True)
                        st.image('flow_trajectory.png', caption="How the search traversed the embedding space", use_container_width=True)
                
                # Remove duplicate songs based on track_name and artists
                unique_tracks = {}
                for i, track_id in enumerate(results["ids"][0]):
                    track_name = results["metadatas"][0][i]['track_name']
                    artists = results["metadatas"][0][i]['artists']
                    track_key = f"{track_name}_{artists}"  # Create unique key from track name and artists
                    
                    if track_key not in unique_tracks:
                        unique_tracks[track_key] = {
                            'id': track_id,
                            'metadata': results["metadatas"][0][i],
                            'distance': results["distances"][0][i] if "distances" in results else None
                        }
                
                # Convert back to original format
                results["ids"][0] = [track['id'] for track in unique_tracks.values()]
                results["metadatas"][0] = [track['metadata'] for track in unique_tracks.values()]
                if "distances" in results:
                    results["distances"][0] = [track['distance'] for track in unique_tracks.values()]
                
                # Display tabs for results
                tab1, tab2, tab3 = st.tabs(["Card View", "Table View", "Analysis"])
                
                with tab1:
                    # Display enhanced cards in a grid
                    col_count = 4  # Number of columns
                    rows = [results["ids"][0][i:i+col_count] for i in range(0, len(results["ids"][0]), col_count)]
                    row_metadatas = [results["metadatas"][0][i:i+col_count] for i in range(0, len(results["metadatas"][0]), col_count)]
                    
                    for row_ids, row_meta in zip(rows, row_metadatas):
                        cols = st.columns(col_count)
                        for i, (id, metadata) in enumerate(zip(row_ids, row_meta)):
                            if i < len(cols):  # Safety check
                                with cols[i]:
                                    track_id = str(id)
                                    
                                    # Get YouTube link
                                    query = f"{metadata['track_name']} {metadata['artists']}"
                                    video_url, audio_url, thumbnail = get_audio_stream(query)
                                    
                                    # Render the music card with enhanced styling
                                    st.markdown(f"""
                                    <div class="music-card">
                                        <div class="music-card-header">
                                            <h4 style="margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: black">{metadata['track_name']}</h4>
                                            <p style="margin: 0; color: white; font-size: 0.9rem;">{metadata['artists']}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Display video player instead of thumbnail
                                    if video_url:
                                        try:
                                            st_player(video_url, height=200)
                                        except Exception as e:
                                            st.error(f"Error loading video: {str(e)}")
                                            st.image("https://i.ibb.co/2tDvBvL/music-placeholder.jpg", use_container_width=True)
                                    else:
                                        st.image("https://i.ibb.co/2tDvBvL/music-placeholder.jpg", use_container_width=True)
                                    
                                    st.markdown(f"""
                                    <div class="music-card-body">
                                        <span class="genre-tag">{metadata['track_genre']}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Favorites button with remove functionality
                                    is_favorite = track_id in [fav.get('id') for fav in st.session_state.favorites]
                                    
                                    if is_favorite:
                                        if st.button(f"❤️ Remove from Favorites", key=f"fav_{track_id}"):
                                            # Remove from favorites
                                            st.session_state.favorites = [fav for fav in st.session_state.favorites if fav.get('id') != track_id]
                                            # Save favorites
                                            with open(FAVORITES_FILE, 'w') as f:
                                                json.dump(st.session_state.favorites, f)
                                            st.success("Removed from favorites!")
                                            st.rerun()
                                    else:
                                        if st.button(f"Add to Favorites", key=f"fav_{track_id}"):
                                            # Add to favorites
                                            favorite = {
                                                'id': track_id,
                                                'track_name': metadata['track_name'],
                                                'artists': metadata['artists'],
                                                'genre': metadata['track_genre'],
                                                'added_date': time.strftime("%Y-%m-%d")
                                            }
                                            st.session_state.favorites.append(favorite)
                                            # Save favorites
                                            with open(FAVORITES_FILE, 'w') as f:
                                                json.dump(st.session_state.favorites, f)
                                            st.success("Added to favorites!")
                                            st.rerun()
                                    
                                    st.markdown("""
                                    </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                
                with tab2:
                    # Enhanced table view
                    st.markdown("""
                    <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: black">
                        <h4>All Recommendations</h4>
                    """, unsafe_allow_html=True)
                    
                    # Create a dataframe for table view with more details
                    results_df = pd.DataFrame([
                        {
                            "Track": metadata['track_name'],
                            "Artist": metadata['artists'],
                            "Genre": metadata['track_genre'],
                            "Match Score": round((1 - results["distances"][0][i]) * 100, 1) if "distances" in results else "N/A",
                        }
                        for i, metadata in enumerate(results["metadatas"][0])
                    ])
                    
                    # Display with better styling
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        column_config={
                            "Match Score": st.column_config.ProgressColumn(
                                "Match Score (%)",
                                format="%f%%",
                                min_value=0,
                                max_value=100,
                            ),
                        },
                        hide_index=True
                    )
                    
                    # Export button
                    if st.button("📥 Export as CSV", use_container_width=False):
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            f"moodsync_recommendations_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tab3:
                    # Enhanced analysis view
                    st.markdown("""
                    <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: black">
                        <h4>Recommendation Analysis</h4>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Count genres
                        genres = [m['track_genre'] for m in results["metadatas"][0]]
                        genre_counts = {genre: genres.count(genre) for genre in set(genres)}
                        
                        # Create a DataFrame for the chart
                        genre_df = pd.DataFrame({
                            'Genre': list(genre_counts.keys()),
                            'Count': list(genre_counts.values())
                        }).sort_values(by='Count', ascending=False)
                        
                        st.markdown("<h5>Genre Distribution</h5>", unsafe_allow_html=True)
                        
                        # Style the chart with custom colors
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(genre_df['Genre'], genre_df['Count'], color='#1DB954')
                        ax.set_xticklabels(genre_df['Genre'], rotation=45, ha='right')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#DDDDDD')
                        ax.spines['bottom'].set_color('#DDDDDD')
                        ax.tick_params(bottom=False, left=False)
                        ax.set_axisbelow(True)
                        ax.yaxis.grid(True, color='#EEEEEE')
                        ax.xaxis.grid(False)
                        
                        # Add value labels to the bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.annotate(f'{height}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha='center', va='bottom')
                        
                        fig.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("<h5>Emotional Match Analysis</h5>", unsafe_allow_html=True)
                        
                        # Create a pie chart showing dominant emotion vs recommendations
                        emotions = sentiment_analysis['emotions']
                        top_emotions = {k: v for k, v in sorted(emotions.items(), key=lambda item: item[1], reverse=True) if v > 0.1}
                        
                        if top_emotions:
                            fig, ax = plt.subplots(figsize=(8, 8))
                            wedges, texts, autotexts = ax.pie(
                                top_emotions.values(), 
                                labels=top_emotions.keys(),
                                autopct='%1.1f%%',
                                startangle=90,
                                wedgeprops={'edgecolor': 'white'},
                                textprops={'fontsize': 12, 'color': 'black'},
                                colors=[get_emotion_color(e) for e in top_emotions.keys()]
                            )
                            
                            # Equal aspect ratio ensures that pie is drawn as a circle
                            ax.axis('equal')
                            plt.setp(autotexts, size=10, weight="bold", color="white")
                            plt.setp(texts, size=12)
                            
                            # Add a title to the pie chart
                            ax.set_title("Top Emotions in Your Image", fontsize=14, pad=20)
                            
                            fig.tight_layout()
                            st.pyplot(fig)
                            
                            # Add explanation text
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px; color: black">
                                <p><b>Dominant Emotion:</b> {dominant_emotion.title()}</p>
                                <p>The recommendations are based on matching songs to these emotional cues in your image.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("Not enough emotion data to visualize.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                st.info("Try again with a different image or search method.")

# Add this function - show_history() implementation
def show_history():
    st.markdown("<h1 class='title-text'>Your History</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Review your past image analyses and music discoveries</p>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        # Empty state with better styling
        st.markdown("""
        <div style="text-align: center; padding: 50px 20px; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style='color: black;'>Your History is Empty</h3>
            <p style="color: #64748b; margin-bottom: 30px;">Start by uploading images in the Discover page to build your history.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get started button
        if st.button("🔍 Go to Discover", use_container_width=False):
            # Navigate to discover page
            st.query_params["page"] = "discover"
            st.rerun()
        return
    
    # History management controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"<p>You have <b>{len(st.session_state.history)}</b> items in your history.</p>", unsafe_allow_html=True)
    
    with col2:
        # Clear history button with confirmation
        if st.button("🗑️ Clear History", use_container_width=True):
            confirm = st.checkbox("Confirm deletion? This cannot be undone.")
            if confirm:
                st.session_state.history = []
                st.success("History cleared successfully!")
                st.rerun()
    
    # Display history items with enhanced styling
    for i, item in enumerate(reversed(st.session_state.history)):
        # Enhanced card-based layout
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; overflow: hidden; display: flex;">
            <div style="width: 200px; min-width: 200px; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center;">
        """, unsafe_allow_html=True)
        
        # Display the image
        if 'image_path' in item and os.path.exists(item['image_path']):
            st.image(item['image_path'], width=200)
        else:
            st.image("https://i.ibb.co/wQhGZFX/sample1.jpg", width=200)
        
        st.markdown("""
            </div>
            <div style="padding: 20px; flex-grow: 1;">
        """, unsafe_allow_html=True)
        
        # History item content
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"<h4>Analysis #{len(st.session_state.history) - i}</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <span class="emotion-tag" style="background-color: {get_emotion_color(item.get('emotion', 'joy'))};">
                    {item.get('emotion', 'Unknown').title()}
                </span>
                <span style="color: #64748b; font-size: 0.9rem; margin-left: 10px;">
                    {item.get('timestamp', 'Unknown')}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Caption with expander
            if 'caption' in item:
                caption = item['caption']
                if len(caption) > 150:
                    st.markdown(f"<p><b>Caption:</b> {caption[:150]}...</p>", unsafe_allow_html=True)
                    with st.expander("Show full caption"):
                        st.write(caption)
                else:
                    st.markdown(f"<p><b>Caption:</b> {caption}</p>", unsafe_allow_html=True)
        
        with col2:
            # Action buttons
            # if st.button("🔄 Reuse", key=f"reuse_{i}"):
            #     st.session_state.reuse_item = item
            #     st.success("Ready to reuse! Go to Discover page to continue.")
            #     # Navigate to discover page
            #     st.query_params["page"] = "discover"
            
            if st.button("🗑️ Remove", key=f"remove_{i}"):
                st.session_state.history.remove(item)
                st.rerun()
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

# Add the main function call at the end of the file
if __name__ == "__main__":
    main()