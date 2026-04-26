"""
DEPLANT – Deep Learning for Plant Disease Prediction
=====================================================
A professional Streamlit web application for predicting plant diseases
using a trained CNN model. Upload a leaf image to get instant predictions.

Usage:
    streamlit run app.py

Authors:
    - Aditya Kumar Gupta
    - Pratishtha Srivastava
    - Dr. Swathy R
"""

import streamlit as st
import numpy as np
from PIL import Image
import time
import os
from google import genai
from google.genai import types
from deep_translator import GoogleTranslator

# ─────────────────────────────────────────────
# Multilanguage Setup
# ─────────────────────────────────────────────
LANGUAGES = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Marathi (मराठी)": "mr",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Gujarati (ગુજરાતી)": "gu"
}

@st.cache_data(show_spinner=False)
def t(text, target_lang):
    if target_lang == "en" or not text.strip():
        return text
    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated
    except Exception:
        return text

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DEPLANT – Plant Disease Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Disease Class Labels (16 classes)
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "Apple – Apple Scab",
    "Apple – Black Rot",
    "Apple – Cedar Apple Rust",
    "Apple – Healthy",
    "Corn (Maize) – Cercospora Leaf Spot",
    "Corn (Maize) – Common Rust",
    "Corn (Maize) – Northern Leaf Blight",
    "Corn (Maize) – Healthy",
    "Grape – Black Rot",
    "Grape – Esca (Black Measles)",
    "Grape – Leaf Blight",
    "Grape – Healthy",
    "Tomato – Bacterial Spot",
    "Tomato – Early Blight",
    "Tomato – Late Blight",
    "Tomato – Healthy",
]

# ─────────────────────────────────────────────
# Custom CSS – Green Accent Theme
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---------- Root variables ---------- */
    :root {
        --green-dark: #1B5E20;
        --green-mid: #2E7D32;
        --green-light: #4CAF50;
        --green-pale: #E8F5E9;
        --green-accent: #00E676;
        --text-dark: #1a1a2e;
        --text-light: #fafafa;
        --card-bg: #ffffff;
        --shadow: 0 4px 24px rgba(0,0,0,0.08);
    }

    /* ---------- Global ---------- */
    /* Remove the hardcoded backgrounds that break dark mode */
    .stApp {
        background-color: transparent;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--green-dark) 0%, var(--green-mid) 100%);
        color: var(--text-light);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-light) !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: var(--text-light) !important;
    }

    /* ---------- Hero Banner ---------- */
    .hero-banner {
        background: linear-gradient(135deg, var(--green-dark), var(--green-light));
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .hero-banner p {
        color: #E8F5E9;
        font-size: 1.15rem;
        max-width: 700px;
        margin: 0 auto;
    }

    /* ---------- Feature / Info Cards ---------- */
    .card {
        background: var(--card-bg);
        border-left: 5px solid var(--green-light);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    }
    .card h3 {
        color: var(--green-dark);
        margin-top: 0;
    }
    .card p {
        color: #555;
    }

    /* ---------- Result Card ---------- */
    .result-card {
        background-color: var(--card-bg);
        border: 2px solid var(--green-light);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: var(--shadow);
    }
    .result-card h2 {
        color: var(--green-dark);
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .result-card .confidence {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--green-mid);
    }

    /* ---------- Author Card ---------- */
    .author-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow);
        border-top: 4px solid var(--green-light);
    }
    .author-card h4 {
        color: var(--green-dark);
        margin-bottom: 0.25rem;
    }
    .author-card p {
        color: #666;
        font-size: 0.9rem;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 3rem;
        background: linear-gradient(90deg, var(--green-dark), var(--green-mid));
        border-radius: 12px;
        color: var(--text-light);
        font-size: 0.95rem;
    }

    /* ---------- Upload area ---------- */
    .stFileUploader > div {
        border: 2px dashed var(--green-light) !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained CNN model from disk."""
    from tensorflow.keras.models import load_model as keras_load_model

    model_path = os.path.join(os.path.dirname(__file__), "plant_disease_model.h5")
    model = keras_load_model(model_path)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the uploaded image for model inference.
    - Resize to 224×224
    - Normalize pixel values to [0, 1]
    - Expand dimensions for batch input
    """
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim
    return img_array


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### Language")
    selected_lang = st.selectbox("Select Language / भाषा चुनें", options=list(LANGUAGES.keys()), index=0)
    lang_code = LANGUAGES[selected_lang]
    st.markdown("---")

    st.markdown(f"## 🌿 {t('DEPLANT', lang_code)}")
    st.markdown(
        t("**Deep Learning for Plant Disease Prediction** — "
        "An intelligent system that leverages Convolutional Neural Networks "
        "to identify plant diseases from leaf images.", lang_code)
    )

    st.markdown("---")
    st.markdown(f"### 🧠 {t('Model Details', lang_code)}")
    st.markdown(
        f"""
        - **{t('Architecture', lang_code)}:** {t('CNN , VGG16 , ResNet', lang_code)}
        - **{t('Dataset', lang_code)}:** PlantVillage
        - **{t('Classes', lang_code)}:** {t('16 disease categories', lang_code)}
        - **{t('Input Size', lang_code)}:** 224 × 224 px
        - **{t('Framework', lang_code)}:** TensorFlow / Keras
        """
    )

    st.markdown("---")
    
    home_lbl = t('Home', lang_code)
    predict_lbl = t('Predict', lang_code)
    agribot_lbl = t('AgriBot', lang_code)
    about_lbl = t('About', lang_code)
    
    page_options = [f"🏠 {home_lbl}", f"🔬 {predict_lbl}", f"💬 {agribot_lbl}", f"ℹ️ {about_lbl}"]
    
    page = st.radio(
        f" 📌 {t('Navigation', lang_code)}",
        options=page_options,
        index=0,
    )

    # Hardcoded API key (invisible to frontend)
    api_key = "AIzaSyCnxNBQBBv1m3YZz6DnbdRGRCiR2anrSRc"


# ─────────────────────────────────────────────
# Home Page
# ─────────────────────────────────────────────
if page == page_options[0]:
    # Hero Banner
    st.markdown(
        f"""
        <div class="hero-banner">
            <h1>🌿 {t('DEPLANT', lang_code)}</h1>
            <p>
                {t('Deep Learning for Plant Disease Prediction — Empowering farmers with AI-driven diagnostics to detect plant diseases early and protect crop yield.', lang_code)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Agriculture-themed header image placeholder
    st.image(
        "https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?w=1200&q=80",
        caption="Harnessing technology for smarter agriculture",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("## ✨ Key Features")

    # Feature cards in a 3-column grid
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>📷 Easy Upload</h3>
                <p>Simply upload a photo of a plant leaf in JPG or PNG format to get started.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>🧠 AI-Powered</h3>
                <p>Uses a trained CNN model to accurately predict diseases from 16 categories.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="card">
                <h3>⚡ Instant Results</h3>
                <p>Get predictions in seconds with confidence scores for informed decisions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(
            """
            <div class="card">
                <h3>🌍 Smart Agriculture</h3>
                <p>Support sustainable farming with early disease detection and prevention.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            """
            <div class="card">
                <h3>📊 Confidence Score</h3>
                <p>View prediction confidence to assess reliability of the diagnosis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            """
            <div class="card">
                <h3>🔒 Privacy First</h3>
                <p>Your images are processed locally and never stored on external servers.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
                                                          
                                                          
# ─────────────────────────────────────────────
# Prediction Page
# ─────────────────────────────────────────────
elif page == page_options[1]:
    st.markdown(
        f"""
        <div class="hero-banner" style="padding: 2rem;">
            <h1 style="font-size: 2rem;">🔬 {t('Disease Prediction', lang_code)}</h1>
            <p>{t('Upload a leaf image to diagnose potential plant diseases', lang_code)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    ) 

    # Two-column layout: upload on left, results on right
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📤 Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of the plant leaf for best results.",
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📸 Uploaded Leaf Image", use_container_width=True)

    with col_result:
        st.markdown("### 🧪 Prediction Results")

        if uploaded_file is not None:
            # Predict button
            if st.button("🚀 Predict Disease", use_container_width=True, type="primary"):
                try:
                    # Progress bar simulation
                    progress_bar = st.progress(0, text="Initializing model...")

                    with st.spinner("🔄 Loading model and analyzing image..."):
                        # Step 1 – Load model
                        progress_bar.progress(20, text="Loading CNN model...")
                        model = load_model()

                        # Step 2 – Preprocess image
                        progress_bar.progress(50, text="Preprocessing image...")
                        img_array = preprocess_image(image)
                        time.sleep(0.3)  # brief pause for UX

                        # Step 3 – Run prediction
                        progress_bar.progress(75, text="Running prediction...")
                        predictions = model.predict(img_array)
                        predicted_index = np.argmax(predictions[0])
                        confidence = float(np.max(predictions[0])) * 100

                        # Step 4 – Complete
                        progress_bar.progress(100, text="✅ Prediction complete!")
                        time.sleep(0.3)

                    # Display results
                    predicted_disease = CLASS_NAMES[predicted_index]

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <h2>🌿 {predicted_disease}</h2>
                            <p style="color: #555; margin-bottom: 0.5rem;">Predicted Disease</p>
                            <div class="confidence">{confidence:.2f}%</div>
                            <p style="color: #777; margin-top: 0.5rem;">Confidence Score</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.success(
                        f"✅ Prediction successful! The model predicts **{predicted_disease}** "
                        f"with **{confidence:.2f}%** confidence."
                    )

                    # Healthy vs diseased feedback
                    if "Healthy" in predicted_disease:
                        st.balloons()
                        st.info("🌱 Great news! The plant appears to be healthy.")
                    else:
                        st.warning(
                            "⚠️ A disease has been detected. Consider consulting an agronomist "
                            "for treatment recommendations."
                        )

                except FileNotFoundError:
                    st.error(
                        "❌ **Model file not found!** Please ensure `plant_disease_model.h5` "
                        "is placed in the project directory."
                    )
                except Exception as e:
                    st.error(f"❌ **Prediction failed:** {str(e)}")
        else:
            st.info("👈 Upload a leaf image on the left to get started.")
                                                   
# ─────────────────────────────────────────────
# AgriBot Page
# ─────────────────────────────────────────────
elif page == page_options[2]:
    st.markdown(
        f"""
        <div class="hero-banner" style="padding: 2rem;">
            <h1 style="font-size: 2rem;">💬 {t('AgriBot', lang_code)}</h1>
            <p>{t('Your AI assistant for plant disease tips, weather context, and farm protection.', lang_code)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_key:
        st.error(
            "🔑 **API Key Required!** Please enter your Google Gemini API Key "
            "in the sidebar to start chatting with AgriBot."
        )
        st.info(
            "Don't have one? Get a free API key from "
            "[Google AI Studio](https://aistudio.google.com/app/apikey)."
        )
    else:
        # Configure Gemini using new google-genai package
        client = genai.Client(api_key=api_key)
        
        target_language_name = selected_lang.split(" (")[0]
        
        # System instructions to keep it focused on agriculture
        system_instruction = (
            f"You are AgriBot, an expert agricultural assistant and agronomist. "
            f"Your main focus is helping farmers deal with plant diseases, pests, "
            f"and providing weather-relevant agricultural advice. "
            f"Prioritize actionable, practical, and highly effective organic and synthetic "
            f"farming techniques. Keep answers concise, helpful, and highly professional. "
            f"IMPORTANT: You MUST respond to the user entirely in {target_language_name}."
        )

        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": t("Hello! I am AgriBot. How can I help you with your crops today?", lang_code)}
            ]

        # Optional Image Upload
        bot_image_upload = st.file_uploader(
            "📸 Show AgriBot a photo (Optional)", 
            type=["jpg", "jpeg", "png"], 
            key="bot_uploader"
        )

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "image" in message and message["image"] is not None:
                    st.image(message["image"], width=300)

        # Chat input
        if prompt := st.chat_input("Ask about diseases, weather effects, or protection techniques..."):
            
            # Read image if provided
            user_img = None
            if bot_image_upload is not None:
                user_img = Image.open(bot_image_upload).convert("RGB")

            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": prompt,
                "image": user_img
            })
            
            with st.chat_message("user"):
                st.markdown(prompt)
                if user_img is not None:
                    st.image(user_img, width=300)

            # Generate and display response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("AgriBot is typing..."):
                    # Try models in order until one works
                    MODELS_TO_TRY = [
                        'gemini-2.0-flash-lite',
                        'gemini-2.0-flash',
                        'gemini-2.5-flash',
                    ]
                    
                    contents = [f"System Instruction: {system_instruction}\n\nUser query: {prompt}"]
                    if user_img is not None:
                        contents.append(user_img)
                    
                    response_text = None
                    last_error = None
                    
                    for model_name in MODELS_TO_TRY:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=contents,
                                config=types.GenerateContentConfig(
                                    tools=[{'google_search': {}}]
                                )
                            )
                            response_text = response.text
                            break  # success — stop trying more models
                        except Exception as e:
                            last_error = e
                            if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                                continue  # quota hit — try next model
                            else:
                                break  # other error — don't retry
                    
                    if response_text:
                        message_placeholder.markdown(response_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    else:
                        quota_msg = (
                            "⚠️ **Daily API quota exhausted for all models.**\n\n"
                            "Your free Gemini API key has reached its daily limit. Here are your options:\n"
                            "1. **Wait** — quota resets every 24 hours (usually midnight US Pacific time).\n"
                            "2. **Get a new API key** — create one at [Google AI Studio](https://aistudio.google.com/app/apikey).\n"
                            "3. **Upgrade your plan** — add billing at [Google AI billing](https://ai.google.dev/gemini-api/docs/billing) for much higher limits."
                        )
                        message_placeholder.markdown(quota_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": quota_msg})

      
# ─────────────────────────────────────────────
# About Page
# ─────────────────────────────────────────────
elif page == page_options[3]:
    st.markdown(
        f"""
        <div class="hero-banner" style="padding: 2rem;">
            <h1 style="font-size: 2rem;">ℹ️ {t('About DEPLANT', lang_code)}</h1>
            <p>{t('Learn more about the project, models, and team', lang_code)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Project Overview
    st.markdown("## 📖 Project Overview")
    st.markdown(
        """
        <div class="card">
            <h3>🎯 Mission</h3>
            <p>
                <strong>DEPLANT</strong> (Deep Learning for Plant Disease Prediction) is an
                AI-powered application designed to help farmers and agricultural professionals
                identify plant diseases quickly and accurately. By uploading a simple photograph
                of a plant leaf, users receive instant predictions backed by deep learning models
                trained on the PlantVillage dataset.
            </p>
            <p>
                The system currently supports <strong>16 disease categories</strong> across
                multiple crop types including Apple, Corn, Grape, and Tomato. Early detection
                enables timely intervention, reducing crop losses and promoting sustainable
                agriculture.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Models Used
    st.markdown("## 🧠 Models & Roadmap")

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown(
            """
            <div class="card" style="border-left-color: #4CAF50;">
                <h3>✅ CNN </h3>
                <p>
                    Convolutional Neural Network trained on the PlantVillage dataset.
                    Achieves strong accuracy on 16-class classification.
                    Currently deployed as the primary model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """
            <div class="card" style="border-left-color: #FFC107;">
                <h3>✅ VGG16 </h3>
                <p>
                    VGG16 architecture used for feature extraction and high-accuracy classification.
                    Optimized for detecting fine-grained patterns in leaf textures.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        st.markdown(
            """
            <div class="card" style="border-left-color: #2196F3;">
                <h3>✅ ResNet </h3>
                <p>
                    ResNet architecture planned for enhanced accuracy using
                    transfer learning. Expected to improve performance on
                    difficult disease categories.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Authors
    st.markdown("## 👥 Project Authors")

    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        st.markdown(
            """
            <div class="author-card">
                <h4>Aditya Kumar Gupta</h4>
                <p>Developer & Researcher</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_a2:
        st.markdown(
            """
            <div class="author-card">
                <h4>Pratishtha Srivastava</h4>
                <p>Developer & Researcher</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_a3:
        st.markdown(
            """
            <div class="author-card">
                <h4>Dr. Swathy R</h4>
                <p>Project Guide & Mentor</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# Footer (shown on all pages except AgriBot)
# ─────────────────────────────────────────────
if page != page_options[2]:
    st.markdown("---")
    st.markdown(
        f"""
        <div class="footer">
            🌾 <strong>{t('Developed for Smart Agriculture', lang_code)}</strong> &nbsp;|&nbsp;
            DEPLANT © 2026 &nbsp;|&nbsp;
            {t('Powered by Deep Learning & Streamlit', lang_code)}
        </div>
        """,
        unsafe_allow_html=True,
    )


















