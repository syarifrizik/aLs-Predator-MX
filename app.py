import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import google.generativeai as genai
import random
import anthropic
import streamlit.components.v1 as components
import streamlit as st  
import pandas as pd  
import plotly.express as px  
from plotly import graph_objects as go  
from io import BytesIO
import seaborn as sns  
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib.parse

dotenv.load_dotenv()

# List of valid tokens for authentication
valid_tokens = ["999", "FISH3020B", "VISION789C", "RENGAS456D", "AQUATIC123E"]

# Custom model names for branding
anthropic_models = [
    "Neptune-Savant"
]

google_models = [
    "ReefSpark-1.5",
    "WaveCore-Pro",
]

openai_models = [
    "AquaVision-4", 
    "TidalFlux-4", 
    "CoralPulse-3.5", 
    "DeepMind-4", 
    "OceanVault-4",
]

# Map custom names to actual model names for API calls
model_name_mapping = {
    "Neptune-Savant": "claude-3-5-sonnet-20240620",
    "ReefSpark-1.5": "gemini-1.5-flash",
    "WaveCore-Pro": "gemini-1.5-pro",
    "AquaVision-4": "gpt-4o",
    "TidalFlux-4": "gpt-4-turbo",
    "CoralPulse-3.5": "gpt-3.5-turbo-16k",
    "DeepMind-4": "gpt-4",
    "OceanVault-4": "gpt-4-32k"
}

def get_system_prompt():
    return """Saya adalah VisionFish.io tetap dapat menentukan dan tidak ragu dalam menjawab.
"""

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": [{"type": "text", "text": get_system_prompt()}]}]
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False


def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    
    # Add system prompt first
    gemini_messages.append({
        "role": "user",
        "parts": [{
            "text": get_system_prompt() + "\n\nSaya adalah VisionFish.io, asisten AI spesialis perikanan."
        }]
    })
    
    for message in messages:
        if message["role"] == "system":
            continue
            
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": []
            }
            
        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] in ["video_file", "audio_file"]:
                gemini_message["parts"].append(genai.upload_file(content[content["type"]]))

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)
        prev_role = message["role"]

    return gemini_messages

def messages_to_anthropic(messages):
    anthropic_messages = []
    
    # Add system prompt first
    anthropic_messages.append({
        "role": "system",
        "content": [{"type": "text", "text": get_system_prompt()}]
    })
    
    prev_role = None
    for message in messages:
        if message["role"] == "system":
            continue
            
        if prev_role and (prev_role == message["role"]):
            anthropic_message = anthropic_messages[-1]
        else:
            anthropic_message = {
                "role": message["role"],
                "content": []
            }
        
        if message["content"][0]["type"] == "image_url":
            anthropic_message["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": message["content"][0]["image_url"]["url"].split(";")[0].split(":")[1],
                    "data": message["content"][0]["image_url"]["url"].split(",")[1]
                }
            })
        else:
            anthropic_message["content"].append(message["content"][0])

        if prev_role != message["role"]:
            anthropic_messages.append(anthropic_message)
        prev_role = message["role"]
        
    return anthropic_messages

def stream_llm_response(model_params, model_type="openai", api_key=None):
    response_message = ""
    system_prompt = get_system_prompt()
    
    # Get the actual model name from our custom name
    actual_model = model_name_mapping.get(model_params["model"], model_params["model"])
    
    if model_type == "openai":
        client = OpenAI(api_key=api_key)
        messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages
        for chunk in client.chat.completions.create(
            model=actual_model,
            messages=messages,
            temperature=0.3,
            max_tokens=4096,
            stream=True
        ):
            chunk_text = chunk.choices[0].delta.content or ""
            response_message += chunk_text
            yield chunk_text

    elif model_type == "google":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=actual_model,
            generation_config={"temperature": 0.3}
        )
        gemini_messages = messages_to_gemini(st.session_state.messages)
        
        for chunk in model.generate_content(contents=gemini_messages, stream=True):
            chunk_text = chunk.text or ""
            response_message += chunk_text
            yield chunk_text

    elif model_type == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        messages = messages_to_anthropic(st.session_state.messages)
        
        with client.messages.stream(
            model=actual_model,
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        ) as stream:
            for text in stream.text_stream:
                response_message += text
                yield text

    st.session_state.messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response_message}]
    })

def get_model_params(model_type):
    params = {
        "temperature": 0.3
    }
    
    if model_type == "openai":
        params["model"] = "AquaVision-4"
    elif model_type == "google":
        params["model"] = "WaveCore-Pro"
    elif model_type == "anthropic":
        params["model"] = "Neptune-Savant"
    
    return params

# Helper functions
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def file_to_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read())

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def process_fish_data(df):  
    # Menghitung jumlah setiap kategori kesegaran berdasarkan kondisi yang ada  
    prima = ((df[['Mata', 'Insang', 'Lendir', 'Daging', 'Bau', 'Tekstur']] == 9).all(axis=1)).sum()  
    advance = ((df[['Mata', 'Insang', 'Lendir', 'Daging', 'Bau', 'Tekstur']] >= 7) & (df[['Mata', 'Insang', 'Lendir', 'Daging', 'Bau', 'Tekstur']] <= 8)).all(axis=1).sum()  
    sedang = ((df[['Mata', 'Insang', 'Lendir', 'Daging', 'Bau', 'Tekstur']] >= 5) & (df[['Mata', 'Insang', 'Lendir', 'Daging', 'Bau', 'Tekstur']] <= 6)).all(axis=1).sum()  
    busuk = ((df[['Mata', 'Insang', 'Lendir', 'Daging', 'Bau', 'Tekstur']] >= 1) & (df[['Mata', 'Insang', 'Lendir', 'Daging', 'Bau', 'Tekstur']] <= 4)).all(axis=1).sum()  

    param_averages = {  
        "Mata": df['Mata'].mean(),  
        "Insang": df['Insang'].mean(),  
        "Lendir": df['Lendir'].mean(),  
        "Daging": df['Daging'].mean(),  
        "Bau": df['Bau'].mean(),  
        "Tekstur": df['Tekstur'].mean(),  
    }  

    processed_df = df.copy()  
    processed_df['Rata-rata'] = processed_df[['Mata', 'Insang', 'Lendir', 'Daging', 'Bau', 'Tekstur']].mean(axis=1)  

    # Logika penentuan 'Kesegaran' berdasarkan 'Rata-rata'  
    processed_df['Kesegaran'] = processed_df['Rata-rata'].apply(  
        lambda x: 'Prima' if x == 9 else   
                ('Advance' if 7 <= x < 9 else   
                ('Sedang' if 5 <= x < 7 else   
                    'Busuk'))  
    )  

    return processed_df, prima, advance, sedang, busuk, param_averages

def get_freshness_prompt():
    return """Analisis Kesegaran Ikan dari Gambar:
Parameter penilaian berdasarkan apa yang dilihat.
Skor:
- Sangat Baik (Excellent): Skor 9
- Masih Baik (Good): Skor 7-8
- Tidak Segar (Moderate): Skor 5-6
- Sangat Tidak Segar (Spoiled): Skor 1-4
Kesimpulan:

Skor: [Penilaian didasarkan pada interpretasi visual dari gambar digital menggunakan skala tersebut (1-9). Tulis skornya di sini, misalnya "X dari 9"].
Setiap kesimpulan disertai dengan alasan."""
    

def get_analisis_ikan_prompt():
    return """Analisis Kesegaran Ikan dari Gambar:
Identifikasi spesies dan klasifikasi ikan dalam gambar, berikan informasi dalam bentuk tabel berikut:

| Kategori    | Detail       |
|-------------|--------------|
| Nama Lokal  | [nama ikan]  |
| Nama Ilmiah | [nama latin] |
| Famili      | [famili]     |

Berikan jawaban dengan yakin tanpa kata-kata keraguan seperti 'mungkin', 'bisa jadi', atau 'kemungkinan'."""


def render_clean_button_interface():
    selected_prompt = None

    # Custom button styling
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #9333EA;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #7928CA;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(147, 51, 234, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)

    # Create buttons only
    if st.button("\U0001F41F Analisis Spesies Ikan", use_container_width=True):
        selected_prompt = get_system_prompt()

    if st.button("\U0001F31F Analisis Kesegaran Ikan", use_container_width=True):
        selected_prompt = get_freshness_prompt()

    return selected_prompt

def main():
    # Initialize session state
    initialize_session_state()

    # --- Page Config ---
    # Custom CSS with modern effects and purple theme
    st.markdown("""  
    <style>  
        /* Animations */  
        @keyframes pulse {  
            0% { opacity: 1; }  
            50% { opacity: 0.5; }  
            100% { opacity: 1; }  
        }  

        @keyframes float {  
            0% { transform: translateY(0px); }  
            50% { transform: translateY(-8px); }  
            100% { transform: translateY(0px); }  
        }  

        @keyframes glow {  
            0% { box-shadow: 0 0 5px rgba(147, 51, 234, 0.2); }  
            50% { box-shadow: 0 0 20px rgba(147, 51, 234, 0.4); }  
            100% { box-shadow: 0 0 5px rgba(147, 51, 234, 0.2); }  
        }  

        @keyframes shimmer {  
            0% { background-position: -1000px 0; }  
            100% { background-position: 1000px 0; }  
        }  

        /* Main Title Styles */  
        .main-title {  
            background: linear-gradient(120deg, #9333EA, #6B21A8);  
            -webkit-background-clip: text;  
            -webkit-text-fill-color: transparent;  
            animation: float 3s ease-in-out infinite;  
            font-weight: 800;  
            text-align: center;  
            font-size: 3em !important;  
            margin: 20px auto !important; /* Auto margin for centering */  
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);  
        }  

        /* Location Badge */  
        .location-badge {  
            background: linear-gradient(135deg, #9333EA, #6B21A8);  
            padding: 12px 20px;  
            border-radius: 25px;  
            color: white;  
            font-size: 1em;  
            margin: 15px auto; /* Centered using auto margin */  
            display: inline-flex;  
            align-items: center;  
            gap: 10px;  
            box-shadow: 0 4px 15px rgba(147, 51, 234, 0.3);  
            backdrop-filter: blur(10px);  
            border: 1px solid rgba(255, 255, 255, 0.2);  
            animation: glow 3s infinite;  
            transition: transform 0.3s ease;  
            text-align: center; /* Center text */  
        }  

        .location-badge:hover {  
            transform: translateY(-3px);  
            box-shadow: 0 6px 20px rgba(75, 0, 130, 0.4);   
        }  

        /* Monitoring Status */  
        .monitoring-status {  
            display: flex;  
            align-items: center;  
            gap: 12px;  
            margin: 15px auto; /* Centered using auto margin */  
            justify-content: center;  
            background: rgba(147, 51, 234, 0.1);  
            padding: 12px 24px;  
            border-radius: 20px;  
            backdrop-filter: blur(8px);  
            max-width: fit-content;  
            border: 1px solid rgba(147, 51, 234, 0.2);  
        }  

        .live-dot {  
            height: 10px;  
            width: 10px;  
            background-color: #9333EA;  
            border-radius: 50%;  
            animation: pulse 1.5s infinite;  
            box-shadow: 0 0 10px rgba(147, 51, 234, 0.5);  
        }  

        /* Chart Container */  
        .chart-container {  
            background: rgba(17, 25, 40, 0.6);  
            border-radius: 25px;  
            padding: 20px;  
            margin: 20px auto; /* Centered using auto margin */  
            box-shadow:   
                inset 0 2px 15px rgba(0, 0, 0, 0.2),  
                0 0 30px rgba(147, 51, 234, 0.15);  
            overflow: hidden;  
            position: relative;  
            transition: all 0.3s ease;  
            text-align: center; /* Center text */  
        }  

        .chart-container::before {  
            content: '';  
            position: absolute;  
            top: -50%;  
            left: -50%;  
            width: 200%;  
            height: 200%;  
            background: radial-gradient(  
                circle,  
                rgba(147, 51, 234, 0.1) 0%,  
                transparent 70%  
            );  
            animation: rotate 15s linear infinite;  
        }  

        /* Dashboard Title */  
        .dashboard-title {  
            color: #9333EA;  
            font-size: 1.5em;  
            margin: 20px 0;  
            text-align: center; /* Centered */  
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);  
            letter-spacing: 1.5px;  
            font-weight: 600;  
        }  

        /* iframe Styling */  
        iframe {  
            border-radius: 20px !important;  
            box-shadow: 0 4px 20px rgba(147, 51, 234, 0.2);  
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);  
            border: 2px solid rgba(147, 51, 234, 0.1);  
            background: white;  
            margin: 20px auto; /* Centered using auto margin */  
            display: block;  
            max-width: 100%; /* Responsive */  
        }  

        iframe:hover {  
            transform: scale(1.01) translateY(-5px);  
            box-shadow: 0 8px 30px rgba(147, 51, 234, 0.3);  
            border-color: rgba(147, 51, 234, 0.3);  
        }  

        /* Footer Styling */  
        .footer-text {  
            background: linear-gradient(90deg, #A855F7, #A855F7);  /* Gradasi warna ungu terang */  
            -webkit-background-clip: text;  
            -webkit-text-fill-color: transparent;  
            font-weight: 600;  
            letter-spacing: 0.8px;  
            display: inline-block;  
            padding: 15px;  
            position: relative;  
            text-align: center; /* Centered */  
        }

        .footer-text::after {  
            content: '';  
            position: absolute;  
            bottom: 0;  
            left: 0;  
            width: 100%;  
            height: 2px;  
            background: linear-gradient(90deg, transparent, #9333EA, transparent);  
            animation: shimmer 5s infinite;  
        }  

        /* Icon Animation */  
        .icon {  
            animation: float 3s ease-in-out infinite;  
            font-size: 3em;  
            margin-bottom: 15px;  
            display: block;  
            text-align: center; /* Centered */  
        }  

        /* Container Wrapper */  
        .content-wrapper {  
            max-width: 1200px;  
            margin: 0 auto;  
            padding: 20px;  
            text-align: center; /* Centered */  
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(147, 51, 234, 0.1);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #9333EA, #6B21A8);
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #6B21A8, #4C1D95);
        }
    </style>  
    """, unsafe_allow_html=True)

    # Header with enhanced animations
    st.markdown("""
        <div class="content-wrapper">
            <div class="icon">
                🤖
            </div>
            <h1 class="main-title">
                VisionFish.io
            </h1>
            <div style="text-align: center;">
                <div class="location-badge">
                    📍 Sungai Rengas Monitoring Station
                </div>
            </div>
            <div class="monitoring-status">
                <div class="live-dot"></div>
                <span style="color: #9333EA; font-weight: 500;">SmartFlow is Monitoring • Live Updates</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ThingSpeak Embed with enhanced container
    thingspeak_url = "https://thingspeak.mathworks.com/channels/2796290/charts/1?bgcolor=%23ffffff&color=%239333EA&dynamic=true&results=60&type=line&update=15"

    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        components.iframe(thingspeak_url, width=800, height=400, scrolling=True)

    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Enhanced footer with animation
    st.markdown("""
        <div style="text-align: center; margin-top: 30px;">
            <p class="footer-text">
                ✨ Real-time monitoring system • Updated every 5 seconds ✨
            </p>
        </div>
    """, unsafe_allow_html=True)

    # --- Side Bar ---
    with st.sidebar:
        st.write("### 🔐 Authentication")
        
        # Token authentication
        access_token = st.text_input("Enter access token:", type="password")
        
        # Check if token is valid
        is_authenticated = access_token in valid_tokens
        st.session_state.is_authenticated = is_authenticated
        
        if not is_authenticated:
            st.warning("Please enter a valid token to access VisionFish.io")
            
            # Add a request link for getting a token
            st.markdown("[Request access token](https://wa.me/0895619313339?text=Halo%20bang%2C%20saya%20ingin%20mendapatkan%20token%20akses%20untuk%20VisionFish.io)", unsafe_allow_html=True)
        else:
            st.success("✅ Authentication successful!")
            
            # Once authenticated, set default API keys from environment variables
            openai_api_key = os.getenv("OPENAI_API_KEY", "")
            google_api_key = os.getenv("GOOGLE_API_KEY", "")
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")

    # Sidebar for upload file
    if "data_penilaian" not in st.session_state:
        st.session_state.data_penilaian = pd.DataFrame({
            'No.': [""],  # Satu baris awal
            'Nama Panelis': [""],  # Nama panelis statis
            'Mata': [""],  # Nilai default
            'Insang':[""],
            'Lendir': [""],
            'Daging': [""],
            'Bau': [""],
            'Tekstur': [""],
        })

    # Check authentication before proceeding with the main content
    if not st.session_state.is_authenticated:
        st.write("#")
        st.write("## 🔒 Please authenticate to access VisionFish.io")
        
        # Display a teaser image or information about the app
        st.info("Enter a valid token in the sidebar to unlock all features")
        
        st.write("""
        ### What VisionFish.io offers:
        - 🐟 Fish species identification
        - 🔍 Freshness analysis
        - 📊 Data visualization and reporting
        - 📱 Real-time monitoring
        """)
        
        # Show map in sidebar even when not authenticated
        with st.sidebar:
            st.write("#")
            st.write("#")
            st.write("#")
            st.write("🗺️ Peta Sungai Rengas:")
            st.components.v1.html('''
            <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d63837.090342372285!2d109.16168223152577!3d-0.016918696446186224!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x2e1d5f74c5e63725%3A0x83498a713bf64183!2sSungai%20Rengas%2C%20Kec.%20Sungai%20Kakap%2C%20Kabupaten%20Kubu%20Raya%2C%20Kalimantan%20Barat!5e0!3m2!1sid!2sid!4v1735501581541!5m2!1sid!2sid" width="300" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
            ''', height=470)
    else:
        # This is where all your authenticated code goes
        client = OpenAI(api_key=openai_api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])

        # --- Sidebar Upload Data ---
        with st.sidebar:
            st.divider()
            
            # Model selection
            available_models = [] + (anthropic_models if anthropic_api_key else []) + (google_models if google_api_key else []) + (openai_models if openai_api_key else [])
            model = st.selectbox("Select a model:", available_models, index=0)
            model_type = None
            if model in openai_models: 
                model_type = "openai"
            elif model in google_models: 
                model_type = "google"
            elif model in anthropic_models:
                model_type = "anthropic"
            
            with st.popover("⚙️ Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            audio_response = st.toggle("Respon audio", value=False)
            if audio_response:
                cols = st.columns(2)
                with cols[0]:
                    tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
                with cols[1]:
                    tts_model = st.selectbox("Pilih model:", ["tts-1", "tts-1-hd"], index=1)

            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Mulai chatan baru", 
                on_click=reset_conversation,
            )
            st.divider()
            
            # Image Upload
            if model in ["AquaVision-4", "TidalFlux-4", "ReefSpark-1.5", "WaveCore-Pro", "Neptune-Savant"]:
                    
                st.write(f"### **Upload {' Document' if model_type=='google' else ''}:**")

                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                        if img_type == "video/mp4":
                            # save the video file
                            video_id = random.randint(100000, 999999)
                            with open(f"video_{video_id}.mp4", "wb") as f:
                                f.write(st.session_state.uploaded_img.read())
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "video_file",
                                        "video_file": f"video_{video_id}.mp4",
                                    }]
                                }
                            )
                        else:
                            raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                            img = get_image_base64(raw_img)
                            st.session_state.messages.append(
                                {
                                    "role": "user", 
                                    "content": [{
                                        "type": "image_url",
                                        "image_url": {"url": f"data:{img_type};base64,{img}"}
                                    }]
                                }
                            )

                cols_img = st.columns(2)

                with cols_img[0]:
                    with st.popover("📁 Upload"):
                        st.file_uploader(
                            f"Upload an image{' or a video' if model_type == 'google' else ''}:", 
                            type=["png", "jpg", "jpeg"] + (["mp4"] if model_type == "google" else []), 
                            accept_multiple_files=False,
                            key="uploaded_img",
                            on_change=add_image_to_messages,
                        )

                with cols_img[1]:                    
                    with st.popover("📸 Camera"):
                        activate_camera = st.checkbox("Activate camera")
                        if activate_camera:
                            st.camera_input(
                                "Take a picture", 
                                key="camera_img",
                                on_change=add_image_to_messages,
                            )

            st.header("Upload Data:")
            uploaded_file = st.file_uploader(
                "Upload Data Penilaian (CSV):",
                type=['csv'],
                help="Upload file CSV yang berisi data penilaian kualitas ikan"
            )
            
            # Expander untuk format data yang dibutuhkan
            with st.expander("Format Bentuk CSV yang Dibutuhkan", expanded=False):
                
                # Tampilkan Editable Dataframe (Data tetap tersimpan di session_state)
                st.subheader("Masukkan Data Penilaian:")
                edited_df = st.data_editor(
                    st.session_state.data_penilaian,
                    num_rows="dynamic",  # User bisa menambah sendiri
                    use_container_width=True
                )

                # Menampilkan peringatan dengan ukuran lebih kecil
                st.markdown(
                    """
                    <div style="font-size: 14px; color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                        Silahkan buat dan download file Anda secara digital dengan mengklik tombol yang sudah disediakan pada kolom tabel di atas. 😊
                    </div>
                    """, unsafe_allow_html=True
                )

                # Link informasi lebih lanjut
                st.markdown("<p style='font-size: 12px;'><a href='https://wa.me/0895619313339'>Info selengkapnya</a></p>", unsafe_allow_html=True)

            st.divider()
            st.write("🗺️ Peta Sungai Rengas:")
            st.components.v1.html('''
            <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d78549.54432253679!2d109.21681479925691!3d-0.0179603793051129!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x2e1d5f74c5e63725%3A0x83498a713bf64183!2sSungai%20Rengas%2C%20Kec.%20Sungai%20Kakap%2C%20Kabupaten%20Kubu%20Raya%2C%20Kalimantan%20Barat!5e1!3m2!1sid!2sid!4v1735743322793!5m2!1sid!2sid" width="300" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
            ''', height=470)

        # Process uploaded CSV file if available
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                processed_df, prima, advance, sedang, busuk, param_averages = process_fish_data(df)

                total_panelists = len(df)
                col1, col2 = st.columns(2)

                # Pie chart
                with col1:
                    total = len(processed_df)
                    categories = ['Prima', 'Advance', 'Sedang', 'Busuk']
                    values = []
                    for category in categories:
                        count = len(processed_df[processed_df['Kesegaran'] == category])
                        percentage = (count / total) * 100
                        values.append(percentage)
                    
                    colors = ['#4A235A', '#8E44AD', '#D2B4DE', '#7D3C98']
                    fig_pie = go.Figure(data=[go.Pie(labels=categories, 
                                                values=values,
                                                hole=.3,
                                                marker_colors=colors)])
                    fig_pie.update_layout(title="Distribusi Kesegaran:")
                    fig_pie.update_traces(textposition='inside', textinfo='percent')
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Radar chart
                with col2:
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=list(param_averages.values()),
                        theta=list(param_averages.keys()),
                        fill='toself',
                        name='Skor Rata-rata',
                        marker=dict(color='purple'),
                        line=dict(color='mediumpurple')))
                    fig_radar.update_layout(title="Radar Chart Skor Rata-rata:")
                    st.plotly_chart(fig_radar, use_container_width=True)

                # Tampilkan data lengkap dan ringkasan
                with st.expander("Lihat Data Lengkap Penilaian"):  
                    def color_freshness(val):  
                        colors = {  
                            'Prima': '#4A235A',
                            'Advance': '#8E44AD',
                            'Sedang': '#D2B4DE',
                            'Busuk': '#7D3C98'
                        }  
                        return f'background-color: {colors.get(val, "#FFFFFF")}'  

                    st.dataframe(processed_df.style.applymap(color_freshness, subset=['Kesegaran'])  
                                .background_gradient(subset=['Rata-rata'], cmap='PuRd_r'),   
                                use_container_width=True, height=400)

                st.subheader("Kesimpulan:")
                freshness = processed_df['Kesegaran'].iloc[0] 
                avg_score = processed_df['Rata-rata'].mean()
                st.metric(label="Tingkat Mutu Ikan", value=freshness, delta=f"{avg_score:.2f} Rata-rata Skor")
                
                # Tambahkan fungsi untuk mengunduh CSV
                def download_csv(df, filename="data_penilaian_processed.csv"):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Data Penilaian (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                # Tambahkan di dalam blok if uploaded_file is not None, setelah st.metric
                st.subheader("Unduh Hasil Analisis:")
                download_csv(processed_df)
            except Exception as e:
                st.error(f"Error dalam memproses file: {str(e)}")
                
        # Chat interface
        # Define hidden system prompts
        def get_fish_analysis_prompt():
            return get_system_prompt()

        # Chat button interface
        prompt = render_clean_button_interface()
        if prompt:
            st.session_state.messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }]
            })
            
            with st.chat_message("assistant"):
                model2key = {
                    "openai": openai_api_key,
                    "google": google_api_key,
                    "anthropic": anthropic_api_key,
                }
                
                response = stream_llm_response(
                    model_params=model_params,
                    model_type=model_type,
                    api_key=model2key[model_type]
                )
                
                st.write_stream(response)
                
                if audio_response:
                    response_text = st.session_state.messages[-1]["content"][0]["text"]
                    audio = client.audio.speech.create(
                        model=tts_model,
                        voice=tts_voice,
                        input=response_text
                    )
                    
                    audio_base64 = base64.b64encode(audio.content).decode('utf-8')
                    st.audio(data=f"data:audio/wav;base64,{audio_base64}", format="audio/wav")

        # # Text input for chat
        # if user_input := st.chat_input("Type your message here..."):
        #     st.session_state.messages.append({
        #         "role": "user",
        #         "content": [{
        #             "type": "text",
        #             "text": user_input
        #         }]
        #     })
            
        #     with st.chat_message("user"):
        #         st.write(user_input)
                
        #     with st.chat_message("assistant"):
        #         model2key = {
        #             "openai": openai_api_key,
        #             "google": google_api_key,
        #             "anthropic": anthropic_api_key,
        #         }
                
        #         response = stream_llm_response(
        #             model_params=model_params,
        #             model_type=model_type,
        #             api_key=model2key[model_type]
        #         )
                
        #         st.write_stream(response)
                
        #         if audio_response:
        #             response_text = st.session_state.messages[-1]["content"][0]["text"]
        #             audio = client.audio.speech.create(
        #                 model=tts_model,
        #                 voice=tts_voice,
        #                 input=response_text
        #             )
                    
        #             audio_base64 = base64.b64encode(audio.content).decode('utf-8')
        #             st.audio(data=f"data:audio/wav;base64,{audio_base64}", format="audio/wav")


if __name__=="__main__":
    main()
