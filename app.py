import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
import base64
from io import BytesIO
import google.generativeai as genai
import anthropic
import streamlit.components.v1 as components
import pandas as pd  
import plotly.express as px  
from plotly import graph_objects as go  
import urllib.parse
import requests
from datetime import datetime

# Load environment variables
dotenv.load_dotenv()

# List of valid tokens for authentication
valid_tokens = [
    "999", "tpi123", "indonesiagelap2025", "IPTEQWONT", "KONTOLODONDIKA", 
    "tpi123", "risko99", "LMX", "Syarifrizik99", "OceanWave", "FishScale", 
    "VisionTide", "RengasFin", "AquaticBlue", "CoralReef", "SeaCurrent", 
    "TunaSwift", "MermaidSong", "NetCast", "DeepAbyss", "SalmonRun", 
    "SharkBite", "PearlDive", "KelpForest"
]

# Custom model names for branding
anthropic_models = ["Neptune-Savant"]
google_models = ["ReefSpark-1.5", "WaveCore-Pro"]
openai_models = ["AquaVision-4", "TidalFlux-4", "CoralPulse-3.5", "DeepMind-4", "OceanVault-4"]

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
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "token_input_submitted" not in st.session_state:
        st.session_state.token_input_submitted = False
    if "data_penilaian" not in st.session_state:
        st.session_state.data_penilaian = pd.DataFrame({
            'No.': [""],
            'Nama Panelis': [""],
            'Mata': [""],
            'Insang': [""],
            'Lendir': [""],
            'Daging': [""],
            'Bau': [""],
            'Tekstur': [""],
        })

def messages_to_gemini(messages):
    """Format messages for Google's Gemini model"""
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
                gemini_message["parts"].append({"text": content["text"]})
            elif content["type"] == "image_url":
                image_data = content["image_url"]["url"].split(",")[1]
                mime_type = content["image_url"]["url"].split(";")[0].split(":")[1]
                gemini_message["parts"].append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_data
                    }
                })
        
        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)
        prev_role = message["role"]

    return gemini_messages

def messages_to_anthropic(messages):
    """Format messages for Anthropic's Claude model"""
    # Add system prompt first
    system_prompt = get_system_prompt()
    
    # Filter out system messages and convert content format
    filtered_messages = []
    for message in messages:
        if message["role"] == "system":
            continue
            
        new_content = []
        for content_item in message["content"]:
            if content_item["type"] == "text":
                new_content.append({
                    "type": "text",
                    "text": content_item["text"]
                })
            elif content_item["type"] == "image_url":
                # Extract base64 data and media type from the URL
                media_type = content_item["image_url"]["url"].split(";")[0].split(":")[1]
                data = content_item["image_url"]["url"].split(",")[1]
                new_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data
                    }
                })
                
        filtered_messages.append({
            "role": message["role"],
            "content": new_content
        })
    
    return filtered_messages

def stream_llm_response(model_params, model_type="openai", api_key=None):
    """Stream responses from the selected LLM"""
    response_message = ""
    system_prompt = get_system_prompt()
    
    # Get the actual model name from our custom name
    actual_model = model_name_mapping.get(model_params["model"], model_params["model"])
    
    if model_type == "openai":
        client = OpenAI(api_key=api_key)
        
        # Format messages for OpenAI
        formatted_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in st.session_state.messages:
            if msg["role"] == "system":
                continue
                
            formatted_msg = {"role": msg["role"], "content": []}
            
            for content in msg["content"]:
                if content["type"] == "text":
                    formatted_msg["content"] = content["text"]
                elif content["type"] == "image_url":
                    if isinstance(formatted_msg["content"], list):
                        formatted_msg["content"] = []
                    
                    if not isinstance(formatted_msg["content"], list):
                        current_content = formatted_msg["content"]
                        formatted_msg["content"] = [{"type": "text", "text": current_content}] if current_content else []
                    
                    formatted_msg["content"].append({
                        "type": "image_url",
                        "image_url": {"url": content["image_url"]["url"]}
                    })
            
            formatted_messages.append(formatted_msg)
        
        try:
            for chunk in client.chat.completions.create(
                model=actual_model,
                messages=formatted_messages,
                temperature=model_params.get("temperature", 0.3),
                max_tokens=4096,
                stream=True
            ):
                chunk_text = chunk.choices[0].delta.content or ""
                response_message += chunk_text
                yield chunk_text
        except Exception as e:
            error_message = f"Error with OpenAI: {str(e)}"
            st.error(error_message)
            yield error_message

    elif model_type == "google":
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=actual_model,
                generation_config={"temperature": model_params.get("temperature", 0.3)}
            )
            
            gemini_messages = messages_to_gemini(st.session_state.messages)
            
            response = model.generate_content(gemini_messages, stream=True)
            
            for chunk in response:
                chunk_text = chunk.text if hasattr(chunk, 'text') else ""
                response_message += chunk_text
                yield chunk_text
        except Exception as e:
            error_message = f"Error with Google AI: {str(e)}"
            st.error(error_message)
            yield error_message

    elif model_type == "anthropic":
        try:
            client = anthropic.Anthropic(api_key=api_key)
            messages = messages_to_anthropic(st.session_state.messages)
            
            with client.messages.stream(
                model=actual_model,
                system=system_prompt,
                messages=messages,
                temperature=model_params.get("temperature", 0.3),
                max_tokens=4096
            ) as stream:
                for text in stream.text_stream:
                    response_message += text
                    yield text
        except Exception as e:
            error_message = f"Error with Anthropic: {str(e)}"
            st.error(error_message)
            yield error_message

    # Add the response to the message history
    st.session_state.messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response_message}]
    })

def get_model_params(model_type, model=None, temperature=0.3):
    """Get model parameters based on the selected model type"""
    params = {
        "temperature": temperature
    }
    
    if model:
        params["model"] = model
    elif model_type == "openai":
        params["model"] = "AquaVision-4"
    elif model_type == "google":
        params["model"] = "WaveCore-Pro"
    elif model_type == "anthropic":
        params["model"] = "Neptune-Savant"
    
    return params

# Helper functions
def get_image_base64(image_raw):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format or "JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    # Make sure to strip 'data:image/jpeg;base64,' if present
    if ',' in base64_string:
        base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def process_fish_data(df):
    """Process fish quality data and return metrics"""
    # Convert all columns to numeric if they're not already
    for col in ['Mata', 'Insang', 'Lendir', 'Daging', 'Bau', 'Tekstur']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values with 0 to avoid calculation errors
    df = df.fillna(0)
    
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
    """Get the prompt for fish freshness analysis"""
    return """Analisis Kesegaran Ikan dari Gambar:
[Jika gambar yang diunggah bukan gambar ikan maka Mohon masukkan gambar ikan secara jelas. Analisis ini hanya berlaku untuk gambar ikan, Saya dapat menganalisis dan menilai kesegaran ikan dalam bentuk gambar]

Parameter penilaian ikan ada 6, namun karena dalam bentuk visual saya hanya dapat melakukan 4 ini secara akurat:
- Mata
- Insang
- Lendir permukaan badan
- Sayatan daging

[Untuk lebih baik lagi maka anda dapat mengecek kondisi ini secara langsung]
- Tekstur daging 
- Bau

Skor:
- Sangat Baik (Excellent): Skor 9
- Masih Baik (Good): Skor 7-8
- Tidak Segar (Moderate): Skor 5-6
- Sangat Tidak Segar (Spoiled): Skor 1-4

Kesimpulan:
Skor: [Tulis skor di sini, misalnya "X dari 9", berdasarkan analisis visual].
Alasan: [Jelaskan alasan skor berdasarkan parameter di atas, misalnya "Warna insang cerah dan sisik mengkilap menunjukkan kesegaran tinggi"]."""
    
def get_analisis_ikan_prompt():
    """Get the prompt for fish species analysis"""
    return """Analisis Spesies Ikan dari Gambar:
[Jika gambar yang diunggah bukan gambar ikan, maka: "Mohon masukkan gambar ikan secara jelas. Analisis ini hanya berlaku untuk gambar ikan." Namun, Jika gambar ikan maka "Saya dapat menjawab ikan dari gambar dengan tepat"]

Identifikasi spesies dan klasifikasi ikan dari gambar adalah:

| Kategori    | Detail       |
|-------------|--------------|
| Nama Lokal  | [nama ikan]  |
| Nama Ilmiah | [nama latin] |
| Famili      | [famili]     |

Analisis dilakukan dengan yakin berdasarkan interpretasi visual dari gambar digital. Sertakan penjelasan singkat tentang ciri-ciri yang digunakan untuk identifikasi, seperti bentuk tubuh, warna, atau pola sisik."""

def setup_page_config():
    """Configure page settings"""
    st.set_page_config(
        page_title="\033[95mVisionFish.io - Analisis Ikan Pintar\033[0m",  # Purple text
        page_icon="🐟",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""  
    <style>  
        /* General Styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Dark background for entire app but keep original layout */
        .stApp {
            background-color: #111827;
            color: #E5E7EB;
        }
        
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
            font-size: 3.5em !important;  
            margin: 20px auto !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);  
        }  

        /* Location Badge */  
        .location-badge {  
            background: linear-gradient(135deg, #9333EA, #6B21A8);  
            padding: 12px 20px;  
            border-radius: 25px;  
            color: white;  
            font-size: 1em;  
            margin: 15px auto;
            display: inline-flex;  
            align-items: center;  
            gap: 10px;  
            box-shadow: 0 4px 15px rgba(147, 51, 234, 0.3);  
            backdrop-filter: blur(10px);  
            border: 1px solid rgba(255, 255, 255, 0.2);  
            animation: glow 3s infinite;  
            transition: transform 0.3s ease;  
            text-align: center;
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
            margin: 15px auto;
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
            background: rgba(31, 41, 55, 0.8);  
            border-radius: 16px;  
            padding: 20px;  
            margin: 20px auto;
            box-shadow: 0 8px 30px rgba(147, 51, 234, 0.15);  
            overflow: hidden;  
            position: relative;  
            transition: all 0.3s ease;  
            text-align: center;
            border: 1px solid rgba(147, 51, 234, 0.1);
        }  

        /* Dashboard Title */  
        .dashboard-title {  
            color: #9333EA;  
            font-size: 1.5em;  
            margin: 20px 0;  
            text-align: center;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  
            letter-spacing: 1px;  
            font-weight: 600;  
        }  

        /* iframe Styling */  
        iframe {  
            border-radius: 16px !important;  
            box-shadow: 0 4px 20px rgba(147, 51, 234, 0.2);  
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);  
            border: 2px solid rgba(147, 51, 234, 0.1);  
            background: white;  
            margin: 20px auto;
            display: block;  
            max-width: 100%;
        }  

        /* Footer Styling */  
        .footer-text {  
            background: linear-gradient(90deg, #A855F7, #A855F7);
            -webkit-background-clip: text;  
            -webkit-text-fill-color: transparent;  
            font-weight: 600;  
            letter-spacing: 0.8px;  
            display: inline-block;  
            padding: 15px;  
            position: relative;  
            text-align: center;
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
            text-align: center;
        }  

        /* Container Wrapper */  
        .content-wrapper {  
            max-width: 1200px;  
            margin: 0 auto;  
            padding: 20px;  
            text-align: center;
        }
        
        /* Auth Container */
        .auth-container {
            background: #1F2937;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 12px 30px rgba(147, 51, 234, 0.15);
            margin: 50px auto;
            max-width: 500px;
            text-align: center;
            transition: all 0.3s ease;
            border: 1px solid rgba(147, 51, 234, 0.3);
        }
        
        .auth-title {
            color: #9333EA;
            font-size: 1.8em;
            margin-bottom: 20px;
            font-weight: 700;
        }
        
        /* Login button */
        .login-btn {
            background: linear-gradient(135deg, #9333EA, #6B21A8);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-block;
            margin-top: 15px;
            box-shadow: 0 4px 15px rgba(147, 51, 234, 0.3);
        }
        
        .login-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(147, 51, 234, 0.4);
        }
        
        /* Button styling */
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
        
        /* Feature boxes */
        .feature-box {
            background: rgba(31, 41, 55, 0.8);
            border-radius: 16px;
            padding: 20px;
            margin: 15px 0;
            border: 1px solid rgba(147, 51, 234, 0.2);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .feature-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(147, 51, 234, 0.2);
            border: 1px solid rgba(147, 51, 234, 0.4);
        }
        
        .feature-icon {
            font-size: 2.5em;
            margin-bottom: 15px;
            background: linear-gradient(135deg, #9333EA, #6B21A8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .feature-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 10px;
            color: #A855F7;
        }
        
        /* Section titles */
        .section-title {
            color: #9333EA;
            font-size: 1.8em;
            font-weight: 700;
            margin: 30px 0 20px 0;
            text-align: left;
            border-bottom: 2px solid rgba(147, 51, 234, 0.3);
            padding-bottom: 10px;
        }
        
        /* Chat container */
        .chat-container {
            background: rgba(31, 41, 55, 0.8);
            border-radius: 16px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(147, 51, 234, 0.2);
        }
        
        /* Image upload area */
        .upload-container {
            background: rgba(31, 41, 55, 0.5);
            border-radius: 16px;
            padding: 20px;
            border: 2px dashed rgba(147, 51, 234, 0.3);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .upload-container:hover {
            border-color: rgba(147, 51, 234, 0.6);
            background: rgba(31, 41, 55, 0.7);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: rgba(31, 41, 55, 0.5);
            border-radius: 10px;
            padding: 5px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 8px;
            color: white;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #9333EA !important;
            color: white !important;
        }
        
        /* Data table styling */
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(147, 51, 234, 0.2);
        }
        
        [data-testid="stDataFrame"] table {
            border-collapse: separate;
            border-spacing: 0;
        }
        
        [data-testid="stDataFrame"] th {
            background-color: #1E293B !important;
            color: #A855F7 !important;
            font-weight: 600;
            padding: 12px 15px !important;
            border-bottom: 2px solid rgba(147, 51, 234, 0.3);
        }
        
        [data-testid="stDataFrame"] td {
            background-color: #2D3748 !important;
            color: white !important;
            padding: 10px 15px !important;
            border-bottom: 1px solid rgba(147, 51, 234, 0.1);
        }
        
        /* Card styling */
        .info-card {
            background: linear-gradient(145deg, rgba(31, 41, 55, 0.8), rgba(31, 41, 55, 0.6));
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(147, 51, 234, 0.2);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            margin: 15px 0;
            transition: all 0.3s ease;
        }
        
        .info-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 25px rgba(147, 51, 234, 0.15);
        }
        
        /* Custom notification */
        .custom-notification {
            background: linear-gradient(135deg, rgba(147, 51, 234, 0.1), rgba(75, 0, 130, 0.1));
            border-left: 4px solid #9333EA;
            padding: 12px 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 0.9em;
        }
        
        /* Upload button */
        .stFileUploader > section > button {
            background-color: #9333EA !important;
            color: white !important;
            border-radius: 10px !important;
        }
        
        /* Metric styling */
        [data-testid="stMetric"] {
            background: rgba(31, 41, 55, 0.6);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(147, 51, 234, 0.2);
            transition: all 0.3s ease;
        }
        
        [data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(147, 51, 234, 0.15);
            border-color: rgba(147, 51, 234, 0.4);
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 1em !important;
            color: #A855F7 !important;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 2em !important;
            font-weight: 700 !important;
            color: white !important;
        }
        
        [data-testid="stMetricDelta"] {
            font-size: 1em !important;
        }
        
        /* Enhancing Streamlit native components */
        .stTextInput > div > div > input {
            background-color: #2D3748;
            color: white;
            border: 1px solid rgba(147, 51, 234, 0.3);
            border-radius: 10px;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #9333EA;
            box-shadow: 0 0 10px rgba(147, 51, 234, 0.3);
        }
        
        /* Dashboard cards */
        .dashboard-card {
            background: rgba(31, 41, 55, 0.7);
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid rgba(147, 51, 234, 0.2);
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .dashboard-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(147, 51, 234, 0.15);
            border-color: rgba(147, 51, 234, 0.4);
        }
        
        .dashboard-value {
            font-size: 2em;
            font-weight: 700;
            color: #A855F7;
            margin: 10px 0;
        }
        
        .dashboard-label {
            font-size: 0.9em;
            color: #E5E7EB;
        }
        
        /* Custom scrollbar for a nice touch */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(147, 51, 234, 0.05);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #9333EA, #6B21A8);
            border-radius: 10px;
        }
    </style>  
    """, unsafe_allow_html=True)

def render_prompt_buttons():
    """Render the prompt selection buttons"""
    st.markdown("""
        <div class="dashboard-title" style="margin-bottom: 15px;">🤖 Pilih Jenis Analisis</div>
    """, unsafe_allow_html=True)
    
    # Create buttons in a more attractive layout
    col1, col2 = st.columns(2)
    
    selected_prompt = None
    
    with col1:
        if st.button("\U0001F41F Analisis Spesies Ikan", use_container_width=True):
            selected_prompt = get_analisis_ikan_prompt()

    with col2:
        if st.button("\U0001F31F Analisis Kesegaran Ikan", use_container_width=True):
            selected_prompt = get_freshness_prompt()

    return selected_prompt

def render_auth_screen():
    """Render the authentication screen"""
    st.markdown("""
        <div class="auth-container">
            <div class="auth-title">🔐 Access VisionFish.io</div>
            <p style="color: #E5E7EB;">Enter your access token to use all features of VisionFish.io</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Token authentication in main area with improved layout
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        access_token = st.text_input("Enter access token:", type="password")
        
        if st.button("Login", key="login_button"):
            # Check if token is valid
            is_authenticated = access_token in valid_tokens
            st.session_state.is_authenticated = is_authenticated
            st.session_state.token_input_submitted = True
            
            if not is_authenticated:
                st.error("Invalid token. Please try again.")
            else:
                st.success("✅ Authentication successful!")
                st.rerun()
        
        # Add a request link for getting a token with better styling
        st.markdown("""
        <div style="text-align: center; margin-top: 15px;">
            <a href="https://wa.me/0895619313339?text=Halo%20bang%2C%20saya%20ingin%20mendapatkan%20token%20akses%20untuk%20VisionFish.io" 
               style="color: #A855F7; text-decoration: none; font-weight: 500;">
               Request access token <span style="font-size: 1.2em;">→</span>
            </a>
        </div>
        """, unsafe_allow_html=True)

def render_unauthenticated_content():
    """Render content for unauthenticated users"""
    # Display a compelling header to attract users
    st.markdown("""
    <div class="content-wrapper">
        <h1 class="main-title">VisionFish.io</h1>
        <p style="text-align: center; font-size: 1.2em; margin-bottom: 30px; max-width: 700px; margin-left: auto; margin-right: auto;">
            Platform analisis ikan berbasis AI untuk membantu industri perikanan Indonesia
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase with improved layout
    st.markdown("<h2 class='dashboard-title' style='margin-top: 30px;'>Fitur Unggulan</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Identifikasi Spesies</div>
            <p>Identifikasi spesies ikan secara instan dari gambar dengan klasifikasi detail</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Visualisasi Data</div>
            <p>Grafik dan laporan canggih untuk analisis kualitas ikan yang komprehensif</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">🌊</div>
            <div class="feature-title">Analisis Kesegaran</div>
            <p>Tentukan tingkat kesegaran ikan dengan presisi ilmiah</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">📱</div>
            <div class="feature-title">Monitoring Real-time</div>
            <p>Pantau kualitas air secara langsung dari stasiun pemantauan</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Showcase a sample analysis to entice users
    st.markdown("<h2 class='dashboard-title' style='margin-top: 40px;'>Contoh Analisis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <style>
            .info-card {
                background: rgba(147, 51, 234, 0.1);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        </style>
        <div class="info-card">
            <div style="font-size: 1.3em; font-weight: 600; color: #A855F7; margin-bottom: 15px;">Analisis Spesies</div>
            <div style="margin-bottom: 15px;">
                <img src="https://media.istockphoto.com/id/1299077582/id/foto/ikan-tongkol-atau-tongkol-kecil-atau-ikan-tongkol-yang-terisolasi-pada-latar-belakang-putih.jpg?s=612x612&w=0&k=20&c=6TxZSHaL3nxhzxm_7ArAQpRw81jkCWQBWN8Yg-7_TVk=" 
                style="width: 100%; border-radius: 10px; margin-bottom: 15px;">
            </div>
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <tr>
                    <th style="padding: 8px; border-bottom: 1px solid rgba(147, 51, 234, 0.2);">Nama Lokal</th>
                    <td style="padding: 8px; border-bottom: 1px solid rgba(147, 51, 234, 0.2);">Tongkol</td>
                </tr>
                <tr>
                    <th style="padding: 8px; border-bottom: 1px solid rgba(147, 51, 234, 0.2);">Nama Ilmiah</th>
                    <td style="padding: 8px; border-bottom: 1px solid rgba(147, 51, 234, 0.2);">Euthynnus affinis</td>
                </tr>
                <tr>
                    <th style="padding: 8px;">Famili</th>
                    <td style="padding: 8px;">Scombridae</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <style>
            .info-card {
                background: rgba(147, 51, 234, 0.1);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        </style>
        <div class="info-card">
            <div style="font-size: 1.3em; font-weight: 600; color: #A855F7; margin-bottom: 15px;">Analisis Kesegaran</div>
            <div style="text-align: center; margin-bottom: 15px;">
                <div style="font-size: 2.5em; font-weight: 700; color: #A855F7;">8.5/10</div>
                <div style="font-size: 1.1em; font-weight: 500; color: #10B981;">Masih Baik (Good)</div>
            </div>
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <tr>
                    <th style="padding: 8px; border-bottom: 1px solid rgba(147, 51, 234, 0.2);">Mata</th>
                    <td style="padding: 8px; border-bottom: 1px solid rgba(147, 51, 234, 0.2);">9/10</td>
                </tr>
                <tr>
                    <th style="padding: 8px; border-bottom: 1px solid rgba(147, 51, 234, 0.2);">Insang</th>
                    <td style="padding: 8px; border-bottom: 1px solid rgba(147, 51, 234, 0.2);">8/10</td>
                </tr>
                <tr>
                    <th style="padding: 8px; border-bottom: 1px solid rgba(147, 51, 234, 0.2);">Lendir</th>
                    <td style="padding: 8px; border-bottom: 1px solid rgba(147, 51, 234, 0.2);">8/10</td>
                </tr>
                <tr>
                    <th style="padding: 8px;">Daging</th>
                    <td style="padding: 8px;">9/10</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    # Add a testimonial section
    st.markdown("<h2 class='dashboard-title' style='margin-top: 40px;'>Digunakan Oleh</h2>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    
    testimonials = [
        {
            "quote": "VisionFish.io membantu mempercepat proses analisis kualitas ikan kami dengan signifikan.",
            "name": "PT Samudra Sejahtera",
            "role": "Distributor Ikan"
        },
        {
            "quote": "Identifikasi spesies yang akurat membantu kami mengelola stok dengan lebih efisien.",
            "name": "TPI Sungai Rengas",
            "role": "Tempat Pelelangan Ikan"
        },
        {
            "quote": "Data kualitas air real-time membantu kami membuat keputusan yang lebih baik untuk budidaya ikan.",
            "name": "Kelompok Nelayan Maju Bersama",
            "role": "Nelayan Lokal"
        }
    ]
    
    for i, col in enumerate(cols):
        with col:
            testimonial = testimonials[i]
            st.markdown(f"""
            <div class="info-card" style="height: 100%;">
                <div style="font-size: 1em; line-height: 1.5; margin-bottom: 15px; font-style: italic; color: #E5E7EB;">
                    "{testimonial["quote"]}"
                </div>
                <div style="font-weight: 600; color: #A855F7;">{testimonial["name"]}</div>
                <div style="font-size: 0.9em; color: #CBD5E0;">{testimonial["role"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show map in sidebar even when not authenticated
    with st.sidebar:
        st.markdown("""
        <div style="margin-top: 30px;">
            <div class="dashboard-title">🗺️ Peta Sungai Rengas</div>
        </div>
        """, unsafe_allow_html=True)
        st.components.v1.html('''
        <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d63837.090342372285!2d109.16168223152577!3d-0.016918696446186224!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x2e1d5f74c5e63725%3A0x83498a713bf64183!2sSungai%20Rengas%2C%20Kec.%20Sungai%20Kakap%2C%20Kabupaten%20Kubu%20Raya%2C%20Kalimantan%20Barat!5e0!3m2!1sid!2sid!4v1735501581541!5m2!1sid!2sid" width="100%" height="450" style="border:0; border-radius: 16px; box-shadow: 0 8px 30px rgba(147, 51, 234, 0.2);" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
        ''', height=470)
        
        # Add contact info in sidebar
        st.markdown("""
        <div class="info-card" style="margin-top: 20px;">
            <div style="font-size: 1.1em; font-weight: 600; color: #A855F7; margin-bottom: 10px;">Kontak Kami</div>
            <p style="margin-bottom: 10px;">Untuk informasi lebih lanjut tentang VisionFish.io atau untuk mendapatkan akses:</p>
            <a href="https://wa.me/0895619313339" style="display: inline-flex; align-items: center; color: #E5E7EB; text-decoration: none; font-weight: 500;">
                <span style="background: #25D366; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; border-radius: 50%; margin-right: 8px;">
                    <span style="color: white; font-size: 14px;">✓</span>
                </span>
                WhatsApp Support
            </a>
        </div>
        """, unsafe_allow_html=True)

def render_dashboard():
    """Render the water quality dashboard with enhanced UI"""
    # Header with enhanced animations
    st.markdown("""
        <div class="content-wrapper">
            <div class="icon">
                🐟
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

    # Dashboard metrics with attractive cards
    st.markdown("<div class='section-title'>📊 Dashboard Kualitas Air</div>", unsafe_allow_html=True)
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.markdown("""
        <div class="dashboard-card">
            <div style="color: #CBD5E0; font-size: 0.9em;">Suhu Air</div>
            <div class="dashboard-value">Coming soon</div>
            <div style="color: #10B981; font-size: 0.8em;">Optimal</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown("""
        <div class="dashboard-card">
            <div style="color: #CBD5E0; font-size: 0.9em;">pH</div>
            <div class="dashboard-value">Coming soon</div>
            <div style="color: #10B981; font-size: 0.8em;">Normal</div>
        </div>
        """, unsafe_allow_html=True)
        
    with metrics_col3:
        st.markdown("""
        <div class="dashboard-card">
            <div style="color: #CBD5E0; font-size: 0.9em;">Oksigen Terlarut</div>
            <div class="dashboard-value">Coming soon</div>
            <div style="color: #10B981; font-size: 0.8em;">mg/L</div>
        </div>
        """, unsafe_allow_html=True)
        
    with metrics_col4:
        st.markdown("""
        <div class="dashboard-card">
            <div style="color: #CBD5E0; font-size: 0.9em;">Kekeruhan</div>
            <div class="dashboard-value">Coming soon</div>
            <div style="color: #FBBF24; font-size: 0.8em;">NTU</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Function to fetch data from ThingSpeak API
    def fetch_thingspeak_data(channel_id, field, results=60):
        api_url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{field}.json?results={results}"
        try:
            response = requests.get(api_url)
            response.raise_for_status()  # Raise an error for bad status codes
            data = response.json()
            feeds = data['feeds']
            # Parse dates and flow rates
            dates = [datetime.strptime(entry['created_at'], "%Y-%m-%dT%H:%M:%SZ") for entry in feeds]
            flow_rates = [float(entry[f'field{field}']) if entry[f'field{field}'] else 0 for entry in feeds]
            return dates, flow_rates
        except requests.RequestException as e:
            st.error(f"Error fetching data from ThingSpeak: {e}")
            return [], []

    # Function to create Plotly chart
    def create_flow_chart(dates, flow_rates):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=flow_rates,
                mode='lines+markers',
                name='Flow Rate',
                line=dict(color='#9333EA'),
                marker=dict(size=8),
                hovertemplate='Date: %{x|%Y-%m-%d %H:%M:%S}<br>Flow Rate: %{y:.2f}<extra></extra>'
            )
        )
        fig.update_layout(
            title="Flow Analytics Dashboard - Sungai Rengas",
            xaxis_title="Date",
            yaxis_title="Flow Rate",
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=None,  # Let Plotly use the full width of the container
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            xaxis=dict(
                tickformat="%H:%M",  # Format x-axis ticks as time
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray'
            ),
            hovermode='x unified'  # Show hover info for all points at the same x-value
        )
        return fig

    # Add custom CSS for the container
    st.markdown("""
        <style>
            .water-quality-chart {
                width: 100%;
                padding: 10px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .water-quality-title {
                text-align: center;
                font-size: 24px;
                margin-bottom: 10px;
                color: #ffff;
            }
        </style>
    """, unsafe_allow_html=True)

    # Fetch data and create the chart
    dates, flow_rates = fetch_thingspeak_data(channel_id=2796290, field=1, results=60)
    if dates and flow_rates:
        fig = create_flow_chart(dates, flow_rates)

        # Display the chart
        st.markdown("""
            <div class="water-quality-chart">
                <div class="water-quality-title">🌊 Grafik Monitoring Kualitas Air</div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig, use_container_width=True)  # Use the full width of the container

        st.markdown("""
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No data available to display the chart.")
    # Enhanced footer with animation
    st.markdown("""
        <div style="text-align: center; margin-top: 30px;">
            <p class="footer-text">
                ✨ Sistem monitoring real-time • Update setiap 5 detik ✨
            </p>
        </div>
    """, unsafe_allow_html=True)

def handle_sidebar_and_model_selection():
    """Handle sidebar UI and model selection"""
    with st.sidebar:
        # Once authenticated, set default API keys from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyB3aHVOIUyzk4sULzjCLjgo4G6-Tc4fiPA")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        # Add logo and sidebar header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 2em; background: linear-gradient(120deg, #9333EA, #6B21A8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;">VisionFish.io</div>
            <div style="font-size: 0.9em; color: #CBD5E0; margin-top: 5px;">Smart Fish Analysis Platform</div>
        </div>
        <div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(147, 51, 234, 0.3), transparent); margin: 20px 0;"></div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='dashboard-title' style='text-align: left; margin-bottom: 15px;'>🤖 Model AI</div>", unsafe_allow_html=True)
        
        # Model selection with better UI
        available_models = [] + (anthropic_models if anthropic_api_key else []) + (google_models if google_api_key else []) + (openai_models if openai_api_key else [])
        model = st.selectbox("Pilih model AI:", available_models, index=0)
        model_type = None
        
        if model in openai_models: 
            model_type = "openai"
        elif model in google_models: 
            model_type = "google"
        elif model in anthropic_models:
            model_type = "anthropic"
        
        with st.expander("⚙️ Parameter model"):
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        # Create model parameters
        model_params = {
            "model": model,
            "temperature": model_temp,
        }

        # Display model info with UI enhancement
        st.markdown(f"""
        <div class="info-card" style="margin-top: 20px;">
            <div style="font-size: 0.9em; color: #CBD5E0; margin-bottom: 8px;">Model yang digunakan:</div>
            <div style="font-weight: 600; color: #A855F7; font-size: 1.1em;">{model}</div>
            <div style="margin-top: 8px; font-size: 0.85em; color: #A0AEC0;">Dioptimalkan untuk analisis ikan</div>
        </div>
        """, unsafe_allow_html=True)

        # Reset conversation button with better styling
        st.markdown("<div style='margin: 25px 0 15px 0;'>", unsafe_allow_html=True)
        if st.button("🗑️ Mulai Chat Baru", use_container_width=True):
            if "messages" in st.session_state:
                st.session_state.pop("messages", None)
                st.rerun()
                
        # Show logout option
        if st.button("🔒 Logout", use_container_width=True):
            st.session_state.is_authenticated = False
            st.session_state.token_input_submitted = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 1px; background: linear-gradient(90deg, transparent, rgba(147, 51, 234, 0.3), transparent); margin: 20px 0;'></div>", unsafe_allow_html=True)
            
        return model, model_type, model_params, openai_api_key, google_api_key, anthropic_api_key

def handle_image_upload():
    """Handle image upload with improved UI"""
    st.markdown("<div class='section-title'>📷 Analisis Gambar Ikan</div>", unsafe_allow_html=True)
    
    # Create tabs for different upload options
    tab1, tab2 = st.tabs(["Upload Gambar", "Ambil Foto"])
    
    uploaded_img = None
    camera_img = None
    
    with tab1:
        st.markdown("""
        <div class="upload-container">
            <div style="font-size: 3em; margin-bottom: 10px;">📤</div>
            <div style="font-weight: 500; color: #A855F7; margin-bottom: 10px;">Upload gambar ikan</div>
            <div style="font-size: 0.9em; color: #CBD5E0;">Format yang didukung: JPG, PNG, JPEG</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_img = st.file_uploader(
            "",  # Empty label
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="uploaded_img",
            label_visibility="collapsed"
        )
    
    with tab2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 15px;">
            <div style="font-weight: 500; color: #A855F7; margin-bottom: 10px;">Ambil foto ikan</div>
            <div style="font-size: 0.9em; color: #CBD5E0;">Gunakan kamera untuk mengambil foto secara langsung</div>
        </div>
        """, unsafe_allow_html=True)
        
        camera_active = st.toggle("📸 Aktifkan Kamera", value=False, key="camera_toggle")
        
        if camera_active:
            camera_img = st.camera_input("", label_visibility="collapsed")
    
    if uploaded_img or camera_img:
        img_input = uploaded_img if uploaded_img else camera_img
        
        # Display the uploaded image
        if img_input:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(img_input)
                st.image(image, caption="Gambar Ikan", use_column_width=True)
            
            with col2:
                st.markdown("""
                <div class="info-card">
                    <div style="font-size: 1.1em; font-weight: 600; color: #A855F7; margin-bottom: 15px;">Instruksi Analisis</div>
                    <p style="margin-bottom: 15px;">Pilih jenis analisis yang ingin Anda lakukan pada gambar ini:</p>
                    <ul style="margin-left: 20px; margin-bottom: 15px;">
                        <li style="margin-bottom: 8px;">Analisis <b>Spesies</b> mengidentifikasi jenis ikan</li>
                        <li style="margin-bottom: 8px;">Analisis <b>Kesegaran</b> menentukan kualitas ikan</li>
                    </ul>
                    <p>Gunakan tombol di bawah untuk memulai analisis sesuai kebutuhan Anda.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add image to session state messages
            img_type = img_input.type
            img_base64 = get_image_base64(image)
            
            image_message = {
                "role": "user", 
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_type};base64,{img_base64}"}
                }]
            }
            
            return image_message
    
    return None

def handle_csv_upload_and_analysis():
    """Handle CSV upload and data analysis with improved UI"""
    st.markdown("<div class='section-title'>📊 Analisis Data Kualitas Ikan</div>", unsafe_allow_html=True)
    
    # Show a better description of the feature
    st.markdown("""
    <div class="custom-notification">
        <div style="font-weight: 600; margin-bottom: 5px;">📝 Analisis data penilaian kualitas ikan</div>
        <p>Upload file CSV dengan data penilaian untuk mendapatkan visualisasi dan analisis mendalam tentang kualitas ikan.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs to organize different parts of the feature
    tabs = st.tabs(["Upload Data", "Format Template", "Tips Penggunaan"])
    
    with tabs[0]:
        uploaded_file = st.file_uploader(
            "Upload Data Penilaian (CSV):",
            type=['csv'],
            help="Upload file CSV yang berisi data penilaian kualitas ikan"
        )
        
        # Process uploaded CSV file if available
        if uploaded_file is not None:
            try:
                st.success("File berhasil diupload! Memproses data...")
                
                # Add a spinner during processing for better UX
                with st.spinner("Menganalisis data..."):
                    df = pd.read_csv(uploaded_file)
                    processed_df, prima, advance, sedang, busuk, param_averages = process_fish_data(df)

                    total_panelists = len(df)
                    
                    # Show summary statistics cards
                    st.markdown("<div style='margin: 20px 0;'><div class='dashboard-title'>Ringkasan Penilaian</div></div>", unsafe_allow_html=True)
                    
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    
                    with stats_col1:
                        st.markdown(f"""
                        <div class="dashboard-card">
                            <div style="color: #CBD5E0; font-size: 0.9em;">Total Sampel</div>
                            <div class="dashboard-value">{total_panelists}</div>
                            <div style="color: #A0AEC0; font-size: 0.8em;">Penilaian</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with stats_col2:
                        avg_score = processed_df['Rata-rata'].mean() if len(processed_df) > 0 else 0
                        color = "#10B981" if avg_score >= 7 else "#FBBF24" if avg_score >= 5 else "#EF4444"
                        st.markdown(f"""
                        <div class="dashboard-card">
                            <div style="color: #CBD5E0; font-size: 0.9em;">Skor Rata-rata</div>
                            <div class="dashboard-value" style="color: {color};">{avg_score:.2f}</div>
                            <div style="color: #A0AEC0; font-size: 0.8em;">dari 9</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with stats_col3:
                        # Get the dominant category
                        categories = ['Prima', 'Advance', 'Sedang', 'Busuk']
                        category_counts = [
                            len(processed_df[processed_df['Kesegaran'] == 'Prima']),
                            len(processed_df[processed_df['Kesegaran'] == 'Advance']),
                            len(processed_df[processed_df['Kesegaran'] == 'Sedang']),
                            len(processed_df[processed_df['Kesegaran'] == 'Busuk'])
                        ]
                        if category_counts:
                            dominant_category = categories[category_counts.index(max(category_counts))]
                        else:
                            dominant_category = "N/A"
                        
                        # Set color based on category
                        category_color = {
                            'Prima': '#10B981',
                            'Advance': '#3B82F6',
                            'Sedang': '#FBBF24',
                            'Busuk': '#EF4444',
                            'N/A': '#A0AEC0'
                        }
                        
                        st.markdown(f"""
                        <div class="dashboard-card">
                            <div style="color: #CBD5E0; font-size: 0.9em;">Status Dominan</div>
                            <div class="dashboard-value" style="color: {category_color[dominant_category]};">{dominant_category}</div>
                            <div style="color: #A0AEC0; font-size: 0.8em;">Tingkat Kesegaran</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with stats_col4:
                        # Find the highest scoring parameter
                        if param_averages:
                            best_param = max(param_averages.items(), key=lambda x: x[1])
                            param_name = best_param[0]
                            param_value = best_param[1]
                        else:
                            param_name = "N/A"
                            param_value = 0
                        
                        st.markdown(f"""
                        <div class="dashboard-card">
                            <div style="color: #CBD5E0; font-size: 0.9em;">Parameter Terbaik</div>
                            <div class="dashboard-value">{param_name}</div>
                            <div style="color: #A0AEC0; font-size: 0.8em;">Skor: {param_value:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Main visualizations
                    col1, col2 = st.columns(2)

                    # Pie chart with improved styling
                    with col1:
                        st.markdown("<div class='dashboard-title' style='font-size: 1.2em;'>Distribusi Kesegaran</div>", unsafe_allow_html=True)
                        
                        total = len(processed_df)
                        categories = ['Prima', 'Advance', 'Sedang', 'Busuk']
                        values = []
                        for category in categories:
                            count = len(processed_df[processed_df['Kesegaran'] == category])
                            percentage = (count / total) * 100 if total > 0 else 0
                            values.append(percentage)
                        
                        colors = ['#4A235A', '#8E44AD', '#D2B4DE', '#7D3C98']
                        fig_pie = go.Figure(data=[go.Pie(labels=categories, 
                                                    values=values,
                                                    hole=.3,
                                                    marker_colors=colors)])
                        fig_pie.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=0, b=30, l=0, r=0),
                            font=dict(color='#CBD5E0'),
                            legend=dict(
                                font=dict(size=12, color='#CBD5E0'),
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)

                    # Radar chart with improved styling
                    with col2:
                        st.markdown("<div class='dashboard-title' style='font-size: 1.2em;'>Profil Kualitas Ikan</div>", unsafe_allow_html=True)
                        
                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=list(param_averages.values()),
                            theta=list(param_averages.keys()),
                            fill='toself',
                            name='Skor Rata-rata',
                            marker=dict(color='#9333EA'),
                            line=dict(color='#7928CA', width=2)))
                        
                        fig_radar.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=0, b=30, l=40, r=40),
                            font=dict(color='#CBD5E0'),
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 9],
                                    color='#64748B',
                                    gridcolor='rgba(147, 51, 234, 0.2)'
                                ),
                                angularaxis=dict(
                                    color='#64748B',
                                    gridcolor='rgba(147, 51, 234, 0.2)'
                                ),
                                bgcolor='rgba(31, 41, 55, 0.5)'
                            )
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                    # Add a bar chart for parameter scores
                    st.markdown("<div class='dashboard-title' style='font-size: 1.2em; margin-top: 20px;'>Skor Per Parameter</div>", unsafe_allow_html=True)
                    
                    params = list(param_averages.keys())
                    values = list(param_averages.values())
                    
                    # Create color gradient based on values
                    colors = [f'rgba(147, 51, 234, {val/9})' for val in values]
                    
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=params,
                        y=values,
                        marker_color=colors,
                        text=[f'{val:.2f}' for val in values],
                        textposition='auto'
                    ))
                    
                    fig_bar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=30, b=30, l=40, r=40),
                        font=dict(color='#CBD5E0'),
                        xaxis=dict(
                            title="Parameter",
                            titlefont=dict(size=14, color='#A0AEC0'),
                            tickfont=dict(color='#A0AEC0'),
                            gridcolor='rgba(147, 51, 234, 0.1)'
                        ),
                        yaxis=dict(
                            title="Skor Rata-rata",
                            titlefont=dict(size=14, color='#A0AEC0'),
                            tickfont=dict(color='#A0AEC0'),
                            gridcolor='rgba(147, 51, 234, 0.1)',
                            range=[0, 9]
                        )
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Tampilkan data lengkap dalam expander yang lebih menarik
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
                    # Improved conclusion section
                    st.markdown("<div class='section-title'>Kesimpulan</div>", unsafe_allow_html=True)

                    # Get the freshness of the first row for demo purposes
                    freshness = processed_df['Kesegaran'].iloc[0] if len(processed_df) > 0 else "Tidak ada data"
                    avg_score = processed_df['Rata-rata'].mean() if len(processed_df) > 0 else 0

                    # Determine the emoji and color based on freshness
                    emoji = '🟢' if freshness == 'Prima' else '🔵' if freshness == 'Advance' else '🟠' if freshness == 'Sedang' else '🔴'
                    color = '#10B981' if freshness == 'Prima' else '#3B82F6' if freshness == 'Advance' else '#FBBF24' if freshness == 'Sedang' else '#EF4444'

                    # Determine the recommendation text
                    recommendation = (
                        'Ikan sangat segar dan optimal untuk dikonsumsi atau diproses lebih lanjut.' if freshness == 'Prima' else 
                        'Ikan masih dalam kondisi baik dan aman untuk dikonsumsi atau diolah.' if freshness == 'Advance' else 
                        'Ikan dalam kondisi sedang, disarankan untuk segera diolah atau dikonsumsi.' if freshness == 'Sedang' else 
                        'Ikan dalam kondisi tidak segar, tidak disarankan untuk dikonsumsi.'
                    )

                    # Construct the conclusion text as a single f-string
                    conclusion_text = f"""
                    <div class="info-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                            <div>
                                <div style="font-size: 1.3em; font-weight: 600; color: #A855F7; margin-bottom: 5px;">Tingkat Mutu Ikan: {freshness}</div>
                                <div style="color: #CBD5E0;">Skor rata-rata: {avg_score:.2f} dari 9</div>
                            </div>
                            <div style="font-size: 2em; color: {color};">{emoji}</div>
                        </div>
                        <p style="margin-bottom: 15px;">Berdasarkan analisis terhadap {total_panelists} sampel penilaian, ikan memiliki kualitas <strong>{freshness.lower()}</strong> dengan skor rata-rata {avg_score:.2f}.</p>
                        {"<div style='margin-bottom: 15px;'><div style='font-weight: 600; margin-bottom: 5px; color: #A0AEC0;'>Parameter Terkuat:</div><div style='display: flex; justify-content: space-between; background: rgba(147, 51, 234, 0.1); padding: 10px; border-radius: 8px;'><div>" + best_param[0] + "</div><div style='font-weight: 600; color: #A855F7;'>" + f"{best_param[1]:.2f}/9" + "</div></div></div>" if param_averages else ""}
                        <div>
                            <div style="font-weight: 600; margin-bottom: 5px; color: #A0AEC0;">Rekomendasi:</div>
                            <p>{recommendation}</p>
                        </div>
                    </div>
                    """

                    # Render the conclusion text
                    st.markdown(conclusion_text, unsafe_allow_html=True)
                    # Add download functionality with better button styling
                    st.markdown("<div class='section-title'>Unduh Hasil Analisis</div>", unsafe_allow_html=True)
                    
                    # Function to download CSV
                    def download_csv(df, filename="data_penilaian_processed.csv"):
                        csv = df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="login-btn" style="text-decoration:none; display: inline-block; margin-right: 10px;">Download Data CSV</a>'
                        return href
                    
                    # Create columns for download buttons
                    dl_col1, dl_col2 = st.columns(2)
                    
                    with dl_col1:
                        st.markdown(download_csv(processed_df), unsafe_allow_html=True)
                    
                    with dl_col2:
                        # Add a PDF/report download option (placeholder functionality)
                        st.markdown("""
                        <a href="#" class="login-btn" style="text-decoration:none; display: inline-block; background: linear-gradient(135deg, #6B21A8, #4A235A);">
                            Download Laporan PDF
                        </a>
                        """, unsafe_allow_html=True)
                        st.markdown("<div style='font-size: 0.8em; color: #A0AEC0; margin-top: 5px;'>Coming soon</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error dalam memproses file: {str(e)}")
                st.markdown("""
                <div class="custom-notification" style="border-left-color: #EF4444;">
                    <div style="font-weight: 600; margin-bottom: 5px;">⚠️ Terjadi kesalahan</div>
                    <p>Pastikan format file CSV Anda sesuai dengan template. Lihat tab "Format Template" untuk informasi lebih lanjut.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Template tab - Show format requirements and template
    with tabs[1]:
        st.markdown("""
        <div class="info-card">
            <div style="font-size: 1.2em; font-weight: 600; color: #A855F7; margin-bottom: 15px;">Format Data CSV</div>
            <p style="margin-bottom: 15px;">File CSV Anda harus memiliki kolom-kolom berikut:</p>
            <ul style="margin-left: 20px; margin-bottom: 20px;">
                <li style="margin-bottom: 8px;"><b>No.</b>: Nomor urut penilaian</li>
                <li style="margin-bottom: 8px;"><b>Nama Panelis</b>: Nama penilai</li>
                <li style="margin-bottom: 8px;"><b>Mata, Insang, Lendir, Daging, Bau, Tekstur</b>: Parameter penilaian (nilai 1-9)</li>
            </ul>
            <div style="font-weight: 600; margin-bottom: 10px; color: #A0AEC0;">Ketentuan nilai:</div>
            <ul style="margin-left: 20px; margin-bottom: 15px;">
                <li style="margin-bottom: 5px;"><b>9</b>: Sangat Baik (Prima)</li>
                <li style="margin-bottom: 5px;"><b>7-8</b>: Masih Baik (Advance)</li>
                <li style="margin-bottom: 5px;"><b>5-6</b>: Tidak Segar (Sedang)</li>
                <li style="margin-bottom: 5px;"><b>1-4</b>: Sangat Tidak Segar (Busuk)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display editable dataframe template
        st.markdown("<div class='dashboard-title' style='font-size: 1.2em; margin-top: 20px;'>Template Data</div>", unsafe_allow_html=True)
        st.markdown("<p style='color: #CBD5E0; font-size: 0.9em; margin-bottom: 15px;'>Gunakan template berikut untuk membuat data Anda sendiri:</p>", unsafe_allow_html=True)
        
        # Improved template with sample data
        sample_data = pd.DataFrame({
            'No.': [1, 2, 3],
            'Nama Panelis': ["Panelis A", "Panelis B", "Panelis C"],
            'Mata': [8, 7, 9],
            'Insang': [7, 8, 9],
            'Lendir': [8, 7, 8],
            'Daging': [7, 8, 9],
            'Bau': [8, 7, 8],
            'Tekstur': [7, 8, 9],
        })
        
        edited_df = st.data_editor(
            sample_data,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True
        )
        
        # Download template button
        if st.button("⬇️ Download Template", use_container_width=True):
            csv = edited_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="template_penilaian_ikan.csv" class="login-btn" style="text-decoration:none; display: block; text-align: center; margin-top: 10px;">Download Template CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Tips tab
    with tabs[2]:
        st.markdown("""
        <div class="info-card">
            <div style="font-size: 1.2em; font-weight: 600; color: #A855F7; margin-bottom: 15px;">Tips Penggunaan</div>
            <div style="margin-bottom: 20px;">
                <div style="font-weight: 600; margin-bottom: 10px; color: #CBD5E0;">1. Persiapan Data</div>
                <ul style="margin-left: 20px; margin-bottom: 15px;">
                    <li style="margin-bottom: 8px;">Pastikan semua kolom wajib terisi</li>
                    <li style="margin-bottom: 8px;">Nilai parameter harus dalam rentang 1-9</li>
                    <li style="margin-bottom: 8px;">Simpan data dalam format CSV</li>
                </ul>
            </div>
            <div style="margin-bottom: 20px;">
                <div style="font-weight: 600; margin-bottom: 10px; color: #CBD5E0;">2. Analisis Hasil</div>
                <ul style="margin-left: 20px; margin-bottom: 15px;">
                    <li style="margin-bottom: 8px;">Perhatikan distribusi nilai kesegaran</li>
                    <li style="margin-bottom: 8px;">Identifikasi parameter dengan skor terendah untuk perbaikan</li>
                    <li style="margin-bottom: 8px;">Bandingkan hasil antar waktu untuk melihat tren</li>
                </ul>
            </div>
            <div style="margin-bottom: 20px;">
                <div style="font-weight: 600; margin-bottom: 10px; color: #CBD5E0;">3. Interpretasi</div>
                <ul style="margin-left: 20px; margin-bottom: 15px;">
                    <li style="margin-bottom: 8px;">Prima (9): Ikan sangat segar, optimal untuk semua penggunaan</li>
                    <li style="margin-bottom: 8px;">Advance (7-8): Ikan masih baik, cocok untuk konsumsi langsung</li>
                    <li style="margin-bottom: 8px;">Sedang (5-6): Perlu segera diolah, tidak untuk sashimi/mentah</li>
                    <li style="margin-bottom: 8px;">Busuk (1-4): Tidak disarankan untuk konsumsi</li>
                </ul>
            </div>
        </div>
        
        <div class="custom-notification" style="margin-top: 20px;">
            <div style="font-weight: 600; margin-bottom: 5px;">💡 Butuh bantuan?</div>
            <p>Untuk pertanyaan lebih lanjut tentang penggunaan fitur analisis data, silakan hubungi tim kami melalui WhatsApp.</p>
            <a href="https://wa.me/0895619313339" style="color: #A855F7; text-decoration: none; font-weight: 500; display: inline-block; margin-top: 10px;">
                Hubungi Support →
            </a>
        </div>
        """, unsafe_allow_html=True)

def process_prompt_response(selected_prompt, model_type, model_params, api_keys):
    """Process a prompt response without showing the prompt to the user"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Add the prompt to session state but don't display it
    st.session_state.messages.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": selected_prompt
        }]
    })
    
    # Show only the model response
    with st.chat_message("assistant"):
        model2key = {
            "openai": api_keys["openai"],
            "google": api_keys["google"],
            "anthropic": api_keys["anthropic"],
        }
        
        response = stream_llm_response(
            model_params=model_params,
            model_type=model_type,
            api_key=model2key[model_type]
        )
        
        st.write_stream(response)

def process_image_analysis(image_message, prompt_text, model_type, model_params, openai_api_key, google_api_key, anthropic_api_key):
    """Process an image analysis request with the selected prompt"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Add the image message to session
    st.session_state.messages.append(image_message)
    
    # Add the prompt for image analysis (but don't display it to the user)
    text_prompt = {
        "role": "user",
        "content": [{
            "type": "text",
            "text": prompt_text
        }]
    }
    
    st.session_state.messages.append(text_prompt)
    
    with st.spinner("Menganalisis gambar..."):
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
    
    # Show a success message
    st.success("Analisis selesai!")
    
    # Show a button to start a new analysis
    if st.button("🔄 Analisis Gambar Baru", use_container_width=True):
        # Clear the messages related to the last image
        if len(st.session_state.messages) >= 3:
            st.session_state.messages = st.session_state.messages[:-3]
        st.rerun()

def main():
    """Main application function"""
    # Setup page and initialize session state
    setup_page_config()
    apply_custom_css()
    initialize_session_state()

    # Check if user is authenticated
    if not st.session_state.is_authenticated:
        if not st.session_state.token_input_submitted:
            render_auth_screen()
        else:
            # If token was submitted but authentication failed
            render_auth_screen()
            render_unauthenticated_content()
    else:
        # User is authenticated - show main application
        # Setup sidebar and get model info
        model, model_type, model_params, openai_api_key, google_api_key, anthropic_api_key = handle_sidebar_and_model_selection()
        
        # Create API keys dictionary for easy access
        api_keys = {
            "openai": openai_api_key,
            "google": google_api_key,
            "anthropic": anthropic_api_key,
        }
        
        # Display the dashboard
        render_dashboard()
        
        # Create tabs to organize main content
        tabs = st.tabs(["💬 AI Assistant", "📸 Image Analysis", "📊 Data Analysis"])
        
        with tabs[0]:
            # Chat interface
            st.markdown("<div class='section-title'>💬 AI Assistant</div>", unsafe_allow_html=True)
            
            # Display the prompt buttons
            prompt = render_prompt_buttons()
            
            # Display the previous messages if there are any
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            
            if "messages" in st.session_state and len(st.session_state.messages) > 0:
                # Display only messages that should be visible to the user
                # (every message except for prompt messages that are processed separately)
                visible_messages = []
                i = 0
                while i < len(st.session_state.messages):
                    # If this is a user message with a prompt button text, skip it and its response
                    if (i < len(st.session_state.messages) - 1 and 
                        st.session_state.messages[i]["role"] == "user" and 
                        i + 1 < len(st.session_state.messages) and
                        st.session_state.messages[i]["content"][0]["type"] == "text" and
                        (st.session_state.messages[i]["content"][0]["text"].startswith("Analisis Kesegaran Ikan dari Gambar:") or
                         st.session_state.messages[i]["content"][0]["text"].startswith("Analisis Spesies Ikan dari Gambar:"))):
                        # Skip the prompt message
                        i += 1
                        # Add only the response to visible messages
                        if i < len(st.session_state.messages):
                            visible_messages.append(st.session_state.messages[i])
                        i += 1
                    else:
                        # Add regular message to visible messages
                        visible_messages.append(st.session_state.messages[i])
                        i += 1
                
                # Display the visible messages
                for message in visible_messages:
                    with st.chat_message(message["role"]):
                        for content in message["content"]:
                            if content["type"] == "text":
                                st.write(content["text"])
                            elif content["type"] == "image_url":      
                                st.image(content["image_url"]["url"])
            else:
                # Show welcome message if no messages
                st.markdown("""
                <div style="text-align: center; padding: 30px 0;">
                    <div style="font-size: 4em; margin-bottom: 20px;">👋</div>
                    <div style="font-size: 1.2em; font-weight: 600; color: #A855F7; margin-bottom: 15px;">
                        Selamat datang di VisionFish.io Assistant
                    </div>
                    <p style="color: #CBD5E0; max-width: 500px; margin: 0 auto;">
                        Asisten AI ini dapat membantu Anda dengan analisis ikan, informasi perikanan, dan praktik terbaik penanganan ikan.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Text input for chat
            user_input = st.chat_input("Tanyakan sesuatu tentang perikanan...")
            
            # Process user inputs
            if prompt:
                # Process prompt response without showing the prompt to the user
                process_prompt_response(prompt, model_type, model_params, api_keys)
                st.rerun()
            
            if user_input:
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                    
                st.session_state.messages.append({
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": user_input
                    }]
                })
                
                with st.chat_message("user"):
                    st.write(user_input)
                    
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
                
                # Rerun to update UI
                st.rerun()
        
        with tabs[1]:
            # Image analysis tab
            image_message = handle_image_upload()
            
            if image_message:
                # Display prompt selection after image upload
                st.markdown("<div class='dashboard-title' style='margin: 20px 0;'>Pilih Jenis Analisis</div>", unsafe_allow_html=True)
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    if st.button("🔍 Analisis Spesies", use_container_width=True):
                        process_image_analysis(image_message, get_analisis_ikan_prompt(), model_type, model_params, openai_api_key, google_api_key, anthropic_api_key)
                
                with analysis_col2:
                    if st.button("🌟 Analisis Kesegaran", use_container_width=True):
                        process_image_analysis(image_message, get_freshness_prompt(), model_type, model_params, openai_api_key, google_api_key, anthropic_api_key)
        
        with tabs[2]:
            # Data analysis tab
            handle_csv_upload_and_analysis()

if __name__=="__main__":
    main()
