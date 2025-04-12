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
import random 
import pytz


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
    return """Saya adalah VisionFish.io khusus seputar perikanan yang tetap dapat menentukan dan tidak ragu dalam menjawab.
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
    if "chat_reset" not in st.session_state:
        st.session_state.chat_reset = False

def messages_to_gemini(messages):
    """Format messages for Google's Gemini model"""
    gemini_messages = []
    prev_role = None
    
    # Add system prompt first
    gemini_messages.append({
        "role": "user",
        "parts": [{
            "text": get_system_prompt() + "\n\nSaya adalah VisionFish.io, asisten spesialis perikanan."
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

def get_freshness_prompt(model):
    """Get the prompt for fish freshness analysis tailored to the model with high accuracy and confidence"""
    base_prompt = """Analisis Kesegaran Ikan dari Gambar:
Jika gambar yang diunggah bukan gambar ikan, kembalikan pesan: "Mohon masukkan gambar ikan secara jelas. Analisis ini hanya berlaku untuk gambar ikan." Jika gambar adalah ikan, lanjutkan analisis berikut dengan penuh keyakinan berdasarkan observasi visual yang akurat.

Lakukan analisis kesegaran ikan berdasarkan parameter visual berikut:
- Mata: Perhatikan kejernihan (jernih vs. keruh), kilau (mengilap vs. kusam), posisi (cembung vs. cekung), dan warna kornea (bening vs. kekuningan).
- Insang: Evaluasi warna (merah cerah vs. merah pucat, abu-abu, atau kecokelatan), kelembapan (basah vs. kering), bau (segar vs. menyengat), dan ada tidaknya lendir berlebih.
- Lendir permukaan badan: Bedakan antara lendir bening tipis (alami, segar) dan lendir tebal keruh (tanda pembusukan). Perhatikan bau (netral vs. busuk).
- Sayatan daging: Tinjau warna (merah segar atau alami vs. kecokelatan/pucat), tekstur (kenyal vs. lembek), dan tanda-tanda kerusakan (berair, bau busuk).

Skor kesegaran untuk setiap parameter (1-9):
- 9 (Sangat Baik): Parameter menunjukkan kesegaran optimal (misalnya mata jernih cembung, insang merah cerah basah).
- 7-8 (Masih Baik): Minor ketidaksempurnaan (misalnya mata sedikit keruh, insang merah agak pucat).
- 5-6 (Tidak Segar): Tanda penurunan kualitas jelas (misalnya mata cekung, insang abu-abu kering).
- 1-4 (Sangat Tidak Segar): Kerusakan signifikan (misalnya mata keruh kekuningan, insang kecokelatan bau).

Hitung skor rata-rata dari semua parameter dan tentukan kategori kesegaran:
- Prima: Rata-rata = 9
- Advance: Rata-rata 7-8.9
- Sedang: Rata-rata 5-6.9
- Busuk: Rata-rata 1-4.9

Kembalikan hasil dalam format berikut:
Kesimpulan:
Skor: [X dari 9, berdasarkan rata-rata parameter].
Kategori: [Prima/Advance/Sedang/Busuk].
Detail:
- Mata: [skor, alasan berdasarkan observasi].
- Insang: [skor, alasan berdasarkan observasi].
- Lendir: [skor, alasan berdasarkan observasi].
- Daging: [skor, alasan berdasarkan observasi].
Alasan Keseluruhan: [Jelaskan dengan yakin mengapa skor ini diberikan, hindari keraguan atau asumsi]."""

    model_prompts = {
        "Neptune-Savant": f"{base_prompt}\n\nAnalisis setiap parameter dengan logika mendalam. Berikan skor dan alasan rinci berdasarkan observasi visual, seperti perubahan warna insang atau tekstur daging.",
        "ReefSpark-1.5": f"{base_prompt}\n\nFokus pada analisis cepat dan akurat. Berikan skor untuk setiap parameter dengan alasan singkat namun pasti, misalnya kejernihan mata atau warna insang.",
        "WaveCore-Pro": f"{base_prompt}\n\nLakukan analisis menyeluruh dengan detail visual maksimal. Berikan skor untuk setiap parameter dan jelaskan secara rinci, misalnya bagaimana lendir menunjukkan kesegaran.",
        "AquaVision-4": f"{base_prompt}\n\nGunakan penglihatan canggih untuk mengevaluasi setiap parameter dengan presisi tinggi. Sertakan alasan visual spesifik, seperti kilau mata atau kelembapan insang.",
        "TidalFlux-4": f"{base_prompt}\n\nBerikan analisis kuat dengan penjelasan terstruktur untuk setiap parameter. Jelaskan skor berdasarkan observasi seperti tekstur daging atau bau lendir.",
        "CoralPulse-3.5": f"{base_prompt}\n\nSederhanakan analisis dengan skor dan ringkasan singkat untuk setiap parameter. Pastikan alasan akurat, misalnya warna insang pucat menunjukkan ketidaksegaran.",
        "DeepMind-4": f"{base_prompt}\n\nGunakan pendekatan andal untuk menilai kesegaran. Berikan skor dan alasan tegas berdasarkan observasi seperti mata cekung atau daging lembek.",
        "OceanVault-4": f"{base_prompt}\n\nBerikan analisis ekstensif dengan skor dan deskripsi mendalam untuk setiap parameter. Jelaskan secara rinci, misalnya bagaimana lendir keruh menunjukkan pembusukan."
    }
    
    return model_prompts.get(model, base_prompt)

def get_analisis_ikan_prompt(model):
    """Get the prompt for fish species analysis tailored to the model with mandatory table"""
    base_prompt = """Analisis Spesies Ikan dari Gambar:
Jika gambar yang diunggah bukan gambar ikan, kembalikan pesan: "Mohon masukkan gambar ikan secara jelas. Analisis ini hanya berlaku untuk gambar ikan." Jika gambar adalah ikan, identifikasi spesies dengan penuh keyakinan dan hindari generalisasi.

Lakukan analisis spesies ikan berdasarkan gambar dengan fokus pada ciri-ciri visual berikut:
- Bentuk tubuh: Perhatikan proporsi tubuh (misalnya ramping, bulat, atau memanjang).
- Pola sisik: Amati ukuran, bentuk, dan pola sisik (misalnya sisik kecil mengilap atau besar kasar).
- Sirip: Perhatikan bentuk, ukuran, dan posisi sirip (misalnya sirip punggung panjang atau pendek).
- Warna: Catat warna dominan dan pola warna (misalnya garis-garis, bintik, atau warna seragam).
- Kepala: Perhatikan bentuk kepala dan mata (misalnya mata besar atau kecil, moncong tumpul atau runcing).
- Nama lokal: Gunakan nama lokal yang umum di Indonesia (misalnya 'tongkol' bukan hanya 'tuna').

Hindari mengacaukan spesies yang mirip, seperti ikan tongkol (Euthynnus affinis) dengan tuna (Thunnus spp.). Jika ragu, jelaskan alasan berdasarkan ciri visual yang terdeteksi.

Kembalikan hasil dalam format tabel berikut (wajib digunakan, tanpa pengecualian):

| Kategori    | Detail       |
|-------------|--------------|
| Nama Lokal  | [nama ikan, gunakan istilah lokal Indonesia]  |
| Nama Ilmiah | [nama latin, pastikan akurat] |
| Famili      | [famili, misalnya Scombridae] |

Setelah tabel, sertakan penjelasan singkat dan tegas tentang ciri-ciri visual spesifik yang digunakan untuk identifikasi, termasuk mengapa spesies ini dipilih dan bukan spesies serupa. Hindari keraguan atau kalimat tidak pasti."""

    model_prompts = {
        "Neptune-Savant": f"{base_prompt}\n\nGunakan penalaran mendalam untuk mengidentifikasi spesies secara akurat. Berikan detail ciri-ciri spesifik, seperti perbedaan antara ikan tongkol dan tuna berdasarkan bentuk sirip atau pola warna.",
        "ReefSpark-1.5": f"{base_prompt}\n\nLakukan identifikasi cepat dan pasti berdasarkan ciri utama. Jika spesies mirip (misalnya tongkol vs. tuna), sebutkan perbedaan kunci seperti ukuran tubuh atau bentuk sirip.",
        "WaveCore-Pro": f"{base_prompt}\n\nBerikan analisis menyeluruh dengan fokus pada semua ciri visual. Jelaskan secara rinci mengapa spesies tertentu dipilih, misalnya berdasarkan pola sisik atau warna tubuh yang khas untuk tongkol.",
        "AquaVision-4": f"{base_prompt}\n\nManfaatkan penglihatan canggih untuk identifikasi presisi tinggi. Sertakan penjelasan visual tajam, seperti perbedaan bentuk kepala antara tongkol dan tuna.",
        "TidalFlux-4": f"{base_prompt}\n\nHasilkan identifikasi kuat dengan tabel dan deskripsi terperinci tentang ciri-ciri spesies, termasuk mengapa tongkol bukan tuna berdasarkan ciri visual.",
        "CoralPulse-3.5": f"{base_prompt}\n\nSederhanakan identifikasi dengan tabel dan ringkasan langsung yang pasti. Sebutkan ciri utama seperti warna atau sirip untuk membedakan tongkol dari tuna.",
        "DeepMind-4": f"{base_prompt}\n\nGunakan metode andal untuk identifikasi spesies yang solid. Pastikan tabel dan penjelasan tegas, misalnya berdasarkan bentuk tubuh tongkol yang lebih ramping.",
        "OceanVault-4": f"{base_prompt}\n\nBerikan identifikasi ekstensif dengan tabel dan penjelasan lengkap tentang setiap ciri spesies, termasuk perbandingan dengan spesies serupa seperti tuna."
    }
    
    return model_prompts.get(model, base_prompt)

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
        ...
        /* Add this to your existing custom CSS */
        .n8n-chat-bubble {
            background-color: #9333EA !important;
            border-radius: 16px !important;
        }

        .n8n-chat-widget {
            border-radius: 16px !important;
            background-color: #111827 !important;
        }

        .n8n-chat-input {
            background-color: #1F2937 !important;
            border: 1px solid rgba(147, 51, 234, 0.3) !important;
        }
    </style>
    """, unsafe_allow_html=True)

def render_visionfish_chat():
    """Render a custom chat component with VisionFish.io themed UI/UX using @n8n/chat"""
    # Check if chat should be reset
    if "chat_reset" not in st.session_state:
        st.session_state.chat_reset = False

    visionfish_chat_html = """
    <div class="visionfish-chat-wrapper" id="visionfish-chat-wrapper">
        <style>
            /* Custom styling for VisionFish.io chat */
            .visionfish-chat-wrapper {
                height: 100%;
                background: #1A1625; /* Dark purple-gray background */
                border-radius: 16px;
                padding: 20px;
                border: 2px solid #9333EA; /* Purple border matching VisionFish theme */
                box-shadow: 0 8px 24px rgba(147, 51, 234, 0.2);
                overflow: hidden;
                font-family: 'Inter', system-ui, sans-serif;
                display: flex;
                flex-direction: column;
            }

            /* Chat header */
            .chat-header {
                background: linear-gradient(135deg, #9333EA, #6B21A8);
                color: #FFFFFF;
                padding: 12px 20px;
                border-radius: 10px 10px 0 0;
                font-size: 18px;
                font-weight: 600;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
                position: relative;
                animation: neonGlow 2s infinite alternate;
            }

            /* Neon glow effect for header */
            @keyframes neonGlow {
                0% {
                    box-shadow: 0 0 10px rgba(147, 51, 234, 0.3);
                }
                100% {
                    box-shadow: 0 0 20px rgba(147, 51, 234, 0.6);
                }
            }

            /* Custom notification style for welcome message */
            .custom-notification {
                background: #2A2438; /* Dark purple background */
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                color: #EDE9FE;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }

            .custom-notification .title {
                font-weight: 600;
                margin-bottom: 5px;
                display: flex;
                align-items: center;
                gap: 5px;
            }

            .custom-notification p {
                margin: 5px 0;
                line-height: 1.5;
            }

            /* Override default styles with higher specificity */
            :host {
                --chat-bg-color: transparent !important;
                --chat-header-bg-color: transparent !important;
                --chat-message-bg-color: #2A2438 !important; /* Dark purple background for messages */
                --chat-message-border-radius: 12px !important;
                --chat-message-text-color: #EDE9FE !important; /* Light purple text for messages */
                --chat-message-sender-color: #A78BFA !important; /* Lighter purple for sender name */
                --chat-message-timestamp-color: #A78BFA !important; /* Lighter purple for timestamp */
                --chat-input-bg-color: #2A2438 !important; /* Dark purple input background */
                --chat-input-border-color: rgba(147, 51, 234, 0.5) !important; /* Purple border for input */
                --chat-input-text-color: #EDE9FE !important; /* Light purple text for input */
                --chat-button-bg-color: #9333EA !important; /* Match VisionFish purple */
                --chat-button-hover-bg-color: #7E22CE !important; /* Darker purple on hover */
                --chat-button-text-color: #FFFFFF !important;
                --chat-text-color: #EDE9FE !important; /* Light purple text for all chat elements */
                --chat-font-family: 'Inter', system-ui, sans-serif !important;
            }

            /* Style for chat messages container */
            .chat-messages-container {
                flex-grow: 1;
                overflow-y: auto;
                padding: 20px;
                background: #111827;
                border-radius: 0 0 10px 10px;
                margin-bottom: 15px;
            }

            /* Style for chat messages */
            .chat-message {
                background: #2A2438 !important; /* Dark purple background for bot */
                border-radius: 12px !important;
                padding: 15px 20px !important;
                margin: 15px 0 !important;
                color: #EDE9FE !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
                transition: all 0.3s ease !important;
                max-width: 80% !important;
                animation: fadeInScale 0.5s ease-in-out;
                position: relative;
            }

            /* Fade-in and scale animation for messages */
            @keyframes fadeInScale {
                0% {
                    opacity: 0;
                    transform: translateY(10px) scale(0.95);
                }
                100% {
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }
            }

            .chat-message:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3) !important;
            }

            /* Particle effect on hover */
            .chat-message:hover::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: radial-gradient(circle, rgba(147, 51, 234, 0.2) 0%, transparent 70%);
                opacity: 0.5;
                pointer-events: none;
                animation: particleBurst 0.5s ease-out;
            }

            @keyframes particleBurst {
                0% {
                    opacity: 0;
                    transform: scale(0);
                }
                50% {
                    opacity: 0.5;
                }
                100% {
                    opacity: 0;
                    transform: scale(1.5);
                }
            }

            /* Bot messages specifically */
            .chat-message-bot {
                background: linear-gradient(135deg, #2A2438, #3B82F6) !important; /* Gradient from dark purple to blue */
                border-left: 4px solid #9333EA !important; /* Brighter purple for bot messages */
                margin-right: 20% !important;
                border-radius: 12px 12px 12px 0 !important; /* Rounded on right bottom */
                position: relative;
            }

            /* Add blue corner to bot messages */
            .chat-message-bot::after {
                content: '';
                position: absolute;
                bottom: 0;
                right: 0;
                width: 20px;
                height: 20px;
                background: #3B82F6; /* Blue color for the corner */
                border-radius: 0 0 12px 0;
                z-index: 1;
            }

            /* User messages specifically */
            .chat-message-user {
                background: #3C3356 !important; /* Slightly lighter purple for user */
                border-right: 4px solid #A78BFA !important; /* Lighter purple for user */
                margin-left: 20% !important;
                margin-right: 0 !important;
                border-radius: 12px 12px 0 12px !important; /* Rounded on left bottom */
            }

            /* Sender and timestamp */
            .chat-message .sender {
                font-size: 12px;
                font-weight: 600;
                margin-bottom: 5px;
            }

            .chat-message-bot .sender {
                color: #9333EA !important; /* Brighter purple for bot sender */
            }

            .chat-message-user .sender {
                color: #A78BFA !important; /* Lighter purple for user sender */
            }

            .chat-message .timestamp {
                font-size: 10px;
                color: #A78BFA !important;
                position: absolute;
                bottom: 5px;
                right: 10px;
                opacity: 0.7;
            }

            /* Style for input field container */
            .chat-input-container {
                background: rgba(42, 36, 56, 0.8) !important; /* Dark purple with glassmorphism effect */
                backdrop-filter: blur(10px); /* Glassmorphism blur effect */
                border: 1px solid rgba(147, 51, 234, 0.5) !important;
                border-radius: 10px !important;
                padding: 12px !important;
                display: flex !important;
                align-items: center !important;
                gap: 12px !important;
                width: 100% !important;
                margin-top: 16px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            }

            .chat-input-container:focus-within {
                border: 1px solid rgba(147, 51, 234, 0.8) !important;
                box-shadow: 0 0 0 3px rgba(147, 51, 234, 0.2) !important;
            }

            /* Style for the input field itself */
            .chat-input-container input {
                background: transparent !important;
                color: #EDE9FE !important;
                border: none !important;
                flex-grow: 1 !important;
                font-size: 15px !important;
                outline: none !important;
                line-height: 1.5 !important;
            }

            /* Style for placeholder text */
            .chat-input-container input::placeholder {
                color: #A78BFA !important;
                font-style: italic !important;
            }

            /* Style for send button */
            .chat-input-container button {
                background: linear-gradient(135deg, #9333EA, #6B21A8) !important;
                border-radius: 8px !important;
                padding: 8px 16px !important;
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
                border: none !important;
                cursor: pointer !important;
                display: flex;
                align-items: center;
                gap: 5px;
                animation: neonGlow 2s infinite alternate;
            }

            .chat-input-container button:hover {
                background: linear-gradient(135deg, #7E22CE, #5B1A8A) !important;
                box-shadow: 0 2px 8px rgba(147, 51, 234, 0.4) !important;
                transform: translateY(-1px) scale(1.05) !important; /* Bounce effect */
            }

            .chat-input-container button:active {
                transform: translateY(1px) !important;
            }

            /* Upload button styles */
            .chat-upload-button {
                background: rgba(40, 46, 66, 0.9) !important;
                border-radius: 8px !important;
                padding: 8px !important;
                color: #A78BFA !important;
                border: 1px solid rgba(147, 51, 234, 0.3) !important;
                transition: all 0.2s ease !important;
            }

            .chat-upload-button:hover {
                background: rgba(50, 56, 76, 1) !important;
                color: #EDE9FE !important;
                border-color: rgba(147, 51, 234, 0.6) !important;
                transform: translateY(-1px) scale(1.05) !important; /* Bounce effect */
            }

            /* Message typing indicator */
            .chat-typing-indicator {
                display: flex !important;
                align-items: center !important;
                gap: 4px !important;
                padding: 8px 12px !important;
                color: #A78BFA !important;
                font-size: 14px !important;
                font-style: italic !important;
            }

            /* Animated dots for typing indicator */
            .chat-typing-indicator::after {
                content: '...';
                display: inline-block;
                animation: dots 1.5s infinite;
            }

            @keyframes dots {
                0%, 20% {
                    content: '.';
                }
                40% {
                    content: '..';
                }
                60%, 100% {
                    content: '...';
                }
            }

            /* Scrollbar customization */
            .chat-messages-container::-webkit-scrollbar {
                width: 6px !important;
            }

            .chat-messages-container::-webkit-scrollbar-track {
                background: rgba(31, 41, 55, 0.5) !important;
                border-radius: 10px !important;
            }

            .chat-messages-container::-webkit-scrollbar-thumb {
                background: rgba(147, 51, 234, 0.5) !important;
                border-radius: 10px !important;
            }

            .chat-messages-container::-webkit-scrollbar-thumb:hover {
                background: rgba(147, 51, 234, 0.8) !important;
            }
            a {
                color: #A78BFA !important; /* Ungu terang yang sesuai tema VisionFish */
                text-decoration: none !important; /* Hilangkan garis bawah default */
                transition: color 0.3s ease !important; /* Efek transisi halus */
            }

            a:hover {
                color: #EDE9FE !important; /* Warna lebih terang saat hover */
                text-decoration: underline !important; /* Garis bawah muncul saat hover */
            }

            a:visited {
                color: #9333EA !important; /* Warna ungu sedikit lebih gelap untuk link yang sudah dikunjungi */
            }
        </style>
        <div class="chat-header">VisionFish Assistant</div>
        <div class="chat-messages-container" id="chat-messages-container">
            <div class="custom-notification">
                <div class="title">🤖 VisionFish Assistant</div>
                <p>Asisten ini dapat membantu Anda dengan pertanyaan seputar perikanan, analisis species ikan, dan tips kesegaran ikan.</p>
                <p>Anda juga dapat mengunggah gambar ikan untuk dianalisis langsung melalui chat.</p>
            </div>
        </div>
        <script type="module">
            import { createChat } from 'https://cdn.jsdelivr.net/npm/@n8n/chat/dist/chat.bundle.es.js';

            // Custom options for VisionFish.io integration
            const chatOptions = {
                webhookUrl: 'https://primary-production-c7f0.up.railway.app/webhook/d49a228d-703d-4a93-8e7a-ed173500fc6e/chat',
                container: document.querySelector('.chat-messages-container'),
                title: 'VisionFish Assistant',
                placeholder: 'Tanyakan tentang analisis ikan...',
                welcomeMessage: '', // Empty welcome message since we use custom notification
                fileUpload: {
                    enabled: true,
                    acceptedFileTypes: ['image/jpeg', 'image/png', 'image/gif'],
                },
                initialState: 'open' // Ensure chat is always open
            };

            // Function to initialize or reset the chat
            function initializeChat() {
                // Clear existing chat content if any
                const wrapper = document.querySelector('.chat-messages-container');
                wrapper.innerHTML = `
                    <div class="custom-notification">
                        <div class="title">🤖 VisionFish Assistant Info</div>
                        <p>Asisten ini dapat membantu Anda dengan pertanyaan seputar perikanan, analisis species ikan, dan tips kesegaran ikan.</p>
                        <p>Tekan icon "💬" di bawah ini untuk memulai chat.</p>
                    </div>
                `;

                // Re-create the chat instance
                const chatInstance = createChat(chatOptions);

                // Auto-scroll to bottom on new messages
                chatInstance.addEventListener('message', (event) => {
                    const chatContainer = document.querySelector('.chat-messages-container');
                    if (chatContainer) {
                        setTimeout(() => {
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }, 100);
                    }
                });

                // Ensure the input container is always visible
                const inputContainer = document.querySelector('.chat-input-container');
                if (inputContainer) {
                    inputContainer.style.display = 'flex';
                    inputContainer.style.opacity = '1';
                    inputContainer.style.visibility = 'visible';
                }

                // Apply custom styles after a short delay to ensure elements are rendered
                setTimeout(() => {
                    const messages = document.querySelectorAll('.chat-message');
                    messages.forEach(message => {
                        if (message.classList.contains('chat-message-bot')) {
                            message.style.background = 'linear-gradient(135deg, #2A2438, #3B82F6)';
                            message.style.borderLeft = '4px solid #9333EA';
                            message.style.marginRight = '20%';
                            message.style.borderRadius = '12px 12px 12px 0';
                        } else if (message.classList.contains('chat-message-user')) {
                            message.style.background = '#3C3356';
                            message.style.borderRight = '4px solid #A78BFA';
                            message.style.marginLeft = '20%';
                            message.style.marginRight = '0';
                            message.style.borderRadius = '12px 12px 0 12px';
                        }
                    });
                }, 100);
            }

            // Initialize chat on page load
            document.addEventListener('DOMContentLoaded', () => {
                initializeChat();
            });

            // Listen for reset signal from Streamlit
            window.addEventListener('message', (event) => {
                if (event.data === 'resetChat') {
                    console.log('Resetting chat...');
                    initializeChat();
                }
            });
        </script>
    </div>
    """
    # Render the chat component in Streamlit
    import streamlit.components.v1 as components
    components.html(visionfish_chat_html, height=600, scrolling=True)

    # If chat reset is requested, send a message to the client to reset the chat
    if st.session_state.chat_reset:
        st.session_state.chat_reset = False
        # Send a message to the client to reset the chat
        st.markdown(
            """
            <script>
                window.parent.postMessage('resetChat', '*');
            </script>
            """,
            unsafe_allow_html=True
        )

def setup_page_config():
    """Configure page settings"""
    st.set_page_config(
        page_title="VisionFish.io - Analisis Ikan Pintar",
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

                /* Chat container */
        .chat-container {
            background: rgba(31, 41, 55, 0.8);
            border-radius: 16px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(147, 51, 234, 0.2);
            height: 600px;  /* Set fixed height for the chat container */
            position: relative;
        }

        /* Add this to your apply_custom_css function */
        .n8n-chat-bubble {
            background-color: #9333EA !important;
            border-radius: 16px !important;
        }

        .n8n-chat-widget {
            border-radius: 16px !important;
            background-color: #111827 !important;
        }

        .n8n-chat-input {
            background-color: #1F2937 !important;
            border: 1px solid rgba(147, 51, 234, 0.3) !important;
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
        /* n8n chat container */
        .n8n-chat-wrapper {
            height: 100%;
            position: relative;
            border-radius: 16px;
            overflow: hidden;
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
        <div class="dashboard-title" style="margin-bottom: 15px;">🤖 Vision Fish.io</div>
    """, unsafe_allow_html=True)

def render_auth_screen():
    """Render the authentication screen with enhanced UI/UX"""
    # Custom CSS for the authentication screen
    st.markdown("""
    <style>
        /* Welcome Overlay */
        .welcome-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            animation: fadeInOverlay 0.5s ease-in-out;
        }
        .welcome-box {
            background: linear-gradient(135deg, #9333EA, #6B21A8);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(147, 51, 234, 0.4);
            animation: slideUp 0.8s ease-out;
            max-width: 500px;
            border: 2px solid rgba(255, 255, 255, 0.2);
        }
        .welcome-title {
            font-size: 2.2em;
            font-weight: 700;
            margin-bottom: 15px;
            text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        }
        .welcome-subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            line-height: 1.5;
        }
        /* Animations */
        @keyframes fadeInOverlay {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        @keyframes slideUp {
            0% { opacity: 0; transform: translateY(50px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeOut {
            0% { opacity: 1; }
            100% { opacity: 0; }
        }
        .auth-container {
            background: linear-gradient(145deg, rgba(31, 41, 55, 0.95), rgba(17, 24, 39, 0.9));
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 12px 30px rgba(147, 51, 234, 0.15);
            margin: 50px auto;
            max-width: 500px;
            text-align: center;
            border: 1px solid rgba(147, 51, 234, 0.3);
            position: relative;
            overflow: hidden;
            animation: fadeIn 1s ease-in-out;
        }
        .auth-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(147, 51, 234, 0.1) 0%, transparent 70%);
            opacity: 0.3;
            z-index: 0;
        }
        .auth-title {
            color: #9333EA;
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            animation: slideIn 0.8s ease-out;
        }
        .auth-subtitle {
            color: #D1D5DB;
            font-size: 1em;
            margin-bottom: 30px;
            line-height: 1.5;
            animation: fadeIn 1.2s ease-in-out;
        }
        .auth-input-container {
            position: relative;
            margin-bottom: 20px;
            z-index: 1;
        }
        .auth-input {
            width: 100%;
            padding: 12px 15px;
            border-radius: 8px;
            border: 1px solid rgba(147, 51, 234, 0.3);
            background: rgba(31, 41, 55, 0.8);
            color: #E5E7EB;
            font-size: 1em;
            transition: all 0.3s ease;
        }
        .auth-input:focus {
            outline: none;
            border-color: #9333EA;
            box-shadow: 0 0 0 3px rgba(147, 51, 234, 0.2);
        }
        .auth-input::placeholder {
            color: #9CA3AF;
            font-style: italic;
        }
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
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: 100%;
            box-shadow: 0 4px 15px rgba(147, 51, 234, 0.3);
            animation: pulse 2s infinite;
        }
        .login-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(147, 51, 234, 0.4);
            background: linear-gradient(135deg, #7E22CE, #5B1A8A);
        }
        .login-btn:active {
            transform: translateY(1px);
        }
        .message-box {
            padding: 10px 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 0.9em;
            display: flex;
            align-items: center;
            gap: 8px;
            animation: slideIn 0.5s ease-out;
        }
        .error-message {
            background: rgba(239, 68, 68, 0.1);
            border-left: 4px solid #EF4444;
            color: #FECACA;
        }
        .success-message {
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid #10B981;
            color: #A7F3D0;
        }
        /* Animations */
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        @keyframes slideIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0% { box-shadow: 0 4px 15px rgba(147, 51, 234, 0.3); }
            50% { box-shadow: 0 4px 25px rgba(147, 51, 234, 0.5); }
            100% { box-shadow: 0 4px 15px rgba(147, 51, 234, 0.3); }
        }
        /* Hide Streamlit's default "Press Enter to submit form" message */
        [data-testid="stFormSubmitButton"] + div {
            display: none !important;
        }
        /* Responsive adjustments for mobile */
        @media (max-width: 768px) {
            .auth-container {
                padding: 20px;
                margin: 20px auto;
                max-width: 90%;
            }
            .auth-title {
                font-size: 1.5em;
                gap: 8px;
            }
            .auth-subtitle {
                font-size: 0.9em;
                margin-bottom: 20px;
            }
            .auth-input {
                padding: 10px 12px;
                font-size: 0.9em;
            }
            .login-btn {
                padding: 10px 20px;
                font-size: 0.9em;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def render_auth_screen():
    # Bagian CSS tetap sama

    # Authentication screen layout
    st.markdown("""
    <div class="auth-container">
        <div class="auth-title">
            <span>🔐</span>VisionFish.io
        </div>
        <p class="auth-subtitle">Enter your access token to unlock all features of VisionFish.io</p>
    </div>
    """, unsafe_allow_html=True)

# Inisialisasi state
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    if 'token_input_submitted' not in st.session_state:
        st.session_state.token_input_submitted = False
    if 'show_welcome' not in st.session_state:
        st.session_state.show_welcome = False

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form(key="login_form", clear_on_submit=True):
            access_token = st.text_input(
                "Enter access token:",
                type="password",
                placeholder="Your access token here...",
                label_visibility="collapsed",
                key="access_token_input"
            )
            submit_button = st.form_submit_button(label="Login", use_container_width=True)

            # Placeholder untuk pesan error
            message_container = st.empty()

            if submit_button and not st.session_state.is_processing:
                st.session_state.is_processing = True
                with st.spinner("Authenticating..."):
                    import time
                    time.sleep(1)
                    is_authenticated = access_token in valid_tokens
                    st.session_state.is_authenticated = is_authenticated
                    st.session_state.token_input_submitted = True
                    
                    if not is_authenticated:
                        message_container.markdown(
                            '<div class="message-box error-message">❌ Invalid token. Please try again.</div>',
                            unsafe_allow_html=True
                        )
                        render_unauthenticated_content()
                    else:
                        message_container.markdown(
                            '<div class="message-box success-message">✅ Authentication successful!</div>',
                            unsafe_allow_html=True
                        )
                        # Tandai untuk menampilkan pesan selamat datang
                        st.session_state.show_welcome = True
                        st.rerun()
                
                st.session_state.is_processing = False

    # Tampilkan pesan selamat datang jika diperlukan
    if st.session_state.get('show_welcome', False):
        st.markdown("""
        <div class="welcome-overlay">
            <div class="welcome-box">
                <div class="welcome-title">🌟 Selamat Datang di VisionFish!</div>
                <div class="welcome-subtitle">Nikmati pengalaman cerdas untuk perikanan modern bersama kami.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Tambahkan delay lalu reset show_welcome
        import time
        time.sleep(3)  # Tampilkan selama 3 detik
        st.session_state.show_welcome = False
        st.rerun()

def render_unauthenticated_content():
    """Render content for unauthenticated users with enhanced UI/UX"""
    # Custom CSS for the unauthenticated content
    st.markdown("""
    <style>
        .content-wrapper {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
            position: relative;
            z-index: 1;
        }
        .main-title {
            background: linear-gradient(120deg, #9333EA, #6B21A8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: float 3s ease-in-out infinite;
            font-weight: 800;
            font-size: 3.5em;
            margin: 20px auto;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .main-subtitle {
            color: #D1D5DB;
            font-size: 1.2em;
            margin: 0 auto 40px auto;
            max-width: 800px;
            line-height: 1.6;
            animation: fadeIn 1.5s ease-in-out;
        }
        .dashboard-title {
            color: #9333EA;
            font-size: 1.8em;
            font-weight: 700;
            margin: 40px 0 20px 0;
            text-align: center;
            border-bottom: 2px solid rgba(147, 51, 234, 0.3);
            padding-bottom: 10px;
            animation: slideIn 1s ease-out;
        }
        .feature-box {
            background: linear-gradient(145deg, rgba(31, 41, 55, 0.95), rgba(17, 24, 39, 0.9));
            border-radius: 16px;
            padding: 25px;
            margin: 15px 0;
            border: 1px solid rgba(147, 51, 234, 0.3);
            transition: all 0.3s ease;
            height: 100%;
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.8s ease-out;
        }
        .feature-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(147, 51, 234, 0.2);
            border: 1px solid rgba(147, 51, 234, 0.5);
        }
        .feature-box::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(147, 51, 234, 0.1) 0%, transparent 70%);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .feature-box:hover::before {
            opacity: 1;
        }
        .feature-icon {
            font-size: 2.5em;
            margin-bottom: 15px;
            background: linear-gradient(135deg, #9333EA, #6B21A8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: float 3s ease-in-out infinite;
        }
        .feature-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 10px;
            color: #A855F7;
        }
        .feature-desc {
            color: #D1D5DB;
            font-size: 1em;
            line-height: 1.5;
        }
        .info-card {
            background: linear-gradient(145deg, rgba(31, 41, 55, 0.95), rgba(17, 24, 39, 0.9));
            border-radius: 16px;
            padding: 20px;
            border: 2px solid rgba(147, 51, 234, 0.3);  /* Uniform border */
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            margin: 15px 0;
            transition: all 0.3s ease;
            animation: fadeInUp 1s ease-out;
        }
        .info-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 25px rgba(147, 51, 234, 0.15);
            border: 2px solid rgba(147, 51, 234, 0.5);
        }
        .analysis-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #A855F7;
            margin-bottom: 15px;
        }
        .analysis-image {
            width: 100%;
            border-radius: 10px;
            margin-bottom: 15px;
            transition: transform 0.3s ease;
        }
        .analysis-image:hover {
            transform: scale(1.05);
        }
        .analysis-table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            color: #E5E7EB;
        }
        .analysis-table th, .analysis-table td {
            padding: 10px;
            border-bottom: 1px solid rgba(147, 51, 234, 0.2);
        }
        .analysis-table th {
            color: #A855F7;
            font-weight: 600;
        }
        .freshness-score {
            font-size: 2.5em;
            font-weight: 700;
            color: #A855F7;
        }
        .freshness-status {
            font-size: 1.1em;
            font-weight: 500;
            color: #10B981;
        }
        .testimonial-quote {
            font-size: 1em;
            line-height: 1.5;
            margin-bottom: 15px;
            font-style: italic;
            color: #E5E7EB;
        }
        .testimonial-name {
            font-weight: 600;
            color: #A855F7;
        }
        .testimonial-role {
            font-size: 0.9em;
            color: #CBD5E0;
        }
        .quick-link {
            color: #A855F7;
            text-decoration: none;
            font-weight: 600;
            font-size: 1rem;
            font-family: 'Inter', sans-serif;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 15px;
            border-radius: 8px;
            transition: all 0.3s ease;
            animation: fadeIn 1.5s ease-in-out;
        }
        .quick-link:hover {
            background: rgba(147, 51, 234, 0.1);
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(147, 51, 234, 0.2);
        }
        .quick-link span {
            font-size: 1.2rem;
        }
        .cta-button {
            background: linear-gradient(135deg, #9333EA, #6B21A8);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 30px;
            box-shadow: 0 4px 15px rgba(147, 51, 234, 0.3);
            animation: pulse 2s infinite;
        }
        .cta-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(147, 51, 234, 0.4);
            background: linear-gradient(135deg, #7E22CE, #5B1A8A);
        }
        /* Animations */
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        @keyframes fadeInUp {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
            100% { transform: translateY(0px); }
        }
        @keyframes pulse {
            0% { box-shadow: 0 4px 15px rgba(147, 51, 234, 0.3); }
            50% { box-shadow: 0 4px 25px rgba(147, 51, 234, 0.5); }
            100% { box-shadow: 0 4px 15px rgba(147, 51, 234, 0.3); }
        }
        /* Responsive adjustments for mobile */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2.5em;
            }
            .main-subtitle {
                font-size: 1em;
                margin-bottom: 30px;
            }
            .dashboard-title {
                font-size: 1.4em;
                margin: 30px 0 15px 0;
            }
            .feature-box {
                padding: 20px;
            }
            .feature-icon {
                font-size: 2em;
            }
            .feature-title {
                font-size: 1.1em;
            }
            .feature-desc {
                font-size: 0.9em;
            }
            .info-card {
                padding: 15px;
            }
            .analysis-title {
                font-size: 1.1em;
            }
            .freshness-score {
                font-size: 2em;
            }
            .freshness-status {
                font-size: 0.9em;
            }
            .analysis-table th, .analysis-table td {
                padding: 8px;
                font-size: 0.9em;
            }
            .testimonial-quote {
                font-size: 0.9em;
            }
            .testimonial-name {
                font-size: 0.95em;
            }
            .testimonial-role {
                font-size: 0.85em;
            }
            .quick-link {
                font-size: 0.9em;
                padding: 8px 12px;
            }
            .quick-link span {
                font-size: 1em;
            }
            .cta-button {
                padding: 10px 20px;
                font-size: 0.9em;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        .custom-header {
            font-size: 24px;
            color: #6A0DAD;  /* Warna ungu */
            margin: 0;
            text-align: center;  /* Memusatkan teks */
        }
        .custom-subtitle {
            font-size: 16px;
            color: #FFFFFF;
            text-align: center;  /* Memusatkan subtitle juga agar seragam */
        }
        @media (max-width: 768px) {
            .custom-header {
                font-size: 24px;
                color: #9333EA
                margin: 0;
                text-align: center !important;
            }
            .custom-subtitle {
                font-size: 14px;
            }
        }
    </style>
    <div>
        <h1 class="custom-header">VisionFish.io</h1>
        <p class="custom-subtitle">
            Solusi analisis ikan terdepan untuk mendukung industri perikanan dengan teknologi OpenCV
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick Links Section
    st.markdown("""
    <div class="content-wrapper">
        <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 40px;">
            <a href="https://syarifrizik.github.io/coming-soon/" target="_blank" class="quick-link">
                <span>📖</span> Penggunaan
            </a>
            <a href="https://drive.google.com/drive/u/1/folders/1afYnOjsv9y95UPDviKoZf0YTe5pHFzWl" target="_blank" class="quick-link">
                <span>📚</span> Dokumentasi
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature showcase with improved layout
    st.markdown("<h2 class='dashboard-title'>Fitur Unggulan</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Identifikasi Spesies</div>
            <p class="feature-desc">Kenali spesies ikan secara instan dari gambar dengan akurasi tinggi</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Visualisasi Data</div>
            <p class="feature-desc">Dapatkan grafik dan laporan mendalam untuk analisis kualitas ikan</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">🌊</div>
            <div class="feature-title">Analisis Kesegaran</div>
            <p class="feature-desc">Ukur tingkat kesegaran ikan dengan presisi ilmiah</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">📱</div>
            <div class="feature-title">Monitoring Real-time</div>
            <p class="feature-desc">Pantau kualitas air secara langsung dari stasiun pemantauan</p>
        </div>
        """, unsafe_allow_html=True)

    # Showcase a sample analysis to entice users
    st.markdown("<h2 class='dashboard-title'>Contoh Analisis</h2>", unsafe_allow_html=True)

    # CSS dan HTML dalam satu blok
    st.markdown("""
    <style>
    .info-card {
        border: 2px solid #8a2be2;
        border-radius: 10px;
        padding: 15px;
        background-color: #1e1e2f;
        color: #ffffff;
        min-height: 400px; /* Menyamakan tinggi minimum card */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Membuat konten terdistribusi dengan baik */
    }
    .analysis-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #8a2be2;
    }
    .analysis-image {
        width: 100%;
        height: 150px; /* Menyamakan tinggi gambar */
        object-fit: cover; /* Memastikan gambar tidak terdistorsi */
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .freshness-score {
        font-size: 28px;
        font-weight: bold;
        color: #ffffff;
    }
    .freshness-status {
        font-size: 16px;
        color: #00cc00;
        margin-top: 5px;
    }
    .data-box {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 10px;
        flex-grow: 1; /* Membuat data-box mengisi ruang yang tersedia */
    }
    .data-item {
        display: flex;
        justify-content: space-between;
        background-color: #2a2a3d;
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        cursor: default;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .data-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .data-label {
        font-weight: bold;
        color: #ffffff;
    }
    .data-value {
        color: #ffffff;
    }
    .green-textbox {
        justify-content: space-between; /* Mengatur teks dan ikon */
        align-items: center; /* Vertikal tengah */
        border: 2px solid #00cc00;
        background-color: #2a2a3d;
    }
    .data-text {
        color: #00cc00;
        font-weight: bold;
    }
    .arrow-icon {
        color: #00cc00;
        font-weight: bold;
        font-size: 12px; /* Memperkecil ikon */
    }
    .login-text {
        font-size: 14px;
        font-style: italic;
        text-align: center;
        margin-top: 10px;
    }
    .gray-text {
        color: #a9a9a9; /* Warna abu-abu untuk kata yang tidak terang */
    }
    .white-text {
        color: #ffffff; /* Warna putih untuk kata yang lebih menonjol */
    }
    </style>
    """, unsafe_allow_html=True)

    # Kolom 1 dan Kolom 2
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="info-card">
            <div>
                <div class="analysis-title">Analisis Spesies</div>
                <div>
                    <img src="https://thegorbalsla.com/wp-content/uploads/2019/08/Ciri-Ciri-Ikan-Tongkol.jpg" 
                    class="analysis-image">
                </div>
                <div class="data-box">
                    <div class="data-item">
                        <span class="data-label">Nama Lokal</span>
                        <span class="data-value">Tongkol</span>
                    </div>
                    <div class="data-item">
                        <span class="data-label">Nama Ilmiah</span>
                        <span class="data-value">Euthynnus affinis</span>
                    </div>
                    <div class="data-item">
                        <span class="data-label">Famili</span>
                        <span class="data-value">Scombridae</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div>
                <div class="analysis-title">Analisis Kesegaran</div>
                <div>
                    <img src="https://thegorbalsla.com/wp-content/uploads/2019/08/Ciri-Ciri-Ikan-Tongkol.jpg" 
                    class="analysis-image">
                </div>
                <div style="text-align: center; margin-bottom: 15px;">
                    <div class="freshness-score">8.5/10</div>
                    <div class="freshness-status">Masih Baik (Good)</div>
                </div>
                <div class="data-box">
                    <div class="data-item green-textbox">
                        <span class="data-text">Lihat Parameter Penilaian</span>
                        <span class="arrow-icon">></span>
                    </div>
                    <div class="login-text">
                        <span class="gray-text">*Silahkan login  terlebih dahulu</span> 
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    # CSS untuk testimonial section
    st.markdown("""
    <style>
    .dashboard-title {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
    }
    .testimonial-card {
        border: 2px solid #8a2be2; /* Border ungu seperti card sebelumnya */
        border-radius: 10px;
        padding: 15px;
        background-color: #1e1e2f;
        color: #ffffff;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .testimonial-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .testimonial-quote {
        font-size: 16px;
        font-style: italic;
        color: #d3d3d3;
        margin-bottom: 15px;
        text-align: center;
    }
    .testimonial-name {
        font-size: 18px;
        font-weight: bold;
        color: #8a2be2; /* Warna ungu untuk nama */
        text-align: center;
    }
    .testimonial-role {
        font-size: 14px;
        color: #ffffff; /* Warna putih untuk role */
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Add a testimonial section
    st.markdown("<h2 class='dashboard-title'>Digunakan Oleh</h2>", unsafe_allow_html=True)

    cols = st.columns(3)

    testimonials = [
        {
            "quote": "VisionFish.io mempercepat analisis kualitas ikan kami secara signifikan.",
            "name": "PT Samudra Sejahtera",
            "role": "Distributor Ikan"
        },
        {
            "quote": "Identifikasi spesies yang akurat membantu kami mengelola stok lebih efisien.",
            "name": "TPI Sungai Rengas",
            "role": "Tempat Pelelangan Ikan"
        },
        {
            "quote": "Data kualitas air real-time sangat mendukung budidaya ikan saya.",
            "name": "Eddy Saputra",
            "role": "Nelayan Lokal"
        }
    ]

    for i, col in enumerate(cols):
        with col:
            testimonial = testimonials[i]
            st.markdown(f"""
            <div class="testimonial-card">
                <div class="testimonial-quote">
                    "{testimonial['quote']}"
                </div>
                <div class="testimonial-name">{testimonial['name']}</div>
                <div class="testimonial-role">{testimonial['role']}</div>
            </div>
            """, unsafe_allow_html=True)



def render_dashboard():
    """Render the water quality dashboard with enhanced UI"""
    # Header with enhanced animations
    st.markdown("""
        <style>
            .custom-title {
                color: #6B21A8;
            }
        </style>
        <div class="content-wrapper">
            <div style="text-align: center; padding: 20px 0; margin-bottom: 20px;">
                <div style="font-size: 3em; margin-bottom: 10px;">🐟</div>
                <div style="font-size: 1.4em; font-weight: 600; color: #A855F7; margin-bottom: 10px;">
                    Vision Fish.io Assistant
                </div>
                <p style="color: #CBD5E0;">
                    Tanyakan tentang ikan, analisis gambar, atau pantau kualitas air
                </p>
            </div>
            <h1 class="custom-title">
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

    # Function to fetch the latest data from ThingSpeak
    def fetch_latest_thingspeak_data(channel_id, field):
        api_url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{field}.json?results=1"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            if not data['feeds']:
                return None
            latest_entry = data['feeds'][-1]
            field_key = f'field{field}'
            value = float(latest_entry[field_key]) if field_key in latest_entry and latest_entry[field_key] is not None else None
            return value
        except requests.RequestException as e:
            st.error(f"Error fetching data for field {field}: {e}")
            return None

    # Define a more sophisticated color and status logic
    def get_status_info(value, optimal_range, warning_range, unit):
        if value is None:
            return "#6B7280", "No Data", "⚪"  # Gray for no data with neutral icon
        
        # Status colors inspired by water quality standards
        optimal_color = "#34D399"  # Soft teal for optimal (better than pure green)
        warning_color = "#FBBF24"  # Warm yellow for warning
        critical_color = "#F87171"  # Soft red for critical
        
        if optimal_range[0] <= value <= optimal_range[1]:
            return optimal_color, "Optimal", "✅"
        elif warning_range[0] <= value <= warning_range[1]:
            return warning_color, "Perhatian", "⚠️"
        else:
            return critical_color, "Kritis", "❌"

    # Dashboard metrics with attractive cards
    st.markdown("""
    <div class='section-title'>
        <span>📊</span> Dashboard Kualitas Air Sungai Rengas
    </div>
    """, unsafe_allow_html=True)

    # Custom CSS for enhanced UI/UX
    st.markdown("""
    <style>
        .section-title {
                color: #9333EA;
                font-size: 1.8em;
                font-weight: 700;
                margin: 30px 0 20px 0;
                text-align: left;
                border-bottom: 2px solid rgba(147, 51, 234, 0.3);
                padding-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            /* Responsive adjustments for mobile */
            @media (max-width: 768px) {
                .section-title {
                    font-size: 1.2em !important;  /* Smaller font size on mobile */
                    margin: 20px 0 15px 0 !important;  /* Reduce margin */
                    padding-bottom: 8px !important;  /* Reduce padding */
                    gap: 8px !important;  /* Reduce gap between icon and text */
                }
                .section-title::before {
                    font-size: 1.2em !important;  /* Adjust icon size on mobile */
                }
            }
        .metric-card {
            background: linear-gradient(145deg, rgba(31, 41, 55, 0.95), rgba(17, 24, 39, 0.9));
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(147, 51, 234, 0.2);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
            text-align: center;
            height: 100%;
            position: relative;
            overflow: hidden;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(147, 51, 234, 0.25);
            border: 1px solid rgba(147, 51, 234, 0.5);
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(147, 51, 234, 0.1) 0%, transparent 70%);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .metric-card:hover::before {
            opacity: 1;
        }
        .metric-title {
            color: #D1D5DB;
            font-size: 0.95em;
            font-weight: 500;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: 700;
            margin: 5px 0;
            transition: color 0.3s ease;
        }
        .metric-status {
            font-size: 0.85em;
            font-weight: 500;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Define your ThingSpeak Channel ID
    channel_id = 2796290

    # Fetch latest data for each parameter
    temperature = fetch_latest_thingspeak_data(channel_id, 1)  # Field 1: Suhu Air (°C)
    ph = fetch_latest_thingspeak_data(channel_id, 2)           # Field 2: pH
    dissolved_oxygen = fetch_latest_thingspeak_data(channel_id, 3)  # Field 3: Oksigen Terlarut (mg/L)
    turbidity = fetch_latest_thingspeak_data(channel_id, 4)    # Field 4: Kekeruhan (NTU)

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        temp_value = f"{temperature:.1f}°C" if temperature is not None else "N/A"
        temp_color, temp_status, temp_icon = get_status_info(temperature, (20, 30), (15, 35), "°C")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🌡️ Suhu Air</div>
            <div class="metric-value" style="color: {temp_color};">{temp_value}</div>
            <div class="metric-status" style="color: {temp_color};">{temp_icon} {temp_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with metrics_col2:
        ph_value = f"{ph:.1f}" if ph is not None else "N/A"
        ph_color, ph_status, ph_icon = get_status_info(ph, (6.5, 8.5), (6.0, 9.0), "")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚗️ pH</div>
            <div class="metric-value" style="color: {ph_color};">{ph_value}</div>
            <div class="metric-status" style="color: {ph_color};">{ph_icon} {ph_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with metrics_col3:
        do_value = f"{dissolved_oxygen:.1f} mg/L" if dissolved_oxygen is not None else "N/A"
        do_color, do_status, do_icon = get_status_info(dissolved_oxygen, (5, 8), (3, 10), "mg/L")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💨 Oksigen Terlarut</div>
            <div class="metric-value" style="color: {do_color};">{do_value}</div>
            <div class="metric-status" style="color: {do_color};">{do_icon} {do_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with metrics_col4:
        turb_value = f"{turbidity:.1f} NTU" if turbidity is not None else "N/A"
        turb_color, turb_status, turb_icon = get_status_info(turbidity, (0, 25), (25, 50), "NTU")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🌫️ Kekeruhan</div>
            <div class="metric-value" style="color: {turb_color};">{turb_value}</div>
            <div class="metric-status" style="color: {turb_color};">{turb_icon} {turb_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Function to fetch data from ThingSpeak API with error handling
    def fetch_thingspeak_data(channel_id, field, results=60, timeout=5):
        api_url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{field}.json?results={results}"
        try:
            response = requests.get(api_url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            feeds = data['feeds']
            dates = [datetime.strptime(entry['created_at'], "%Y-%m-%dT%H:%M:%SZ") for entry in feeds]
            flow_rates = [float(entry[f'field{field}']) if entry[f'field{field}'] else None for entry in feeds]
            return dates, flow_rates
        except (requests.RequestException, ValueError, KeyError) as e:
            st.error(f"Error fetching data: {str(e)}")
            return [], []

    # Function to create Plotly chart with dark theme, panning enabled, and mobile-friendly title
    def create_flow_chart(dates, flow_rates):
        fig = go.Figure()
        # Candlestick chart for flow range
        valid_flow_rates = [f for f in flow_rates if f is not None]
        min_rate = min(valid_flow_rates) if valid_flow_rates else 0
        max_rate = max(valid_flow_rates) if valid_flow_rates else 0
        fig.add_trace(go.Candlestick(
            x=dates,
            open=[min_rate] * len(dates),
            high=flow_rates,
            low=flow_rates,
            close=[max_rate] * len(dates),
            increasing_line_color='#10B981',
            decreasing_line_color='#EF4444',
            name='Flow Range'
        ))
        # Scatter plot for actual flow rates
        fig.add_trace(go.Scatter(
            x=dates,
            y=flow_rates,
            mode='lines+markers',
            name='Flow Rate',
            line=dict(color='#9333EA', width=2, shape='spline'),
            marker=dict(size=6, color='#A855F7', line=dict(width=1, color='#D1D5DB')),
            hovertemplate='Tanggal: %{x|%Y-%m-%d %H:%M}<br>Flow Rate: %{y:.2f} L/m<extra></extra>'
        ))
        fig.update_layout(
            title={
                'text': "Flow Analytics Dashboard - Sungai Rengas",
                'font': {'size': 18, 'color': '#EDE9FE', 'family': "Arial"},
                'x': 0.5, 'xanchor': 'center',
                'y': 0.95, 'yanchor': 'top'
            },
            xaxis_title="Waktu",
            yaxis_title="Flow Rate (L/m)",
            plot_bgcolor='#1F2937',
            paper_bgcolor='#1F2937',
            width=None,
            height=350,
            margin=dict(l=50, r=50, t=100, b=50),
            showlegend=True,
            dragmode='pan',
            xaxis=dict(
                tickformat="%H:%M",
                tickfont=dict(size=10, color='#D1D5DB'),
                titlefont=dict(size=12, color='#D1D5DB'),
                showgrid=True,
                gridcolor='#374151',
                zeroline=False,
                fixedrange=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1H", step="hour", stepmode="backward"),
                        dict(count=6, label="6H", step="hour", stepmode="backward"),
                        dict(count=12, label="12H", step="hour", stepmode="backward"),
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    bgcolor='#1F2937',
                    bordercolor='#4B5563',
                    borderwidth=1,
                    font=dict(size=10, color='#D1D5DB'),
                    x=0.5,
                    y=1.1,
                    xanchor='center'
                ),
                rangeslider=dict(visible=True, bgcolor='#4B5563', thickness=0.1),
                type="date"
            ),
            yaxis=dict(
                tickfont=dict(size=10, color='#D1D5DB'),
                titlefont=dict(size=12, color='#D1D5DB'),
                showgrid=True,
                gridcolor='#374151',
                zeroline=False,
                fixedrange=False
            ),
            hovermode='x unified',
            modebar=dict(
                bgcolor='rgba(0,0,0,0)',
                color='#9CA3AF',
                activecolor='#9333EA',
                orientation='h',
                add=['zoomIn', 'zoomOut', 'autoScale', 'resetScale']
            )
        )
        return fig

    # Custom CSS to shrink modebar icons and ensure title visibility on mobile
    st.markdown("""
    <style>
        .modebar-container .modebar-group a {
            width: 16px !important;
            height: 16px !important;
            padding: 2px !important;
        }
        .modebar-container .modebar-group svg {
            width: 12px !important;
            height: 12px !important;
        }
        .modebar-container {
            padding: 2px !important;
            top: 60px !important;
        }
        .js-plotly-plot .plotly .main-svg {
            overflow: visible !important;
        }
        .rangeselector-btn {
            font-size: 12px !important;
            padding: 4px 8px !important;
        }
        @media (max-width: 768px) {
            .gtitle {
                font-size: 14px !important;
                transform: translateY(-10px) !important;
            }
            .modebar-container {
                top: 50px !important;
            }
            .js-plotly-plot {
                height: 300px !important;
            }
            .rangeselector-btn {
                font-size: 10px !important;
                padding: 3px 6px !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Fetch data and display the chart
    dates, flow_rates = fetch_thingspeak_data(channel_id=2796290, field=1, results=60)
    if dates and any(f is not None for f in flow_rates):
        fig = create_flow_chart(dates, flow_rates)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tidak ada data untuk ditampilkan.")

    # Close the chart container
    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced footer with animation, centered on both Android and desktop
    st.markdown("""
        <div class="footer-text" style="font-size: 0.9em; margin: 30px auto; display: flex; justify-content: center; align-items: center;">
            ✨ Sistem monitoring real-time • Update setiap 5 detik ✨
        </div>
    """, unsafe_allow_html=True)


def handle_sidebar_and_model_selection():
    with st.sidebar:
        # Sidebar header (tetap sama)
        st.markdown("""
        <div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, #A855F7, #7C3AED); border-radius: 10px; margin-bottom: 30px;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #EDE9FE; font-family: 'Inter', sans-serif;">🐟 VisionFish</div>
            <div style="font-size: 0.875rem; color: #D1D5DB; font-family: 'Inter', sans-serif;">Smart Fishery Assistant</div>
        </div>
        """, unsafe_allow_html=True)

        # User info (tetap sama)
        st.markdown("""
        <div class="info-card" style="margin-bottom: 30px; background: #1F2937; border: 1px solid rgba(147, 51, 234, 0.3); border-radius: 8px;">
            <div style="display: flex; align-items: center; gap: 12px; padding: 12px;">
                <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #A855F7, #7C3AED); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #EDE9FE; font-size: 1.2rem; box-shadow: 0 4px 10px rgba(147, 51, 234, 0.3);">👤</div>
                <div>
                    <div style="font-weight: 600; color: #EDE9FE; font-size: 1rem; font-family: 'Inter', sans-serif;">Premium</div>
                    <div style="font-size: 0.875rem; color: #D1D5DB; font-family: 'Inter', sans-serif;">Trial User</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick links (tetap sama)
        with st.expander("🔗 Tautan Cepat", expanded=False):
            st.markdown("""
            <style>
                .quick-links-container {
                    max-height: 200px;
                    overflow-y: auto;
                    padding-right: 5px;
                }
                .quick-links-container::-webkit-scrollbar {
                    width: 6px;
                }
                .quick-links-container::-webkit-scrollbar-track {
                    background: #374151;
                    border-radius: 10px;
                }
                .quick-links-container::-webkit-scrollbar-thumb {
                    background: #A855F7;
                    border-radius: 10px;
                }
                .quick-links-container::-webkit-scrollbar-thumb:hover {
                    background: #C084FC;
                }
                .quick-link:hover {
                    background: rgba(147, 51, 234, 0.1);
                    border-radius: 5px;
                    padding: 5px;
                    transition: background 0.2s ease;
                }
            </style>
            <nav class="quick-links-container" aria-label="Quick Links">
                <div style="display: flex; flex-direction: column; gap: 12px;">
                    <a href="https://syarifrizik.github.io/coming-soon/" target="_blank" class="quick-link" style="color: #A855F7; text-decoration: none; font-weight: 600; font-size: 1rem; font-family: 'Inter', sans-serif; display: flex; align-items: center; gap: 8px;" aria-label="Penggunaan">
                        <span style="font-size: 1.2rem;">📖</span> Penggunaan
                    </a>
                    <a href="https://drive.google.com/drive/u/1/folders/1afYnOjsv9y95UPDviKoZf0YTe5pHFzWl" target="_blank" class="quick-link" style="color: #A855F7; text-decoration: none; font-weight: 600; font-size: 1rem; font-family: 'Inter', sans-serif; display: flex; align-items: center; gap: 8px;" aria-label="Dokumentasi">
                        <span style="font-size: 1.2rem;">📚</span> Dokumentasi
                    </a>
                    <a href="https://wa.me/+62895619313339" target="_blank" class="quick-link" style="color: #A855F7; text-decoration: none; font-weight: 600; font-size: 1rem; font-family: 'Inter', sans-serif; display: flex; align-items: center; gap: 8px;" aria-label="Support">
                        <span style="font-size: 1.2rem;">📞</span> Support
                    </a>
                </div>
            </nav>
            """, unsafe_allow_html=True)

        # Logout button dengan reset penuh
        st.markdown("<div style='margin-top: 30 PXpx; margin-bottom: 30px;'>", unsafe_allow_html=True)
        if st.button("🔒 Logout", use_container_width=True, key="logout_btn"):
            # Reset semua state autentikasi
            st.session_state.clear()  # Hapus semua session state
            st.session_state.is_authenticated = False
            st.session_state.token_input_submitted = False
            st.session_state.is_processing = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Version info (tetap sama)
        st.markdown("""
        <div style="text-align: center; padding: 10px 0; background: #1F2937; border-top: 1px solid rgba(147, 51, 234, 0.2);">
            <div style="font-size: 0.875rem; color: #EDE9FE; font-weight: 600; font-family: 'Inter', sans-serif;">VisionFish v3.1</div>
            <div style="font-size: 0.75rem; color: #D1D5DB; font-family: 'Inter', sans-serif;">©2025 Copyright</div>
        </div>
        """, unsafe_allow_html=True)

# API KEY
    # Return placeholder values (tetap sama)
    model = "default_model"
    model_type = None
    model_params = {"model": model, "temperature": 0.3}
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    google_api_key = os.getenv("GOOGLE_API_KEY", "")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    return model, model_type, model_params, openai_api_key, google_api_key, anthropic_api_key

def handle_image_upload():
    # Bagian awal tetap sama
    st.markdown("""
    <div class='section-title'>
        <span>📷</span> Analisis Gambar Ikan
    </div>
    """, unsafe_allow_html=True)
    
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
            "", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="uploaded_img", label_visibility="collapsed"
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
        
        if img_input:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                try:
                    image = Image.open(img_input)
                    # Hapus use_container_width untuk kompatibilitas
                    st.image(image, caption="Gambar Ikan")
                except Exception as e:
                    st.error(f"Error membuka gambar: {str(e)}")
                    return None, None, None, None, None, None, None

            # API KEY
            with col2:
                openai_api_key = os.getenv("OPENAI_API_KEY", "")
                google_api_key = os.getenv("GOOGLE_API_KEY", "")
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
                
                st.markdown("""
                <div class="info-card" style="overflow: visible; z-index: 1;">
                    <div style="font-size: 1.1em; font-weight: 600; color: #A855F7; margin-bottom: 15px;">Instruksi Analisis</div>
                    <p style="margin-bottom: 15px;">Pilih jenis analisis yang ingin Anda lakukan pada gambar ini:</p>
                    <ul style="margin-left: 20px; margin-bottom: 15px;">
                        <li style="margin-bottom: 8px;">Analisis <b>Spesies</b> mengidentifikasi jenis ikan</li>
                        <li style="margin-bottom: 8px;">Analisis <b>Kesegaran</b> menentukan kualitas ikan</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div class='dashboard-title' style='text-align: left; margin-bottom: 15px;'>🤖 Model Vision Fish</div>", unsafe_allow_html=True)
                
                # Model name mapping tetap sama seperti sebelumnya
                model_name_mapping = {
                    "AquaDepth-Elite": "gpt-4o",  
                    "NeptuneFlow-Prime": "gemini-1.5-flash",  
                    "NeptuneWave-Regal": "gemini-1.5-pro", 
                }
                
                anthropic_models = ["ReefEye-Prime"]
                google_models = ["NeptuneFlow-Prime", "NeptuneWave-Regal"]
                openai_models = ["AquaDepth-Elite"]
                
                available_models = [] + (anthropic_models if anthropic_api_key else []) + (google_models if google_api_key else []) + (openai_models if openai_api_key else [])
                
                model = st.selectbox("Pilih model Vision Fish:", available_models, index=0)
                model_type = None
                
                if model in openai_models: 
                    model_type = "openai"
                elif model in google_models: 
                    model_type = "google"
                elif model in anthropic_models:
                    model_type = "anthropic"
                
                # Menghilangkan bagian parameter model
                st.markdown(f"<div style='margin-top: 10px; font-size: 0.9em; color: #555;'>Model yang dipilih: <b>{model}</b></div>", unsafe_allow_html=True)

                model_params = {
                    "model": model_name_mapping.get(model, model),
                    # "temperature": model_temp,  # Dihapus
                }

                # Definisi model_descriptions dengan kata-kata premium yang mudah dipahami
                model_descriptions = {
                    # Kelompok Aqua (OpenAI)
                    "AquaDepth-Elite": "Keunggulan luar biasa untuk analisis gambar dan teks yang mendalam",
                    # "AquaFlow-Max": "Kehebatan visual dan narasi yang memimpin dengan elegan",
                    # "AquaPulse-Lite": "Kejernihan sempurna untuk wawasan cepat dan berkelas",
                    # "AquaCore-Classic": "Klasik abadi dengan ketajaman visual dan intelektual",
                    # "AquaDepth-XL": "Kemewahan tak tertandingi untuk eksplorasi gambar dan teks",
                    
                    # Kelompok Neptune (Google)
                    "NeptuneFlow-Prime": "Kecepatan dan ketepatan unggul untuk penglihatan dan analisis",
                    "NeptuneWave-Regal": "Keagungan dalam penglihatan dan pemahaman yang presisi",
                    
                    # Kelompok Reef (Anthropic)
                    "ReefCore-Illustrious": "Kecemerlangan yang memadukan gambar dan narasi dengan sempurna"
                }


                # Pisahkan CSS ke string terpisah untuk menghindari konflik dengan f-string
                css = """
                <style>
                    .upgrade-badge {
                        background: #FFD700; /* Kuning */
                        color: #1F2937; /* Hitam */
                        padding: 4px 10px;
                        border-radius: 8px;
                        font-size: 0.65em;
                        font-weight: 600;
                        box-shadow: 0 2px 8px rgba(255, 215, 0, 0.3);
                        display: inline-flex;
                        align-items: center;
                        gap: 3px;
                        transition: all 0.3s ease;
                        position: relative; /* Pastikan pseudo-element relatif terhadap badge */
                    }
                    .upgrade-badge::before {
                        content: '';
                        position: absolute;
                        top: -2px;
                        left: -2px;
                        right: -2px;
                        bottom: -2px;
                        border: 1px solid transparent;
                        border-radius: 12px;
                        background: linear-gradient(45deg, #FFD700, #FFA500, #FFD700) border-box;
                        animation: rotateBorder 3s linear infinite;
                        z-index: -1;
                    }
                    .upgrade-badge:hover {
                        transform: scale(1.05);
                        box-shadow: 0 4px 12px rgba(255, 215, 0, 0.5);
                    }
                    .upgrade-badge a {
                        color: #1F2937;
                        text-decoration: none;
                        font-family: 'Inter', sans-serif;
                    }
                    .upgrade-badge a:hover {
                        text-decoration: none;
                    }
                    @keyframes rotateBorder {
                        0% { background-position: 0% 50%; }
                        100% { background-position: 400% 50%; }
                    }
                </style>
                """

                # Gabungkan CSS dengan HTML dalam f-string
                st.markdown(f"""
                {css}
                <div class="info-card" style="margin-top: 20px;">
                    <div style="font-size: 0.9em; color: #CBD5E0; margin-bottom: 8px;">Model yang digunakan:</div>
                    <div style="font-weight: 600; color: #A855F7; font-size: 1.1em;">{model}</div>
                    <div style="margin-top: 8px; font-size: 0.85em; color: #A0AEC0;">{model_descriptions.get(model, "Dioptimalkan untuk analisis ikan")}</div>
                </div>
                <div style="display: flex; justify-content: flex-end; margin-top: 10px;">
                    <div class="upgrade-badge">
                        <a href="https://wa.me/+62895619313339" target="_blank">Tambah Fitur Premium</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='margin: 25px 0 15px 0;'>", unsafe_allow_html=True)
                if st.button("🗑️ Mulai Chat Baru", use_container_width=True):
                    if "messages" in st.session_state:
                        st.session_state.pop("messages", None)
                    st.session_state.chat_reset = True
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            img_type = img_input.type
            img_base64 = get_image_base64(image)
            
            image_message = {
                "role": "user", 
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_type};base64,{img_base64}"}
                }]
            }
            
            return image_message, model, model_type, model_params, openai_api_key, google_api_key, anthropic_api_key
    
    return None, None, None, None, None, None, None



def fetch_weather_data(location):
    """
    Fetch weather data from OpenWeatherMap API for a given location.
    Returns None if the request fails for any reason, without raising an exception.
    """
    try:
        api_key = "e21eb6272044ad9e9600f8fdf7c2d8c1"  # Ganti dengan API key Anda
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric&lang=id"
        response = requests.get(url)
        # Periksa status kode secara manual tanpa melemparkan exception
        if response.status_code != 200:
            return None  # Kembalikan None jika status kode bukan 200 (misalnya, 404)
        data = response.json()
        # Pastikan data memiliki kunci yang diperlukan
        if "coord" not in data or "main" not in data or "wind" not in data or "weather" not in data:
            return None
        return {
            "location": data["name"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "description": data["weather"][0]["description"],
            "timestamp": datetime.fromtimestamp(data["dt"]).strftime("%d %B %Y, Pukul %H:%M:%S"),
            "coord": data["coord"]  # Pastikan koordinat disertakan
        }
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return None  # Kembalikan None tanpa melemparkan exception

def interpret_weather(weather_data):
    """Provide interpretation of weather data for fishery with pros and cons"""
    temp = weather_data["temperature"]
    humidity = weather_data["humidity"]
    wind_speed = weather_data["wind_speed"]
    description = weather_data["description"].lower()

    interpretation = {
        "Suhu": {"value": f"{temp}°C", "meaning": "", "pros": "", "cons": "", "score": 0, "icon": "🌡️"},
        "Kelembapan": {"value": f"{humidity}%", "meaning": "", "pros": "", "cons": "", "score": 0, "icon": "💧"},
        "Kecepatan Angin": {"value": f"{wind_speed} m/s", "meaning": "", "pros": "", "cons": "", "score": 0, "icon": "💨"},
        "Kondisi Hujan": {"value": description, "meaning": "", "pros": "", "cons": "", "score": 0, "icon": "☁️"}
    }

    if temp < 20:
        interpretation["Suhu"]["meaning"] = "Suhu dingin, mungkin memperlambat aktivitas ikan."
        interpretation["Suhu"]["pros"] = "Ikan tertentu seperti ikan air dingin (salmon) bisa lebih aktif."
        interpretation["Suhu"]["cons"] = "Metabolisme ikan tropis melambat, pertumbuhan terhambat."
        interpretation["Suhu"]["score"] = -1
    elif 20 <= temp <= 30:
        interpretation["Suhu"]["meaning"] = "Suhu ideal untuk sebagian besar ikan tropis."
        interpretation["Suhu"]["pros"] = "Mendukung pertumbuhan dan aktivitas ikan optimal."
        interpretation["Suhu"]["cons"] = "Jika terlalu lama panas, bisa meningkatkan risiko penyakit."
        interpretation["Suhu"]["score"] = 1
    else:
        interpretation["Suhu"]["meaning"] = "Suhu panas, bisa menyebabkan stres pada ikan."
        interpretation["Suhu"]["pros"] = "Meningkatkan metabolisme untuk ikan tertentu."
        interpretation["Suhu"]["cons"] = "Kadar oksigen turun, risiko kematian ikan meningkat."
        interpretation["Suhu"]["score"] = -1

    if humidity < 50:
        interpretation["Kelembapan"]["meaning"] = "Udara kering, bisa memengaruhi evaporasi air."
        interpretation["Kelembapan"]["pros"] = "Mengurangi risiko jamur di peralatan."
        interpretation["Kelembapan"]["cons"] = "Evaporasi cepat, perlu tambah air di kolam."
        interpretation["Kelembapan"]["score"] = -0.5
    elif 50 <= humidity <= 80:
        interpretation["Kelembapan"]["meaning"] = "Kelembapan normal, baik untuk lingkungan perikanan."
        interpretation["Kelembapan"]["pros"] = "Stabil untuk ekosistem air dan udara."
        interpretation["Kelembapan"]["cons"] = "Tidak ada dampak negatif signifikan."
        interpretation["Kelembapan"]["score"] = 0.5
    else:
        interpretation["Kelembapan"]["meaning"] = "Kelembapan tinggi, meningkatkan risiko kelembapan berlebih."
        interpretation["Kelembapan"]["pros"] = "Menjaga air tetap stabil di kolam."
        interpretation["Kelembapan"]["cons"] = "Risiko jamur dan bakteri meningkat."
        interpretation["Kelembapan"]["score"] = -0.5

    if wind_speed < 3:
        interpretation["Kecepatan Angin"]["meaning"] = "Angin lemah, kondisi tenang."
        interpretation["Kecepatan Angin"]["pros"] = "Aman untuk operasi perikanan di laut."
        interpretation["Kecepatan Angin"]["cons"] = "Kurang sirkulasi udara di kolam tertutup."
        interpretation["Kecepatan Angin"]["score"] = 0.5
    elif 3 <= wind_speed <= 10:
        interpretation["Kecepatan Angin"]["meaning"] = "Angin sedang, baik untuk sirkulasi air."
        interpretation["Kecepatan Angin"]["pros"] = "Meningkatkan oksigenasi air di kolam."
        interpretation["Kecepatan Angin"]["cons"] = "Bisa mengganggu jaring kecil."
        interpretation["Kecepatan Angin"]["score"] = 0.5
    else:
        interpretation["Kecepatan Angin"]["meaning"] = "Angin kencang, berpotensi berbahaya."
        interpretation["Kecepatan Angin"]["pros"] = "Sirkulasi air sangat baik."
        interpretation["Kecepatan Angin"]["cons"] = "Risiko kerusakan peralatan dan gelombang tinggi."
        interpretation["Kecepatan Angin"]["score"] = -1

    if "hujan" not in description.lower():
        interpretation["Kondisi Hujan"]["meaning"] = "Tidak ada hujan, kondisi lebih nyaman."
        interpretation["Kondisi Hujan"]["pros"] = "Visibilitas air baik, memancing lebih mudah."
        interpretation["Kondisi Hujan"]["cons"] = "Tidak ada dampak negatif signifikan."
        interpretation["Kondisi Hujan"]["score"] = 1
    else:
        interpretation["Kondisi Hujan"]["meaning"] = "Hujan, dapat memengaruhi visibilitas air."
        interpretation["Kondisi Hujan"]["pros"] = "Ikan mungkin lebih aktif di permukaan."
        interpretation["Kondisi Hujan"]["cons"] = "Air keruh, memancing lebih sulit."
        interpretation["Kondisi Hujan"]["score"] = -1

    return interpretation

def get_fishing_season_and_conditions(weather_data, interpretation):
    """Determine fishing season and conditions with integrated interpretation and user-friendly messages"""
    current_month = datetime.now().month

    if 4 <= current_month <= 10:
        season = "Musim Kering"
        season_desc = "Ini adalah musim kering, biasanya waktu yang baik untuk memancing karena cuaca lebih stabil dan ikan lebih aktif di perairan dangkal."
        season_score = 1
    else:
        season = "Musim Hujan"
        season_desc = "Ini adalah musim hujan, ikan mungkin lebih tersebar karena aliran air meningkat, tetapi memancing bisa lebih sulit karena kondisi cuaca."
        season_score = -0.5

    total_score = (
        season_score +
        interpretation["Suhu"]["score"] +
        interpretation["Kelembapan"]["score"] +
        interpretation["Kecepatan Angin"]["score"] +
        interpretation["Kondisi Hujan"]["score"]
    )

    if total_score >= 1:
        fishing_condition = "Baik"
        condition_summary = "Kondisi ini sangat mendukung untuk memancing. Peluang menemukan ikan cukup tinggi!"
        condition_icon = "✅"
    elif 0 <= total_score < 1:
        fishing_condition = "Cukup Baik"
        condition_summary = "Kondisi ini cukup mendukung, tetapi ada beberapa faktor yang perlu diperhatikan. Jika ikan sulit ditemukan, coba gunakan umpan yang lebih menarik atau pindah ke perairan yang lebih teduh."
        condition_icon = "⚠️"
    else:
        fishing_condition = "Kurang Baik"
        condition_summary = "Kondisi ini kurang ideal untuk memancing. Jika Anda kesulitan menemukan ikan, coba lagi di pagi atau sore hari saat suhu lebih sejuk, atau cari spot dengan arus air yang lebih stabil."
        condition_icon = "❌"

    condition_reasons = []
    for param, details in interpretation.items():
        condition_reasons.append({
            "param": param,
            "value": details["value"],
            "meaning": details["meaning"],
            "pros": details["pros"],
            "cons": details["cons"],
            "icon": details["icon"]
        })

    source_info = "Berdasarkan pola musiman tropis dan pengaruh cuaca terhadap perilaku ikan (Farmers' Almanac, Fishing Booker)."

    return {
        "season": season,
        "season_description": season_desc,
        "condition": fishing_condition,
        "condition_summary": condition_summary,
        "condition_icon": condition_icon,
        "condition_reasons": condition_reasons,
        "source": source_info
    }

def fetch_hourly_forecast(location):
    """
    Fetch hourly weather forecast (24 hours, 3-hour intervals) from OpenWeatherMap API.
    Returns None if the request fails for any reason.
    """
    # Periksa apakah weather_data tersedia di session state
    if 'weather_data' not in st.session_state or not st.session_state.weather_data:
        return None  # Kembalikan None tanpa pesan error

    weather_data = st.session_state.weather_data
    # Periksa apakah koordinat tersedia dalam weather_data
    if "coord" not in weather_data:
        return None  # Kembalikan None tanpa pesan error

    try:
        api_key = "e21eb6272044ad9e9600f8fdf7c2d8c1"  # Ganti dengan API key Anda
        lat = weather_data["coord"]["lat"]
        lon = weather_data["coord"]["lon"]
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=id"
        response = requests.get(url)
        # Periksa status kode secara manual tanpa melemparkan exception
        if response.status_code != 200:
            return None  # Kembalikan None jika status kode bukan 200
        data = response.json()
        # Ambil data untuk 24 jam ke depan (8 interval, 3 jam sekali)
        hourly_data = []
        wib_timezone = pytz.timezone("Asia/Jakarta")  # Zona waktu WIB
        for i in range(8):  # 8 interval untuk 24 jam (3 jam sekali)
            forecast = data["list"][i]
            # Konversi timestamp ke zona waktu WIB
            forecast_time = datetime.fromtimestamp(forecast["dt"], tz=pytz.UTC).astimezone(wib_timezone)
            hourly_data.append({
                "time": forecast_time.strftime("%H:%M"),  # Format waktu dalam WIB
                "temperature": round(forecast["main"]["temp"], 2),
                "icon": forecast["weather"][0]["icon"]
            })
        return hourly_data
    except (requests.exceptions.RequestException, ValueError, KeyError):
        return None  # Kembalikan None tanpa pesan error
    
def display_weather_info():
    """Display weather information with improved UI/UX"""
    st.markdown("""
    <div class='section-title'>
        <span>🌤️</span> Lihat Cuaca untuk Perikanan
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="custom-notification">
        <div style="font-weight: 600; margin-bottom: 5px;">ℹ️ Info Cuaca</div>
        <p>Lihat cuaca terkini berdasarkan lokasi untuk mendukung aktivitas perikanan Anda. Tekan Enter atau klik ikon untuk melihat hasil.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        .weather-search-container {
            display: flex;
            align-items: center;
            background: rgba(31, 41, 55, 0.8);
            border-radius: 12px;
            padding: 8px;
            border: 1px solid rgba(147, 51, 234, 0.3);
            transition: all 0.3s ease;
            max-width: 600px;
            margin: 0 auto;
        }
        .weather-search-container:hover {
            border-color: rgba(147, 51, 234, 0.6);
            box-shadow: 0 0 12px rgba(147, 51, 234, 0.3);
        }
        .weather-search-input {
            flex-grow: 1;
            border: none;
            background: transparent;
            color: #EDE9FE;
            padding: 10px;
            font-size: 1.1em;
            outline: none;
        }
        .weather-search-input::placeholder {
            color: #A78BFA;
            font-style: italic;
        }
        .weather-search-button {
            background: linear-gradient(135deg, #9333EA, #6B21A8);
            border: none;
            border-radius: 8px;
            padding: 10px 16px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .weather-search-button:hover {
            background: linear-gradient(135deg, #7E22CE, #5B1A8A);
            box-shadow: 0 4px 12px rgba(147, 51, 234, 0.5);
            transform: translateY(-2px);
        }
        .weather-card {
            background: linear-gradient(145deg, rgba(31, 41, 55, 0.9), rgba(45, 55, 72, 0.7));
            border-radius: 16px;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid rgba(147, 51, 234, 0.2);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            min-height: 100px;
        }
        .weather-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 25px rgba(147, 51, 234, 0.15);
        }
        .weather-icon {
            font-size: 1.2em;
            margin-right: 10px;
        }
        .condition-card {
            background: linear-gradient(145deg, rgba(31, 41, 55, 0.9), rgba(45, 55, 72, 0.7));
            border-radius: 16px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid rgba(147, 51, 234, 0.2);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .condition-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 25px rgba(147, 51, 234, 0.15);
        }
        .condition-icon {
            font-size: 1.5em;
            margin-right: 10px;
        }
        .condition-detail {
            margin: 10px 0;
            padding: 15px;
            background: rgba(147, 51, 234, 0.1);
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .condition-detail:hover {
            background: rgba(147, 51, 234, 0.15);
            transform: translateX(5px);
        }
        /* Box Container untuk Ramalan Memancing dan elemen di bawahnya */
        .forecast-box {
            background: rgba(31, 41, 55, 0.8);
            border-radius: 16px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(147, 51, 234, 0.2);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        }
        /* Styling untuk Expander */
        [data-testid="stExpander"] {
            background: transparent;
            border: none;
        }
        [data-testid="stExpanderToggleIcon"] {
            color: #A855F7;
        }
        [data-testid="stExpander"] > div:first-child {
            font-weight: 600;
            color: #A855F7;
            font-size: 1em;
        }
    </style>
    """, unsafe_allow_html=True)

    # Search bar untuk lokasi menggunakan st.form agar mendukung Enter
    # Search bar untuk lokasi menggunakan st.form agar mendukung Enter
    with st.form(key="weather_search_form", clear_on_submit=False):
        col_search, col_button = st.columns([5, 1])
        with col_search:
            location = st.text_input("", placeholder="Contoh: Jakarta, ID", key="weather_search_input", label_visibility="collapsed")
        with col_button:
            search_button = st.form_submit_button("🔍")

    # Proses pencarian saat tombol ditekan atau Enter ditekan
    if search_button:
        if not location or location.strip() == "":
            st.error("Gagal mengambil data cuaca. Pastikan format lokasi benar (City name, City ID). Lihat [panduan penulisan lokasi](https://openweathermap.org/current#cityid) untuk informasi lebih lanjut.")
            # Reset state jika pencarian gagal
            st.session_state.initial_search = False
            if 'weather_data' in st.session_state:
                del st.session_state.weather_data
            if 'hourly_forecast' in st.session_state:
                del st.session_state.hourly_forecast
        else:
            with st.spinner("Mengambil data cuaca..."):
                weather_data = fetch_weather_data(location)
                if weather_data:
                    # Gunakan waktu lokal saat ini dengan zona waktu WIB (UTC+7)
                    wib_timezone = pytz.timezone("Asia/Jakarta")  # Zona waktu WIB
                    current_time = datetime.now(wib_timezone).strftime("%d %B %Y, Pukul %H:%M:%S")
                    weather_data['timestamp'] = current_time  # Perbarui timestamp dengan waktu sekarang di WIB
                    st.session_state.weather_data = weather_data
                    st.session_state.initial_search = True
                    # Reset slider ke skor awal berdasarkan data API
                    interpretation = interpret_weather(weather_data)
                    fishing_info = get_fishing_season_and_conditions(weather_data, interpretation)
                    # Hitung skor awal berdasarkan data cuaca dari API
                    suhu = weather_data['temperature']
                    if suhu > 32:
                        suhu_score = 0  # Buruk
                    elif 28 <= suhu <= 32:
                        suhu_score = 0.5  # Cukup baik
                    elif 25 <= suhu < 28:
                        suhu_score = 1  # Bagus
                    else:
                        suhu_score = 0  # Buruk

                    kelembapan = weather_data['humidity']
                    if kelembapan > 85:
                        kelembapan_score = 0  # Buruk
                    elif 70 <= kelembapan <= 85:
                        kelembapan_score = 0.5  # Cukup baik
                    elif 55 <= kelembapan < 70:
                        kelembapan_score = 1  # Bagus
                    else:
                        kelembapan_score = 0  # Buruk

                    angin = weather_data['wind_speed']
                    if angin > 8:
                        angin_score = 0  # Buruk
                    elif 4 <= angin <= 8:
                        angin_score = 0.5  # Cukup baik
                    else:
                        angin_score = 1  # Bagus

                    kondisi = weather_data['description'].lower()
                    if "hujan" in kondisi or "rintik" in kondisi or "petir" in kondisi:
                        kondisi_score = 0  # Buruk
                    elif "cerah" in kondisi:
                        kondisi_score = 1  # Bagus
                    else:
                        kondisi_score = 0.5  # Cukup baik

                    total_score = suhu_score + kelembapan_score + angin_score + kondisi_score
                    normalized_score = (total_score / 4) * 5  # Skala 0-4 diubah ke 0-5

                    st.session_state.slider_key = random.randint(1, 1000000)  # Key acak untuk memaksa slider reset
                    # Reset hourly forecast data
                    if 'hourly_forecast' in st.session_state:
                        del st.session_state.hourly_forecast
                    st.session_state.slider_changed = False  # Reset status slider
                    st.session_state.last_slider_value = normalized_score  # Update nilai slider
                else:
                    st.error("Gagal mengambil data cuaca. Pastikan format lokasi benar (City name, City ID). Lihat [panduan penulisan lokasi](https://openweathermap.org/current#cityid) untuk informasi lebih lanjut.")
                    # Reset state jika pencarian gagal
                    st.session_state.initial_search = False
                    if 'weather_data' in st.session_state:
                        del st.session_state.weather_data
                    if 'hourly_forecast' in st.session_state:
                        del st.session_state.hourly_forecast
                        
    # Gunakan data cuaca dari session state jika tersedia
    if 'weather_data' in st.session_state and st.session_state.get('initial_search', False):
        weather_data = st.session_state.weather_data

        # Cuaca Terkini dengan kartu yang lebih kecil
        st.markdown("<div class='dashboard-title' style='font-size: 1.2em; margin-top: 20px;'>Cuaca Terkini</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="weather-card">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span class="weather-icon">📍</span>
                    <div style="font-size: 0.8em; color: #CBD5E0;">Lokasi</div>
                </div>
                <div style="font-size: 1em; font-weight: 600; color: #A855F7;">{weather_data['location']}</div>
                <div style="font-size: 0.7em; color: #A0AEC0; margin-top: 3px;">{weather_data['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="weather-card">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span class="weather-icon">🌡️</span>
                    <div style="font-size: 0.8em; color: #CBD5E0;">Suhu</div>
                </div>
                <div style="font-size: 1em; font-weight: 600; color: #10B981;">{weather_data['temperature']}°C</div>
                <div style="font-size: 0.7em; color: #A0AEC0; margin-top: 3px;">{weather_data['description']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="weather-card">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span class="weather-icon">💧</span>
                    <div style="font-size: 0.8em; color: #CBD5E0;">Kelembapan</div>
                </div>
                <div style="font-size: 1em; font-weight: 600; color: #3B82F6;">{weather_data['humidity']}%</div>
                <div style="font-size: 0.7em; color: #A0AEC0; margin-top: 3px;">Tekanan: {weather_data['pressure']} hPa</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="weather-card">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span class="weather-icon">💨</span>
                    <div style="font-size: 0.8em; color: #CBD5E0;">Kecepatan Angin</div>
                </div>
                <div style="font-size: 1em; font-weight: 600; color: #A855F7;">{weather_data['wind_speed']} m/s</div>
                <div style="font-size: 0.7em; color: #A0AEC0; margin-top: 3px;">
                    {"Angin kencang, dapat memengaruhi stabilitas perahu." if weather_data['wind_speed'] > 10 else "Angin sedang, cukup aman untuk memancing." if weather_data['wind_speed'] > 5 else "Angin lemah, kondisi ideal untuk memancing."}
                </div>
            </div>
            """, unsafe_allow_html=True)

        

        # Hourly Forecast (24 jam ke depan, interval 3 jam)
        # Hanya jalankan jika weather_data tersedia
        if 'weather_data' in st.session_state and st.session_state.get('initial_search', False):
            st.markdown("<div class='dashboard-title' style='font-size: 1.2em; margin-top: 20px;'>Ramalan Per 3 Jam (24 Jam)</div>", unsafe_allow_html=True)

            # Simpan data forecast di session state
            if 'hourly_forecast' not in st.session_state:
                st.session_state.hourly_forecast = fetch_hourly_forecast(location)

            hourly_forecast = st.session_state.hourly_forecast
            if hourly_forecast:
                # Gunakan kolom untuk menampilkan 4 kartu per baris
                for i in range(0, len(hourly_forecast), 4):
                    cols = st.columns(4)
                    for j in range(4):
                        if i + j < len(hourly_forecast):
                            hour = hourly_forecast[i + j]
                            # Tentukan ikon cuaca dan status berdasarkan kode ikon dari API
                            if hour['icon'] == '01d':
                                weather_icon = "☀️"  # Cerah (siang)
                                status = "Cerah"
                                status_color = "#10B981"  # Hijau untuk cerah
                            elif hour['icon'] == '01n':
                                weather_icon = "🌙"  # Cerah (malam)
                                status = "Cerah"
                                status_color = "#10B981"
                            elif hour['icon'] in ['02d', '03d', '04d']:
                                weather_icon = "⛅"  # Berawan (siang)
                                status = "Berawan"
                                status_color = "#FBBF24"  # Kuning untuk berawan
                            elif hour['icon'] in ['02n', '03n', '04n']:
                                weather_icon = "🌥️"  # Berawan (malam)
                                status = "Berawan"
                                status_color = "#FBBF24"
                            elif hour['icon'] in ['09d', '09n', '10d', '10n']:
                                weather_icon = "🌦️"  # Hujan rintik-rintik
                                status = "Hujan"
                                status_color = "#3B82F6"  # Biru untuk hujan
                            elif hour['icon'] in ['11d', '11n']:
                                weather_icon = "⛈️"  # Hujan petir
                                status = "Petir"
                                status_color = "#EF4444"  # Merah untuk petir
                            else:
                                weather_icon = "☁️"  # Default: berawan
                                status = "Berawan"
                                status_color = "#FBBF24"

                            with cols[j]:
                                st.markdown(f"""
                                <div class='weather-card'>
                                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                        <span class="weather-icon">⏰</span>
                                        <div style="font-size: 0.8em; color: #CBD5E0;">{hour['time']}</div>
                                    </div>
                                    <div style="font-size: 1em; font-weight: 600; color: #EDE9FE;">{weather_icon} {hour['temperature']}°C</div>
                                    <div style="font-size: 0.7em; color: {status_color}; margin-top: 3px;">{status}</div>
                                </div>
                                """, unsafe_allow_html=True)
        else:
            st.warning("Tidak dapat mengambil data Ramalan per 3 jam.")

        # Ramalan Memancing dengan box
        # Ramalan Memancing dengan box
        st.markdown("<div class='forecast-box'>", unsafe_allow_html=True)
        st.markdown("<div class='dashboard-title' style='font-size: 1.2em; margin-top: 0;'>Ramalan Memancing</div>", unsafe_allow_html=True)

        # Inisialisasi fishing_info dengan nilai default
        fishing_info = {"source": "Belum ada data cuaca."}  # Nilai default jika belum ada data

        # Hitung skor awal berdasarkan data cuaca dari API jika tersedia
        if 'weather_data' in st.session_state and st.session_state.get('initial_search', False):
            weather_data = st.session_state.weather_data  # Ambil data cuaca dari session state

            # Analisis Suhu
            suhu = weather_data['temperature']
            if suhu > 32:
                suhu_score = 0  # Buruk
            elif 28 <= suhu <= 32:
                suhu_score = 0.5  # Cukup baik
            elif 25 <= suhu < 28:
                suhu_score = 1  # Bagus
            else:
                suhu_score = 0  # Buruk

            # Analisis Kelembapan
            kelembapan = weather_data['humidity']
            if kelembapan > 85:
                kelembapan_score = 0  # Buruk
            elif 70 <= kelembapan <= 85:
                kelembapan_score = 0.5  # Cukup baik
            elif 55 <= kelembapan < 70:
                kelembapan_score = 1  # Bagus
            else:
                kelembapan_score = 0  # Buruk

            # Analisis Kecepatan Angin
            angin = weather_data['wind_speed']
            if angin > 8:
                angin_score = 0  # Buruk
            elif 4 <= angin <= 8:
                angin_score = 0.5  # Cukup baik
            else:
                angin_score = 1  # Bagus

            # Analisis Kondisi Hujan
            kondisi = weather_data['description'].lower()
            if "hujan" in kondisi or "rintik" in kondisi or "petir" in kondisi:
                kondisi_score = 0  # Buruk
            elif "cerah" in kondisi:
                kondisi_score = 1  # Bagus
            else:
                kondisi_score = 0.5  # Cukup baik

            # Hitung total skor dan normalisasi ke skala 0-5
            total_score = suhu_score + kelembapan_score + angin_score + kondisi_score
            normalized_score = (total_score / 4) * 5  # Skala 0-4 diubah ke 0-5

            # Inisialisasi fishing_info dengan data API
            interpretation = interpret_weather(weather_data)
            fishing_info = get_fishing_season_and_conditions(weather_data, interpretation)
        else:
            # Nilai default jika belum ada data cuaca
            normalized_score = 0.0

        # Reset status slider agar nilai awal selalu mengikuti data API terbaru
        st.session_state.slider_changed = False
        st.session_state.last_slider_value = normalized_score

        # Slider untuk simulasi skor
        simulated_score = st.slider("Simulasi Skor Ramalan", min_value=0.0, max_value=5.0, value=normalized_score, step=0.1, key=f"simulated_score_slider_{st.session_state.get('slider_key', 0)}")
        st.markdown("<div style='font-size: 0.7em; color: #A0AEC0; text-align: center; margin: 5px 0;'><i>Geser untuk mensimulasikan kondisi cuaca (Sumber: Farmers' Almanac, Fishing Booker). Tekan 🔍 pada bar pencarian untuk kembali ke posisi lokasi semula.</i></div>", unsafe_allow_html=True)

        # Tentukan apakah slider telah digeser oleh pengguna
        if 'last_slider_value' not in st.session_state:
            st.session_state.last_slider_value = normalized_score
        if abs(simulated_score - st.session_state.last_slider_value) > 0.1:  # Deteksi perubahan slider
            st.session_state.slider_changed = True
        st.session_state.last_slider_value = simulated_score

        # Gunakan data API jika slider belum digeser, gunakan data simulasi jika slider digeser
        if not st.session_state.get('slider_changed', False):
            # Gunakan data API asli jika tersedia
            if 'weather_data' in st.session_state and st.session_state.get('initial_search', False):
                current_data = weather_data
                current_fishing_info = fishing_info
                current_score = normalized_score
            else:
                # Nilai default jika belum ada data cuaca
                current_data = {
                    "temperature": 0.0,
                    "humidity": 0,
                    "wind_speed": 0.0,
                    "description": "Tidak ada data"
                }
                current_fishing_info = fishing_info
                current_score = 0.0
        else:
            # Simulasi data acak berdasarkan skor (0: buruk, 5: optimal)
            if simulated_score <= 1:  # Kondisi buruk
                simulated_temp = random.uniform(35, 40)  # Suhu sangat panas
                simulated_humidity = random.uniform(85, 95)  # Kelembapan sangat tinggi
                simulated_wind_speed = random.uniform(10, 20)  # Angin kencang
                simulated_weather_desc = "Hujan Lebat"
            elif simulated_score <= 2:  # Kondisi agak buruk
                simulated_temp = random.uniform(32, 35)  # Suhu panas
                simulated_humidity = random.uniform(75, 85)  # Kelembapan tinggi
                simulated_wind_speed = random.uniform(7, 10)  # Angin agak kencang
                simulated_weather_desc = "Rintik-Rintik"
            elif simulated_score <= 3:  # Kondisi sedang
                simulated_temp = random.uniform(30, 32)  # Suhu agak panas
                simulated_humidity = random.uniform(65, 75)  # Kelembapan normal
                simulated_wind_speed = random.uniform(4, 7)  # Angin sedang
                simulated_weather_desc = "Berawan"
            else:  # Kondisi baik
                simulated_temp = random.uniform(25, 30)  # Suhu ideal
                simulated_humidity = random.uniform(55, 65)  # Kelembapan optimal
                simulated_wind_speed = random.uniform(1, 4)  # Angin lemah
                simulated_weather_desc = "Cerah"

            # Simulasi data
            simulated_data = {
                "temperature": round(simulated_temp, 2),
                "humidity": round(simulated_humidity),
                "wind_speed": round(simulated_wind_speed, 1),
                "description": simulated_weather_desc
            }
            current_data = simulated_data
            current_fishing_info = get_fishing_season_and_conditions(simulated_data, interpret_weather(simulated_data))
            current_score = simulated_score

        # Tentukan kondisi cuaca dan memancing berdasarkan current_score (yang mencerminkan simulated_score)
        current_hour = datetime.now().hour
        is_daytime = 6 <= current_hour < 18  # Pagi hingga sore (06:00-18:00)

        # Sesuaikan kondisi cuaca berdasarkan waktu dan nilai slider (current_score)
        if is_daytime:
            if current_score <= 1:  # Buruk
                current_weather_desc = "Panas dan Hujan"
                current_weather_icon = "⛈️"
                weather_desc_text = "Cuaca buruk, memancing sangat sulit karena panas dan hujan."
            elif current_score <= 2:  # Agak buruk
                current_weather_desc = "Panas"
                current_weather_icon = "☀️"
                weather_desc_text = "Cuaca panas, memancing mungkin lebih sulit karena ikan cenderung mencari tempat teduh."
            elif current_score <= 3:  # Sedang
                current_weather_desc = "Berawan"
                current_weather_icon = "⛅"
                weather_desc_text = "Cuaca berawan, cukup baik untuk memancing."
            else:  # Bagus
                current_weather_desc = "Cerah"
                current_weather_icon = "☀️"
                weather_desc_text = "Cuaca cerah, sangat mendukung untuk memancing."
        else:  # Malam
            if current_score <= 1:  # Buruk
                current_weather_desc = "Hujan dan Dingin"
                current_weather_icon = "⛈️"
                weather_desc_text = "Cuaca buruk, memancing sangat sulit karena hujan dan dingin."
            elif current_score <= 2:  # Agak buruk
                current_weather_desc = "Hangat"
                current_weather_icon = "🌙"  # Ikon bulan untuk malam
                weather_desc_text = "Cuaca hangat di malam hari, memancing masih memungkinkan tetapi ikan mungkin kurang aktif."
            elif current_score <= 3:  # Sedang
                current_weather_desc = "Berawan"
                current_weather_icon = "☁️"  # Ikon berawan untuk malam
                weather_desc_text = "Cuaca berawan di malam hari, cukup baik untuk memancing."
            else:  # Bagus
                current_weather_desc = "Cerah"
                current_weather_icon = "🌙"  # Ikon bulan untuk malam
                weather_desc_text = "Cuaca cerah di malam hari, sangat mendukung untuk memancing."

        current_condition = "Baik" if current_score >= 3 else "Cukup Baik" if current_score >= 1 else "Kurang Baik"
        current_color = "#10B981" if current_score >= 3 else "#FBBF24" if current_score >= 1 else "#EF4444"
        current_icon = "✅" if current_score >= 3 else "⚠️" if current_score >= 1 else "❌"

        # Tampilan Ramalan Memancing
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
            <div class="weather-card" style="padding: 10px; min-height: 100px;">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span class="weather-icon" style="font-size: 1.2em;">{current_weather_icon}</span>
                    <div style="font-size: 0.8em; color: #CBD5E0;">Kondisi Cuaca</div>
                </div>
                <div style="font-size: 1em; font-weight: 600; color: #A855F7;">{current_weather_desc}</div>
                <div style="font-size: 0.7em; color: #A0AEC0; margin-top: 3px;">{weather_desc_text}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="weather-card" style="padding: 10px; min-height: 100px;">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span class="weather-icon" style="font-size: 1.2em;">{current_icon}</span>
                    <div style="font-size: 0.8em; color: #CBD5E0;">Kondisi Memancing</div>
                </div>
                <div style="font-size: 1em; font-weight: 600; color: {current_color};">{current_condition}</div>
                <div style="font-size: 0.7em; color: #A0AEC0; margin-top: 3px;">Skor: {current_score:.1f} dari 5</div>
            </div>
            """, unsafe_allow_html=True)

        # Detail Analisis dengan Expander
        with st.expander("Detail Analisis"):
            # Analisis Suhu
            suhu = current_data['temperature']
            if suhu > 32:
                suhu_desc = "Suhu terlalu panas, kondisi ini buruk untuk sebagian besar ikan."
                suhu_pros = "Meningkatkan metabolisme ikan dalam jangka pendek."
                suhu_cons = "Meningkatkan risiko stres panas, penyakit, dan kematian ikan."
                suhu_score = 0  # Buruk
            elif 28 <= suhu <= 32:
                suhu_desc = "Suhu agak panas, kondisi ini cukup baik untuk ikan tropis."
                suhu_pros = "Mendukung aktivitas ikan, tetapi perlu pemantauan."
                suhu_cons = "Dapat menyebabkan stres jika berlangsung lama."
                suhu_score = 0.5  # Cukup baik
            elif 25 <= suhu < 28:
                suhu_desc = "Suhu ideal untuk sebagian besar ikan tropis."
                suhu_pros = "Mendukung pertumbuhan dan aktivitas ikan optimal."
                suhu_cons = "Tidak ada dampak negatif signifikan."
                suhu_score = 1  # Bagus
            else:
                suhu_desc = "Suhu terlalu dingin, kondisi ini kurang optimal untuk ikan tropis."
                suhu_pros = "Mengurangi risiko penyakit akibat panas."
                suhu_cons = "Memperlambat metabolisme dan aktivitas ikan."
                suhu_score = 0  # Buruk

            # Analisis Kelembapan
            kelembapan = current_data['humidity']
            if kelembapan > 85:
                kelembapan_desc = "Kelembapan terlalu tinggi, kondisi ini buruk untuk perikanan."
                kelembapan_pros = "Mendukung lingkungan lembap untuk ikan tertentu."
                kelembapan_cons = "Meningkatkan risiko jamur dan penyakit di kolam."
                kelembapan_score = 0  # Buruk
            elif 70 <= kelembapan <= 85:
                kelembapan_desc = "Kelembapan agak tinggi, kondisi ini cukup baik untuk perikanan."
                kelembapan_pros = "Mendukung ekosistem air yang stabil."
                kelembapan_cons = "Dapat sedikit meningkatkan risiko jamur."
                kelembapan_score = 0.5  # Cukup baik
            elif 55 <= kelembapan < 70:
                kelembapan_desc = "Kelembapan ideal, kondisi ini sangat baik untuk perikanan."
                kelembapan_pros = "Stabil untuk ekosistem air dan udara."
                kelembapan_cons = "Tidak ada dampak negatif signifikan."
                kelembapan_score = 1  # Bagus
            else:
                kelembapan_desc = "Kelembapan terlalu rendah, kondisi ini kurang optimal."
                kelembapan_pros = "Mengurangi risiko jamur di lingkungan."
                kelembapan_cons = "Dapat menyebabkan kekeringan pada permukaan air."
                kelembapan_score = 0  # Buruk

            # Analisis Kecepatan Angin
            angin = current_data['wind_speed']
            if angin > 8:
                angin_desc = "Angin kencang, kondisi ini buruk untuk perikanan."
                angin_pros = "Meningkatkan sirkulasi udara di kolam terbuka."
                angin_cons = "Berisiko untuk stabilitas perahu dan operasi laut."
                angin_score = 0  # Buruk
            elif 4 <= angin <= 8:
                angin_desc = "Angin sedang, kondisi ini cukup aman untuk memancing."
                angin_pros = "Mendukung sirkulasi udara tanpa risiko besar."
                angin_cons = "Dapat sedikit mengganggu stabilitas perahu kecil."
                angin_score = 0.5  # Cukup baik
            else:
                angin_desc = "Angin lemah, kondisi ini ideal untuk perikanan."
                angin_pros = "Aman untuk operasi perikanan di laut."
                angin_cons = "Tidak ada dampak negatif signifikan."
                angin_score = 1  # Bagus

            # Analisis Kondisi Hujan
            kondisi = current_data['description'].lower()
            if "hujan" in kondisi or "rintik" in kondisi or "petir" in kondisi:
                kondisi_desc = "Hujan, kondisi ini buruk untuk memancing."
                kondisi_pros = "Meningkatkan kadar oksigen di air."
                kondisi_cons = "Visibilitas air buruk, memancing lebih sulit."
                kondisi_score = 0  # Buruk
            elif "cerah" in kondisi:
                kondisi_desc = "Cerah, kondisi ini ideal untuk memancing."
                kondisi_pros = "Visibilitas air baik, memancing lebih mudah."
                kondisi_cons = "Tidak ada dampak negatif signifikan."
                kondisi_score = 1  # Bagus
            else:
                kondisi_desc = "Berawan, kondisi ini cukup baik untuk memancing."
                kondisi_pros = "Suhu lebih stabil, ikan lebih aktif."
                kondisi_cons = "Cahaya matahari berkurang, mungkin memengaruhi visibilitas."
                kondisi_score = 0.5  # Cukup baik

            # Tampilkan data dengan analisis
            st.markdown(f"""
            <div class="weather-card" style="padding: 10px; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span class="weather-icon" style="font-size: 1.2em;">🌡️</span>
                    <div style="font-size: 0.8em; color: #CBD5E0;">Suhu</div>
                </div>
                <div style="font-size: 1em; font-weight: 600; color: #10B981;">{suhu}°C</div>
                <div style="font-size: 0.7em; color: #A0AEC0; margin-top: 3px;">
                    {suhu_desc}<br>
                    <strong>Kelebihan:</strong> {suhu_pros}<br>
                    <strong>Kekurangan:</strong> {suhu_cons}
                </div>
            </div>
            <div class="weather-card" style="padding: 10px; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span class="weather-icon" style="font-size: 1.2em;">💧</span>
                    <div style="font-size: 0.8em; color: #CBD5E0;">Kelembapan</div>
                </div>
                <div style="font-size: 1em; font-weight: 600; color: #3B82F6;">{kelembapan}%</div>
                <div style="font-size: 0.7em; color: #A0AEC0; margin-top: 3px;">
                    {kelembapan_desc}<br>
                    <strong>Kelebihan:</strong> {kelembapan_pros}<br>
                    <strong>Kekurangan:</strong> {kelembapan_cons}
                </div>
            </div>
            <div class="weather-card" style="padding: 10px; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span class="weather-icon" style="font-size: 1.2em;">💨</span>
                    <div style="font-size: 0.8em; color: #CBD5E0;">Kecepatan Angin</div>
                </div>
                <div style="font-size: 1em; font-weight: 600; color: #A855F7;">{angin} m/s</div>
                <div style="font-size: 0.7em; color: #A0AEC0; margin-top: 3px;">
                    {angin_desc}<br>
                    <strong>Kelebihan:</strong> {angin_pros}<br>
                    <strong>Kekurangan:</strong> {angin_cons}
                </div>
            </div>
            <div class="weather-card" style="padding: 10px; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <span class="weather-icon" style="font-size: 1.2em;">☁️</span>
                    <div style="font-size: 0.8em; color: #CBD5E0;">Kondisi Hujan</div>
                </div>
                <div style="font-size: 1em; font-weight: 600; color: #EDE9FE;">{current_data['description']}</div>
                <div style="font-size: 0.7em; color: #A0AEC0; margin-top: 3px;">
                    {kondisi_desc}<br>
                    <strong>Kelebihan:</strong> {kondisi_pros}<br>
                    <strong>Kekurangan:</strong> {kondisi_cons}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Tutup box
        st.markdown("</div>", unsafe_allow_html=True)

def handle_csv_upload_and_analysis():
    """Handle CSV upload and data analysis with improved UI"""
    # Handle CSV upload and data analysis with improved UI
    st.markdown("""
    <div class='section-title'>
        <span>📊</span> Analisis Data Kualitas Ikan
    </div>
    """, unsafe_allow_html=True)
    
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
            <a href="https://wa.me/+62895619313339" style="color: #A855F7; text-decoration: none; font-weight: 500; display: inline-block; margin-top: 10px;">
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

def process_image_analysis(image_message, prompt, model_type, model_params, openai_api_key, google_api_key, anthropic_api_key):
    """Process image analysis using provided model info"""
    if image_message:
        with st.spinner("Menganalisis gambar..."):
            # Map model type to API key
            model2key = {
                "openai": openai_api_key,
                "google": google_api_key,
                "anthropic": anthropic_api_key,
            }
            
            # Initialize session state messages if not present
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Clear previous messages
            st.session_state.messages = []
            
            # Combine image_message with prompt as a single user message
            combined_message = {
                "role": "user",
                "content": [
                    image_message["content"][0],  # The image_url part
                    {"type": "text", "text": prompt}
                ]
            }
            
            st.session_state.messages.append(combined_message)
            
            # Call stream_llm_response
            response = stream_llm_response(
                model_params=model_params,
                model_type=model_type,
                api_key=model2key[model_type]
            )
            
            with st.chat_message("assistant"):
                st.write_stream(response)
def main():
    """Main application function"""
    # Setup page dan inisialisasi
    setup_page_config()
    apply_custom_css()
    initialize_session_state()

    # Tampilkan layar autentikasi jika belum login
    if not st.session_state.get('is_authenticated', False):
        render_auth_screen()
    else:
        # User sudah autentikasi
        model, model_type, model_params, openai_api_key, google_api_key, anthropic_api_key = handle_sidebar_and_model_selection()
        
        api_keys = {
            "openai": openai_api_key,
            "google": google_api_key,
            "anthropic": anthropic_api_key,
        }
        
        render_dashboard()
        
        tabs = st.tabs(["💬 Assistant", "📸 Image Analysis", "📊 Data Analysis", "🌤️ Weather Info"])
        
        with tabs[0]:
            prompt = render_prompt_buttons()
            render_visionfish_chat()
            user_input = None
            
            if prompt:
                process_prompt_response(prompt, model_type, model_params, api_keys)
                st.rerun()
            
            if user_input:
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                    
                st.session_state.messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": user_input}]
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
                
                st.rerun()

        with tabs[1]:
            result = handle_image_upload()
            if result[0] is not None:
                image_message, model, model_type, model_params, openai_api_key, google_api_key, anthropic_api_key = result
                
                st.markdown("<div class='dashboard-title' style='margin: 20px 0;'>Pilih Jenis Analisis</div>", unsafe_allow_html=True)
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    if st.button("🔍 Analisis Spesies", use_container_width=True):
                        process_image_analysis(image_message, get_analisis_ikan_prompt(model), model_type, model_params, openai_api_key, google_api_key, anthropic_api_key)
                
                with analysis_col2:
                    if st.button("🌟 Analisis Kesegaran", use_container_width=True):
                        process_image_analysis(image_message, get_freshness_prompt(model), model_type, model_params, openai_api_key, google_api_key, anthropic_api_key)
        
        with tabs[2]:
            handle_csv_upload_and_analysis()

        with tabs[3]:
                display_weather_info()
if __name__ == "__main__":
    main()
