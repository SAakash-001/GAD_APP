#https://drive.google.com/drive/folders/1y8WXczClP9TJg0pt-n0lgQW_-jeWm9Y7?usp=sharing

# model_loader.py

import streamlit as st
import os
import requests
import pickle
from tqdm import tqdm

# --- Model and Encoder URLs ---
# IMPORTANT: Replace these with your actual direct download links.
# For Google Drive, you need to convert the sharing link to a direct download link.
# A useful tool for this: https://www.gilthonwe.com/files/google-drive-direct-link-generator
MODEL_URLS = {
    "gender_model.pkl": "https://drive.google.com/u/0/uc?id=1tmCjAANLNpVE2axNpHPo6IjF8Y42MPP2&export=download",
    "age_model.pkl": "https://drive.google.com/u/0/uc?id=18YQNrUPGOEDvV3FXAOoqtFD_ujiyrXtb&export=download",
    "age_label_encoder.pkl": "https://drive.google.com/u/0/uc?id=10RlIXKbTQY2aUXHQrqv9DTqNjK2xXXPR&export=download"
}

# --- Download and Caching Logic ---
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_file(url, save_path):
    """Downloads a file from a URL and saves it, showing a progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"Downloaded {os.path.basename(save_path)} successfully.")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading {os.path.basename(save_path)}: {e}")
        return False

@st.cache_resource
def load_models_and_encoders():
    """
    Checks for models, downloads them if missing, and loads them into memory.
    Using @st.cache_resource ensures this heavy operation runs only once.
    """
    models = {}
    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(SAVE_DIR, filename)
        if not os.path.exists(filepath):
            st.info(f"Downloading {filename}... (this may take a moment)")
            if not download_file(url, filepath):
                st.error(f"Failed to download required model file: {filename}. The app cannot continue.")
                return None # Stop if a model fails to download
    
    # Load the models from the local files
    try:
        with open(os.path.join(SAVE_DIR, 'gender_model.pkl'), 'rb') as f:
            models['gender_model'] = pickle.load(f)
        with open(os.path.join(SAVE_DIR, 'age_model.pkl'), 'rb') as f:
            models['age_model'] = pickle.load(f)
        with open(os.path.join(SAVE_DIR, 'age_label_encoder.pkl'), 'rb') as f:
            models['age_label_encoder'] = pickle.load(f)
        
        st.success("Models loaded successfully!")
        return models
    except FileNotFoundError as e:
        st.error(f"Could not load models. File not found: {e}. Please check paths.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        return None
