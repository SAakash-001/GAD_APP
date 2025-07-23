# model.py

import pickle
import numpy as np
import librosa
import os
import streamlit as st
import requests
from tqdm import tqdm

# --- Configuration: Model Hosting and Local Paths ---
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# IMPORTANT: Replace these placeholder URLs with your actual direct download links.
# If using Google Drive, use a direct link generator like https://www.gilthonwe.com/files/google-drive-direct-link-generator
MODEL_URLS = {
    "gender_model.pkl": "YOUR_DIRECT_DOWNLOAD_LINK_FOR_GENDER_MODEL",
    "age_model.pkl": "YOUR_DIRECT_DOWNLOAD_LINK_FOR_AGE_MODEL",
    "age_label_encoder.pkl": "YOUR_DIRECT_DOWNLOAD_LINK_FOR_AGE_ENCODER"
}

# --- Helper Function to Download Models ---
def download_file(url, save_path):
    """Downloads a file from a URL, showing a progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Use st.progress for a Streamlit-native progress bar
        progress_bar = st.progress(0)
        progress_status = st.empty()
        
        bytes_downloaded = 0
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bytes_downloaded += len(chunk)
                # Update progress bar
                percent_complete = int((bytes_downloaded / total_size) * 100)
                progress_bar.progress(percent_complete)
                progress_status.text(f"Downloading {os.path.basename(save_path)}: {percent_complete}%")
        
        progress_bar.empty()
        progress_status.empty()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download {os.path.basename(save_path)}. Error: {e}")
        return False

# --- Model Loading with Caching and On-Demand Download ---
@st.cache_resource
def load_all_artifacts():
    """
    Checks for models, downloads if missing, and loads them.
    The @st.cache_resource decorator ensures this function runs only once.
    """
    artifacts = {}
    all_files_present = True

    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(SAVE_DIR, filename)
        if not os.path.exists(filepath):
            st.warning(f"Model file '{filename}' not found. Downloading...")
            if not download_file(url, filepath):
                st.error(f"Could not download {filename}. The app cannot make predictions.")
                all_files_present = False
    
    if not all_files_present:
        return None

    try:
        with open(os.path.join(SAVE_DIR, 'gender_model.pkl'), 'rb') as f:
            artifacts['gender_model'] = pickle.load(f)
        with open(os.path.join(SAVE_DIR, 'age_model.pkl'), 'rb') as f:
            artifacts['age_model'] = pickle.load(f)
        with open(os.path.join(SAVE_DIR, 'age_label_encoder.pkl'), 'rb') as f:
            artifacts['age_label_encoder'] = pickle.load(f)
        st.success("✅ Models loaded successfully!")
        return artifacts
    except Exception as e:
        st.error(f"An error occurred while loading model files: {e}")
        return None

# Load the models ONCE when the app starts.
loaded_artifacts = load_all_artifacts()


# --- Feature Extraction (This must match the training script) ---
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mean_mfcc = np.mean(mfcc, axis=1)
        std_mfcc = np.std(mfcc, axis=1)
        feature_vector = np.hstack((mean_mfcc, std_mfcc))
        return feature_vector
    except Exception as e:
        st.error(f"Could not process audio file: {e}")
        return None


# --- Prediction Function ---
def predict(audio_path):
    """Predicts gender and age group from an audio file."""
    if loaded_artifacts is None:
        return "Error: Models are not loaded.", "Please check application logs."
    
    if not os.path.exists(audio_path):
        return "Error: Input audio file not found.", None

    features = extract_features(audio_path)
    if features is None:
        return "Error: Could not extract features from audio.", None
        
    feat = features.reshape(1, -1)
    
    try:
        # Get models from the loaded artifacts dictionary
        gender_clf = loaded_artifacts['gender_model']
        age_clf = loaded_artifacts['age_model']
        age_le = loaded_artifacts['age_label_encoder']

        # Predict gender
        gender_pred = gender_clf.predict(feat)[0]
        gender = 'Male' if gender_pred == 1 else 'Female'
        
        # Predict age group
        age_pred_encoded = age_clf.predict(feat)[0]
        age_group = age_le.inverse_transform([age_pred_encoded])[0]
        
        age_group_display = age_group.capitalize()

        return gender, age_group_display
    except Exception as e:
        return f"Error during prediction: {str(e)}", None
