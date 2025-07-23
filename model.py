# model.py

import pickle
import numpy as np
import librosa
import os
import streamlit as st
import requests

# --- Configuration: Model Hosting and Local Paths ---
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Your Google Drive download links ---
MODEL_URLS = {
    "gender_model.pkl": "https://drive.google.com/u/0/uc?id=1tmCjAANLNpVE2axNpHPo6IjF8Y42MPP2&export=download",
    "age_model.pkl": "https://drive.google.com/u/0/uc?id=18YQNrUPGOEDvV3FXAOoqtFD_ujiyrXtb&export=download",
    "age_label_encoder.pkl": "https://drive.google.com/u/0/uc?id=10RlIXKbTQY2aUXHQrqv9DTqNjK2xXXPR&export=download"
}

# --- NEW: Robust Helper Function to Download Models from Google Drive ---
def download_file_from_google_drive(id, destination):
    """
    Downloads a large file from Google Drive, handling the virus scan warning.
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Use st.progress for a Streamlit-native progress bar
    progress_bar = st.progress(0)
    progress_status = st.empty()
    total_size = int(response.headers.get('content-length', 0))
    bytes_downloaded = 0
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                bytes_downloaded += len(chunk)
                if total_size > 0:
                    percent_complete = int((bytes_downloaded / total_size) * 100)
                    progress_bar.progress(percent_complete)
                    progress_status.text(f"Downloading {os.path.basename(destination)}: {percent_complete}%")

    progress_bar.empty()
    progress_status.empty()

# --- Model Loading with Caching and On-Demand Download ---
@st.cache_resource
def load_all_artifacts():
    """
    Checks for models, downloads if missing, and loads them.
    The @st.cache_resource decorator ensures this function runs only once.
    """
    artifacts = {}
    for filename, url in MODEL_URLS.items():
        filepath = os.path.join(SAVE_DIR, filename)
        if not os.path.exists(filepath):
            st.warning(f"Model file '{filename}' not found. Downloading...")
            try:
                # Extract the file ID from the URL
                file_id = url.split('id=')[1].split('&')[0]
                download_file_from_google_drive(file_id, filepath)
                st.info(f"'{filename}' downloaded successfully.")
            except Exception as e:
                st.error(f"Failed to download {filename}. Error: {e}")
                return None  # Stop if any file fails to download

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
        # The error message for the user is now more specific
        return "Error: Models are not loaded.", "Please check the app logs for download/loading errors."
    
    if not os.path.exists(audio_path):
        return "Error: Input audio file not found.", None

    features = extract_features(audio_path)
    if features is None:
        return "Error: Could not extract features from audio.", None
        
    feat = features.reshape(1, -1)
    
    try:
        gender_clf = loaded_artifacts['gender_model']
        age_clf = loaded_artifacts['age_model']
        age_le = loaded_artifacts['age_label_encoder']

        gender_pred = gender_clf.predict(feat)[0]
        gender = 'Male' if gender_pred == 1 else 'Female'
        
        age_pred_encoded = age_clf.predict(feat)[0]
        age_group = age_le.inverse_transform([age_pred_encoded])[0]
        
        age_group_display = age_group.capitalize()
        return gender, age_group_display
    except Exception as e:
        return f"Error during prediction: {str(e)}", None

