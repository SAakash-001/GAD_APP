# model.py

import pickle
import numpy as np
import librosa
import os
import streamlit as st

# --- Model Loading with Caching ---
@st.cache_resource
def load_all_artifacts():
    """
    Loads all model artifacts from the local 'saved_models' directory.
    The @st.cache_resource decorator ensures this function runs only once.
    """
    try:
        artifacts = {}
        with open(os.path.join('saved_models', 'gender_model.pkl'), 'rb') as f:
            artifacts['gender_model'] = pickle.load(f)
        with open(os.path.join('saved_models', 'age_model.pkl'), 'rb') as f:
            artifacts['age_model'] = pickle.load(f)
        with open(os.path.join('saved_models', 'age_label_encoder.pkl'), 'rb') as f:
            artifacts['age_label_encoder'] = pickle.load(f)
        
        st.success("✅ Models loaded successfully!")
        return artifacts
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the 'saved_models' directory and its contents are in the repository.")
        return None
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

