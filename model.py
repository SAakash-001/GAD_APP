import pickle
import numpy as np
import librosa
import os

# --- Load Models and the New Label Encoder ---
try:
    with open('saved_models/gender_model.pkl', 'rb') as f:
        gender_clf = pickle.load(f)
    # Load the new age classifier model
    with open('saved_models/age_model.pkl', 'rb') as f:
        age_clf = pickle.load(f)
    # Load the label encoder to decode predictions
    with open('saved_models/age_label_encoder.pkl', 'rb') as f:
        age_le = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Models not found. Please run the train_model.py script again.")

# --- Feature Extraction (This must match the training script) ---
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mean_mfcc = np.mean(mfcc, axis=1)
        std_mfcc = np.std(mfcc, axis=1)
        feature_vector = np.hstack((mean_mfcc, std_mfcc))
        return feature_vector
    except Exception:
        return None

# --- Prediction Function ---
def predict(audio_path):
    """Predicts gender and age GROUP from an audio file."""
    if not os.path.exists(audio_path):
        return "Error: File does not exist", None

    features = extract_features(audio_path)
    if features is None:
        return "Error: Could not process audio file", None
        
    feat = features.reshape(1, -1)

    if feat.shape[1] != gender_clf.n_features_in_:
        return "Error: Feature shape mismatch.", None
    
    try:
        # Predict gender (no change here)
        gender_pred = gender_clf.predict(feat)[0]
        gender = 'Male' if gender_pred == 1 else 'Female'
        
        # Predict the encoded age group
        age_pred_encoded = age_clf.predict(feat)[0]
        # Use the label encoder to get the original text label (e.g., 'twenties')
        age_group = age_le.inverse_transform([age_pred_encoded])[0]
        
        # Capitalize for better display
        age_group_display = age_group.capitalize()

        return gender, age_group_display
    except Exception as e:
        return f"Error during prediction: {str(e)}", None
