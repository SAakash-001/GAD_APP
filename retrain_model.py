# retrain_model.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
import librosa
import shutil

# --- Configuration ---
ORIGINAL_CSV_PATH = 'data/cv-valid-train.csv'
ORIGINAL_AUDIO_DIR = 'data/cv-valid-train'
FEEDBACK_CSV_PATH = 'user_submissions/feedback.csv'
FEEDBACK_AUDIO_DIR = 'user_submissions/audio'
MODEL_SAVE_DIR = 'saved_models'
ARCHIVE_DIR = 'user_submissions/archive'

# --- Feature Extraction (Must be identical to train_model.py) ---
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mean_mfcc = np.mean(mfcc, axis=1)
        std_mfcc = np.std(mfcc, axis=1)
        return np.hstack((mean_mfcc, std_mfcc))
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# --- Main Retraining Logic ---
def run_retraining():
    print("🚀 Starting retraining pipeline...")

    # 1. Check for new feedback data
    if not os.path.exists(FEEDBACK_CSV_PATH):
        print("No new feedback data found to process. Exiting.")
        return

    # 2. Load and Combine DataFrames
    print("Loading original and new feedback datasets...")
    df_original = pd.read_csv(ORIGINAL_CSV_PATH)
    df_feedback = pd.read_csv(FEEDBACK_CSV_PATH)

    # Assign base directories for locating audio files
    df_original['source_dir'] = ORIGINAL_AUDIO_DIR
    df_feedback['source_dir'] = FEEDBACK_AUDIO_DIR
    
    # Standardize column names if needed (e.g., 'path' vs 'filename')
    original_audio_col = 'filename' if 'filename' in df_original.columns else 'path'
    
    # Combine dataframes
    combined_data = []

    # Process original data
    for _, row in df_original.iterrows():
        combined_data.append([row[original_audio_col], row['gender'], row['age'], row['source_dir']])

    # Process feedback data
    for _, row in df_feedback.iterrows():
        combined_data.append([row['filename'], row['gender'], row['age'], row['source_dir']])
        
    df_combined = pd.DataFrame(combined_data, columns=['filename', 'gender', 'age', 'source_dir'])

    # 3. Prepare data for modeling
    print("Preparing combined data...")
    df_combined.dropna(subset=['gender', 'age'], inplace=True)
    df_combined = df_combined[df_combined['gender'].isin(['male', 'female'])].copy()
    df_combined['age_group'] = df_combined['age'].astype(str).str.strip().str.lower()
    df_combined['gender_numeric'] = df_combined['gender'].apply(lambda g: 1 if g == 'male' else 0)

    # 4. Extract features from all audio files
    print(f"Extracting features from {len(df_combined)} total audio files...")
    features = []
    for _, row in tqdm(df_combined.iterrows(), total=df_combined.shape[0], desc="Processing audio"):
        audio_path = os.path.join(row['source_dir'], row['filename'])
        if os.path.exists(audio_path):
            feature = extract_features(audio_path)
            if feature is not None:
                features.append((feature, row['gender_numeric'], row['age_group']))

    if not features:
        print("Could not extract any valid features. Aborting retraining.")
        return
        
    X, y_gender, y_age_group = zip(*features)
    X = np.array(X)
    y_gender = np.array(y_gender)
    y_age_group = np.array(y_age_group)
    print(f"Successfully extracted features for {len(X)} samples.")
    
    # 5. Retrain models on the full, combined dataset
    age_le = LabelEncoder()
    y_age_encoded = age_le.fit_transform(y_age_group)

    print("\nTraining Gender Classifier...")
    gender_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    gender_clf.fit(X, y_gender)

    print("Training Age Group Classifier...")
    age_clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
    age_clf.fit(X, y_age_encoded)

    # 6. Save the newly trained models and the label encoder
    print("\nSaving updated models...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    with open(os.path.join(MODEL_SAVE_DIR, 'gender_model.pkl'), 'wb') as f:
        pickle.dump(gender_clf, f)
    with open(os.path.join(MODEL_SAVE_DIR, 'age_model.pkl'), 'wb') as f:
        pickle.dump(age_clf, f)
    with open(os.path.join(MODEL_SAVE_DIR, 'age_label_encoder.pkl'), 'wb') as f:
        pickle.dump(age_le, f)

    # 7. Archive the processed feedback data to prevent re-using it
    print("Archiving processed feedback data...")
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # Move the CSV file
    archive_csv_path = os.path.join(ARCHIVE_DIR, f'feedback_{timestamp}.csv')
    shutil.move(FEEDBACK_CSV_PATH, archive_csv_path)
    
    # Move the audio files
    archive_audio_dir = os.path.join(ARCHIVE_DIR, 'audio')
    os.makedirs(archive_audio_dir, exist_ok=True)
    for fname in df_feedback['filename']:
        src = os.path.join(FEEDBACK_AUDIO_DIR, fname)
        dst = os.path.join(archive_audio_dir, fname)
        if os.path.exists(src):
            shutil.move(src, dst)

    print("✅ Retraining pipeline complete. Models have been updated.")

if __name__ == '__main__':
    run_retraining()

