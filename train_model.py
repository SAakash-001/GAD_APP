import os
import librosa
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm

# --- Configuration ---
CSV_PATH = 'data/cv-valid-train.csv'
AUDIO_DIR = 'data/cv-valid-train'
MODEL_SAVE_DIR = 'saved_models'

# --- Feature Extraction (No changes needed here) ---
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mean_mfcc = np.mean(mfcc, axis=1)
        std_mfcc = np.std(mfcc, axis=1)
        feature_vector = np.hstack((mean_mfcc, std_mfcc))
        return feature_vector
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# --- Main Training Logic ---
print("Loading metadata...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Metadata file not found at {CSV_PATH}. Please check the path.")
df = pd.read_csv(CSV_PATH)

print("Preparing data...")
df.dropna(subset=['gender', 'age'], inplace=True)
df = df[df['gender'].isin(['male', 'female'])].copy()
# Prepare labels for classification
df['age_group'] = df['age'].astype(str).str.strip().str.lower()
df['gender_numeric'] = df['gender'].apply(lambda g: 1 if g == 'male' else 0)

print(f"Extracting features from {len(df)} audio files...")
audio_col = 'filename' if 'filename' in df.columns else 'path'
features = []
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing audio"):
    audio_path = os.path.join(AUDIO_DIR, row[audio_col])
    if os.path.exists(audio_path):
        feature = extract_features(audio_path)
        if feature is not None:
            # Append the age_group string label
            features.append((feature, row['gender_numeric'], row['age_group']))

X, y_gender, y_age_group = zip(*features)
X = np.array(X)
y_gender = np.array(y_gender)
y_age_group = np.array(y_age_group)

print(f"Successfully extracted features for {len(X)} samples.")

# --- Encode Age Group Labels ---
age_le = LabelEncoder()
y_age_encoded = age_le.fit_transform(y_age_group)

# --- Train Gender Classification Model (No changes) ---
print("\nTraining Gender Classifier...")
X_train_g, X_test_g, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.2, random_state=42, stratify=y_gender)
gender_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gender_clf.fit(X_train_g, y_gender_train)
print(f"Gender model accuracy: {gender_clf.score(X_test_g, y_gender_test):.2%}")

# --- Train Age Group Classification Model ---
print("\nTraining Age Group Classifier...")
X_train_a, X_test_a, y_age_train, y_age_test = train_test_split(X, y_age_encoded, test_size=0.2, random_state=42, stratify=y_age_encoded)
# Use a classifier, not a regressor
age_clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
age_clf.fit(X_train_a, y_age_train)
y_age_pred = age_clf.predict(X_test_a)
# Use accuracy score instead of R-squared
print(f"Age Group model accuracy: {accuracy_score(y_age_test, y_age_pred):.2%}")

# --- Save Models and the New Label Encoder ---
print("\nSaving models...")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
with open(os.path.join(MODEL_SAVE_DIR, 'gender_model.pkl'), 'wb') as f:
    pickle.dump(gender_clf, f)
with open(os.path.join(MODEL_SAVE_DIR, 'age_model.pkl'), 'wb') as f:
    pickle.dump(age_clf, f)
# We must save the encoder to decode the predictions later
with open(os.path.join(MODEL_SAVE_DIR, 'age_label_encoder.pkl'), 'wb') as f:
    pickle.dump(age_le, f)

print("✅ Model training complete. Models are saved in the 'saved_models' directory.")
