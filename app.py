# app.py

import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from model import predict
import time
import uuid
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Vocalytics - AI Voice Analysis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Application Header ---
st.title("🎙️ Vocalytics: AI Voice Analysis")
st.markdown("Welcome to Vocalytics! Analyze a voice sample and help improve our AI by providing feedback on the prediction.")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.markdown("Select your input method.")
    input_method = st.radio(
        "Input Method:",
        ["Record Audio", "Upload Audio File", "Upload Video File"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.info("Your privacy is respected. Submitted audio is used anonymously to improve model accuracy.")

# --- Main Page Layout ---
col1, col2 = st.columns([0.6, 0.4], gap="large")

with col1:
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

    if input_method == "Record Audio":
        st.subheader("Record Your Voice")
        audio_bytes = audio_recorder(
            pause_threshold=2.0, sample_rate=41_000, text="",
            recording_color="#e8b62c", neutral_color="#6aa36f", icon_name="microphone", icon_size="3x"
        )
        if audio_bytes:
            st.session_state.audio_bytes = audio_bytes
    elif input_method == "Upload Audio File":
        st.subheader("Upload an Audio File")
        uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a"], label_visibility="collapsed")
        if uploaded_file:
            st.session_state.audio_bytes = uploaded_file.read()
    elif input_method == "Upload Video File":
        st.subheader("Upload a Video File")
        uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"], label_visibility="collapsed")
        if uploaded_file:
            with st.spinner("Extracting audio from video..."):
                # ... (video extraction logic is unchanged)
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                clip = VideoFileClip(tfile.name)
                audio_tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                clip.audio.write_audiofile(audio_tfile.name, logger=None)
                with open(audio_tfile.name, "rb") as f:
                    st.session_state.audio_bytes = f.read()
                clip.close()
                os.unlink(tfile.name)
                os.unlink(audio_tfile.name)


# --- Results and Prediction Column ---
with col2:
    st.subheader("Analysis Results")
    if st.session_state.audio_bytes:
        st.markdown("##### 🎧 Your Audio Sample")
        st.audio(st.session_state.audio_bytes, format="audio/wav")

        if "prediction_done" not in st.session_state:
            st.session_state.prediction_done = False

        predict_button = st.button("🎯 Analyze Voice", use_container_width=True, type="primary")

        if predict_button:
            with st.spinner("🤖 The AI is analyzing the voice..."):
                temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_audio_file.write(st.session_state.audio_bytes)
                temp_audio_file.close()
                try:
                    gender, age_group = predict(temp_audio_file.name)
                    st.session_state.prediction = (gender, age_group)
                    st.session_state.prediction_done = True
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    os.unlink(temp_audio_file.name)
        
        if st.session_state.get("prediction_done"):
            gender, age_group = st.session_state.prediction
            if "Error" not in str(gender):
                st.success("Analysis Complete!")
                st.metric(label="Predicted Gender", value=gender)
                st.metric(label="Predicted Age Group", value=age_group.capitalize())

                # --- NEW: Feedback Form ---
                st.markdown("---")
                with st.form("feedback_form"):
                    st.write("**Help Improve Our Model**")
                    st.write("If the prediction was off, please correct it and submit.")
                    
                    correct_gender = st.selectbox("Correct Gender:", ["Male", "Female"], index=0 if gender == "Male" else 1)
                    age_options = ["Teens", "Twenties", "Thirties", "Forties", "Fifties", "Sixties", "Seventies", "Eighties"]
                    try:
                        age_index = age_options.index(age_group.capitalize())
                    except ValueError:
                        age_index = 0 # Default if prediction is not in list
                    correct_age = st.selectbox("Correct Age Group:", age_options, index=age_index)

                    submitted = st.form_submit_button("Submit Feedback")

                    if submitted:
                        # Define paths for user submissions
                        submission_dir = "user_submissions"
                        audio_dir = os.path.join(submission_dir, "audio")
                        metadata_path = os.path.join(submission_dir, "feedback.csv")
                        os.makedirs(audio_dir, exist_ok=True)

                        # Save audio file with a unique name
                        audio_filename = f"{uuid.uuid4()}.wav"
                        audio_filepath = os.path.join(audio_dir, audio_filename)
                        with open(audio_filepath, "wb") as f:
                            f.write(st.session_state.audio_bytes)

                        # Save metadata to a CSV file
                        feedback_data = {
                            'filename': [audio_filename],
                            'gender': [correct_gender.lower()],
                            'age': [correct_age.lower()]
                        }
                        new_df = pd.DataFrame(feedback_data)
                        header = not os.path.exists(metadata_path)
                        new_df.to_csv(metadata_path, mode='a', header=header, index=False)
                        
                        st.success("✅ Thank you! Your feedback is saved and will be used to improve the model.")
            else:
                st.error(f"Prediction failed: {gender}")

        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state.audio_bytes = None
            st.session_state.prediction_done = False
            st.rerun()

    else:
        st.info("The prediction results will appear here once you provide an audio sample.")
        

