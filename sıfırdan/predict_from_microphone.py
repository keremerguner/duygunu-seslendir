import sounddevice as sd
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
import librosa

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Function to record audio
def record_audio(duration=5, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until the recording is finished
    print("Recording finished")
    return np.squeeze(recording)

# Function to save recorded audio to a file
def save_audio(filename, data, fs=44100):
    write(filename, fs, data)

# Function to convert audio to MFCC
def audio_to_mfcc(audio_data, sr=44100, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# Function to predict emotion from audio
def predict_emotion(audio_data):
    mfcc_features = audio_to_mfcc(audio_data)
    # Check the shape of mfcc_features
    print(f"MFCC features shape: {mfcc_features.shape}")

    # Reshape to match model input shape (1, time_steps, 1)
    mfcc_features = mfcc_features.reshape(1, -1, 1)

    # Check the shape of reshaped mfcc_features
    print(f"Reshaped MFCC features shape: {mfcc_features.shape}")
    
    prediction = model.predict(mfcc_features)
    emotion = np.argmax(prediction)
    return emotion

# Dictionary to map emotion index to emotion label
emotion_labels = {0: 'Angry', 1: 'Calm', 2: 'Happy', 3: 'Sad'}

# Record audio
duration = 5  # seconds
audio_data = record_audio(duration=duration)

# Save the recorded audio to a file (optional)
save_audio('recorded_audio.wav', audio_data)

# Predict emotion from recorded audio
emotion_index = predict_emotion(audio_data)
emotion = emotion_labels[emotion_index]

print(f"Predicted Emotion: {emotion}")
