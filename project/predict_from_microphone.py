import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa

# Load the trained model
model = tf.keras.models.load_model('../sıfırdan/model.h5')

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
    # Reshape to match model input shape (1, time_steps, 1)
    mfcc_features = mfcc_features.reshape(1, -1, 1)
    prediction = model.predict(mfcc_features)
    emotion_index = np.argmax(prediction)
    confidence = prediction[0][emotion_index] * 100  # Get confidence in percentage
    return emotion_index, confidence

# Function to predict emotion from audio file
def predict_emotion_from_audio_file(file_path):
    audio_data, _ = librosa.load(file_path, sr=44100)
    return predict_emotion(audio_data)

# Dictionary to map emotion index to emotion label
emotion_labels = {0: 'Angry', 1: 'Calm', 2: 'Happy', 3: 'Sad'}

def predict_emotion_from_microphone():
    audio_data = record_audio()
    return predict_emotion(audio_data)
