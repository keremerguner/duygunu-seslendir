import numpy as np
import tensorflow as tf
import librosa
import os

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Function to load audio file and convert to MFCC
def load_audio(file_path, sr=44100, n_mfcc=13):
    audio_data, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

# Function to predict emotion from audio file
def predict_emotion_from_file(file_path):
    mfcc_features = load_audio(file_path)
    # Reshape to match model input shape (1, time_steps, 1)
    mfcc_features = mfcc_features.reshape(1, -1, 1)
    prediction = model.predict(mfcc_features)
    emotion = np.argmax(prediction)
    return emotion

# Dictionary to map emotion index to emotion label
emotion_labels = {0: 'Angry', 1: 'Calm', 2: 'Happy', 3: 'Sad'}
label_dict = {v: k for k, v in emotion_labels.items()}  # Reverse the emotion_labels dictionary

# Path to the directory containing test audio files and their labels
test_audio_dir = '../test/'  # Replace with your test audio directory path

# List to store true and predicted labels
true_labels = []
predicted_labels = []

# Iterate through the test audio files
for label in os.listdir(test_audio_dir):
    label_dir = os.path.join(test_audio_dir, label)
    if os.path.isdir(label_dir):
        for file in os.listdir(label_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(label_dir, file)
                true_labels.append(label_dict[label])  # Add true label
                predicted_label = predict_emotion_from_file(file_path)
                predicted_labels.append(predicted_label)  # Add predicted label

# Calculate accuracy
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
accuracy = np.mean(true_labels == predicted_labels)

print(f'Accuracy: {accuracy * 100:.2f}%')
