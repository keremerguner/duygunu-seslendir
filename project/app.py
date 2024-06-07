from flask import Flask, render_template, jsonify
import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import threading
import time

app = Flask(__name__)

model = load_model('./modell.h5', compile=False)

emotions = ["Mutlu", "Üzgün", "Kızgın", "Sakin"]

def predict_emotion(mfcc):
    input_data = np.expand_dims(mfcc, axis=0)
    predictions = model.predict(input_data)[0]
    emotion_confidences = {emotions[i]: predictions[i] * 100 for i in range(len(emotions))}
    return emotion_confidences

def record_audio():
    duration = 2  # Kayıt süresini 1 saniyeye düşürüyoruz
    sample_rate = 44100
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()
    return audio_data.flatten(), sample_rate

def emotion_thread():
    global current_emotion_confidences, running
    while running:
        audio_data, sample_rate = record_audio()
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0)
        current_emotion_confidences = predict_emotion(mfcc)
        time.sleep(1)  # Tahminler arasındaki bekleme süresini 1 saniyeye düşürüyoruz

current_emotion_confidences = {emotion: 0.0 for emotion in emotions}
running = False
thread = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emotion')
def emotion():
    global current_emotion_confidences
    return jsonify(current_emotion_confidences)

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global running, thread
    if not running:
        running = True
        thread = threading.Thread(target=emotion_thread, daemon=True)
        thread.start()
    return jsonify({"status": "started"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global running
    running = False
    if thread:
        thread.join()
    return jsonify({"status": "stopped"})

@app.route('/record_once', methods=['POST'])
def record_once():
    audio_data, sample_rate = record_audio()
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    emotion_confidences = predict_emotion(mfcc)
    return jsonify(emotion_confidences)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
