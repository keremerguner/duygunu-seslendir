import os
from flask import Flask, render_template, request, jsonify  
import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)

app.config['SECRET_KEY'] = 'your_secret_key'

model_path = '../efe/modeldeneme123.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model dosyası '{model_path}' bulunamadı.")

model = load_model(model_path)  

def predict_emotion(mfcc):
    input_data = np.expand_dims(mfcc, axis=0)
    predictions = model.predict(input_data)[0]  
    return predictions.tolist()  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ses kaydı için gerekli parametreler
    duration = 3  # Kayıt süresi (saniye)
    sample_rate = 44100  # Örnek hızı

    # Ses kaydı yapma
    print("Lütfen konuşun...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()

    # Ses dosyasını MFCC'ye dönüştürme
    mfcc = librosa.feature.mfcc(y=audio_data.flatten(), sr=sample_rate, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    
    # Duygu tahmini
    predicted_emotion = predict_emotion(mfcc)
    
    # Tahmin sonuçlarını JSON formatında döndür
    return jsonify(predicted_emotion)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
