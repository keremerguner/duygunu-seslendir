import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Modeli yükleme
model = load_model('model.h5')  # Modelinizi buraya yükleyin

def predict_emotion(audio_file_path, model):
    # Ses dosyasını yükle ve MFCC çıkar
    y, sr = librosa.load(audio_file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    
    # Modeli kullanarak tahmin yapma
    input_data = np.expand_dims(mfcc, axis=0)
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions[0])
    
    # Tahmin edilen sınıfı döndürme
    return predicted_class, predictions[0][predicted_class]

# Test etmek istediğiniz ses dosyasının yolu
# audio_file_path = '../test/Happy/6783_mt_koy.wav'  # Test edilecek ses dosyasının yolu
# audio_file_path = '../test/Angry/6783_kz_tirtil.wav'  # Test edilecek ses dosyasının yolu
# audio_file_path = '../test/Calm/7895_sk_acik.wav'  # Test edilecek ses dosyasının yolu
# audio_file_path = '../test/Sad/6783_hl_sepet.wav'  # Test edilecek ses dosyasının yolu
audio_file_path = './firildak.wav'  # Test edilecek ses dosyasının yolu


# Tahmini yapma
predicted_class, confidence = predict_emotion(audio_file_path, model)

# Tahmin sonuçlarını yazdırma
emotions = {0: 'Angry', 1: 'Calm', 2: 'Happy', 3: 'Sad'}
predicted_emotion = emotions[predicted_class]
print("Tahmin edilen duygu: ", predicted_emotion)
print("Tahmin güveni (yüzde): ", confidence * 100)
