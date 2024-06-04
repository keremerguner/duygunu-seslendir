import os
import numpy as np
import librosa
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Veri yükleme ve ön işleme fonksiyonu
def load_data(directory):
    labels = []
    features = []
    for label in os.listdir(directory):
        # Her kategori klasörü için
        class_path = os.path.join(directory, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                # Her ses dosyası için
                file_path = os.path.join(class_path, file)
                if file.endswith('.wav'):
                    # Ses dosyasını yükle ve MFCC çıkar
                    y, sr = librosa.load(file_path, sr=None)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfcc = np.mean(mfcc.T, axis=0)
                    features.append(mfcc)
                    labels.append(label)
    return features, labels

# Veriyi yükle
features, labels = load_data('../Sound Source/')

# Etiketleri sayısal hale getir
label_dict = {label: num for num, label in enumerate(sorted(set(labels)))}
numeric_labels = [label_dict[label] for label in labels]

# Özellikler ve etiketler array'e çevir
X = np.array(features)
y = to_categorical(numeric_labels)

# Eğitim ve test kümelerini böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Modeli oluşturma
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Modeli derleme
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Model değerlendirme
scores = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])
model.save('model.h5')
print("Eğitilmiş model 'model.h5' olarak kaydedildi.")

# Eğitim ve test kümelerini dosyalara kaydetme
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Eğitim ve test kümeleri kaydedildi.")
# Eğitilmiş modeli yükleme
model = load_model('./model.h5')
print("Eğitilmiş model yüklendi.")

# Test verilerini yükleme
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Modeli kullanarak tahminler yapma
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Sınıflandırma raporu ve karışıklık matrisi
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(true_classes, predicted_classes))
print(confusion_matrix(true_classes, predicted_classes))