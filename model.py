import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Baca data fitur dari CSV
data = pd.read_csv('car_features.csv')

# Pisahkan fitur dan label
X = data[['Area', 'Aspect_Ratio']]
y = data['Label']

# Bagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
accuracy = model.score(X_test, y_test)
print(f'Akurasi model: {accuracy * 100:.2f}%')

# Simpan model ke file .pkl
with open('car_classifier.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model berhasil disimpan ke car_classifier.pkl")
