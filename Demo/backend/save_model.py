# save_model.py
from model import KNNModel
import pandas as pd
import joblib
import random

def split_data(X, y, train_size=0.8):
    data = list(zip(X, y))
    random.shuffle(data)
    split_idx = int(len(data) * train_size)
    train_data = data[:split_idx]
    X_train, y_train = zip(*train_data)
    return list(X_train), list(y_train)

# Đọc dữ liệu thật
data = pd.read_csv("clean_data.csv")
features = ['tempmax', 'tempmin', 'temp', 'humidity', 'winddir', 'cloudcover', 'sealevelpressure']
X = data[features].dropna().values
y = data['preciptype'].dropna().values

# Chia và train
X_train, y_train = split_data(X, y)
model = KNNModel(k=55)
model.fit(X_train, y_train)

# Lưu model
joblib.dump(model, 'knn_model.joblib')
print("Model saved successfully")