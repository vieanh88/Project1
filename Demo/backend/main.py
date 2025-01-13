from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from model import KNNModel  # Import KNNModel từ module model.py

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WeatherData(BaseModel):
    tempmax: float
    tempmin: float
    temp: float
    humidity: float
    winddir: float
    cloudcover: float
    sealevelpressure: float

# Đường dẫn đến model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'knn_model.joblib')

print(f"Current working directory: {os.getcwd()}")
print(f"Base directory: {BASE_DIR}")
print(f"Model path: {MODEL_PATH}")
print(f"Model file exists: {os.path.exists(MODEL_PATH)}")

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    raise

@app.post("/predict")
async def predict_weather(data: WeatherData):
    try:
        features = [[
            data.tempmax, data.tempmin, data.temp,
            data.humidity, data.winddir, data.cloudcover,
            data.sealevelpressure
        ]]

        # Dự đoán
        prediction = model.predict(features)[0]
        
        # Chuyển đổi kết quả 0, 1 thành Không mưa, Có mưa
        weather_mapping = {0: "KHÔNG MƯA", 1: "CÓ MƯA"}
        prediction_label = weather_mapping.get(prediction, "Không xác định")  # Dự phòng nếu giá trị không khớp
        
        return {"prediction": prediction_label}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise