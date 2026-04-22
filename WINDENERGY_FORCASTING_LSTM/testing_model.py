import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
DATA_PATH = r"C:\Users\ABARNAA\Downloads\open-meteo-9.75N77.75E229m.csv"
SEQ_LENGTH = 48
PRED_LENGTH = 24
TRAIN_SPLIT = 0.8
MODELS_DIR = "models"
EPOCHS = 50
BATCH_SIZE = 32

def load_data(path):
    df = pd.read_csv(path)
    # Use features: Power and Wind Speed
    df = df[['Time', 'Power', 'wind_speed_10m (km/h)']]
    return df

def create_sequences(data, seq_length, pred_length):
    x, y = [], []
    # data is (N, 2) -> (Power, Wind Speed)
    # y is the next 24 hours of Power only
    for i in range(len(data) - seq_length - pred_length + 1):
        x.append(data[i : i + seq_length])
        y.append(data[i + seq_length : i + seq_length + pred_length, 0]) # Index 0 is Power
    return np.array(x), np.array(y)

def build_model(seq_length, pred_length):
    model = Sequential([
        LSTM(128, input_shape=(seq_length, 2), return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(pred_length)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    df = load_data(DATA_PATH)
    
    split_idx = int(len(df) * TRAIN_SPLIT)
    test_df = df.iloc[split_idx:]
    
    print(f"\n--- Testing for SEQ_LENGTH = {SEQ_LENGTH} ---")
    
    scaler = MinMaxScaler()
    # Features: Power (col 0), Wind Speed (col 1)
    features = test_df[['Power', 'wind_speed_10m (km/h)']].values
    scaler.fit(features)
    scaled_data = scaler.transform(features)
    
    X, y = create_sequences(scaled_data, SEQ_LENGTH, PRED_LENGTH)
    
    model = build_model(SEQ_LENGTH, PRED_LENGTH)
    # Early Stopping
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[early_stop])
    
    # Save as .h5
    model.save(os.path.join(MODELS_DIR, f'testing_lstm_model_{SEQ_LENGTH}.h5'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f'testing_scaler_{SEQ_LENGTH}.pkl'))
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    metrics = {
        str(SEQ_LENGTH): {
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2)
        }
    }
        
    with open(os.path.join(MODELS_DIR, 'testing_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
