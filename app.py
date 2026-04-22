import os
import json
import math
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory
from datetime import timedelta, timezone, datetime

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======= SOLAR CONFIG =======
SOLAR_RESULTS_DIR = os.path.join(BASE_DIR, "solar_pytorch_results")
SOLAR_DATA_PATH = os.path.join(BASE_DIR, "solar_thermal_time_series_openmeteo.csv")
SOLAR_TARGET_COL = 'Electrical_Power_kW'
SOLAR_SEQ_LEN = 72  # 3-day lookback for solar

# ======= WIND CONFIG =======
WIND_RESULTS_DIR = os.path.join(BASE_DIR, "wind_pytorch_results")
WIND_DATA_PATH = os.path.join(BASE_DIR, "WINDENERGY_FORCASTING_LSTM", "data", "Wind_open-meteo-9.75N77.75E229m.csv")
WIND_TARGET_COL = 'Power'
WIND_SEQ_LEN = 48   # 2-day lookback for wind

# Cache
_df_cache = {}
_model_cache = {}


# ======= UNIVARIATE LSTM MODEL DEFINITION (PyTorch) =======
class UnivariateLSTMModel(nn.Module):
    """Univariate BiLSTM model — same architecture for both solar and wind."""
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, dropout=0.2):
        super(UnivariateLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        fc_input = hidden_size * 2
        self.fc_block = nn.Sequential(
            nn.LayerNorm(fc_input),
            nn.Linear(fc_input, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc_block(out)
        return out


# ================================================================
#                    DATA LOADERS
# ================================================================

def get_solar_data():
    if "solar_main" not in _df_cache:
        df = pd.read_csv(SOLAR_DATA_PATH)
        df['time'] = pd.to_datetime(df['time'], dayfirst=True)
        df = df.sort_values('time').reset_index(drop=True)
        _df_cache["solar_main"] = df
    return _df_cache["solar_main"]


def get_wind_data():
    """Load wind data CSV."""
    if "wind_main" not in _df_cache:
        if os.path.exists(WIND_DATA_PATH):
            df = pd.read_csv(WIND_DATA_PATH)
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'])
                df = df.sort_values('Time').reset_index(drop=True)
            _df_cache["wind_main"] = df
        else:
            _df_cache["wind_main"] = None
    return _df_cache["wind_main"]


def get_solar_predictions(seq_len):
    key = f"solar_pred_{seq_len}"
    if key not in _df_cache:
        path = os.path.join(SOLAR_RESULTS_DIR, f"predictions_seq_{seq_len}.csv")
        if os.path.exists(path):
            _df_cache[key] = pd.read_csv(path)
        else:
            return None
    return _df_cache[key]


# ================================================================
#                    SOLAR MODEL LOADING & INFERENCE
# ================================================================

def get_solar_model_and_scaler():
    """Load the trained univariate solar seq=72 model and scaler."""
    if "solar_model" not in _model_cache:
        device = torch.device("cpu")

        scaler = joblib.load(os.path.join(SOLAR_RESULTS_DIR, "scaler.save"))

        model = UnivariateLSTMModel(input_size=1).to(device)
        model_path = os.path.join(SOLAR_RESULTS_DIR, f"best_model_seq_{SOLAR_SEQ_LEN}.pth")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()

        _model_cache["solar_model"] = model
        _model_cache["solar_scaler"] = scaler
        _model_cache["solar_device"] = device

    return (
        _model_cache["solar_model"],
        _model_cache["solar_scaler"],
        _model_cache["solar_device"],
    )


def predict_solar_24h():
    """Use the univariate solar seq=72 LSTM to forecast next 24 hours."""
    model, scaler, device = get_solar_model_and_scaler()
    df = get_solar_data()

    # UNIVARIATE: Only use the target column
    last_data = df[[SOLAR_TARGET_COL]].values[-SOLAR_SEQ_LEN:]
    scaled_window = scaler.transform(last_data)

    predictions = []
    timestamps = []
    current_window = scaled_window.copy()

    now_ist = datetime.now(IST)

    for step in range(24):
        future_time = now_ist + timedelta(hours=step + 1)
        hour_of_day = future_time.hour

        # Solar energy is only available during sunlight hours (approx 6AM-6PM)
        is_nighttime = hour_of_day < 6 or hour_of_day >= 19

        if is_nighttime:
            pred_real = 0.0
        else:
            x = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_scaled = model(x).cpu().numpy().flatten()[0]

            pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
            pred_real = max(0, pred_real)

        predictions.append(round(float(pred_real), 2))
        timestamps.append(future_time.strftime("%I:%M %p"))

        # Slide window: append new prediction, remove oldest
        if is_nighttime:
            new_val_scaled = scaler.transform([[0.0]])[0]
        else:
            new_val_scaled = np.array([[pred_scaled]])

        current_window = np.vstack([current_window[1:], new_val_scaled])

    return predictions, timestamps


# ================================================================
#                    WIND MODEL LOADING & INFERENCE
# ================================================================

def get_wind_model_and_scaler():
    """Load the trained univariate wind seq=48 model and scaler."""
    if "wind_model" not in _model_cache:
        device = torch.device("cpu")

        scaler_path = os.path.join(WIND_RESULTS_DIR, "scaler.save")
        model_path = os.path.join(WIND_RESULTS_DIR, f"best_model_seq_{WIND_SEQ_LEN}.pth")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            _model_cache["wind_model"] = None
            return None, None, None

        scaler = joblib.load(scaler_path)

        model = UnivariateLSTMModel(input_size=1).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()

        _model_cache["wind_model"] = model
        _model_cache["wind_scaler"] = scaler
        _model_cache["wind_device"] = device

    if _model_cache.get("wind_model") is None:
        return None, None, None

    return (
        _model_cache["wind_model"],
        _model_cache["wind_scaler"],
        _model_cache["wind_device"],
    )


def predict_wind_24h():
    """Use the univariate wind seq=48 LSTM to forecast next 24 hours."""
    model, scaler, device = get_wind_model_and_scaler()

    if model is None:
        return None, None

    wind_df = get_wind_data()
    if wind_df is None:
        return None, None

    # UNIVARIATE: Only use Power column
    last_data = wind_df[[WIND_TARGET_COL]].values[-WIND_SEQ_LEN:]
    scaled_window = scaler.transform(last_data)

    predictions = []
    timestamps = []
    current_window = scaled_window.copy()

    now_ist = datetime.now(IST)

    for step in range(24):
        future_time = now_ist + timedelta(hours=step + 1)

        x = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = model(x).cpu().numpy().flatten()[0]

        pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
        pred_real = max(0, pred_real)  # Power can't be negative

        predictions.append(round(float(pred_real), 2))
        timestamps.append(future_time.strftime("%I:%M %p"))

        # Slide window
        new_val_scaled = np.array([[pred_scaled]])
        current_window = np.vstack([current_window[1:], new_val_scaled])

    return predictions, timestamps


# ================================================================
#                    METRICS LOADING
# ================================================================

def get_wind_metrics():
    """Load wind model metrics."""
    metrics = {"training": None, "testing": None}

    train_path = os.path.join(WIND_RESULTS_DIR, "training_metrics.json")
    test_path = os.path.join(WIND_RESULTS_DIR, "testing_metrics.json")

    if os.path.exists(train_path):
        with open(train_path) as f:
            metrics["training"] = json.load(f)

    if os.path.exists(test_path):
        with open(test_path) as f:
            metrics["testing"] = json.load(f)

    return metrics


# ================================================================
#                    API ROUTES — SOLAR
# ================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    df = get_solar_data()
    stats = {
        "total_rows": len(df),
        "date_range": {
            "start": df['time'].min().strftime("%Y-%m-%d"),
            "end": df['time'].max().strftime("%Y-%m-%d"),
        },
        "columns": list(df.columns),
        "feature_stats": {}
    }
    for col in df.columns:
        if col == "time":
            continue
        stats["feature_stats"][col] = {
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2),
        }
    return jsonify(stats)


@app.route("/api/metrics")
def api_metrics():
    path = os.path.join(SOLAR_RESULTS_DIR, "metrics.json")
    with open(path, "r") as f:
        metrics = json.load(f)
    return jsonify(metrics)


@app.route("/api/data")
def api_data():
    df = get_solar_data()
    page = int(request.args.get("page", 1))
    size = int(request.args.get("size", 50))
    total = len(df)
    total_pages = math.ceil(total / size)
    start = (page - 1) * size
    end = min(start + size, total)
    subset = df.iloc[start:end].copy()
    subset['time'] = subset['time'].dt.strftime("%Y-%m-%d %H:%M")
    return jsonify({
        "page": page,
        "size": size,
        "total": total,
        "total_pages": total_pages,
        "data": subset.to_dict(orient="records")
    })


@app.route("/api/predictions/<int:seq_len>")
def api_predictions(seq_len):
    pred_df = get_solar_predictions(seq_len)
    if pred_df is None:
        return jsonify({"error": "Not found"}), 404
    sample_size = int(request.args.get("samples", 500))
    if len(pred_df) > sample_size:
        step = len(pred_df) // sample_size
        sampled = pred_df.iloc[::step].head(sample_size)
    else:
        sampled = pred_df
    return jsonify({
        "seq_len": seq_len,
        "total": len(pred_df),
        "actual": [round(v, 2) for v in sampled["Actual"].tolist()],
        "predicted": [round(v, 2) for v in sampled["Predicted"].tolist()],
    })


@app.route("/api/forecast24")
def api_forecast24():
    """Solar: Run the univariate seq=72 LSTM model to predict next 24 hours."""
    try:
        predictions, timestamps = predict_solar_24h()
        return jsonify({
            "forecast_timestamps": timestamps,
            "forecast_predicted": predictions,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train_test_split/<int:seq_len>")
def api_train_test_split(seq_len):
    pred_df = get_solar_predictions(seq_len)
    if pred_df is None:
        return jsonify({"error": "Not found"}), 404
    total = len(pred_df)
    sample_size = int(request.args.get("samples", 500))
    if total > sample_size:
        step = total // sample_size
        sampled = pred_df.iloc[::step].head(sample_size)
    else:
        sampled = pred_df
    return jsonify({
        "seq_len": seq_len,
        "total": total,
        "test_actual": [round(v, 2) for v in sampled["Actual"].tolist()],
        "test_predicted": [round(v, 2) for v in sampled["Predicted"].tolist()],
    })


@app.route("/api/features")
def api_features():
    df = get_solar_data()
    sample_size = int(request.args.get("samples", 2000))
    if len(df) > sample_size:
        step = len(df) // sample_size
        sampled = df.iloc[::step].head(sample_size)
    else:
        sampled = df
    return jsonify({
        "time": sampled['time'].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        "DNI_W_m2": [round(v, 1) for v in sampled['DNI_W_m2'].tolist()],
        "Ambient_Temp_C": [round(v, 1) for v in sampled['Ambient_Temp_C'].tolist()],
        "Wind_Speed_mps": [round(v, 1) for v in sampled['Wind_Speed_mps'].tolist()],
        "Electrical_Power_kW": [round(v, 1) for v in sampled['Electrical_Power_kW'].tolist()],
        "Thermal_Power_kW": [round(v, 1) for v in sampled['Thermal_Power_kW'].tolist()],
    })


# ================================================================
#                    API ROUTES — WIND
# ================================================================

@app.route("/api/wind/forecast24")
def api_wind_forecast24():
    """Wind: Run the univariate seq=48 LSTM model to predict next 24 hours."""
    try:
        predictions, timestamps = predict_wind_24h()
        if predictions is None:
            return jsonify({"error": "Wind model not available."}), 503

        return jsonify({
            "forecast_timestamps": timestamps,
            "forecast_predicted": predictions,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/wind/metrics")
def api_wind_metrics():
    """Return wind model training & testing metrics."""
    metrics = get_wind_metrics()
    if metrics["training"] is None and metrics["testing"] is None:
        return jsonify({"error": "Wind metrics not found"}), 404
    return jsonify(metrics)


@app.route("/api/wind/predictions")
def api_wind_predictions():
    """Return wind model test predictions."""
    path = os.path.join(WIND_RESULTS_DIR, f"predictions_seq_{WIND_SEQ_LEN}.csv")
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    pred_df = pd.read_csv(path)
    sample_size = int(request.args.get("samples", 500))
    if len(pred_df) > sample_size:
        step = len(pred_df) // sample_size
        sampled = pred_df.iloc[::step].head(sample_size)
    else:
        sampled = pred_df
    return jsonify({
        "seq_len": WIND_SEQ_LEN,
        "total": len(pred_df),
        "actual": [round(v, 2) for v in sampled["Actual"].tolist()],
        "predicted": [round(v, 2) for v in sampled["Predicted"].tolist()],
    })


# ================================================================
#                    API ROUTES — COMBINED
# ================================================================

@app.route("/api/combined/forecast24")
def api_combined_forecast24():
    """Combined solar + wind 24H forecast."""
    result = {"solar": None, "wind": None, "combined": None}

    try:
        solar_pred, solar_ts = predict_solar_24h()
        result["solar"] = {
            "timestamps": solar_ts,
            "predicted": solar_pred,
        }
    except Exception as e:
        result["solar_error"] = str(e)

    try:
        wind_pred, wind_ts = predict_wind_24h()
        if wind_pred is not None:
            result["wind"] = {
                "timestamps": wind_ts,
                "predicted": wind_pred,
            }
    except Exception as e:
        result["wind_error"] = str(e)

    # Combined total
    if result["solar"] and result["wind"]:
        combined = [
            round(s + w, 2)
            for s, w in zip(result["solar"]["predicted"], result["wind"]["predicted"])
        ]
        result["combined"] = {
            "timestamps": result["solar"]["timestamps"],
            "predicted": combined,
        }

    return jsonify(result)


@app.route("/results/<path:filename>")
def serve_result_file(filename):
    return send_from_directory(SOLAR_RESULTS_DIR, filename)


@app.route("/wind_results/<path:filename>")
def serve_wind_result_file(filename):
    return send_from_directory(WIND_RESULTS_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 55)
    print("  Wind-Solar Energy Forecasting Dashboard")
    print("  Solar: Univariate LSTM (72H Seq -> 24H Forecast)")
    print("  Wind:  Univariate LSTM (48H Seq -> 24H Forecast)")
    print("  Framework: PyTorch (both models)")
    print(f"  Open: http://localhost:{port}")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
