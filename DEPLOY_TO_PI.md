# Deploy Solar-Wind Forecasting on Raspberry Pi

## Quick Overview

| What | Details |
|------|---------|
| **Pi Model** | Raspberry Pi 4 (4GB+ RAM recommended) |
| **OS** | Raspberry Pi OS (64-bit recommended) |
| **Python** | 3.9 or 3.10 |
| **Framework** | PyTorch (CPU only) |
| **Web Server** | Flask on port 5000 |

---

## Step 1: Prepare Files on Your PC (Optional Cleanup)

Before copying, remove unnecessary training files to save space:

```bash
cd "A:\solar forecasting"
python prepare_for_pi.py
```

Or manually - you only need these files/folders on the Pi:

```
solar_forecasting/
├── app.py                              <- Flask app (REQUIRED)
├── requirements.txt                    <- Dependencies
├── solar_thermal_time_series_openmeteo.csv  <- Solar data
├── templates/
│   └── index.html                      <- Dashboard UI
├── solar_pytorch_results/
│   ├── best_model_seq_72.pth           <- Solar model (~15MB)
│   ├── scaler.save                     <- Solar scaler
│   ├── metrics.json                    <- Solar metrics
│   ├── predictions_seq_72.csv          <- Test predictions
│   ├── loss_curve_seq_72.png           <- Loss curve image
│   ├── train_pred_plot_seq_72.png
│   └── test_pred_plot_seq_72.png
├── wind_pytorch_results/
│   ├── best_model_seq_48.pth           <- Wind model (~15MB)
│   ├── scaler.save                     <- Wind scaler
│   ├── metrics.json
│   ├── training_metrics.json
│   ├── testing_metrics.json
│   ├── predictions_seq_48.csv
│   ├── loss_curve_seq_48.png
│   ├── train_pred_plot_seq_48.png
│   └── test_pred_plot_seq_48.png
└── WINDENERGY_FORCASTING_LSTM/
    └── data/
        └── Wind_open-meteo-9.75N77.75E229m.csv  <- Wind data
```

---

## Step 2: Copy Files to Raspberry Pi

### Option A: USB Drive
```bash
sudo mkdir -p /home/pi/solar_forecasting
cp -r /media/pi/USB_DRIVE/solar\ forecasting/* /home/pi/solar_forecasting/
```

### Option B: SCP (over network from Windows)
```powershell
scp -r "A:\solar forecasting\*" pi@<PI_IP>:/home/pi/solar_forecasting/
```

### Option C: FileZilla (GUI)
1. Install FileZilla on your PC
2. Connect to Pi using SFTP (port 22)
3. Drag and drop the folder

---

## Step 3: Install Dependencies on Raspberry Pi

SSH into your Pi or open a terminal:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv

# Navigate to project
cd /home/pi/solar_forecasting

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (except PyTorch)
pip install flask numpy pandas scikit-learn joblib matplotlib

# Install PyTorch for Raspberry Pi (CPU only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you get memory errors during install, increase swap:
```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## Step 4: Test Run

```bash
cd /home/pi/solar_forecasting
source venv/bin/activate
python3 app.py
```

You should see:
```
=======================================================
  Wind-Solar Energy Forecasting Dashboard
  Solar: Univariate LSTM (72H Seq -> 24H Forecast)
  Wind:  Univariate LSTM (48H Seq -> 24H Forecast)
  Open: http://localhost:5000
=======================================================
```

Open in browser: http://<PI_IP>:5000

IMPORTANT: To access from other devices on your network, edit app.py
and change the last line to:

    app.run(debug=False, host='0.0.0.0', port=5000)

Then access via http://<PI_IP_ADDRESS>:5000 from any device on the same network.

---

## Step 5: Run on Boot (Auto-start with systemd)

```bash
sudo nano /etc/systemd/system/solar-forecast.service
```

Paste this:
```ini
[Unit]
Description=Solar Wind Forecasting Dashboard
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/solar_forecasting
ExecStart=/home/pi/solar_forecasting/venv/bin/python3 app.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Then enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable solar-forecast.service
sudo systemctl start solar-forecast.service

# Check status
sudo systemctl status solar-forecast.service

# View logs
journalctl -u solar-forecast.service -f
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| PyTorch won't install | pip install torch --no-cache-dir |
| Out of memory | Increase swap (see Step 3) |
| Port already in use | sudo lsof -i :5000 then kill process |
| Model loads slowly | First load ~10-15 sec, then cached |
| Can't access from phone | Set host='0.0.0.0' in app.py |
| Permission denied | chmod -R 755 /home/pi/solar_forecasting |

## Performance Notes

- First request takes 10-15 seconds (model loading)
- Subsequent requests are fast (~1-2 sec, models cached)
- RAM usage: ~500MB with both models
- Recommended: Raspberry Pi 4, 4GB+ RAM
