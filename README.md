# Anomaly Detection in Time-Series Data

Real-time IoT sensor anomaly detection using LSTM autoencoders and Isolation Forest, with n8n-based alert automation.

## Features
- LSTM autoencoder for unsupervised temporal anomaly detection
- Isolation Forest for ensemble anomaly scoring
- Real-time data ingestion via FastAPI
- Confidence-scored anomaly alerts
- n8n workflow integration for automated notifications
- 35% lower false positive rate vs rule-based baseline

## Metrics
- False Positives: -35% vs baseline
- Detection: Real-time (<200ms latency)

## Stack
Python · PyTorch · LSTM · Isolation Forest · FastAPI · n8n

## Setup
```bash
pip install -r requirements.txt
python train.py     # Train LSTM autoencoder
uvicorn api:app --reload
```
