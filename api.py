"""Real-time anomaly detection API."""
  import numpy as np, torch, joblib
  from fastapi import FastAPI
  from pydantic import BaseModel

  SEQ_LEN, N_FEATURES, HIDDEN_DIM = 30, 5, 64

  class LSTMAutoencoder(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.encoder = torch.nn.LSTM(N_FEATURES, HIDDEN_DIM, batch_first=True)
          self.decoder = torch.nn.LSTM(HIDDEN_DIM, N_FEATURES, batch_first=True)
      def forward(self, x):
          _, (h, _) = self.encoder(x)
          return self.decoder(h.squeeze(0).unsqueeze(1).repeat(1, x.size(1), 1))[0]

  app = FastAPI(title="Anomaly Detection API", version="1.0.0")
  model = LSTMAutoencoder(); model.eval()
  try:
      model.load_state_dict(torch.load("lstm_ae.pt", map_location="cpu"))
      iso = joblib.load("iso_forest.joblib")
      ready = True
  except: ready = False

  class SensorData(BaseModel):
      sequence: list[list[float]]  # shape: [seq_len, n_features]

  @app.post("/detect")
  def detect(data: SensorData):
      if not ready: return {"error": "Model not trained yet. Run train.py first."}
      x = torch.tensor([data.sequence], dtype=torch.float32)
      with torch.no_grad():
          recon = model(x)
          error = float(((recon - x)**2).mean())
      score = float(iso.decision_function([[error]])[0])
      is_anomaly = iso.predict([[error]])[0] == -1
      return {"is_anomaly": bool(is_anomaly), "reconstruction_error": round(error, 6),
              "anomaly_score": round(score, 4), "confidence": round(min(abs(score)*2, 1.0), 3)}

  @app.get("/health")
  def health(): return {"status": "online", "model_ready": ready}
  