"""Train LSTM Autoencoder + Isolation Forest for time-series anomaly detection."""
  import numpy as np
  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader, TensorDataset
  from sklearn.ensemble import IsolationForest
  import joblib

  SEQ_LEN, N_FEATURES, HIDDEN_DIM = 30, 5, 64

  class LSTMAutoencoder(nn.Module):
      def __init__(self):
          super().__init__()
          self.encoder = nn.LSTM(N_FEATURES, HIDDEN_DIM, batch_first=True)
          self.decoder = nn.LSTM(HIDDEN_DIM, N_FEATURES, batch_first=True)

      def forward(self, x):
          _, (h, _) = self.encoder(x)
          # Repeat hidden state across sequence length for decoder
          repeated = h.squeeze(0).unsqueeze(1).repeat(1, x.size(1), 1)
          out, _ = self.decoder(repeated)
          return out

  def generate_data(n=2000, anomaly_ratio=0.05):
      """Synthetic multivariate time-series with injected anomalies."""
      t = np.linspace(0, 4*np.pi, n)
      data = np.column_stack([np.sin(t), np.cos(t), 0.5*np.sin(2*t),
                               0.3*np.cos(3*t), np.random.normal(0, 0.1, n)])
      labels = np.zeros(n, dtype=int)
      n_anom = int(n * anomaly_ratio)
      anom_idx = np.random.choice(n, n_anom, replace=False)
      data[anom_idx] += np.random.normal(0, 3, (n_anom, N_FEATURES))
      labels[anom_idx] = 1
      return data, labels

  def make_sequences(data, seq_len):
      seqs = [data[i:i+seq_len] for i in range(len(data)-seq_len)]
      return np.array(seqs, dtype=np.float32)

  data, labels = generate_data()
  seqs = make_sequences(data, SEQ_LEN)
  normal_seqs = seqs[labels[:len(seqs)] == 0]

  loader = DataLoader(TensorDataset(torch.tensor(normal_seqs)), batch_size=64, shuffle=True)
  model = LSTMAutoencoder()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.MSELoss()

  print("Training LSTM Autoencoder on normal sequences...")
  for epoch in range(20):
      total_loss = 0
      for (batch,) in loader:
          optimizer.zero_grad()
          recon = model(batch)
          loss = criterion(recon, batch)
          loss.backward(); optimizer.step()
          total_loss += loss.item()
      if (epoch + 1) % 5 == 0:
          print(f"Epoch {epoch+1:2d}/20 — Loss: {total_loss/len(loader):.6f}")

  torch.save(model.state_dict(), "lstm_ae.pt")

  # ── Isolation Forest on reconstruction errors ─────────────────────────────
  model.eval()
  with torch.no_grad():
      recon_errors = ((model(torch.tensor(seqs)) - torch.tensor(seqs))**2).mean(dim=(1,2)).numpy()

  iso_forest = IsolationForest(contamination=0.05, random_state=42)
  iso_forest.fit(recon_errors.reshape(-1, 1))
  joblib.dump(iso_forest, "iso_forest.joblib")
  print("Saved: lstm_ae.pt | iso_forest.joblib")
  