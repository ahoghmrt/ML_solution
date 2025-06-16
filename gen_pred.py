import numpy as np
import joblib
from tensorflow import keras

# Load scalers and model
scaler_wave = joblib.load("training_plots/waveform_scaler.pkl")
scaler_t0 = joblib.load("training_plots/t0_scaler.pkl")
scaler_amp = joblib.load("training_plots/amp_scaler.pkl")
signal_model = keras.models.load_model("signal_model.keras")

# Load input waveforms
data = np.load("ml_training_data/training_data_signals.npz")
X = data["waveforms"]

# Normalize waveforms
X_scaled = scaler_wave.transform(X)
X_scaled = np.expand_dims(X_scaled, axis=-1)

# Predict and inverse-transform
pred = signal_model.predict(X_scaled)
pred_t0s = scaler_t0.inverse_transform(pred[:, 0::3])
pred_amps = scaler_amp.inverse_transform(pred[:, 1::3])
presence = pred[:, 2::3]

# Save
np.savez("ml_training_data/predicted_signals.npz",
         t0s=pred_t0s,
         amps=pred_amps,
         presence=presence)

print("✅ Saved ml_training_data/predicted_signals.npz")
