import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def main(epochs=30, batch_size=64, test_size=0.2):
    # -----------------------------
    # Load Dataset
    # -----------------------------
    data = np.load("ml_training_data/training_data_signals.npz")
    X_wave = data["waveforms"]                     # shape: (samples, 120)
    y = data["labels"]                             # shape: (samples, max_signals, 2)
    time = data["time"]

    max_signals = y.shape[1]

    logger.info(f"Loaded {X_wave.shape[0]} waveforms ({X_wave.shape[1]} time bins, max_signals={max_signals})")
    logger.debug(f"Label shape: {y.shape}, Time shape: {time.shape}")

    # -----------------------------
    # Normalize targets (t0 and amplitude separately)
    # -----------------------------
    t0s = y[:, :, 0]
    amps = y[:, :, 1]

    scaler_t0 = StandardScaler()
    scaler_amp = StandardScaler()

    t0s_norm = scaler_t0.fit_transform(t0s)
    amps_norm = scaler_amp.fit_transform(amps)

    y_norm = np.stack([t0s_norm, amps_norm], axis=-1)
    y_flat = y_norm.reshape((y.shape[0], max_signals * 2))

    # Save target scalers
    os.makedirs("training_plots", exist_ok=True)
    joblib.dump(scaler_t0, "training_plots/t0_scaler.pkl")
    joblib.dump(scaler_amp, "training_plots/amp_scaler.pkl")

    # -----------------------------
    # Normalize inputs (waveforms)
    # -----------------------------
    scaler_wave = StandardScaler()
    X_wave_scaled = scaler_wave.fit_transform(X_wave)
    X_wave_scaled = np.expand_dims(X_wave_scaled, axis=-1)

    # Save waveform scaler
    joblib.dump(scaler_wave, "training_plots/waveform_scaler.pkl")


    input_wave = layers.Input(shape=(X_wave.shape[1], 1), name="waveform_input")

    # Conv1D processing
    x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(input_wave)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)

    # Dense processing
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(max_signals * 2, name="signal_output")(x)

    model = keras.Model(inputs=input_wave, outputs=output)
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    model.summary()


    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_wave_scaled, y_flat, test_size=test_size, random_state=42
    )

    # -----------------------------
    # Callbacks
    # -----------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='best_signal_model.keras',
            monitor='val_mae',
            save_best_only=True,
            verbose=1
        )
    ]

    # -----------------------------
    # Train
    # -----------------------------
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # -----------------------------
    # Save Model and History
    # -----------------------------
    model.save("signal_model.keras")
    pd.DataFrame(history.history).to_csv("training_plots/signal_model_history.csv", index=False)

    # -----------------------------
    # Plot Training History
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss (MAE)')
    plt.plot(history.history['val_loss'], label='Val Loss (MAE)')
    plt.title("Signal Model Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_plots/signal_model_training.png")
    plt.close()
    logger.info("Saved model to 'signal_model.keras' and training plot")

    # -----------------------------
    # Final Evaluation on Validation Set
    # -----------------------------
    y_val_pred = model.predict(X_val)

    t0_idx = list(range(0, max_signals * 2, 2))
    amp_idx = list(range(1, max_signals * 2, 2))

    # Inverse-transform to original scale
    y_val_t0 = scaler_t0.inverse_transform(y_val[:, t0_idx])
    y_val_amp = scaler_amp.inverse_transform(y_val[:, amp_idx])
    y_pred_t0 = scaler_t0.inverse_transform(y_val_pred[:, t0_idx])
    y_pred_amp = scaler_amp.inverse_transform(y_val_pred[:, amp_idx])

    t0_mae = mean_absolute_error(y_val_t0, y_pred_t0)
    amp_mae = mean_absolute_error(y_val_amp, y_pred_amp)
    t0_rmse = np.sqrt(mean_squared_error(y_val_t0, y_pred_t0))
    amp_rmse = np.sqrt(mean_squared_error(y_val_amp, y_pred_amp))

    t0_pearson, _ = pearsonr(y_val_t0.ravel(), y_pred_t0.ravel())
    t0_spearman, _ = spearmanr(y_val_t0.ravel(), y_pred_t0.ravel())
    amp_pearson, _ = pearsonr(y_val_amp.ravel(), y_pred_amp.ravel())
    amp_spearman, _ = spearmanr(y_val_amp.ravel(), y_pred_amp.ravel())

    logger.info(f"t0  - MAE: {t0_mae:.4f} ns, RMSE: {t0_rmse:.4f} ns, "
                f"Pearson: {t0_pearson:.4f}, Spearman: {t0_spearman:.4f}")
    logger.info(f"amp - MAE: {amp_mae:.4f}, RMSE: {amp_rmse:.4f}, "
                f"Pearson: {amp_pearson:.4f}, Spearman: {amp_spearman:.4f}")

    # Per-signal-slot MAE
    per_slot_t0_mae = []
    per_slot_amp_mae = []
    for s in range(max_signals):
        slot_t0 = mean_absolute_error(y_val_t0[:, s], y_pred_t0[:, s])
        slot_amp = mean_absolute_error(y_val_amp[:, s], y_pred_amp[:, s])
        per_slot_t0_mae.append(slot_t0)
        per_slot_amp_mae.append(slot_amp)
        logger.debug(f"  Slot {s}: t0 MAE = {slot_t0:.4f} ns, amp MAE = {slot_amp:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    slots = np.arange(max_signals)
    ax1.bar(slots, per_slot_t0_mae, color='steelblue', edgecolor='black')
    ax1.set_xlabel("Signal Slot")
    ax1.set_ylabel("MAE (ns)")
    ax1.set_title("Per-Slot t0 MAE")
    ax1.set_xticks(slots)
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(slots, per_slot_amp_mae, color='salmon', edgecolor='black')
    ax2.set_xlabel("Signal Slot")
    ax2.set_ylabel("MAE")
    ax2.set_title("Per-Slot Amplitude MAE")
    ax2.set_xticks(slots)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_plots/signal_model_per_slot_mae.png")
    plt.close()
    logger.info("Saved per-slot MAE plot to 'training_plots/signal_model_per_slot_mae.png'")

    return {
        't0_mae': float(t0_mae), 't0_rmse': float(t0_rmse),
        't0_pearson': float(t0_pearson), 't0_spearman': float(t0_spearman),
        'amp_mae': float(amp_mae), 'amp_rmse': float(amp_rmse),
        'amp_pearson': float(amp_pearson), 'amp_spearman': float(amp_spearman),
    }


if __name__ == "__main__":
    main()
