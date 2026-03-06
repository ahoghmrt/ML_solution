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
from scipy.optimize import linear_sum_assignment
import config as cfg

logger = logging.getLogger(__name__)


try:
    _register = keras.saving.register_keras_serializable
except AttributeError:
    _register = tf.keras.utils.register_keras_serializable


@_register(package="signal_model")
class WeightedHuberLoss(keras.losses.Loss):
    """Huber loss with per-component weights (higher weight on t0)."""
    def __init__(self, max_signals=cfg.MAX_SIGNALS, delta=cfg.HUBER_DELTA,
                 t0_weight=cfg.T0_LOSS_WEIGHT, **kwargs):
        super().__init__(**kwargs)
        self.max_signals = max_signals
        self.delta = delta
        self.t0_weight = t0_weight
        w = []
        for _ in range(max_signals):
            w.extend([t0_weight, 1.0])
        self.weights = tf.constant(w, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        abs_error = tf.abs(error)
        huber = tf.where(abs_error <= self.delta,
                         0.5 * tf.square(error),
                         self.delta * (abs_error - 0.5 * self.delta))
        return tf.reduce_mean(huber * self.weights)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_signals": self.max_signals,
            "delta": self.delta,
            "t0_weight": self.t0_weight,
        })
        return config


def _build_signal_loss(max_signals):
    """Build loss function based on config."""
    if cfg.SIGNAL_LOSS_TYPE == "weighted_huber":
        return WeightedHuberLoss(max_signals=max_signals)
    else:
        return 'mae'


class _EpochLogger(keras.callbacks.Callback):
    """Logs epoch metrics to the logger every N epochs, plus first and last."""
    def __init__(self, total_epochs, interval=cfg.EPOCH_LOG_INTERVAL):
        super().__init__()
        self.total_epochs = total_epochs
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        if ep == 1 or ep == self.total_epochs or ep % self.interval == 0:
            parts = [f"{k}: {v:.4f}" for k, v in (logs or {}).items()]
            logger.info(f"[Signal] Epoch {ep}/{self.total_epochs} - {' - '.join(parts)}")


class _HistoryWrapper:
    """Wraps a dict to look like keras History for plotting compatibility."""
    def __init__(self, history_dict):
        self.history = history_dict


def _pit_training_loop(model, X_train, y_train, X_val, y_val,
                       signal_counts_train, signal_counts_val,
                       max_signals, epochs, batch_size, log_dir=None):
    """Custom training loop with permutation-invariant matching."""
    optimizer = keras.optimizers.Adam()
    loss_fn = _build_signal_loss(max_signals)
    if isinstance(loss_fn, str):
        loss_fn = keras.losses.MeanAbsoluteError()

    tb_writer = None
    if log_dir:
        tb_writer = tf.summary.create_file_writer(log_dir)

    best_val_loss = np.inf
    patience_counter = 0
    lr_patience_counter = 0
    history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

    best_weights = model.get_weights()

    for epoch in range(epochs):
        # Shuffle training data
        perm = np.random.permutation(len(X_train))
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]
        counts_shuf = signal_counts_train[perm]

        epoch_losses = []
        epoch_maes = []

        # Mini-batch loop
        for start in range(0, len(X_shuf), batch_size):
            end = min(start + batch_size, len(X_shuf))
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]
            counts_batch = counts_shuf[start:end]

            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                y_pred_np = y_pred.numpy()

                # Hungarian-reorder y_true to match y_pred
                y_reordered = y_batch.copy()
                for bi in range(len(X_batch)):
                    n_active = int(counts_batch[bi])
                    if n_active < 2:
                        continue
                    # Build cost matrix in normalized space
                    cost = np.zeros((max_signals, n_active))
                    for s in range(max_signals):
                        for t in range(n_active):
                            cost[s, t] = abs(y_pred_np[bi, 2*s] - y_batch[bi, 2*t]) + \
                                         abs(y_pred_np[bi, 2*s+1] - y_batch[bi, 2*t+1])
                    row_ind, col_ind = linear_sum_assignment(cost)
                    # Build reordered target: place matched true signals into optimal slots
                    new_target = np.zeros(max_signals * 2)
                    for r, c in zip(row_ind, col_ind):
                        new_target[2*r] = y_batch[bi, 2*c]
                        new_target[2*r+1] = y_batch[bi, 2*c+1]
                    y_reordered[bi] = new_target

                y_reordered_tf = tf.constant(y_reordered, dtype=tf.float32)
                batch_loss = loss_fn(y_reordered_tf, y_pred)

            grads = tape.gradient(batch_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_losses.append(float(batch_loss))
            epoch_maes.append(float(tf.reduce_mean(tf.abs(y_reordered_tf - y_pred))))

        # Validation
        val_pred = model(X_val, training=False)
        val_pred_np = val_pred.numpy()
        y_val_reordered = y_val.copy()
        for bi in range(len(X_val)):
            n_active = int(signal_counts_val[bi])
            if n_active < 2:
                continue
            cost = np.zeros((max_signals, n_active))
            for s in range(max_signals):
                for t in range(n_active):
                    cost[s, t] = abs(val_pred_np[bi, 2*s] - y_val[bi, 2*t]) + \
                                 abs(val_pred_np[bi, 2*s+1] - y_val[bi, 2*t+1])
            row_ind, col_ind = linear_sum_assignment(cost)
            new_target = np.zeros(max_signals * 2)
            for r, c in zip(row_ind, col_ind):
                new_target[2*r] = y_val[bi, 2*c]
                new_target[2*r+1] = y_val[bi, 2*c+1]
            y_val_reordered[bi] = new_target

        val_loss = float(loss_fn(tf.constant(y_val_reordered, dtype=tf.float32), val_pred))
        val_mae = float(tf.reduce_mean(tf.abs(tf.constant(y_val_reordered, dtype=tf.float32) - val_pred)))

        train_loss = np.mean(epoch_losses)
        train_mae = np.mean(epoch_maes)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        ep = epoch + 1
        if ep == 1 or ep == epochs or ep % cfg.EPOCH_LOG_INTERVAL == 0:
            logger.info(f"[Signal-PIT] Epoch {ep}/{epochs} - loss: {train_loss:.4f} - mae: {train_mae:.4f} "
                         f"- val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f}")

        if tb_writer:
            with tb_writer.as_default():
                tf.summary.scalar('epoch_loss', train_loss, step=epoch)
                tf.summary.scalar('epoch_mae', train_mae, step=epoch)
                tf.summary.scalar('epoch_val_loss', val_loss, step=epoch)
                tf.summary.scalar('epoch_val_mae', val_mae, step=epoch)
                tf.summary.scalar('epoch_learning_rate', float(optimizer.learning_rate), step=epoch)
            tb_writer.flush()

        # ModelCheckpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            model.save("best_signal_model.keras")
            patience_counter = 0
            lr_patience_counter = 0
        else:
            patience_counter += 1
            lr_patience_counter += 1

        # ReduceLROnPlateau
        if lr_patience_counter >= cfg.LR_REDUCE_PATIENCE:
            old_lr = float(optimizer.learning_rate)
            new_lr = max(old_lr * cfg.LR_REDUCE_FACTOR, cfg.LR_MIN)
            optimizer.learning_rate.assign(new_lr)
            logger.info(f"Reducing learning rate: {old_lr:.2e} -> {new_lr:.2e}")
            lr_patience_counter = 0

        # EarlyStopping
        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    model.set_weights(best_weights)
    return _HistoryWrapper(history)


def main(epochs=cfg.SIGNAL_MODEL_EPOCHS, batch_size=cfg.SIGNAL_MODEL_BATCH_SIZE, test_size=cfg.TEST_SIZE, log_dir=None, use_pit=None):
    # -----------------------------
    # Load Dataset
    # -----------------------------
    data = np.load(os.path.join(cfg.DIR_ML_DATA, "training_data_signals.npz"))
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
    os.makedirs(cfg.DIR_TRAINING_PLOTS, exist_ok=True)
    joblib.dump(scaler_t0, os.path.join(cfg.DIR_TRAINING_PLOTS, "t0_scaler.pkl"))
    joblib.dump(scaler_amp, os.path.join(cfg.DIR_TRAINING_PLOTS, "amp_scaler.pkl"))

    # -----------------------------
    # Normalize inputs (waveforms)
    # -----------------------------
    scaler_wave = StandardScaler()
    X_wave_scaled = scaler_wave.fit_transform(X_wave)
    X_wave_scaled = np.expand_dims(X_wave_scaled, axis=-1)

    # Save waveform scaler
    joblib.dump(scaler_wave, os.path.join(cfg.DIR_TRAINING_PLOTS, "waveform_scaler.pkl"))


    input_wave = layers.Input(shape=(X_wave.shape[1], 1), name="waveform_input")

    # Conv1D processing
    x = layers.Conv1D(cfg.CONV_FILTERS[0], kernel_size=cfg.CONV_KERNEL_SIZE, activation='relu', padding='same')(input_wave)
    if cfg.USE_BATCHNORM:
        x = layers.BatchNormalization()(x)
    x = layers.Conv1D(cfg.CONV_FILTERS[1], kernel_size=cfg.CONV_KERNEL_SIZE, activation='relu', padding='same')(x)
    if cfg.USE_BATCHNORM:
        x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(cfg.DENSE_UNITS[0], activation='relu')(x)
    if cfg.USE_BATCHNORM:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg.DROPOUT_RATE)(x)
    x = layers.Dense(cfg.DENSE_UNITS[1], activation='relu')(x)
    output = layers.Dense(max_signals * 2, name="signal_output")(x)

    model = keras.Model(inputs=input_wave, outputs=output)

    # Build loss function
    loss_fn = _build_signal_loss(max_signals)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])
    model.summary()


    # Load signal counts for PIT
    count_data = np.load(os.path.join(cfg.DIR_ML_DATA, "training_data_counts.npz"))
    signal_counts = count_data["labels"]  # (N,)

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_val, y_train, y_val, counts_train, counts_val = train_test_split(
        X_wave_scaled, y_flat, signal_counts,
        test_size=test_size, random_state=cfg.RANDOM_STATE
    )

    # -----------------------------
    # Train
    # -----------------------------
    pit_enabled = use_pit if use_pit is not None else cfg.USE_PIT_LOSS
    if pit_enabled:
        logger.info("Using permutation-invariant training (PIT)")
        history = _pit_training_loop(
            model, X_train, y_train, X_val, y_val,
            counts_train, counts_val,
            max_signals, epochs, batch_size, log_dir=log_dir
        )
    else:
        callbacks = [
            _EpochLogger(epochs),
            keras.callbacks.EarlyStopping(
                monitor='val_mae',
                patience=cfg.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=cfg.LR_REDUCE_FACTOR,
                patience=cfg.LR_REDUCE_PATIENCE,
                min_lr=cfg.LR_MIN,
                verbose=0
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='best_signal_model.keras',
                monitor='val_mae',
                save_best_only=True,
                verbose=0
            )
        ]
        if log_dir:
            callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))
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
    pd.DataFrame(history.history).to_csv(os.path.join(cfg.DIR_TRAINING_PLOTS, "signal_model_history.csv"), index=False)

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
    plt.savefig(os.path.join(cfg.DIR_TRAINING_PLOTS, "signal_model_training.png"))
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
    plt.savefig(os.path.join(cfg.DIR_TRAINING_PLOTS, "signal_model_per_slot_mae.png"))
    plt.close()
    logger.info(f"Saved per-slot MAE plot to '{cfg.DIR_TRAINING_PLOTS}/signal_model_per_slot_mae.png'")

    return {
        't0_mae': float(t0_mae), 't0_rmse': float(t0_rmse),
        't0_pearson': float(t0_pearson), 't0_spearman': float(t0_spearman),
        'amp_mae': float(amp_mae), 'amp_rmse': float(amp_rmse),
        'amp_pearson': float(amp_pearson), 'amp_spearman': float(amp_spearman),
    }


if __name__ == "__main__":
    main()
