import numpy as np
import os
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import config as cfg

logger = logging.getLogger(__name__)


def _setup_device(use_gpu=None):
    """Configure TensorFlow device. Call before building models."""
    if use_gpu is False:
        tf.config.set_visible_devices([], 'GPU')
        logger.info("Device: CPU (GPU disabled via --no-gpu)")
        return
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if cfg.GPU_MEMORY_GROWTH:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Device: GPU ({len(gpus)} available)")
    else:
        logger.info("Device: CPU (no GPU found)")


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
            logger.info(f"[Count] Epoch {ep}/{self.total_epochs} - {' - '.join(parts)}")


def main(epochs=cfg.COUNT_MODEL_EPOCHS, batch_size=cfg.COUNT_MODEL_BATCH_SIZE, test_size=cfg.TEST_SIZE, log_dir=None, use_gpu=None):
    _setup_device(use_gpu)
    # -----------------------------
    # Load Dataset
    # -----------------------------
    data = np.load(os.path.join(cfg.DIR_ML_DATA, "training_data_counts.npz"))
    X = data["waveforms"]                      # shape: (samples, 120)
    y = data["labels"]                         # shape: (samples,)
    time = data["time"]

    logger.info(f"Loaded {X.shape[0]} waveforms ({X.shape[1]} time bins)")
    logger.debug(f"Label shape: {y.shape}")

    # -----------------------------
    # Normalize Inputs
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.expand_dims(X_scaled, axis=-1)

    # Save waveform scaler for inference
    os.makedirs(cfg.DIR_TRAINING_PLOTS, exist_ok=True)
    joblib.dump(scaler, os.path.join(cfg.DIR_TRAINING_PLOTS, "count_waveform_scaler.pkl"))

    # -----------------------------
    # Build Model (Model = CONV, Input = waveform, Output = signal count class)
    # -----------------------------

    input_layer = layers.Input(shape=(X.shape[1], 1), name="waveform_input")

    x = layers.Conv1D(cfg.CONV_FILTERS[0], kernel_size=cfg.CONV_KERNEL_SIZE, activation='relu', padding='same')(input_layer)
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

    num_classes = len(cfg.SIGNAL_COUNTS)
    output = layers.Dense(num_classes, activation='softmax', name="count_output")(x)

    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=test_size, random_state=cfg.RANDOM_STATE
    )

    # -----------------------------
    # Class weights
    # -----------------------------
    class_weight_dict = None
    if cfg.USE_CLASS_WEIGHTS:
        classes_present = np.unique(y_train.astype(int))
        weights = compute_class_weight('balanced', classes=classes_present, y=y_train.astype(int))
        class_weight_dict = {int(c): float(w) for c, w in zip(classes_present, weights)}
        logger.info(f"Class weights: {class_weight_dict}")

    # -----------------------------
    # Callbacks
    # -----------------------------
    callbacks = [
        _EpochLogger(epochs),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
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
            filepath='best_count_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
    ]
    if log_dir:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))

    # -----------------------------
    # Train
    # -----------------------------
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

    # -----------------------------
    # Save Model and History
    # -----------------------------
    os.makedirs(cfg.DIR_TRAINING_PLOTS, exist_ok=True)
    model.save("signal_count_model.keras")
    pd.DataFrame(history.history).to_csv(os.path.join(cfg.DIR_TRAINING_PLOTS, "signal_count_model_history.csv"), index=False)

    # -----------------------------
    # Plot Training History
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Signal Count Model Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.DIR_TRAINING_PLOTS, "signal_count_model_training.png"))
    plt.close()
    logger.info("Saved model to 'signal_count_model.keras' and training plot")

    # -----------------------------
    # Final Evaluation on Validation Set
    # -----------------------------
    y_val_probs = model.predict(X_val)
    y_val_pred = np.argmax(y_val_probs, axis=1)
    val_accuracy = np.mean(y_val_pred == y_val)
    logger.info(f"Final validation accuracy: {val_accuracy:.4f}")

    report = classification_report(y_val, y_val_pred, zero_division=0)
    logger.info(f"Classification Report:\n{report}")

    # ROC AUC and PR AUC (one-vs-rest)
    classes = np.arange(len(cfg.SIGNAL_COUNTS))
    y_val_bin = label_binarize(y_val, classes=classes)
    present_classes = [c for c in classes if c in y_val]
    if len(present_classes) > 1:
        roc_auc = roc_auc_score(y_val_bin, y_val_probs, average='macro', multi_class='ovr')
        pr_auc = average_precision_score(y_val_bin, y_val_probs, average='macro')
        logger.info(f"ROC AUC (macro OVR): {roc_auc:.4f}")
        logger.info(f"PR AUC (macro): {pr_auc:.4f}")
    else:
        roc_auc = None
        pr_auc = None
        logger.warning("Not enough classes in validation set for ROC/PR AUC")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title("Count Model Confusion Matrix (Validation)")
    ax.set_xlabel("Predicted Count")
    ax.set_ylabel("True Count")
    ax.set_xticks(classes)
    ax.set_yticks(classes)
    plt.colorbar(im, ax=ax)
    for i in range(len(cfg.SIGNAL_COUNTS)):
        for j in range(len(cfg.SIGNAL_COUNTS)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.DIR_TRAINING_PLOTS, "count_model_confusion_matrix.png"))
    plt.close()
    logger.info(f"Saved confusion matrix to '{cfg.DIR_TRAINING_PLOTS}/count_model_confusion_matrix.png'")

    return {
        'accuracy': float(val_accuracy),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'pr_auc': float(pr_auc) if pr_auc is not None else None,
    }


if __name__ == "__main__":
    main()
