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
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


def main(epochs=40, batch_size=128, test_size=0.2):
    # -----------------------------
    # Load Dataset
    # -----------------------------
    data = np.load("ml_training_data/training_data_counts.npz")
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
    os.makedirs("training_plots", exist_ok=True)
    joblib.dump(scaler, "training_plots/count_waveform_scaler.pkl")

    # -----------------------------
    # Build Model (Model = CONV, Input = waveform, Output = signal count class)
    # -----------------------------

    input_layer = layers.Input(shape=(X.shape[1], 1), name="waveform_input")

    # Two Conv1D Layers
    x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(input_layer)
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)

    # Flatten preserves all learned features
    x = layers.Flatten()(x)

    # Dense Block
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output for classification (0 to 6)
    output = layers.Dense(7, activation='softmax', name="count_output")(x)

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
        X_scaled, y, test_size=test_size, random_state=42
    )

    # -----------------------------
    # Callbacks
    # -----------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
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
            filepath='best_count_model.keras',
            monitor='val_accuracy',
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
    os.makedirs("training_plots", exist_ok=True)
    model.save("signal_count_model.keras")
    pd.DataFrame(history.history).to_csv("training_plots/signal_count_model_history.csv", index=False)

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
    plt.savefig("training_plots/signal_count_model_training.png")
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
    classes = np.arange(7)
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
    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig("training_plots/count_model_confusion_matrix.png")
    plt.close()
    logger.info("Saved confusion matrix to 'training_plots/count_model_confusion_matrix.png'")

    return {
        'accuracy': float(val_accuracy),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'pr_auc': float(pr_auc) if pr_auc is not None else None,
    }


if __name__ == "__main__":
    main()
