import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ----------------------------
# Load training data
# ----------------------------
data = np.load("ml_training_data/training_data_counts.npz")
X = data["waveforms"]
y = data["labels"]

print(f"✅ Loaded dataset: {X.shape[0]} samples, {X.shape[1]} time bins")
print(f"✅ Signal count labels shape: {y.shape}, Min: {y.min()}, Max: {y.max()}")

# ----------------------------
# Normalize waveforms
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Split train/val
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------------
# Build shallow MLP model
# ----------------------------
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(np.max(y) + 1, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# Callbacks
# ----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# ----------------------------
# Train the model
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# ----------------------------
# Save model
# ----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/signal_count_model.keras")
print("✅ Model saved to models/signal_count_model.keras")

# ----------------------------
# Plot training history (with smoothing)
# ----------------------------
os.makedirs("training_plots", exist_ok=True)

def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(smooth_curve(history.history["loss"]), label="Train Loss (Smooth)")
plt.plot(smooth_curve(history.history["val_loss"]), label="Val Loss (Smooth)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(smooth_curve(history.history["accuracy"]), label="Train Acc (Smooth)")
plt.plot(smooth_curve(history.history["val_accuracy"]), label="Val Acc (Smooth)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig("training_plots/count_model_training.png")
plt.show()


# 📈 Plot generalization gap
val_acc = history.history['val_accuracy']
train_acc = history.history['accuracy']
gap = np.array(val_acc) - np.array(train_acc)

plt.figure(figsize=(6, 4))
plt.plot(gap, label="Validation - Training Accuracy")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Accuracy Gap")
plt.title("Generalization Gap Over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("training_plots/generalization_gap.png")
plt.show()