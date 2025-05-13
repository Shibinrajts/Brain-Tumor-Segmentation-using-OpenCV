import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_preprocessing import load_images_masks
from model import unet_model

# Paths - adjust these to match your folders
TRAIN_IMG_DIR = "images"
TRAIN_MASK_DIR = "masks"

if __name__ == "__main__":
    # Load data
    X, y = load_images_masks(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    print(f"Loaded data shapes: X={X.shape}, y={y.shape}")

    # Ensure correct shape
    if X.ndim != 4 or y.ndim != 4:
        X = np.stack(X, axis=0)
        y = np.stack(y, axis=0)
        print(f"Stacked arrays: X={X.shape}, y={y.shape}")

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train/Val shapes: X_train={X_train.shape}, X_val={X_val.shape}")

    # Build model
    model = unet_model(input_size=X.shape[1:])
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = ModelCheckpoint(
        "checkpoints/unet_best.h5", save_best_only=True, verbose=1
    )
    early_stop = EarlyStopping(patience=10, verbose=1)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=[checkpoint, early_stop]
    )
    model.save("unet_final.h5")