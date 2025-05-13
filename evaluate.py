import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import load_images_masks

def dice_coeff(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

if __name__ == "__main__":
    # Evaluate on the same 'images' and 'mask' folders or on separate test folders
    X_test, y_test = load_images_masks("images", "mask")
    model = load_model("checkpoints/unet_best.h5")
    preds = model.predict(X_test)
    dices = [dice_coeff(y_test[i], (preds[i] > 0.5).astype(np.float32)) for i in range(len(preds))]
    print(f"Mean Dice Coefficient: {np.mean(dices):.4f}")
import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import load_images_masks

def dice_coeff(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

if __name__ == "__main__":
    X_test, y_test = load_images_masks("data/test/images", "data/test/masks")
    model = load_model("checkpoints/unet_best.h5")
    preds = model.predict(X_test)
    dices = [dice_coeff(y_test[i], (preds[i] > 0.5).astype(np.float32)) for i in range(len(preds))]
    print(f"Mean Dice Coefficient: {np.mean(dices):.4f}")

