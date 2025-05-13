import os
import cv2
import numpy as np
from glob import glob

def load_images_masks(image_dir, mask_dir, img_size=(256, 256)):
    images = []
    masks = []
    img_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

    for img_path, msk_path in zip(img_paths, mask_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, img_size)
        msk = cv2.resize(msk, img_size)

        img = img.astype(np.float32) / 255.0
        msk = (msk > 127).astype(np.float32)

        images.append(img[..., np.newaxis])
        masks.append(msk[..., np.newaxis])

    X = np.array(images)
    y = np.array(masks)
    return X, y

if __name__ == "__main__":
    # Example usage
    X, y = load_images_masks(
        image_dir="/path/to/images", mask_dir="/path/to/masks"
    )
    print(f"Loaded {len(X)} images and {len(y)} masks")