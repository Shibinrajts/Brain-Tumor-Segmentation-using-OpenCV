
# Brain Tumor Segmentation using Computer Vision

**Team**  
- Shibin Raj T S  
- Sherin Raj S S  
- Vaishakh V S  
- Abhijith B A  

---

## 🚀 Project Overview

This repository contains a complete pipeline for **segmenting brain tumors** in MRI scans using a U-Net–based deep learning model and a Streamlit web interface. Given an input MRI slice, the app predicts a probability mask of the tumor region, converts it into a binary mask, and overlays the segmentation on the original image for easy visualization and download.

---

## 📋 Requirements

All Python dependencies are listed in [`requirements.txt`](./requirements.txt). To install:

```bash
pip install -r requirements.txt
🛠 How It Works
Data Loading & Preprocessing

Grayscale MRI images and corresponding ground-truth masks are resized to a fixed resolution (256×256), normalized, and stacked into NumPy arrays.

Model Architecture

A standard U-Net encoder–decoder with skip connections implemented in TensorFlow/Keras.

Binary cross-entropy loss + Adam optimizer.

Training

Data is split into train/validation sets (default 80/20).

Best model checkpoint (unet_best.h5) is saved via ModelCheckpoint; early stopping halts training on plateau.

Final weights saved as unet_final.h5.

Inference & Streamlit App

Upload a new MRI slice via the web UI.

Model predicts a probability mask (float values 0–1) which you can visualize.

Adjustable threshold slider converts probabilities into a binary mask (0 or 255).

Optional morphological post-processing and contour extraction highlight the largest connected tumor region.

Overlay of segmentation on the original image, plus download buttons for masks.

Dataset Brain Tumor MRI Dataset (https://drive.google.com/file/d/14bNmRNEldTI8QTJC8njMA2t4hE9DldyS/view)

💻 Installation & Setup

# 1. Clone repository
git clone https://github.com/<your-username>/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.\.venv\Scripts\activate       # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare data
#    - Place all MRI scans in an `images/` folder
#    - Place corresponding ground truth masks in a `mask/` folder
#    - Filenames must align one-to-one

# 5. Train your model (produces checkpoints/unet_best.h5 & unet_final.h5)
python train.py

# 6. Launch the Streamlit demo
streamlit run app.py
▶️ Typical Usage
Train:
python train.py
Evaluate (optional):
python evaluate.py
Run Demo:
streamlit run app.py
In Browser

Upload an MRI image file (PNG/JPG).

(Optional) Upload a custom .h5 model.

Adjust the “Mask threshold” slider to capture faint tumors.

View/download the raw probability map, binary mask, and overlay.

📂 Repository Structure
bash
Copy
Edit
brain-tumor-segmentation/
├── images/               # MRI input images
├── mask/                 # Ground-truth masks
├── checkpoints/          # Saved model weights
├── data_preprocessing.py
├── model.py
├── train.py
├── evaluate.py
├── inference.py
├── app.py
├── requirements.txt
└── README.md
🤝 Contributions
Feel free to open issues or submit pull requests for improvements, new architectures, or additional post-processing steps!

🔒 License This project is for educational and research purposes only. Not for clinical use without medical approval.
---

## `requirements.txt`

```txt
tensorflow>=2.11
opencv-python
streamlit
numpy
scikit-learn
Pillow
h5py

