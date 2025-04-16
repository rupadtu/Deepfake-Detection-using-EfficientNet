# Deepfake Detection using EfficientNet

This project implements a deep learning model using EfficientNet to detect deepfake images. The approach involves training a binary classification model using EfficientNet-B0, a lightweight and powerful convolutional neural network. The dataset consists of real and fake images, and the model learns to distinguish between them using transfer learning, data augmentation, and cross-validation techniques. The training pipeline includes optimization strategies such as AdamW optimizer, learning rate scheduling, and gradient clipping to improve convergence and generalization.

![Validation Accuracy](assets/validation%20accuracy.png)

---

## 📌 Project Overview
This project leverages the power of the EfficientNet architecture to detect Deepfake images with high precision and recall. Deepfakes pose a significant threat to the authenticity of digital media, and detecting them effectively is a modern challenge in computer vision.

We trained and evaluated our model using a well-structured and stratified pipeline, applying advanced augmentation and optimization strategies. Our solution addresses key challenges such as dataset imbalance, model overfitting, and generalization on unseen data.

---

## ✅ Key Highlights
- **Model Architecture:** EfficientNet-B0, fine-tuned for binary classification (Fake vs Real).
- **High Performance:**
  - 🔹 Final **Validation Accuracy**: `94.23%`
  - 🔹 Final **Training Accuracy**: `99.24%`
  - 🔹 Final **Training Loss**: `0.0257`
  - 🔹 Test Accuracy: `92.5%`
- **Balanced Dataset Handling:** Stratified K-Fold Cross-Validation (k=5) to ensure robust generalization.
- **Augmentation & Normalization:** Ensures the model learns key facial patterns rather than dataset-specific artifacts.

---

## 🧠 How It Works

1. **Dataset Preparation:**
   - Organized in `Fake` and `Real` folders.
   - Downsampled and preprocessed with resizing and normalization.

2. **Modeling:**
   - Based on `EfficientNet-B0` from the `timm` library.
   - Last classifier layer modified for binary output.

3. **Training:**
   - BCEWithLogitsLoss as the loss function.
   - Optimizer: Adam with learning rate `1e-4`.
   - Stratified 5-Fold Cross-Validation.
   - **Model Checkpointing:** Saved best model weights for each fold in `saved_models/`:
     - `best_model_fold0.pth`
     - `best_model_fold1.pth`
     - `best_model_fold2.pth`
     - `best_model_fold3.pth`
     - `best_model_fold4.pth`

4. **Evaluation:**
   - Classification Report & Confusion Matrix used to assess generalization.

---

## 📊 Results

### 📉 Confusion Matrix
![Confusion Matrix](assets/confusion%20matrix.png)

---

## 🛠️ Tech Stack
- Python 3.10+
- PyTorch & torchvision
- `timm` (for EfficientNet)
- scikit-learn
- Matplotlib & PIL
- tqdm

---

## 📂 Project Structure
```
deepfake_detection/
├── dataset/
│   ├── train/
│   ├── validation/
│   ├── test/
├── saved_models/
│   ├── best_model_fold0.pth
│   ├── best_model_fold1.pth
│   ├── best_model_fold2.pth
│   ├── best_model_fold3.pth
│   └── best_model_fold4.pth
├── models/
│   └── efficientnet_model.py
├── utils/
│   ├── data_loader.py
│   └── dataset_utils.py
├── train.py
├── test.py
├── config.py
└── main.py
```

---

## 🎯 Why This Project Matters?
- **Demonstrates Real-World Application of CNNs** on a trending, high-impact problem.
- Shows capability to handle **imbalanced datasets**, **fine-tune state-of-the-art models**, and apply **model evaluation metrics** professionally.
- Implements **robust training pipelines**, clean code architecture, and visual performance reporting.
- Includes **multi-fold training** and **model checkpointing**, reflecting best practices in MLOps.

---

## 📌 Future Work
- Incorporating attention mechanisms (e.g., CBAM)
- Extend to video-based deepfake detection
- Ensemble multiple EfficientNet variants for further accuracy boosts

---

## 🙌 Acknowledgements
Inspired by ongoing research in Deepfake detection and visual forensics

