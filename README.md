# Deepfake Detection using EfficientNet

This project implements a deep learning model using EfficientNet to detect deepfake images. The approach involves training a binary classification model using EfficientNet-B0, a lightweight and powerful convolutional neural network. The dataset consists of real and fake images, and the model learns to distinguish between them using transfer learning, data augmentation, and cross-validation techniques. The training pipeline includes optimization strategies such as AdamW optimizer, learning rate scheduling, and gradient clipping to improve convergence and generalization.

## Dataset Structure
The dataset should be structured as follows:
```
/train
    /real
        img1.jpg
        img2.jpg
    /fake
        img3.jpg
        img4.jpg
/test
    img5.jpg
    img6.jpg
```

## Installation
Clone the repository and install dependencies:
```sh
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

## Training the Model
To train the model, run:
```sh
python main.py
```
This will perform 5-fold cross-validation and save the best model weights.

## Testing the Model
To apply the model to test data, run:
```sh
python test.py
```
This generates a `submission.csv` file with predictions.

## File Descriptions
- `config.py` – Stores model configurations.
- `dataset.py` – Handles dataset loading.
- `train_utils.py` – Contains training and validation functions.
- `model.py` – Defines the EfficientNet model.
- `main.py` – Trains the model.
- `test.py` – Runs inference on test images.

## Notes
- Ensure your dataset is properly structured.
- Modify `config.py` if needed.

For any issues, feel free to contribute or raise an issue!

