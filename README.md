# Deepfake-Detection-using-EfficientNet
# Deepfake Detection with EfficientNet

This project implements a deep learning model using EfficientNet to detect deepfake images. It includes data preprocessing, training, and testing scripts.

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

