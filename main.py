from train import train_model
from test import test_model
from utils.dataset_utils import downsample_dataset

if __name__ == "__main__":
    train_model()
    downsample_dataset('./dataset/test', './dataset/test_small', keep_per_class=2500)
    test_model()