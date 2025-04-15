import torch

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1
    img_size = 224
    batch_size = 32
    num_workers = 0
    pin_memory = True
    epochs = 5
    lr = 1e-4
    k_folds = 5
    seed = 42

    train_path = './dataset/train_small'
    val_path = './dataset/val_small'
    test_path = './dataset/test_small'