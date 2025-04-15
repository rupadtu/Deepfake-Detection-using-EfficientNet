from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from models.efficientnet_model import EfficientNetModel
from utils.data_loader import DeepfakeDataset
from config import Config
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def test_model():
    transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = DeepfakeDataset(Config.test_path, transform=transform)
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)

    model = EfficientNetModel()
    model.load_state_dict(torch.load('best_model_fold0.pth'))
    model = model.to(Config.device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(Config.device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            y_pred.extend(preds.flatten().tolist())
            y_true.extend(labels.int().numpy().tolist())

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix on Test Data")
    plt.show()