import torch.nn as nn
import timm
from config import Config

class EfficientNetModel(nn.Module):
    def __init__(self):
        super(EfficientNetModel, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(in_features, Config.num_classes)

    def forward(self, x):
        return self.base_model(x)