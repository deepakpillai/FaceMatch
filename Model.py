import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt

class FaceMatchCNNModel(nn.Module):
    def __init__(self, hidden_unites):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_unites, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_unites, out_channels=hidden_unites, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_unites, out_channels=hidden_unites, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_unites, out_channels=hidden_unites, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_unites, out_channels=hidden_unites, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_unites, out_channels=3, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.dense_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=289, out_features=3)
        )

    def forward(self, x: torch.Tensor):
        first_layer = self.conv_layer_1(x)
        second_layer = self.conv_layer_2(first_layer)
        third_layer = self.conv_layer_3(second_layer)
        print(third_layer.shape)
        return self.dense_layer(third_layer)


class FaceEmbedding():
    def __init__(self):
        pass

    def get_face_embedding(self, x: torch.Tensor):
        with torch.inference_mode():
            torch.manual_seed(3)
            model = FaceMatchCNNModel(64)
            return model(x)
