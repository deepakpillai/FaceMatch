import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import vgg16

class FaceMatchCNNModel(nn.Module):
    def __init__(self, hidden_unites):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_unites, kernel_size=(3, 3), stride=1),
            # nn.ReLU(),
            nn.Conv2d(in_channels=hidden_unites, out_channels=hidden_unites, kernel_size=(3, 3), stride=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_unites, out_channels=hidden_unites, kernel_size=(3, 3), stride=1),
            # nn.ReLU(),
            nn.Conv2d(in_channels=hidden_unites, out_channels=hidden_unites, kernel_size=(3, 3), stride=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_unites, out_channels=hidden_unites, kernel_size=(3, 3), stride=1),
            # nn.ReLU(),
            nn.Conv2d(in_channels=hidden_unites, out_channels=1, kernel_size=(3, 3), stride=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.dense_layer = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(in_features=289, out_features=3)
        )

    def forward(self, x: torch.Tensor):
        first_layer = self.conv_layer_1(x)
        second_layer = self.conv_layer_2(first_layer)
        third_layer = self.conv_layer_3(second_layer)
        print(third_layer.shape)
        dense_layer =  self.dense_layer(third_layer)
        print("dense_layer")
        print(dense_layer.shape)
        return dense_layer


class VGG_Model():
    def __init__(self):
        pass

    def load_model(self):
        model = vgg16(pretrained=True)
        model = nn.Sequential(*list(model.children()))[:-4]
        return model

    def pre_process_image(self, data):
        image_transforms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor()
        ])

        pre_processed_data = image_transforms(data)
        return pre_processed_data.unsqueeze(0)

    def get_embedding(self, data):
        model = self.load_model()
        with torch.inference_mode():
            embedding = model(data)
            # print(f"embedding {embedding.shape}")
            flatten = nn.Flatten()
            embedding = flatten(embedding)
            # print(f"Flatten {embedding.shape}")
            return embedding


class FaceEmbedding():
    def __init__(self):
        pass

    def get_face_embedding(self, x: torch.Tensor):
        with torch.inference_mode():
            # torch.manual_seed(3)
            embed = VGG_Model().get_embedding(x)
            return embed
