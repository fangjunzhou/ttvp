import torch

import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
from torchvision.utils import save_image
import time


class VFFFeatures:
    def __init__(self):
        super(VFFFeatures, self).__init__()
        # self.chosen_features = [0, 5, 10, 19, 28]
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT).features[:37]

    def forward(self, x):
        # features = []
        # for i, layer in enumerate(self.model):
        #     x = layer(x)
        #     if i in self.chosen_features:
        #         features.append(x)
        return self.model.forward(x)


def main():
    print(torch.__version__)


if __name__ == '__main__':
    main()
