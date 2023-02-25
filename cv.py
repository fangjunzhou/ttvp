import torch

import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
from torchvision.utils import save_image
import time


class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        # self.chosen_features = [0, 5, 10, 19, 28]
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT).features[:36]

    def forward(self, x):
        # features = []
        # for i, layer in enumerate(self.model):
        #     x = layer(x)
        #     if i in self.chosen_features:
        #         features.append(x)
        return self.model.forward(x)


class ContentToViews(nn.Module):
    def __init__(self):
        super(ContentToViews, self).__init__()
        self.vgg = VGGFeatures() # the output shape is (1, 512, 14, 14)
        # This is the part that we need to train
        # The input is the output of VGG
        # The output is a scalar value
        self.other_layers = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )


    def forward(self, input):
        """
        The output should be a scalar value,
        The input should be the same as VGG input
        """
        out = self.vgg(input)
        out = out.view(out.size(0), -1)
        out = self.other_layers(out)
        return out



def main():
    print(torch.__version__)


if __name__ == '__main__':
    main()
