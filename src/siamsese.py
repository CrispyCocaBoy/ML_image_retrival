import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet50"):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models
        self.backbone = models.__dict__[backbone](weights=ResNet50_Weights.DEFAULT, progress=True)

        # Freeze the backbone network
        for param in self.backbone_full.parameters():
            param.requires_grad = False

        # Get the number of features that are outputted by the last layer of backbone network.
        out_features = self.backbone.fc.in_features

        # Rimuovi la classificazione finale
        self.backbone = nn.Sequential(
            *list(self.backbone.children())[:-1],
            nn.Flatten()
        )
        
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
        # Riferimento ma non implementato
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5)
        )

        # Start to modify the model
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),
            nn.Dropout(p=0.5)
        )


        # Create an embedding layer to get the similarity between the two images
        self.embedding = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        

    def forward(self, img1, img2):
        '''
        Returns the similarity value between two images.

            Parameters:
                    img1 (torch.Tensor): shape=[b, 3, 224, 224]
                    img2 (torch.Tensor): shape=[b, 3, 224, 224]

            where b = batch size

            Returns:
                    output (torch.Tensor): shape=[b, 1], Similarity of each pair of images
        '''

        # Pass the both images through the backbone network to get their seperate feature vectors
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        feat1 = self.head(feat1)
        feat2 = self.head(feat2)

        feat1 = self.embedding(feat1)
        feat2 = self.embedding(feat2)

        
        return feat1, feat2
    



