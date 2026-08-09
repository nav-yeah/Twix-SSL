import torch.nn as nn
import torchvision.models as models


class BarlowTwinsModel(nn.Module):

    def __init__(self, projection_dim=128, hidden_dim=512):
        super().__init__()

        resnet = models.resnet18(weights=None)
        self.feature_dim = resnet.fc.in_features  
        resnet.fc = nn.Identity()  #removing the classification layer
        self.encoder = resnet

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x):
        h = self.encoder(x)     
        z = self.projector(h)  
        return z


if __name__ == "__main__":
    import torch
    model = BarlowTwinsModel()
    x = torch.randn(4, 3, 96, 96)
    z = model(x)
    print("Output shape:", z.shape)  