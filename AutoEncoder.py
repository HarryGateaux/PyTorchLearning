from torch import nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):

    def __init__(self, inSz, latentSz = 128):

        super().__init__()

        self.encoder = nn.Sequential(
        nn.Linear(inSz, 128),
        nn.ReLU(True),
        nn.Linear(128, latentSz),
        nn.ReLU(True)
        )


        self.decoder = nn.Sequential(
        nn.Linear(latentSz, 128),
        nn.ReLU(True),
        nn.Linear(128, inSz),
        nn.ReLU(True)
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x