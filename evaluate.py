import torch
import torchvision
import os
import numpy as np
from AutoEncoder import AutoEncoder
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("latent", type=int, help = "number of dimensions in latent space", default= 20)

args = parser.parse_args()

if __name__ == "__main__":

    model = AutoEncoder(28 * 28,  args.latent)
    model.load_state_dict(torch.load("ae.pth"))
    model.eval()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )

    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=40, shuffle=True, num_workers=4, pin_memory=True
    )

    batch, _ = next(iter(train_loader))

    print(batch.shape)
    grid_img = torchvision.utils.make_grid(batch, nrow=10)

    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig("inputs.png")


    outputs = model(batch.view(-1, 28 * 28))
    latents = model.encoder(batch.view(-1, 28 * 28))

    inp = torch.tensor([2, 8, 8,  2,  0], dtype = torch.float32)
    inp = inp.view(1, 5)
    
    test = model.decoder(inp)
    test = test.reshape(28, 28)

    plt.imshow(test.detach().numpy())
    plt.savefig("test.png")
    outputs = outputs.reshape(40, 1, 28, 28)
    grid_img = torchvision.utils.make_grid(outputs, nrow=10)
    plt.imshow(grid_img.permute(1, 2, 0).detach().numpy())
    plt.savefig("outputs.png")