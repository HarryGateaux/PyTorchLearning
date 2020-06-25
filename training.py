import torch
from torch import optim, nn
import torchvision
import os
from AutoEncoder import AutoEncoder
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("latent", type=int, help = "number of dimensions in latent space", default= 20)
parser.add_argument("epochs", type = int, help = "number of epochs", default= 20)

args = parser.parse_args()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae = AutoEncoder(28 * 28, args.latent).to(device)

    optimizer = optim.Adam(ae.parameters(), lr=1e-3)

    criterion = nn.MSELoss()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    epochs = args.epochs

    for i in range(epochs):
        loss = 0

        for data, _ in train_loader:
            
            data = data.view(-1, 784).to(device)

            optimizer.zero_grad()

            outputs = ae(data)
            
            train_loss = criterion(outputs, data)

            train_loss.backward()
            
            optimizer.step()

            loss += train_loss.item()

        loss /= len(train_loader)

        print(f"epoch : {i} , loss : {loss}")

    path = os.getcwd()
    torch.save(ae.state_dict(), path + "/ae.pth")


