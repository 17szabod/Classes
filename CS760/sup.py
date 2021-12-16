# Implements supervised learning methods on images from Ken's report
# Got some ideas from https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid


def collect_data(data_dir):
    # load the train and test data
    dataset = ImageFolder(data_dir, transform=Compose([
        Resize((100, 100)), ToTensor()
    ]))
    return dataset


def show_batch(dl):
    """Plot images grid of single batch"""
    images = []
    labs = []
    for img, labels in dl:
        images += img
        labs += labels
    fig, ax = plt.subplots(figsize = (16,12))
    ax.imshow(make_grid(images,nrow=int(len(images)/4)).permute(1,2,0))
    plt.title("Dataset of Districting Plans")
    plt.show()


class CNN(torch.nn.Module):
    def __init__(self, rand):
        super(CNN, self).__init__()
        self.network = torch.nn.Sequential(
            # take pixel channels and expand
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # In general, we like small kernels because we're trying to learn based on small boundaries
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            # Try learning larger kernels to capture some different information, worst case can be trained to identity
            torch.nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            # Switch to linear
            torch.nn.Flatten(),
            torch.nn.Linear(2*2592, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return float(test_loss)


if __name__ == '__main__':
    batch_size = 4
    epochs = 50
    model = CNN(rand=True)
    dataset = collect_data("data/images/district_pictures")
    train_data, test_data = random_split(dataset, [int(np.ceil(len(dataset)*.85)), int(np.floor(len(dataset)*.15))])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    # Visualize the data
    show_batch(train_dataloader)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    ys = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        loss = test(test_dataloader, model, loss_fn)
        ys.append(loss)
    plt.plot(ys, 'b-')
    plt.title("Average Training Loss as a Function of Epochs")
    plt.show()
    print("Done!")
    exit()

