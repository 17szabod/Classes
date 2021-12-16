import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose


class SimpleNN(torch.nn.Module):
    def __init__(self, rand):
        super(SimpleNN, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.L1 = torch.nn.Linear(28*28, 300)
        torch.nn.init.uniform(self.L1.weight.data, a=-1, b=1) if rand else torch.nn.init.constant(self.L1.weight.data, val=0)
        torch.nn.init.uniform(self.L1.bias.data, a=-1, b=1) if rand else torch.nn.init.constant(self.L1.bias.data, val=0)
        self.sig = torch.nn.Sigmoid()
        self.L2 = torch.nn.Linear(300, 10)
        torch.nn.init.uniform(self.L2.weight.data, a=-1, b=1) if rand else torch.nn.init.constant(self.L2.weight.data, val=0)
        torch.nn.init.uniform(self.L2.bias.data, a=-1, b=1) if rand else torch.nn.init.constant(self.L2.bias.data, val=0)
        self.smax = torch.nn.Softmax()

    def forward(self, x):
        x = self.flatten(x)
        x = self.sig(self.L1(x))
        x = self.smax(self.L2(x))
        return x


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

        if batch % 100 == 0:
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
    return 100*correct


# Derivative of the sigmoid function
def dsigmoid(x):
    s = sigmoid(x)
    return s*(1-s)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0, dtype=x.dtype)


# derivative of the cross entropy loss of the softmax function (much nicer than just dsoftmax)
# https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
def d_l_softmax(x, y):
    return softmax(x) - y


def sigmoid(x):
    return 1/(1+np.exp(-x))


# Used these sources as well as lecture notes:
# https://ljvmiranda921.github.io/notebook/2017/02/17/artificial-neural-networks/#implementation
# https://towardsdatascience.com/coding-a-2-layer-neural-network-from-scratch-in-python-4dd022d19fd2
class MyNN():
    def __init__(self, lr):
        self.W1 = np.random.randn(28*28, 300)
        self.b1 = np.random.randn(300, )
        self.W2 = np.random.randn(300, 10)
        self.b2 = np.random.randn(10, )

        self.sig = sigmoid
        self.smax = softmax

        self.lr = lr
        self.grads = {}
        self.cache = {}
        self.loss = []

    def Forward(self, x):
        x = x.flatten()
        self.cache['A0'] = x
        # a = np.matmul(self.W1.T, x)
        x = np.matmul(self.W1.T, x) + self.b1
        self.cache['L1'] = x  # Layer 1 pre-activation
        x = self.sig(x)
        self.cache['A1'] = x  # Layer 1 activation
        x = np.matmul(self.W2.T, x) + self.b2
        self.cache['L2'] = x  # Layer 2 pre-activation
        y = self.smax(x)
        self.cache['A2'] = x  # Layer 2 activation
        return y

    def Backward(self, y):
        dLoss_L2 = d_l_softmax(self.cache['A2'], y)
        dLoss_A1 = self.W2@dLoss_L2
        dLoss_W2 = np.outer(dLoss_L2, self.cache['A1'])
        dLoss_b2 = dLoss_L2  #np.sum(dLoss_L2, axis=1, keepdims=True)

        dLoss_L1 = dLoss_A1 * dsigmoid(self.cache['L1'])
        dLoss_W1 = np.outer(dLoss_L1, self.cache['A0'])
        dLoss_b1 = dLoss_L1  #np.sum(dLoss_L1, axis=1, keepdims=True)
        return np.asanyarray([dLoss_W1, dLoss_b1, dLoss_W2, dLoss_b2], dtype=object)

    def Loss(self, y, yhat):
        return -np.sum(y*np.log(yhat))

    # For a dataloader D, performs SGD
    def Train(self, D):
        ct = 0
        for X_batch, y_batch in D:
            dL = []
            losses = []
            for i in range(X_batch.shape[0]):
                X, y = np.asarray(X_batch[i]), y_batch[i]
                y = np.eye(10)[y]  # change y into a 1-hot vector
                yhat = self.Forward(X)
                losses.append(self.Loss(y, yhat))
                dL = self.Backward(y) if i == 0 else dL + self.Backward(y)
            self.W1 = self.W1 - self.lr * np.mean(dL[0], axis=-1)
            self.b1 = self.b1 - self.lr * np.mean(dL[1], axis=-1)
            self.W2 = self.W2 - self.lr * np.mean(dL[2], axis=-1)
            self.b2 = self.b2 - self.lr * np.mean(dL[3], axis=-1)
            self.loss.append(np.mean(losses))
            # print("Loss: {0}  [{1}/{2}]".format(self.loss[-1], ct*y_batch.shape[0], y_batch.shape[0]*len(D)))
            if ct > len(D):
                break
            ct += 1
        print("Average training loss: {0}".format(np.mean(self.loss)))
        self.loss = []
        return np.mean(self.loss)

    def Test(self, D):
        losses = []
        cor = 0
        ct = 0
        for X_batch, y_batch in D:
            for i in range(len(X_batch)):
                X, y1 = np.asarray(X_batch[i]), y_batch[i]
                y = np.eye(10)[y1]  # change y into a 1-hot vector
                yhat = self.Forward(X)
                losses.append(self.Loss(y, yhat))
                cor += 1 if y1 == np.argmax(yhat) else 0
            if ct > 20:
                break
            ct += 1
        print("Accuracy: {0}, Average loss on test set: {1}".format(cor/(D.batch_size*20), np.mean(losses)))
        return cor/(D.batch_size*20), np.mean(losses)


if __name__ == '__main__':
    model = SimpleNN(rand=True)
    train_mnist = torchvision.datasets.MNIST('/home/daniel/PycharmProjects/CS760ML/data/', download=True, train=True, transform=ToTensor())
    test_mnist = torchvision.datasets.MNIST('/home/daniel/PycharmProjects/CS760ML/data/', download=True, train=False, transform=ToTensor())
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    batch_size = 128
    # Create data loaders.
    train_dataloader = DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_mnist, batch_size=batch_size, shuffle=True)
    # for X, y in test_dataloader:
    #     print("Shape of X [N, C, H, W]: ", X.shape)
    #     print("Shape of y: ", y.shape, y.dtype)
    #     break
    epochs = 5
    xs = [0,]
    ys = [0.1,]
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy = test(test_dataloader, model, loss_fn)
        xs.append(xs[-1] + 6000)
        ys.append(accuracy)
    plt.plot(xs, ys, 'b-')
    print("Done!")
    model = SimpleNN(rand=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    xs = [0,]
    ys = [0.1,]
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy = test(test_dataloader, model, loss_fn)
        xs.append(xs[-1] + 6000)
        ys.append(accuracy)
    plt.plot(xs, ys, 'r-')
    plt.title("Pytorch Neural Net with Random Initialization vs Zero Initialization")
    plt.legend(["Uniform random init", "Zeros init"])
    # myNN = MyNN(.1)
    # epochs = 5
    # xs = [0,]
    # ys = [0.1,]
    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train_loss = myNN.Train(train_dataloader)
    #     accuracy, test_loss = myNN.Test(test_dataloader)
    #     xs.append(xs[-1] + 6000)
    #     ys.append(accuracy*100)
    # plt.plot(xs, ys, 'r-')
    plt.show()

