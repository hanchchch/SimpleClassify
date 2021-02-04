import torch
import torch.nn as nn
from torch import optim
from sklearn.datasets import load_digits

from random import randint

from model import NeuralNet, loss_fn, device


digits = load_digits()

X = torch.tensor(digits['data'], dtype=torch.float32).to(device)
Y = torch.tensor(digits['target'], dtype=torch.int64).to(device)

model = NeuralNet()
optimizer = optim.Adam(model.parameters())

i = 100
for epoch in range(i):
    optimizer.zero_grad()
    y_predict = model(X)
    loss = loss_fn(y_predict, Y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, i, loss.item()))
