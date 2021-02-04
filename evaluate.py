import torch
from sklearn.datasets import load_digits
from random import randint
from train import model
from model import device
import matplotlib.pyplot as plt

digits = load_digits()

for _ in range(10):
    with torch.no_grad():
        X = torch.tensor(digits['data'], dtype = torch.float32).to(device)
        Y = torch.tensor(digits['target'], dtype = torch.int64).to(device)
        
        hypothesis = model(X)
        prediction = torch.argmax(hypothesis, 1)
        correct_prediction = prediction == Y
        accuracy = correct_prediction.float().mean()
        print('Accuracy: ', accuracy.item())
        
        r = randint(0, X.shape[0]-1)
        X_single_data = X[r][:].float().to(device)
        Y_single_data = Y[r].to(device)
        
        print('Label: ', Y_single_data.item())
        single_prediction = model(X_single_data)
        print('Prediction: ', torch.argmax(single_prediction, dim = 0).item())
        
        plt.imshow(X_single_data.view(8, 8), cmap = 'Greys', interpolation = 'nearest')
        plt.show()
