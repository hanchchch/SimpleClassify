import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, 32) 
        self.fc2 = nn.Linear(32, 16)  
        self.fc3 = nn.Linear(16, 10)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out

loss_fn = nn.CrossEntropyLoss().to(device)