import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.loss = nn.L1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        layer1 = F.leaky_relu(self.fc1(data))
        layer2 = F.leaky_relu(self.fc2(layer1))
        layer3 = self.fc3(layer2)

        return layer3
