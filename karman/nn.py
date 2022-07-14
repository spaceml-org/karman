from torch import nn

class FFNN(nn.Module):
    def __init__(self, num_features):
        super(FFNN, self).__init__()
        self.name = 'Three Layer FFNN'
        self.num_features = num_features
        self.fc1 = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.fc1(x)
        return x
