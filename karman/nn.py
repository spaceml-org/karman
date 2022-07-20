import torch
from torch import nn

class FFNN(nn.Module):
    def __init__(self, num_features):
        super(FFNN, self).__init__()
        self.name = 'Four Layer FFNN'
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

class FeedForwardDensityPredictor(nn.Module):
    def __init__(self, num_features):
        super(FeedForwardDensityPredictor, self).__init__()
        self.ffnn = FFNN(num_features)

    def forward(self, batch):
        x = self.ffnn(batch['static_features'])
        return x


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, output_size=10, lstm_size=100, lstm_depth=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_size = lstm_size
        self.lstm_depth = lstm_depth
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.lstm_size,
            num_layers=lstm_depth,
            batch_first=True,
            dropout=dropout
        )
        self.fc1 = nn.Linear(lstm_size, self.output_size)
        self.dropout1 = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size=x.size(0)
        x, _ = self.lstm(x)
        x = self.dropout1(x[:, -1,:])
        x = torch.relu(x)
        x = self.fc1(x)
        x = x.view(batch_size, -1)
        return x[:,-self.output_size:]

class FullFeatureDensityPredictor(nn.Module):
    def __init__(self,
                 input_size_thermo,
                 input_size_fism2_flare,
                 input_size_fism2_daily,
                 input_size_omni,
                 output_size_fism2_flare=20,
                 output_size_fism2_daily=20,
                 output_size_omni=20,
                 dropout=0.):
        super().__init__()
        self.model_lstm_fism2_flare=LSTMPredictor(input_size=input_size_fism2_flare, output_size=output_size_fism2_flare, dropout=dropout)
        self.model_lstm_fism2_daily=LSTMPredictor(input_size=input_size_fism2_daily, output_size=output_size_fism2_daily, dropout=dropout)
        self.model_lstm_omni=LSTMPredictor(input_size=input_size_omni, output_size=output_size_omni, dropout=dropout)
        self.ffnn=FFNN(num_features=input_size_thermo+output_size_fism2_flare+output_size_fism2_daily+output_size_omni)

    def forward(self, batch):
        flare_features = self.model_lstm_fism2_flare(batch['fism2_flare'])
        omni_features = self.model_lstm_omni(batch['omni'])
        flare_daily_features = self.model_lstm_fism2_daily(batch['fism2_daily'])

        concatenated_features = torch.cat([
            batch['static_features'],
            flare_features,
            flare_daily_features,
            omni_features
        ], dim=1)
        density = self.ffnn(concatenated_features)
        return density
