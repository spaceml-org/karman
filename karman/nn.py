import torch
from torch import nn

class FFNN(nn.Module):
    def __init__(self, num_features, dropout=0.):
        super(FFNN, self).__init__()
        self.name = 'Four Layer FFNN'
        self.num_features = num_features
        self.fc1 = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.fc1(x)
        return x

class FeedForwardDensityPredictor(nn.Module):
    def __init__(self, num_features, dropout=0.):
        super(FeedForwardDensityPredictor, self).__init__()
        self.ffnn = FFNN(num_features, dropout)

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

    def forward(self, x):
        # Reset the hidden state
        h0 = torch.zeros(self.lstm_depth, x.size(0), self.lstm_size).to(x.device)
        c0 = torch.zeros(self.lstm_depth, x.size(0), self.lstm_size).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        # Just use the final item in the sequence.
        x = self.fc1(x[:, -1,:])
        return x

class OmniDensityPredictor(nn.Module):
    def __init__(self,
                 input_size_thermo,
                 input_size_omni,
                 output_size_omni=20,
                 dropout_lstm=0.,
                 dropout_ffnn=0.):
        super().__init__()
        self.model_lstm_omni=LSTMPredictor(input_size=input_size_omni, output_size=output_size_omni, dropout=dropout_lstm)
        self.ffnn=FFNN(num_features=input_size_thermo+output_size_omni, dropout=dropout_ffnn)

    def forward(self, batch):
        omni_features = self.model_lstm_omni(batch['omni'])

        concatenated_features = torch.cat([
            batch['static_features'],
            omni_features
        ], dim=1)
        density = self.ffnn(concatenated_features)
        return density

class Fism2DailyDensityPredictor(nn.Module):
    def __init__(self,
                 input_size_thermo,
                 input_size_fism2_daily,
                 output_size_fism2_daily=20,
                 dropout_lstm=0.,
                 dropout_ffnn=0.):
        super().__init__()
        self.model_lstm_fism2_daily=LSTMPredictor(input_size=input_size_fism2_daily, output_size=output_size_fism2_daily, dropout=dropout_lstm)
        self.ffnn=FFNN(num_features=input_size_thermo+output_size_fism2_daily, dropout=dropout_ffnn)

    def forward(self, batch):
        flare_daily_features = self.model_lstm_fism2_daily(batch['fism2_daily'])

        concatenated_features = torch.cat([
            batch['static_features'],
            flare_daily_features,
        ], dim=1)
        density = self.ffnn(concatenated_features)
        return density

class Fism2FlareDensityPredictor(nn.Module):
    def __init__(self,
                 input_size_thermo,
                 input_size_fism2_flare,
                 output_size_fism2_flare=20,
                 dropout_lstm=0.,
                 dropout_ffnn=0.):
        super().__init__()
        self.model_lstm_fism2_flare=LSTMPredictor(input_size=input_size_fism2_flare, output_size=output_size_fism2_flare, dropout=dropout_lstm)
        self.ffnn=FFNN(num_features=input_size_thermo+output_size_fism2_flare, dropout=dropout_ffnn)

    def forward(self, batch):
        flare_features = self.model_lstm_fism2_flare(batch['fism2_flare'])

        concatenated_features = torch.cat([
            batch['static_features'],
            flare_features,
        ], dim=1)
        density = self.ffnn(concatenated_features)
        return density

class FullFeatureDensityPredictor(nn.Module):
    def __init__(self,
                 input_size_thermo,
                 input_size_fism2_flare,
                 input_size_fism2_daily,
                 input_size_omni,
                 output_size_fism2_flare=20,
                 output_size_fism2_daily=20,
                 output_size_omni=20,
                 dropout_lstm=0.,
                 dropout_ffnn=0.):
        super().__init__()
        self.model_lstm_fism2_flare=LSTMPredictor(input_size=input_size_fism2_flare, output_size=output_size_fism2_flare, dropout=dropout_lstm)
        self.model_lstm_fism2_daily=LSTMPredictor(input_size=input_size_fism2_daily, output_size=output_size_fism2_daily, dropout=dropout_lstm)
        self.model_lstm_omni=LSTMPredictor(input_size=input_size_omni, output_size=output_size_omni, dropout=dropout_lstm)
        self.ffnn=FFNN(num_features=input_size_thermo+output_size_fism2_flare+output_size_fism2_daily+output_size_omni, dropout=dropout_ffnn)

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
