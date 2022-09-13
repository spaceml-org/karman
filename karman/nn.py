import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, dropout=0., hidden_size=500, out_features=100):
        super(FeedForward, self).__init__()
        self.name = 'FeedForward'
        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.LazyLinear(hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),

            nn.LazyLinear(hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),

            nn.LazyLinear(hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),

            nn.LazyLinear(hidden_size),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),

            nn.LazyLinear(out_features),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class FullFeatureFeedForward(nn.Module):
    def __init__(self, dropout=0.0, hidden_size=200, out_features=50):
        super(FullFeatureFeedForward, self).__init__()
        self.dropout = dropout
        self.name = 'Full Feature Feed Forward'
        self.fc_thermo = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.fc_omni = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.fc_fism2_daily = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.fc_fism2_flare = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.regressor = FeedForward(hidden_size=hidden_size, out_features=1)

    def forward(self, x):
        thermo_features = self.fc_thermo(x['instantaneous_features'])
        omni_features = self.fc_omni(x['omni'])
        fism2_daily_features = self.fc_fism2_daily(x['fism2_daily_stan_bands'])
        fism2_flare_features = self.fc_fism2_flare(x['fism2_flare_stan_bands'])
        concatenated_features = torch.cat([
            thermo_features,
            omni_features,
            fism2_daily_features,
            fism2_flare_features
        ], dim=1)
        return self.regressor(concatenated_features)

class NoFism2FlareAndDailyFeedForward(nn.Module):
    def __init__(self, dropout=0.0, hidden_size=200, out_features=50):
        super(NoFism2FlareAndDailyFeedForward, self).__init__()
        self.dropout = dropout
        self.name = 'Full Feature Feed Forward'
        self.fc_thermo = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.fc_omni = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.regressor = FeedForward(hidden_size=hidden_size, out_features=1)

    def forward(self, x):
        thermo_features = self.fc_thermo(x['instantaneous_features'])
        omni_features = self.fc_omni(x['omni'])
        concatenated_features = torch.cat([
            thermo_features,
            omni_features
        ], dim=1)
        return self.regressor(concatenated_features)

class NoFism2FlareAndDailyAndOmniFeedForward(nn.Module):
    def __init__(self, dropout=0.0, hidden_size=200, out_features=50):
        super(NoFism2FlareAndDailyFeedForward, self).__init__()
        self.dropout = dropout
        self.name = 'No FISM2 Flare, Daily and No Omni Feed Forward'
        self.fc_thermo = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.regressor = FeedForward(hidden_size=hidden_size, out_features=1)

    def forward(self, x):
        thermo_features = self.fc_thermo(x['instantaneous_features'])
        return self.regressor(thermo_features)


class NoFism2FlareFeedForward(nn.Module):
    def __init__(self, dropout=0.0, hidden_size=200, out_features=50):
        super(NoFism2FlareFeedForward, self).__init__()
        self.dropout = dropout
        self.name = 'Full Feature Feed Forward'
        self.fc_thermo = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.fc_omni = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.fc_fism2_daily = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.regressor = FeedForward(hidden_size=hidden_size, out_features=1)

    def forward(self, x):
        thermo_features = self.fc_thermo(x['instantaneous_features'])
        omni_features = self.fc_omni(x['omni'])
        fism2_daily_features = self.fc_fism2_daily(x['fism2_daily_stan_bands'])
        concatenated_features = torch.cat([
            thermo_features,
            omni_features,
            fism2_daily_features
        ], dim=1)
        return self.regressor(concatenated_features)

class NoFism2DailyFeedForward(nn.Module):
    def __init__(self, dropout=0.0, hidden_size=200, out_features=50):
        super(NoFism2DailyFeedForward, self).__init__()
        self.dropout = dropout
        self.name = 'Full Feature Feed Forward'
        self.fc_thermo = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.fc_omni = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.fc_fism2_flare = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.regressor = FeedForward(hidden_size=hidden_size, out_features=1)

    def forward(self, x):
        thermo_features = self.fc_thermo(x['instantaneous_features'])
        omni_features = self.fc_omni(x['omni'])
        fism2_flare_features = self.fc_fism2_flare(x['fism2_flare_stan_bands'])
        concatenated_features = torch.cat([
            thermo_features,
            omni_features,
            fism2_flare_features
        ], dim=1)
        return self.regressor(concatenated_features)

class NoOmniFeedForward(nn.Module):
    def __init__(self, dropout=0.0, hidden_size=200, out_features=50):
        super(NoOmniFeedForward, self).__init__()
        self.dropout = dropout
        self.name = 'Full Feature Feed Forward'
        self.fc_thermo = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.fc_fism2_daily = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.fc_fism2_flare = FeedForward(dropout=dropout, hidden_size=hidden_size, out_features=out_features)
        self.regressor = FeedForward(hidden_size=hidden_size, out_features=1)

    def forward(self, x):
        thermo_features = self.fc_thermo(x['instantaneous_features'])
        fism2_daily_features = self.fc_fism2_daily(x['fism2_daily_stan_bands'])
        fism2_flare_features = self.fc_fism2_flare(x['fism2_flare_stan_bands'])
        concatenated_features = torch.cat([
            thermo_features,
            fism2_daily_features,
            fism2_flare_features
        ], dim=1)
        return self.regressor(concatenated_features)

class FFNN(nn.Module):
    def __init__(self, num_features, dropout=0.):
        super(FFNN, self).__init__()
        self.name = 'Four Layer FFNN'
        self.num_features = num_features
        # This is for backwards compatibility reasons.
        # Some previous models dont expect a dropout layer
        # and so the code breaks if a model isnt expecting
        # then when loaded up at a later date.
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
    def __init__(self, input_size, output_size=10, lstm_size=190, lstm_depth=2, dropout=0.2):
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

class WindowCNN(nn.Module):
    def __init__(self, outsize=100):
        super().__init__()
        self.outsize = outsize

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 5, 3, stride=1),
            nn.Conv2d(5, 5, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(5, 5, 3, stride=1),
            nn.Conv2d(5, 5, 3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.fc = nn.LazyLinear(self.outsize)

    def forward(self, x):
        x = x.unsqueeze(1)
        cnn_features = self.cnn(x)
        return self.fc(cnn_features)

class OmniDensityPredictor(nn.Module):
    def __init__(self,
                 input_size_thermo,
                 input_size_omni,
                 lstm_depth=2,
                 output_size_omni=20,
                 dropout_lstm=0.,
                 dropout_ffnn=0.):
        super().__init__()
        self.model_lstm_omni=LSTMPredictor(input_size=input_size_omni, output_size=output_size_omni, dropout=dropout_lstm, lstm_depth=lstm_depth)
        self.ffnn=FFNN(num_features=input_size_thermo+output_size_omni, dropout=dropout_ffnn)
        self.dropout=dropout_lstm

    def forward(self, batch):
        omni_features = self.model_lstm_omni(batch['omni'])

        concatenated_features = torch.cat([
            batch['instantaneous_features'],
            omni_features
        ], dim=1)
        density = self.ffnn(concatenated_features)
        return density

class Fism2DailyDensityPredictor(nn.Module):
    def __init__(self,
                 input_size_thermo,
                 input_size_fism2_daily,
                 output_size_fism2_daily=20,
                 lstm_depth=2,
                 dropout_lstm=0.,
                 dropout_ffnn=0.):
        super().__init__()
        self.model_lstm_fism2_daily=LSTMPredictor(input_size=input_size_fism2_daily, output_size=output_size_fism2_daily, lstm_depth=lstm_depth, dropout=dropout_lstm)
        self.ffnn=FFNN(num_features=input_size_thermo+output_size_fism2_daily, dropout=dropout_ffnn)
        self.dropout=dropout_lstm

    def forward(self, batch):
        flare_daily_features = self.model_lstm_fism2_daily(batch['fism2_daily_stan_bands'])

        concatenated_features = torch.cat([
            batch['instantaneous_features'],
            flare_daily_features,
        ], dim=1)
        density = self.ffnn(concatenated_features)
        return density

class Fism2FlareDensityPredictor(nn.Module):
    def __init__(self,
                 input_size_thermo,
                 input_size_fism2_flare,
                 output_size_fism2_flare=20,
                 lstm_depth=2,
                 dropout_lstm=0.,
                 dropout_ffnn=0.):
        super().__init__()
        self.model_lstm_fism2_flare=LSTMPredictor(input_size=input_size_fism2_flare, output_size=output_size_fism2_flare, lstm_depth=lstm_depth, dropout=dropout_lstm)
        self.ffnn=FFNN(num_features=input_size_thermo+output_size_fism2_flare, dropout=dropout_ffnn)
        self.dropout=dropout_lstm

    def forward(self, batch):
        flare_features = self.model_lstm_fism2_flare(batch['fism2_daily_stan_bands'])

        concatenated_features = torch.cat([
            batch['instantaneous_features'],
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
        flare_features = self.model_lstm_fism2_flare(batch['fism2_flare_stan_bands'])
        omni_features = self.model_lstm_omni(batch['omni'])
        flare_daily_features = self.model_lstm_fism2_daily(batch['fism2_daily_stan_bands'])

        concatenated_features = torch.cat([
            batch['instantaneous_features'],
            flare_features,
            flare_daily_features,
            omni_features
        ], dim=1)
        density = self.ffnn(concatenated_features)
        return density


class CNNDensityPredictor(nn.Module):
    def __init__(self,
                 input_size_thermo,
                 cnn_output_size):
        super().__init__()
        self.fism2_flare_cnn=WindowCNN(cnn_output_size)
        self.fism2_daily_cnn=WindowCNN(cnn_output_size)
        self.omni_cnn=WindowCNN(cnn_output_size)
        self.ffnn=FFNN(num_features=input_size_thermo+3*cnn_output_size)

    def forward(self, batch):
        flare_features = self.fism2_flare_cnn(batch['fism2_flare_stan_bands'])
        omni_features = self.omni_cnn(batch['omni'])
        flare_daily_features = self.fism2_daily_cnn(batch['fism2_daily_stan_bands'])

        concatenated_features = torch.cat([
            batch['instantaneous_features'],
            flare_features,
            flare_daily_features,
            omni_features
        ], dim=1)
        density = self.ffnn(concatenated_features)
        return density

class OneGiantFeedForward(nn.Module):
    def __init__(self, dropout=0.0, hidden_size=200, out_features=50):
        super(OneGiantFeedForward, self).__init__()
        self.dropout = dropout
        self.regressor = FeedForward(hidden_size=hidden_size, out_features=1)

    def forward(self, x):
        thermo_data = x['instantaneous_features']
        omni_data = x['omni']
        fism2_daily_data = x['fism2_daily_stan_bands']
        fism2_flare_data = x['fism2_flare_stan_bands']
        concatenated_data = torch.cat([
            thermo_data,
            omni_data,
            fism2_daily_data,
            fism2_flare_data
        ], dim=1)
        return self.regressor(concatenated_data)
