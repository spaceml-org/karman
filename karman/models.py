from torch import nn
import torch

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
        super(NoFism2FlareFeedForward, self).__init__()
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
