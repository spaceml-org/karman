import torch
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

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, output_size=10, lstm_size=512, lstm_depth=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_size = lstm_size
        self.lstm_depth = lstm_depth
        self.dropout = dropout
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_size, num_layers=lstm_depth, batch_first=True, dropout=dropout if dropout else 0)
        self.fc1 = nn.Linear(lstm_size, self.output_size)
        if dropout is not None:
            self.dropout1 = nn.Dropout(p=dropout)
    def forward(self, x):
        batch_size=x.size(0)
#        print(x.shape)
        x, _ = self.lstm(x)
#        print(x.shape)
        if self.dropout:
            x = self.dropout1(x)
        x = torch.relu(x)
#        print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = x.view(batch_size, -1)
        return x[:,-self.output_size:]
    
class DensityPredictor(nn.Module):
    def __init__(self, 
                 input_size_thermo,
                 input_size_fism2_flare, 
                 input_size_fism2_daily,
                 input_size_omni,
                 output_size_fism2_flare=10,
                 output_size_fism2_daily=10,
                 output_size_omni=10,
                 dropout=0.):
        super().__init__()
        self.model_lstm_fism2_flare=LSTMPredictor(input_size=input_size_fism2_flare, output_size=output_size_fism2_flare, dropout=dropout)
        self.model_lstm_fism2_daily=LSTMPredictor(input_size=input_size_fism2_daily, output_size=output_size_fism2_daily, dropout=dropout)
        self.model_lstm_omni=LSTMPredictor(input_size=input_size_omni, output_size=output_size_omni, lstm_size=128, dropout=dropout)
        self.ffnn=FFNN(num_features=input_size_thermo+output_size_fism2_flare+output_size_fism2_daily+output_size_omni)
    def forward(self, x):
        x=torch.cat((x[0], self.model_lstm_fism2_flare(x[1]), self.model_lstm_fism2_daily(x[2]), self.model_lstm_omni(x[3])),axis=1)
        x=self.ffnn(x)
        return x