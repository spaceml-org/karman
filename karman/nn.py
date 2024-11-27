import torch
from torch import nn
import torch.nn.init as init

class SimpleNetwork(nn.Module):
    """Generic fully connected MLP with adjustable depth."""

    def __init__(
        self,
        input_dim,
        hidden_layer_dims=[256, 256, 256],
        output_dim=1,
        act=nn.LeakyReLU(negative_slope=0.01),
        dropout=0.0,
    ):
        """
        Args:
            - input_dim (int): input dimension
            - hid_dims (List[int]): list of hidden layer dimensions
            - output_dim (int): output dimension
            - non_linearity (str): type of non-linearity in hidden layers
            - dropout (float): dropout rate (applied each layer)
        """
        super(SimpleNetwork, self).__init__()

        dims = [input_dim] + hidden_layer_dims

        self.dropout = nn.Dropout(p=dropout)
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.act = act
        self.acts = nn.ModuleList([self.act for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, act in zip(self.fcs, self.acts):
            x = act(fc(self.dropout(x)))
        # non activated final layer
        return self.fc_out(x)

class LSTMModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size=1, 
                 lstm_size=190, 
                 lstm_depth=2, 
                 dropout=0.0):
        super(LSTMModel,self).__init__()
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
        # Reset the hidden state -> Not needed, it's done by default
        #h0 = torch.zeros(self.lstm_depth, x.size(0), self.lstm_size).to(x.device)
        #c0 = torch.zeros(self.lstm_depth, x.size(0), self.lstm_size).to(x.device)
        x, _ = self.lstm(x)#, (h0, c0))
        # Just use the final item in the sequence.
        x = self.fc1(x[:, -1,:])
        return x

class LSTMDensityPredictor(nn.Module):
    def __init__(self,
                 input_size_static,
                 input_size_timedependent,
                 ffnn_hidden_layer_dims=[256, 256, 256],
                 lstm_depth=2,
                 lstm_size=190,
                 output_size_timedependent=20,
                 dropout_lstm=0.,
                 dropout_ffnn=0.):
        super(LSTMDensityPredictor,self).__init__()
        self.model_lstm_omni=LSTMModel(input_size=input_size_timedependent, 
                                       output_size=output_size_timedependent, 
                                       dropout=dropout_lstm, 
                                       lstm_size=lstm_size,
                                       lstm_depth=lstm_depth)
        self.ffnn=SimpleNetwork(input_dim=input_size_static+output_size_timedependent, 
                                       hidden_layer_dims=ffnn_hidden_layer_dims,
                                       output_dim=1,
                                       dropout=dropout_ffnn)

    def forward(self, batch):
        #let's pass the time dependent features through an LSTM
        timedependent_features = self.model_lstm_omni(batch['historical_ts_numeric'])
        #and concatenate the output with the static features and pass it through a feedforward neural network:
        concatenated_features = torch.cat([batch['static_feats_numeric'],timedependent_features], dim=1)
        density = self.ffnn(concatenated_features)
        return density


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                bias.data[: n // 3].fill_(-1.0)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
