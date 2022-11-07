import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.mean, self.std = None, None
        self.max, self.min = None, None

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        x = self.normalize(x)
        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        predictions = self.inverse(predictions)
        return predictions[:, -1]

    def normalize(self, x):

        b, t, f = x.size()
        x = x.reshape(b, -1)
        self.mean = torch.mean(x, dim=1, keepdim=True)
        self.std = torch.std(x, dim=1, keepdim=True)
        # print(self.mean.size())
        norm_x = (x - self.mean)/ self.std
        norm_x = norm_x.reshape(b, t, f)
        return norm_x

        # b, t, f = x.size()
        # x = x.reshape(b, -1)
        # self.max, _ = torch.max(x, dim=1, keepdim=True)
        # self.min, _ = torch.min(x, dim=1, keepdim=True)
        # # # print(self.mean.size())
        # # norm_x = (x - self.mean)/ self.std
        # norm_x = (x - self.min) / (self.max - self.min)
        # norm_x = norm_x.reshape(b, t, f)
        # return norm_x


    def inverse(self, x):
        x = x * self.std + self.mean
        # x = x * (self.max - self.min) + self.min
        return x