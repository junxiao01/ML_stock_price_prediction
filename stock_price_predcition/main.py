import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data.dataset import SingleStockDataset
from torch.utils.data import DataLoader
from models.arch.LSTMModel_arch import LSTMModel

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0, keepdims=True)
        self.sd = np.std(x, axis=0, keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

def prepare_data_x(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output

def main():

    # read_data
    stock_data = pd.read_csv('./dataset/MSFT.csv')

    # extract the information of adjust close price
    stock_adj_close_price = stock_data['Adj Close'].to_numpy()
    # stock_adj_close_price = stock_data['Adj Close'].pct_change().to_numpy()[1:]
    data_date = stock_data['Date'].tolist()

    plt.figure(figsize=(25,5))
    plt.plot(data_date, stock_adj_close_price)
    plt.xticks(np.arange(0, len(data_date), 90), rotation='vertical')
    plt.savefig('./results/MSFT_price.png')

    ## Data normalization
    # Normal_transform = Normalizer()
    # normalized_price = Normal_transform.fit_transform(stock_adj_close_price)
    normalized_price = stock_adj_close_price

    ## prepare data and its corresponding label
    window_size = 5
    data_x, data_unseen =  prepare_data_x(normalized_price, window_size=window_size)
    data_y = prepare_data_y(normalized_price, window_size)

    #### Split dataset

    train_split_size = 0.8
    split_range = int(data_y.shape[0] * train_split_size)

    data_x_train = data_x[:split_range]
    data_y_train = data_y[:split_range]

    data_x_val = data_x[split_range:]
    data_y_val = data_y[split_range:]

    dataset_train = SingleStockDataset(data_x_train, data_y_train)
    dataset_test = SingleStockDataset(data_x_val, data_y_val)

    train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=False)

    device = 'cuda'
    model = LSTMModel(input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2)
    model.to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    num_epoch = 120
    best_loss = float('inf')
    for epoch in range(num_epoch):

        model.train()
        train_loss = 0.0
        for idx, (x, y) in enumerate(train_dataloader):

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.contiguous(), y.contiguous())
            loss.backward()
            optimizer.step()

            train_loss += (loss.detach().item() / x.shape[0])

        if (epoch + 1) % 20 == 0:
            model.eval()
            val_loss = 0.0
            for idx, (x, y) in enumerate(val_dataloader):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out.contiguous(), y.contiguous())

            val_loss += (loss.detach().item() / x.shape[0])
            if val_loss < best_loss:
                torch.save(model.state_dict(), './experiments/best.pt')
            lr = scheduler.get_last_lr()[0]
            print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
                  .format(epoch + 1, num_epoch, train_loss, val_loss, lr))

        scheduler.step()

    model = LSTMModel(input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2)
    model.load_state_dict(torch.load('./experiments/best.pt'))
    model.to(device)
    model.eval()
    pred_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(device)
        out = model(x)
        out = out.cpu().detach().numpy()
        pred_val = np.concatenate((pred_val, out))

    # to_plot_y_gt = Normal_transform.inverse_transform(data_y_val)
    # to_plot_y_pred = Normal_transform.inverse_transform(pred_val)

    to_plot_y_gt = data_y_val
    to_plot_y_pred = pred_val
    to_plot_date = data_date[split_range+window_size:]

    plt.figure(figsize=(25, 5))
    plt.plot(to_plot_date, to_plot_y_gt, label='Ground_truth')
    plt.plot(to_plot_date, to_plot_y_pred, label='predicted data')
    plt.xticks(np.arange(0, len(to_plot_date), 20), rotation='vertical')
    plt.legend()
    plt.savefig('./results/pred_MSFT_results.png')


if __name__ == '__main__':
    main()