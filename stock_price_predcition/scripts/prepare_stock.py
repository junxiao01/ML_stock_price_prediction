import yfinance
import yfinance as yf
import os

def download_stock(symbol, start_date='2010-01-01', end_date='2022-11-01', save_root='../dataset'):

    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    symbol = symbol
    save_path = save_root + '/' + symbol + '.csv'
    start_date = start_date
    end_date = end_date
    stock = yfinance.download(symbol, start=start_date, end=end_date, interval='1d')
    stock.to_csv(save_path)

    return None


if __name__ == '__main__':
    download_stock('INTC')
