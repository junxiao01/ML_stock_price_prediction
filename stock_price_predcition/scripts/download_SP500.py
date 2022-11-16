import pandas as pd
import os
import yfinance
import argparse
from tqdm import tqdm

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--start_date", type=str, default="2006-01-01", help="the start date of stocks you would like to exrtact")
    parser.add_argument('-e', "--end_date", type=str, default="2022-11-01", help="the end date of stocks you would like to extract")
    parser.add_argument("--save_path", type=str, default="/Volumes/My_Passport/Master_of_Finance/SP500_data", help="the path to save the stock files")

    args = parser.parse_args()

    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    symbols = sorted(df['Symbol'].tolist())
    symbols.remove('BF.B')
    symbols.remove('BRK.B')

    start_date = args.start_date
    end_date = args.end_date
    save_path = args.save_path

    for symbol in tqdm(symbols):
        download_stock(symbol, start_date=start_date, end_date=end_date, save_root=save_path)

def check_file():
    file_path = '/Volumes/My_Passport/Master_of_Finance/SP500_data/snp500.csv'
    f = pd.read_csv(file_path)
    print(f.head(5))

if __name__ == '__main__':
    main()
    # check_file()