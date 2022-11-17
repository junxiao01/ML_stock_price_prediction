import pandas as pd
import numpy as np
import glob
import os

class Dataset():
    def __init__(self,
                 folder_root,
                 train_period='2020-01-01--2021-10-31',
                 val_period='2021-11-01--2021-12-31',
                 test_period='2022-01-01--2022-10-31'):

        train_time = train_period.split('--')
        val_time = val_period.split('--')
        test_time = test_period.split('--')

        self.train_start, self.train_end = train_time[0], train_time[1]
        self.val_start, self.val_end = val_time[0], val_time[1]
        self.test_start, self.test_end = test_time[0], test_time[1]

        self.folder_root = folder_root

    def read_csv(self, path):

        # read the dataset .csv
        data = pd.read_csv(path)

        # clean 

    def traveral_files(self, folder_roots):

        csv_files = glob.glob(os.path.join(self.folder_root, '*.csv'))
        return csv_files
    def get_data(self):

        csv_files = self.traveral_files(self.folder_root)
        for csv_path in csv_files:
            raw_csv = self.read_csv(csv_path)

            csv_factor = self.compute_factor(csv)

    def compute_factor(self, data):
        '''
        :param data: panda file, index, Date, Open, High, Low, Close, Adj Close, Volume
        :return:
        '''







