import os
import numpy as np
import pandas as pd
import torch
import model.timefeatures




class StandardScaler():
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def load_st_dataset(dataset,args):

    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]
    elif dataset == 'HJ':
        data_path = os.path.join('../data/ETT/HJ.csv')
        df_raw = np.array(pd.read_csv(data_path))
        df_raw = df_raw[:, :]
        num_row = df_raw.shape[0]
        lose_num_row = num_row*0.5


        df_raw_tmp = pd.read_csv(data_path)
        num_train = int(len(df_raw)*args.train_ratio)

        indices = torch.randperm(num_train)[:int(lose_num_row)]

        num_test = int(len(df_raw) * args.test_ratio)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, num_train + num_vali + num_test]
        scaler = StandardScaler()
        df_data = np.array(df_raw)[0:4000, 1:].astype('float32')
        train_data = df_data[border1s[0]:border2s[0]]
        scaler.fit(train_data)
        data = scaler.transform(df_data)
        data = sigmoid(data)



    return data,scaler
