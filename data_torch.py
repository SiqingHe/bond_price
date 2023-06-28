import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from utils import get_data,invert_scale,get_pkl
import pickle

class mydataset(Dataset):
    def __init__(self,data,targets):
        self.data=data
        self.targets=targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        x=self.data[index]
        y=self.targets[index]
        return x,y


if __name__=="__main__":
    # X_train_path=r"D:\python_code\LSTM-master\model_bond\timedata\x_train.json"
    # y_train_path=r"D:\python_code\LSTM-master\model_bond\timedata\y_train.json"
    # X_test_path=r"D:\python_code\LSTM-master\model_bond\timedata\x_test.json"
    # y_test_path=r"D:\python_code\LSTM-master\model_bond\timedata\y_test.json"
    # batch_size=100
    # # X_test=np.array(get_data(X_train_path))
    # # y_test=np.array(get_data(y_train_path))
    # X_tr_list=get_data(X_train_path)
    # y_tr_list=get_data(y_train_path)
    # X_train=np.array(X_tr_list)
    # y_train=np.array(y_tr_list)
    
    X_tr_path=r"D:\python_code\LSTM-master\model_bond\timedata\x_train.pkl"
    y_tr_path=r"D:\python_code\LSTM-master\model_bond\timedata\y_train.pkl"
    X_tr_list=get_pkl(X_tr_path)
    y_tr_list=get_pkl(y_tr_path)
    X_train=np.array(X_tr_list)
    y_train=np.array(y_tr_list)
    # with open(X_tr_save,"wb") as wr:
    #     pickle.dump(X_tr_list,wr)
    # with open(y_tr_save,"wb") as wry:
    #     pickle.dump(y_tr_list,wry)
    # print(X_train.shape)
    # print(X_train[0:10,:,:])
    # print(y_train.shape)
    # print(y_train[0:10])
    
    # train_lstm(model_path,X_train,y_train,batch_size,nb_epoch=1, neurons=5)
    pkl_path=r"D:\python_code\LSTM-master\model_bond\timedata\scale_y.pkl"
    with open(pkl_path,"rb") as rd:
        scaley=pickle.load(rd)
    y_train=invert_scale(scaley,y_train)
    dataset=mydataset(X_train,y_train)
    datald=DataLoader(dataset,batch_size=5,shuffle=False)
    for idx,data in datald:
        # X,y=data
        print(idx)
        print("X"*20)
        print(data)
        # print(X)
        # print("y"*20)
        # print(y)
        print("*"*20)