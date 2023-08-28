import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from utils import dropnull

def excel_overview(excel_path):
    ex_df=pd.read_csv(excel_path,header=0,index_col=0)
    print(ex_df.head())
    print(ex_df.iloc[0])
    print(ex_df.columns)
    print(ex_df.index)
    print(ex_df.shape)
    return ex_df

def train_split(excel_path,saveDir):
    ex_df=excel_overview(excel_path)
    ex_df=dropnull(ex_df)
    # ex_df.fillna(0,inplace=True)
    Path(saveDir).mkdir(exist_ok=True,parents=True)
    save_list=["train","valid","test"]
    savedic=defaultdict(list)
    # for savei in save_list:
    #     savedic[savei]=defaultdict(list)
    for _,item in ex_df.iterrows():
        value=np.random.rand()
        item_list=item.to_list()
        new_list=[item_list[0]]+item_list[2:]+[item_list[1]]
        if value<0.8:
            savedic["train"].append(new_list)
        elif value>=0.8 and value<0.9:
            savedic["valid"].append(new_list)
        else:
            savedic["test"].append(new_list)
    for key,value in savedic.items():
        saveJpath=str(Path(saveDir).joinpath(key+".json"))
        with open(saveJpath,"w") as wr:
            json.dump(value,wr,indent=1,ensure_ascii=False)
def invert_scale(scaler, X, value):
    # print(X)
    # print(X.shape,type(X))
    new_row = [x for x in X] + [value]
    new_row=np.hstack((X,value))
    array = np.array(new_row)
    # array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    print(inverted.shape)
    return inverted[:,-1]

def scale(train, test):
    # 根据训练数据建立缩放器
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # 转换train data
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # 转换test data
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

def get_data(json_path):
    with open(json_path,"r") as wr:
        data=json.load(wr)
    return data

# def invert_scale_sg(scaler,y):
#     array = np.array(y)
#     # array = array.reshape(1, len(array))
#     inverted = scaler.inverse_transform(array)
#     return inverted
if __name__=="__main__":
    excel_path=r"D:\python_code\LSTM-master\bond_data\train.csv"
    # excel_overview(excel_path)
    train_split(excel_path,
                saveDir=r"D:\python_code\LSTM-master\model_bond\bond_trdataNonull")
    # valid_path=r"D:\python_code\LSTM-master\model_bond\bond_trdata\valid.json"
    # # from lstm_test import get_data
    # valid_data=get_data(valid_path)
    # tt1,tt2=np.array(valid_data[0:5]),np.array(valid_data[6:8])
    # # print(tt1)
    # # print(tt2)
    # scalett,tt1_scale,tt2_scale=scale(tt1,tt2)
    # tt1_x=tt1_scale[:,0:-1]
    # tt1_y=tt1_scale[:,-1].reshape(-1,1)
    # print(tt1_x.shape,tt1_y.shape)
    # # print(tt1_scale)
    # # print("---------------------------")
    # # print(tt1_y)
    # aa=invert_scale(scalett,tt1_x,tt1_y)
    # print(aa)
    # print(tt1[:,-1])
    # tt1_yi=invert_scale_sg(scalett,tt1_y)
    # print(tt1_yi)