import json
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

def get_data(json_path):
    with open(json_path,"r") as wr:
        data=json.load(wr)
    return data

def invert_scale(scaler, X):
    X_ar = np.array(X)
    # print(len(X_ar.shape))
    if len(X_ar.shape)<2:
        X_ar=X_ar.reshape(-1,1)
    # array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(X_ar)
    # print(inverted.shape)
    return inverted

def get_pkl(pkl_path):
    with open(pkl_path,"rb") as rd:
        target=pickle.load(rd)
    return target

def scaler_trans(train):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    if len(train.shape)<2:
        train=train.reshape(-1,1)
    scaler = scaler.fit(train)
    # 转换train data
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    return train_scaled,scaler

def jsonSave(saveJpath,dic):
    with open(saveJpath,"w") as wr:
        json.dump(dic,wr,indent=1,ensure_ascii=False)
        
def dropnull(df):
    ndf=df.dropna()
    return ndf