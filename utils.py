from __future__ import unicode_literals
import json
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path

def get_data(json_path):
    with open(json_path,"r",encoding="utf-8-sig") as wr:
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
    with open(saveJpath,"w",encoding="utf-8-sig") as wr:
        json.dump(dic,wr,indent=1,ensure_ascii=False)
        
def dropnull(df):
    ndf=df.dropna()
    return ndf

def read_excel(excel_path):
    ex_df=pd.read_excel(excel_path,header=0)
    return ex_df

def read_csv(excel_path):
    ex_df=pd.read_csv(excel_path,header=0,index_col=0)
    return ex_df

def csv_save(save_pd,save_path):
    save_pd.to_csv(save_path,encoding="utf_8_sig")
    
def read_excel_batch(excel_dir):
    saveLs=[]
    for excelpath in Path(excel_dir).glob("*.xlsx"):
        excel_pd=read_excel(str(excelpath))
        saveLs.append(excel_pd)
    return saveLs

def read_csv_batch(excel_dir):
    saveLs=[]
    for excelpath in Path(excel_dir).glob("*.csv"):
        excel_pd=read_csv(str(excelpath))
        saveLs.append(excel_pd)
    return saveLs

def column_combine(pdlist:list(),style="left"):
    combine1=pd.merge(pdlist[0],pdlist[1],how=style,on=["债券ID","日期"])
    return combine1