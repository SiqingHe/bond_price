from __future__ import unicode_literals
import json
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pathlib import Path
import os
import datetime
import logging
import time

def json_read(json_path,encode="utf-8-sig"):
    with open(json_path,"r",encoding=encode) as wr:
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

def read_excel(excel_path,header=None,index_col=None):
    ex_df=pd.read_excel(excel_path,header=header,index_col=index_col)
    return ex_df

def read_csv(excel_path,header=None,index_col=None):
    ex_df=pd.read_csv(excel_path,header=header,index_col=index_col)
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
    if isinstance(excel_dir,list):
        iter_obj=excel_dir
    elif Path(excel_dir).is_dir():
        iter_obj=Path(excel_dir).glob("*.csv")
    else:raise("unexpected input type")
    for excelpath in iter_obj:
        excel_pd=read_csv(str(excelpath),header=0,index_col=0)
        saveLs.append(excel_pd)
    return saveLs

table_dic={".csv":pd.read_csv,".xlsx":pd.read_excel}

def read_table_iter(excel_dir,suffix,header=None,index_col=None):
    for excelpath in Path(excel_dir).glob("*".format(suffix)):
        excel_pd=table_dic[suffix](str(excelpath),header=header,index_col=index_col)
        yield excel_pd

def column_combine(pdlist:list(),style="left"):
    combine1=pd.merge(pdlist[0],pdlist[1],how=style,left_on=["债券简称","日期"],right_on=["债券简称",'日期'])
    return combine1

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def tm_format_trans(tt):
    tr = datetime.datetime.strptime(str(tt),"%Y%m%d") 
    return tr.strftime("%Y-%m-%d")

def timestamp2date(tic,mode="%Y%m%d"):
    timeArray = time.localtime(tic)
    date = time.strftime(mode,timeArray)
    return date

def configSave(cfg,saveDir):
    ticn=datetime.datetime.now().strftime('%y.%m.%d.%H.%M.%S')
    basePath=Path(datetime.datetime.now().strftime('%y.%m.%d'))
    # saveCfg=Path(saveDir).joinpath(basePath).joinpath("config.yaml")
    saveCfg=Path(saveDir).joinpath("config_{}.yaml".format(ticn))
    Path(saveCfg).parent.mkdir(exist_ok=True,parents=True)
    with open(saveCfg,"w") as file:
        file.write(cfg.dump())
        
def log_set(filename,log_level,filemode="w"):
    logging.basicConfig(filename=filename,
                        filemode=filemode,
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S",
                        level=log_level
                        
                        )
    # encoding="utf-8"