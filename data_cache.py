import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from utils import get_data,invert_scale,get_pkl
import pickle
from tqdm import tqdm
import pandas as pd
# from config import xgboost_cfg
from dataPy import dtUtils
import joblib
from pathlib import Path
import h5py
import time
# cfg = xgboost_cfg.cfg
import numba
from siameseNetwork import similarity
from collections import defaultdict
from config import xgboost_cfg
xgb_cfg = xgboost_cfg.cfg



def data_cache(dataPd,idx_types,idx_nums):
    res_item = []
    for i in tqdm(range(dataPd.shape[0])):
        item = dataPd.iloc[i,:]
        yval = item.values[-1]
        dataPd_bf = dataPd.iloc[max(0,i-20000):i,:]
        df_positive = dataPd_bf[np.abs(dataPd_bf["yield"]-yval)<0.01]
        if df_positive.shape[0]>0:
            rd_pos = df_positive.iloc[np.random.choice(df_positive.shape[0]),:]
            pos_item = item.values,rd_pos.values,1
            res_item.append(pos_item)
        df_negative = dataPd_bf[np.abs(dataPd_bf["yield"]-yval)>0.1]
        if df_negative.shape[0]>0:
            rd_ng = df_negative.iloc[np.random.choice(df_negative.shape[0]),:]
            ng_item= item.values,rd_ng.values,0
            res_item.append(ng_item)

def positive_sift(item1,item2,idx_types,idx_nums):
    values1 = item1.values
    values2 = item2.values
    dis = similarity(values1[0:-1],values2[0:-1],idx_types,idx_nums)
    return ((dis<0.2).numpy()[0] and (abs(values1[-1]-values2[-1])<0.01))
            
def data_cache2(dataPd,idx_types,idx_nums,save_dir):
    Path(save_dir).mkdir(exist_ok = True,parents = True)
    res_item = []
    # save_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\similar_visual"
    for i in tqdm(range(dataPd.shape[0])):
        # if i>30:break
        item = dataPd.iloc[i,:]
        yval = item.values[-1]
        dataPd_bf = dataPd.iloc[max(0,i-20000):i,:]
        # pos_bool = dataPd_bf.apply(lambda x:positive_sift(x,item,idx_types,idx_nums),axis =1 )
        # df_positive = dataPd_bf[pos_bool]
        df_positive = dataPd_bf[np.abs(dataPd_bf["yield"]-yval)<0.005]
        if df_positive.shape[0]>0:
            rd_pos = df_positive.iloc[np.random.choice(df_positive.shape[0]),:]
            # df_positive["dis"] = df_positive.apply(lambda x:similarity(item.values[0:-1],x.values[0:-1],idx_types,idx_nums)[0],axis = 1)
            # df_positive.to_csv(str(Path(save_path).joinpath(str(i)+"_similar.csv")))
            # for _,rd_pos in df_positive.iterrows():
            #     # pos_item = item.values,rd_pos.values,1
            #     dis  = similarity(item.values[0:-1],rd_pos.values[0:-1],idx_types,idx_nums)
                # print(dis)
                # if dis > 0.1:
                #     print(item,rd_pos)
            pos_item = item.values,rd_pos.values,1
            res_item.append(pos_item)
        # df_negative = dataPd_bf[(np.abs(dataPd_bf["yield"]-yval)>0.01) & (np.abs(dataPd_bf["yield"]-yval)<0.05)]
        df_negative = dataPd_bf[(np.abs(dataPd_bf["yield"]-yval)>0.01)]
        if df_negative.shape[0]>0:
            rd_ng = df_negative.iloc[np.random.choice(df_negative.shape[0]),:]
            ng_item= item.values,rd_ng.values,0
            res_item.append(ng_item)
    save_path = str(Path(save_dir).joinpath("data0930_nostd1.pkl"))
    with open(save_path, 'wb') as file:
        pickle.dump(res_item,file)
if __name__ == "__main__":
    pass
    train_path =  r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0818to0818\allData.csv"
    dfs = pd.read_csv(train_path)
    dfs = dfs[dfs['date']<1664553600]
    dfs = dfs[xgb_cfg.X_COLUMN+xgb_cfg.Y_COLUMN]
    # dfs_copy = dfs.copy()
    # dfs_copy2 = dfs.copy()
    # nums_clm = [_ for _  in xgb_cfg.X_COLUMN if _ not in xgb_cfg.TYPE_COLUMN]
    # dfs_copy = dfs.apply(lambda x: (x - x.mean()) / x.std())
    # dfs_copy.fillna(-1,inplace = True)
    # for _ in nums_clm:
    #     dfs_copy2[_] = dfs_copy[_]
    idx_types = [xgb_cfg.X_COLUMN.index(_) for _ in xgb_cfg.TYPE_COLUMN]
    idx_nums = [xgb_cfg.X_COLUMN.index(_) for _  in xgb_cfg.X_COLUMN if _ not in xgb_cfg.TYPE_COLUMN]
    data_cache2(dfs,
                 idx_types,
                 idx_nums,
                 save_dir = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0818to0818")
    # pkl_path=r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0818to0818\data0930.pkl"
    # with open(pkl_path,"rb") as rd:
    #     datacache=pickle.load(rd)
    # print(datacache)

