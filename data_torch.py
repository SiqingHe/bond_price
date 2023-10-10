import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from utils import get_data,invert_scale,get_pkl
import pickle
from tqdm import tqdm
import pandas as pd
from config import xgboost_cfg
from dataPy import dtUtils
import joblib
from pathlib import Path
import h5py
import time
cfg = xgboost_cfg.cfg
import numba
from siameseNetwork import similarity


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
class dataIter(Dataset):
    def __init__(self,dataPd,idx_types,idx_nums):
        self.dataPd = dataPd
        self.idx_types = idx_types
        self.idx_nums = idx_nums
        self.len = int((self.dataPd.shape[0]-1)*self.dataPd.shape[0]/2)
        # self.targets=targets
    def __len__(self):
        return int((self.dataPd.shape[0]-1)*self.dataPd.shape[0]/2)
    def __getitem__(self, index):
        # while True:
        col = int((index*2+0.25)**0.5+0.5)
        row = int(index - (col-1)*col/2)
        x=self.dataPd.iloc[col,:].values
        y=self.dataPd.iloc[row,:].values
        dis = similarity(x[0:-1],y[0:-1],self.idx_types,self.idx_nums)
        # print(x,y,dis,x[-1]-y[-1])
        if abs(x[-1]-y[-1])<=0.05 and dis<0.2:
            label = 1
            return x,y,label
        elif abs(x[-1]-y[-1])>0.1:
            label = 0
            return x,y,label
        else:
            # index = (index + 1)%self.len
            return self.__getitem__((index + 1) % len(self))
                
        # return index,col,row
        # print(index,col,row)
class dataInference(Dataset):
    def __init__(self,dataPd,inferItem):
        self.dataPd = dataPd
        self.inferItem = inferItem
        # self.targets=targets
    def __len__(self):
        return self.dataPd.shape[0]
    def __getitem__(self, index):
        return self.inferItem[0:-1],self.dataPd.iloc[index][0:-1]      
       
    
class metricDataset(Dataset):
    def __init__(self,dfs,x_column,y_column):
        self.dfs = dfs
        self.data = dfs[x_column]
        self.targets = dfs[y_column]
        self.x_column = x_column
        self.y_column = y_column
    def __len__(self):
        return self.dfs.shape[0]
    def __getitem__(self, index):
        self.dfs.sort_values(by = "deal_time",inplace = True)
        # triplets = []
        # labels = set(item['label'] for item in dataset)  # 假设每个数据样本都有一个标签
        # total = dfs.shape[0]
        num = 0
        row = self.dfs.iloc[index,:]
            # if num > 10 : break
        anchor = []
        anchor_feature = row[cfg.X_COLUMN].values.tolist()
        anchor_label = row[cfg.Y_COLUMN].values.tolist()[0]
        anchor_zt = row["ISSUERUPDATED"]

        anchor = anchor_feature 
        # + [anchor_label]
        tic = time.time()
        positive_candidates = self.dfs[(abs(self.dfs[cfg.Y_COLUMN[0]]-anchor_label)<0.01) & \
                                (self.dfs["deal_time"]<anchor_feature[0]) & (self.dfs["ISSUERUPDATED"] == anchor_zt)]
        if positive_candidates.shape[0]<1:
            positive_candidates = self.dfs[(abs(self.dfs[cfg.Y_COLUMN[0]]-anchor_label)<0.05) & (self.dfs["deal_time"]<anchor_feature[0])]
        negative_candidates = self.dfs[(abs(self.dfs[cfg.Y_COLUMN[0]]-anchor_label)>=0.1) & (self.dfs["deal_time"]<anchor_feature[0])]
        
        atic = time.time()
        print(atic-tic)
        # print(positive_candidates.shape[0],negative_candidates.shape[0])
        if positive_candidates.shape[0]<1 or negative_candidates.shape[0]<1:
            return self.__getitem__((index + 1) % len(self))
        # positive = {}
        # negative = {}
        positive = []
        negative = []
        ps_shape = positive_candidates.shape[0]
        ng_shape = negative_candidates.shape[0]
        ps_choose = positive_candidates.iloc[ps_shape-1]
        ng_choose = negative_candidates.iloc[ng_shape-1]
        positive =  ps_choose[cfg.X_COLUMN].values.tolist() 

        negative =  ng_choose[cfg.X_COLUMN].values.tolist() 
        return {'anchor': anchor, 'positive': positive, 'negative': negative}

class metricCacheDataset(Dataset):
    def __init__(self,file):
        self.data = file["my_dict_dataset"][:]
    # #     cached_data = dataset[:]
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        id_data = self.data[index]
        return id_data[0],id_data[1],id_data[2]

# def similarity(array1,array2,type_idls):
#     tp1 = array1[type_idls]
#     tp2 = array2[type_idls]
#     dis1 = np.sum(tp1 == tp2)/len(array1)
#     numls = set(range(len(array1)))-set(type_idls)
#     num1 = array1[numls]
#     num2 = array2[numls]
#     dot_product = np.dot(num1, num2)
#     norm_vector1 = np.linalg.norm(num1)
#     norm_vector2 = np.linalg.norm(num2)
#     if norm_vector1 == 0 or norm_vector2 == 0:
#         dis2 = 1
#     else:
#         cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
#         dis2 = 1 - cosine_similarity
#     dis = dis1*len(tp1)/len(array1) + dis2*len(num1)/len(array2)
#     return dis
# @numba.jit
def triplet_get1(dfs,row,idx):
    anchor_feature = row[cfg.X_COLUMN].values.tolist()
    anchor_label = row[cfg.Y_COLUMN].values.tolist()[0]
    anchor_zt = row["ISSUERUPDATED"]
    anchor = anchor_feature + [anchor_label]
    dfs_time = dfs.iloc[:idx,]
    # positive_candidates = dfs_time.loc[((dfs_time[cfg.Y_COLUMN[0]]-anchor_label<0.01) | \
    #                         (dfs_time[cfg.Y_COLUMN[0]]-anchor_label> -0.01)) & \
    #                         (dfs_time["ISSUERUPDATED"] == anchor_zt)]
    positive_candidates = dfs_time.loc[((dfs_time[cfg.Y_COLUMN[0]]-anchor_label<0.01) | \
                            (dfs_time[cfg.Y_COLUMN[0]]-anchor_label> -0.01)) & \
                            (dfs_time["ISSUERUPDATED"] == anchor_zt)]
    if positive_candidates.shape[0]<1:
        positive_candidates = dfs_time.loc[((dfs_time[cfg.Y_COLUMN[0]]-anchor_label<0.01) | \
                                    (dfs_time[cfg.Y_COLUMN[0]]-anchor_label> -0.01))
                                    & (dfs_time["deal_time"]<anchor_feature[0])]
    negative_candidates = dfs_time.loc[((dfs_time[cfg.Y_COLUMN[0]]-anchor_label>=0.1) | \
                                (dfs_time[cfg.Y_COLUMN[0]]-anchor_label<=-0.1))]
    if positive_candidates.shape[0]<1 or negative_candidates.shape[0]<1:
        return None
    ps_shape = positive_candidates.shape[0]
    ng_shape = negative_candidates.shape[0]
    ps_choose = positive_candidates.iloc[ps_shape-1]
    ng_choose = negative_candidates.iloc[np.random.choice(ng_shape)]
    del positive_candidates
    del negative_candidates
    del dfs
    positive = ps_choose[cfg.X_COLUMN].values.tolist() + ps_choose[cfg.Y_COLUMN].values.tolist()
    negative = ng_choose[cfg.X_COLUMN].values.tolist() + ng_choose[cfg.Y_COLUMN].values.tolist()
    return anchor,positive,negative
def cache_triplets(dfs,save_dir,label,batch_num = 10000):
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    dfs.sort_values(by = "deal_time",inplace = True)
    # labels = set(item['label'] for item in dataset)  # 假设每个数据样本都有一个标签
    # total = dfs.shape[0]
    num = 0
    save_path = str(Path(save_dir).joinpath("{}_{}.pkl".format(label,num)))
    
    # file = h5py.File(save_path, "w")
    # max_shape = (0,batch_num,3,len(cfg.X_COLUMN)+1)
    # dataset = file.create_dataset("my_dict_dataset", shape=max_shape, maxshape =(None,batch_num,3,len(cfg.X_COLUMN)+1))
    triples = []
    iter_num = 0
    added,add1,add2 = 0,0,0
    dfs_column = dfs.columns.tolist()
    x_index = [dfs_column.index(_) for _ in cfg.X_COLUMN]
    y_index = dfs_column.index(cfg.Y_COLUMN[0])
    issue_index = dfs_column.index("ISSUERUPDATED")
    deal_index = dfs_column.index("deal_time")
    dfs_values = dfs.values
    for idx,(_,row) in tqdm(enumerate(dfs.iterrows()),total = dfs.shape[0]):
        res = triplet_get1(dfs,row,idx)
        if res is None:
            continue
        # time.sleep(0.01)
        triples.append(res)
    # triples = []
    # for idx,(_,row) in tqdm(enumerate(dfs.iterrows()),total = dfs.shape[0]):
    #     if idx<2:continue
    #     positive_ls = []
    #     negative_ls = []
    #     anchor = row[cfg.X_COLUMN].values.tolist() + row[cfg.Y_COLUMN].values.tolist()
    #     for idx2,(_,row2) in enumerate(dfs.iloc[:idx,].iterrows()):
    #         ngps = row2[cfg.X_COLUMN].values.tolist() + row2[cfg.Y_COLUMN].values.tolist()
    #         if np.abs(ngps[-1]-anchor[-1])<0.05:
    #             positive_ls.append(ngps)
    #         elif np.abs(ngps[y_index]-anchor[y_index])>0.1:
    #             negative_ls.append(ngps)
    #     if not positive_ls or not negative_ls:
    #         continue
    #     for i in positive_ls:
    #         for j in negative_ls:
    #             triples.append([anchor,i,j])
        # print(row)
    # for i in tqdm(range(dfs_values.shape[0])):
    #     row = dfs_values[i,:]
    #     anchor = []
    #     anchor_feature = row[x_index]
    #     anchor_label = row[y_index]
    #     anchor = anchor_feature.tolist() + [anchor_label]
    #     issue = row[issue_index]
    #     deal_time = row[deal_index]
    #     tic = time.time()
    #     p_cdt = ((dfs_values[:,issue_index]==issue) & \
    #                      ((dfs_values[:,y_index]<anchor_label+0.01)|(dfs_values[:,y_index]>anchor_label-0.01)) & \
    #                      (dfs_values[:,deal_index]<deal_time))
        
    #     n_cdt = ((dfs_values[:,y_index]>anchor_label+0.1)|(dfs_values[:,y_index]<anchor_label-0.1)) & \
    #                      (dfs_values[:,deal_index]<deal_time)
    #     # positive_candidates = dfs_values[p_cdt]
    #     # negative_candidates = dfs_values[n_cdt]
    #     # p_chs = dfs_values[p_cdt][-1,:]
    #     # n_chs = dfs_values[n_cdt][np.random.choice(range(dfs_values[n_cdt].shape[0])),:]
    #     try:
    #         p_chs = dfs_values[p_cdt][-1,:]
    #         n_chs = dfs_values[n_cdt][np.random.choice(range(dfs_values[n_cdt].shape[0])),:]
    #     except IndexError as e:
    #         # print(e)
    #         continue
    #     added += (time.time()-tic)
    #     if (num%batch_num==0 and num>0):
    #         print(added)
    #     # if positive_candidates.shape[0]==0 or negative_candidates.shape[0]==0:
    #     #     continue
    #     # p_chs = positive_candidates[-1,:]
    #     # n_chs = negative_candidates[np.random.choice(range(negative_candidates.shape[0])),:]
    #     ps_feature = p_chs[x_index]
    #     ps_label = p_chs[y_index]
    #     positive = ps_feature.tolist() + [ps_label]
    #     ng_feature = n_chs[x_index]
    #     ng_label = n_chs[y_index]
    #     negative = ng_feature.tolist() + [ng_label]
    #     triples.append([anchor,positive,negative])
    #     num += 1
    #     pass
    # for _,row in tqdm(dfs.iterrows(),total = dfs.shape[0]):
    #     # if num > 10 : break
    #     tic1  = time.time()
    #     iter_num += 1
    #     if (num%batch_num==0 and num>0) or iter_num==dfs.shape[0]: 
    #         # current_size = dataset.shape[0]
    #         # dataset.resize(current_size + 1, axis=0)
    #         # dataset[current_size] = np.array(triples)
    #         # file.close()
    #         # with open(save_path, 'wb') as file:
    #         #     pickle.dump(triples,file)
    #         save_path = str(Path(save_dir).joinpath("{}_{}.pkl".format(label,num)))
    #         # file = h5py.File(save_path, "w")
    #         # max_shape = (0,batch_num,3,len(cfg.X_COLUMN)+1)
    #         # dataset = file.create_dataset("my_dict_dataset", shape=max_shape, maxshape =(None,batch_num,3,len(cfg.X_COLUMN)+1))
    #         triples = []
    #     anchor = []
    #     anchor_feature = row[cfg.X_COLUMN].values.tolist()
    #     anchor_label = row[cfg.Y_COLUMN].values.tolist()[0]
    #     # anchor_bid = row[""]
    #     anchor_zt = row["ISSUERUPDATED"]
    #     anchor = anchor_feature + [anchor_label]
    #     # anchor["label"] = anchor_label
    #     tic = time.time()
    #     add1 += tic - tic1
    #     positive_candidates = dfs[((dfs[cfg.Y_COLUMN[0]]-anchor_label<0.01) | \
    #                                 (dfs[cfg.Y_COLUMN[0]]-anchor_label> -0.01)) & \
    #                               (dfs["deal_time"]<anchor_feature[0]) & (dfs["ISSUERUPDATED"] == anchor_zt)]
    #     if positive_candidates.shape[0]<1:
    #         positive_candidates = dfs[((dfs[cfg.Y_COLUMN[0]]-anchor_label<0.01) | \
    #                                     (dfs[cfg.Y_COLUMN[0]]-anchor_label> -0.01))
    #                                   & (dfs["deal_time"]<anchor_feature[0])]
    #     negative_candidates = dfs[((dfs[cfg.Y_COLUMN[0]]-anchor_label>=0.1) | \
    #                               (dfs[cfg.Y_COLUMN[0]]-anchor_label<=-0.1)) \
    #                               & (dfs["deal_time"]<anchor_feature[0])]
        
    #     atic = time.time()
    #     added += atic - tic
    #     if (num%batch_num==0 and num>0) or iter_num==dfs.shape[0]:
    #         print(add1)
    #         print(added)
    #         print(add2)
    #         added = 0
    #         add2 = 0
    #         add1 = 0
    #     # print(positive_candidates.shape[0],negative_candidates.shape[0])
    #     if positive_candidates.shape[0]<1 or negative_candidates.shape[0]<1:
    #         continue
    #     # for _,pitem in positive_candidates.iterrows():
    #     #     for _,nitem in negative_candidates.iterrows():
    #     #         positive = {}
    #     #         negative = {}
    #     #         positive["feature"] = pitem[cfg.X_COLUMN].values.tolist()
    #     #         positive["label"] = pitem[cfg.Y_COLUMN].values.tolist()[0]
    #     #         negative["feature"] = nitem[cfg.X_COLUMN].values.tolist()
    #     #         negative["label"] = nitem[cfg.Y_COLUMN].values.tolist()[0]
    #     positive = {}
    #     negative = {}
    #     ps_shape = positive_candidates.shape[0]
    #     ng_shape = negative_candidates.shape[0]
    #     ps_choose = positive_candidates.iloc[ps_shape-1]
    #     ng_choose = negative_candidates.iloc[ng_shape-1]
    #     # positive["feature"] = ps_choose[cfg.X_COLUMN].values.tolist()
    #     # positive["label"] = ps_choose[cfg.Y_COLUMN].values.tolist()[0]
        
    #     positive = ps_choose[cfg.X_COLUMN].values.tolist() + ps_choose[cfg.Y_COLUMN].values.tolist()
    #     # negative["feature"] = ng_choose[cfg.X_COLUMN].values.tolist()
    #     # negative["label"] = ng_choose[cfg.Y_COLUMN].values.tolist()[0]
        
    #     negative = ng_choose[cfg.X_COLUMN].values.tolist() + ng_choose[cfg.Y_COLUMN].values.tolist()
    #     # triplets.append({'anchor': anchor, 'positive': positive, 'negative': negative})
        
    #     triples.append([anchor,positive,negative])
    #     # current_size = dataset.shape[0]
    #     # dataset.resize(current_size + 1, axis=0)
    #     # dataset[current_size] = np.array([anchor,positive,negative])
    #     num += 1
    #     tic2 = time.time()
    #     add2 += tic2 - atic
    # file.close()
    return triples

def create_triples(dfs):
    dfs.sort_values(by = "deal_time",inplace = True)
    # triplets = []
    # labels = set(item['label'] for item in dataset)  # 假设每个数据样本都有一个标签
    # total = dfs.shape[0]
    num = 0
    triples = []
    for _,row in tqdm(dfs.iterrows(),total = dfs.shape[0]):
        anchor = []
        anchor_feature = row[cfg.X_COLUMN].values.tolist()
        anchor_label = row[cfg.Y_COLUMN].values.tolist()[0]
        anchor_zt = row["ISSUERUPDATED"]

        anchor = anchor_feature 
        # + [anchor_label]
        positive_candidates = dfs[(abs(dfs[cfg.Y_COLUMN[0]]-anchor_label)<0.01) & \
                                (dfs["deal_time"]<anchor_feature[0]) & (dfs["ISSUERUPDATED"] == anchor_zt)]
        if positive_candidates.shape[0]<1:
            positive_candidates = dfs[(abs(dfs[cfg.Y_COLUMN[0]]-anchor_label)<0.05) & (dfs["deal_time"]<anchor_feature[0])]
        negative_candidates = dfs[(abs(dfs[cfg.Y_COLUMN[0]]-anchor_label)>=0.1) & (dfs["deal_time"]<anchor_feature[0])]
        
        # print(positive_candidates.shape[0],negative_candidates.shape[0])
        if positive_candidates.shape[0]<1 or negative_candidates.shape[0]<1:
            continue
        # positive = {}
        # negative = {}
        positive = []
        negative = []
        ps_shape = positive_candidates.shape[0]
        ng_shape = negative_candidates.shape[0]
        ps_choose = positive_candidates.iloc[ps_shape-1]
        ng_choose = negative_candidates.iloc[ng_shape-1]
        positive =  ps_choose[cfg.X_COLUMN].values.tolist() 

        negative =  ng_choose[cfg.X_COLUMN].values.tolist() 
        triples.append({'anchor': anchor, 'positive': positive, 'negative': negative})
    pass

def testData():
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

if __name__=="__main__":
    pass
    # test_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0818\test.csv"
    # dfs = pd.read_csv(test_path)
    # triples = create_triplets(dfs)
    # saveDir = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_pkl0910"
    # data_path = Path(saveDir).joinpath("test.pkl")
    # joblib.dump(triples, data_path)
    # # pkl_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_pkl0910\test.pkl"
    # # test = dtUtils.get_pkl(pkl_path)
    # pass
    # import pickle
    # test_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_pkl0910\tt.pkl"
    # with open(test_path, 'wb') as file:
    # # 在此之后，可以逐步写入数据
    #     for i in range(3):
    #         data = [i+1 for _ in range(3)]
    #         pickle.dump(data, file)
    data_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0818\valid.csv"
    dfs = pd.read_csv(data_path)
    dfs = dfs[cfg.X_COLUMN+cfg.Y_COLUMN]
    dfs_copy = dfs.copy()
    dfs_copy2 = dfs.copy()
    nums_clm = [_ for _  in cfg.X_COLUMN if _ not in cfg.TYPE_COLUMN]
    dfs_copy = dfs.apply(lambda x: (x - x.mean()) / x.std())
    dfs_copy.fillna(-1,inplace = True)
    for _ in nums_clm:
        dfs_copy2[_] = dfs_copy[_]
    #     dfs_copy[_] = dfs.apply(lambda x: (x[_] - x[_].mean()) / x[_].std(), axis = 1)
    idx_types = [cfg.X_COLUMN.index(_) for _ in cfg.TYPE_COLUMN]
    idx_nums = [cfg.X_COLUMN.index(_) for _  in cfg.X_COLUMN if _ not in cfg.TYPE_COLUMN]
    dataset = dataIter(dfs,idx_types,idx_nums)
    batch_size = 3
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num = 0
    for batch in dataloader:
        if num>10:break
        print(batch)
        num += 1
    # dataset = metricDataset(dfs,cfg.X_COLUMN,cfg.Y_COLUMN)
    # batch_size = 2
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # print(dataloader.__len__)
    # print(len(dataloader))5000
    # for batch in dataloader:
    #     print(batch)
    # save_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0818\valid"
    # cache_triplets(dfs,save_path,label= "valid",batch_num = 5000)
    # file = h5py.File(save_path, "r")

    # # # 从数据集中读取数据
    # # if "my_dict_dataset" in file:
    # #     dataset = file["my_dict_dataset"]
    # #     cached_data = dataset[:]  # 读取整个数据集到内存中
    # dataset = metricCacheDataset(file)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    # num = 0
    # for batch in dataloader:
    #     if num>3:break
    #     anchor,positive,negative = batch
    #     print("anchor",anchor)
    #     print("anchor",anchor.shape)
    #     print("positive",positive)
    #     print("negative",negative)
    #     num += 1
    # #     # 打印读取的数据
    # #     print(cached_data)

    # # 关闭文件
    # file.close()
        