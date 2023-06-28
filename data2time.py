import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import pickle
from pathlib import Path

def dropnull(df):
    ndf=df.dropna()
    return ndf

def data2time(linedict,
              timeft=["received_time_diff","trade_price","trade_size","trade_type","curve_based_price"],
              keepitem=["bond_id","weight","current_coupon","is_callable"]):
    data_ls=[[] for _ in range(10)]
    for k,v in linedict.items():
        k_split=k.split("last")
        if k in keepitem:
            for _ in data_ls:_.append(v)
        if len(k_split)>1:
            time=k_split[1].replace("last","")
            itime=int(time)
            if len(data_ls[itime-1])<5:
                data_ls[itime-1].append(itime)
                data_ls[itime-1].append(v)
            else:
                data_ls[itime-1].append(v)
        # print(data_ls)
    return data_ls

def data2timeLine(lines,
              start_id=10,
              keepid=[0,2,3,5]):
    data_ls=[[] for _ in range(10)]
    line_len=len(lines)
    for i in range(line_len):
        if i in keepid:
            for ii in data_ls:ii.append(lines[i])
        if i>=start_id:
            num=(i-start_id)//5
            # rem=(i-start_id)%5
            if len(data_ls[num])<5:
                data_ls[num].append(num+1)
            data_ls[num].append(lines[i])
        # print(data_ls)
    return data_ls

def data2t(data_ls,len=2):
    dt_ar=np.array(data_ls)
    x_org,y_org=dt_ar[:,[0,1,2,3,4,5,7,8,9]],dt_ar[:,6]
    # print(x_org,y_org)
    X,y=[],[]
    t_len=x_org.shape[0]
    for i in range(t_len-len+1):
        X.append(x_org[i:i+len,:])
        y.append(y_org[i+len-1])
    return X,y

def scaler_trans(train):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # 转换train data
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    return train_scaled,scaler

def jsonSave(saveJpath,dic):
    with open(saveJpath,"w") as wr:
        json.dump(dic,wr,indent=1,ensure_ascii=False)

def read_test(csv_path,save_path):
    if not Path(save_path).exists():
        Path(save_path).mkdir(exist_ok=True,parents=True)
    csv_res=pd.read_csv(csv_path,index_col=0,header=0)
    drop_csv=dropnull(csv_res)
    # dfvalue=drop_csv.values
    
    # 
    value_trans=[]
    # dt_values=scaler_trans(drop_csv.values)
    for _,item in drop_csv.iterrows():
    # for item_list in dt_values:
        # if _>10:
        #     break
        item_dict=dict(item)
        # item_list=item.to_list()
        trans=data2time(item_dict)
        # trans=data2timeLine(item_list)
        value_trans+=trans
        # trans1=data2timeLine(item_list)
        # print(trans)
        # print(trans1)
        # print(trans==trans1)
        # X,y=data2t(trans)
        
        # print(trans)
    value_trans=np.array(value_trans)
    y_all=value_trans[:,6].reshape(-1,1)
    value_trans,scaler_all=scaler_trans(value_trans)
    y_trans,scale_y=scaler_trans(y_all)
    dt_len=len(value_trans)
    X_train,y_train=[],[]
    X_test,y_test=[],[]
    
    for i in range(0,dt_len,10):
        X,y=data2t(value_trans[i:i+10])
        X_train+=X[0:7]
        y_train+=y[0:7]
        X_test+=X[7:]
        y_test+=y[7:]
    jsonSave(Path(save_path).joinpath("x_train.json"),np.array(X_train).tolist())
    print(np.array(X_train).shape)
    jsonSave(Path(save_path).joinpath("y_train.json"),np.array(y_train).tolist())
    print(np.array(y_train).shape)
    jsonSave(Path(save_path).joinpath("x_test.json"),np.array(X_test).tolist())
    jsonSave(Path(save_path).joinpath("y_test.json"),np.array(y_test).tolist())
    jsonSave(Path(save_path).joinpath("y_trans.json"),y_trans.tolist())
    jsonSave(Path(save_path).joinpath("vy_trans.json"),value_trans[:,6].tolist())
    pkl_save=str(Path(save_path).joinpath("scale_y.pkl"))
    with open(pkl_save,"wb") as wr:
        pickle.dump(scale_y,wr)
    return scale_y
if __name__=="__main__":
    pass
    csv_path=r"D:\python_code\LSTM-master\bond_data\train.csv"
    read_test(csv_path,
              save_path=r"D:\python_code\LSTM-master\model_bond\timedata")