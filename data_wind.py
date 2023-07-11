import pandas as pd
import numpy as np
from WindPy import w
from config.wind_column import column1,column2,column3,column
import time
w.start()
from data_pre import bondid_path_corr
from pathlib import Path
from utils import jsonSave
from collections import defaultdict
from tqdm import tqdm
import json

def wind_fetch(bond_id,date_start,date_end,column_ls,save_path):
    column1,column2,column3=column_ls
    wind_data1=w.wsd(bond_id,column1,date_start, date_end, 
                    "unit=1;ratingAgency=104;industryType=1;type=All;ShowBlank=-1;Currency=CNY;PriceAdj=YTM")
    wind_data2=w.wsd(bond_id,column2,date_start, date_end, 
                    "type=1;ratingAgency=101;ShowBlank=-1;Currency=CNY;PriceAdj=YTM")
    wind_data3=w.wsd(bond_id,column3,date_start, date_end, 
                    "ratingAgency=18;ShowBlank=-1;Currency=CNY;PriceAdj=YTM")
    print(wind_data1)
    if wind_data1.ErrorCode!=0:
        return 0
    if wind_data2.ErrorCode!=0:
        return -1
    if wind_data3.ErrorCode!=0:
        return -2
    data=wind_data1.Data+wind_data2.Data
    columns=wind_data1.Fields+wind_data2.Fields
    data.insert(12,wind_data3.Data[0])
    columns.insert(12,"{}2".format(column3).upper())
    df = pd.DataFrame(data,columns=wind_data1.Times,index=columns).T
    df.fillna(-1,inplace=True)
    df.to_csv(save_path,encoding="utf-8-sig")
    return 1

def fetch_1day(bond_id,date,column_ls):
    column1,column2,column3=column_ls
    # column=""
    wind_data1=w.wss(bond_id,column1,"tradeDate={};unit=1;ratingAgency=104;type=All;industryType=1;date={}".format(date,date))
    wind_data2=w.wss(bond_id,column2,"tradeDate={};type=1;ratingAgency=101;date={}".format(date,date))
    wind_data3=w.wss(bond_id,column3,"tradeDate={};ratingAgency=18;date={}".format(date,date))
    # print(wind_data1)
    # print(wind_data2)
    # print(wind_data3)
    if wind_data1.ErrorCode!=0:
        return 0
    if wind_data2.ErrorCode!=0:
        return -1
    if wind_data3.ErrorCode!=0:
        return -2
    data=wind_data1.Data+wind_data2.Data
    columns=wind_data1.Fields+wind_data2.Fields
    data.insert(11,wind_data3.Data[0])
    columns.insert(11,"{}2".format(column3).upper())
    # print(data)
    # df = pd.DataFrame(data,columns=wind_data1.Times,index=columns).T
    # df.fillna(-1,inplace=True)
    # df.to_csv(save_path,encoding="utf-8-sig")
    return data,columns

def fetch_id(bond_id,date_ls,column_ls,save_path,except_txt):
    data_ls=[]
    except_list=[0,-1,-2]
    date_end=[]
    cl=0
    for date in date_ls:
        res=fetch_1day(bond_id,date,column_ls)
        if res not in except_list:
            dt,cl=res
            data_ls.append(dt)
            date_end.append(date)
        else:
            with open(except_txt,"a") as wr:
                wr.write(str(bond_id)+" "+str(date)+" "+str(res)+"\n")
    if cl:
        data=np.hstack((data_ls)).tolist()
        df=pd.DataFrame(data,columns=date_end,index=cl).T
        df.to_csv(save_path,encoding="utf-8-sig")

def fetch_batch(bond_dir,save_dir,date_start,date_end,column_ls,record_txt):
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(exist_ok=True,parents=True)
    bondid_dic=bondid_path_corr(bond_dir)
    for k in bondid_dic.keys():
        save_path=str(Path(save_dir).joinpath("wind{}.csv".format(k)))
        mark=wind_fetch(k,date_start,date_end,column_ls,save_path)
        if mark!=1:
            with open(record_txt,"a") as wr:
                wr.write(k+" "+str(mark)+"\n")
                
def bondid_times_get(excel_dir,save_json):
    time_dic={}
    for filepath in Path(excel_dir).glob("*.csv"):
        name=filepath.stem
        name_split=name.split("_")
        bond_id=name_split[0]
        excel_pd=pd.read_csv(str(filepath),header=0,index_col=0)
        time_ls=excel_pd["日期"].to_list()
        if bond_id not in time_dic:
            time_dic[bond_id]=time_ls
        else:
            print("unexpect repected {}".format(filepath.name))
    jsonSave(save_json,time_dic)
    
def fetchid_batch(idtime_js,column_ls,save_dir,except_txt):
    num=0
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(exist_ok=True,parents=True)
    with open(idtime_js,"r",encoding="utf-8-sig") as fr:
        time_dic=json.load(fr)
    for bondid,date_ls in tqdm(time_dic.items()):
        # if num>2:
        #     break
        save_path=str(Path(save_dir).joinpath(bondid+".csv"))
        fetch_id(bondid,date_ls,column_ls,save_path,except_txt)
        num+=1
if __name__=="__main__":
    pass
    # bond_id="7103.IB"
    # date_start="2023-03-02"
    # date_end="2023-03-02"
    # save_path=r"D:\python_code\LSTM-master\bond_price\real_data\wind7103.IB_ts.csv"
    column_ls=[column1,column2,column3]
    # wind_fetch(bond_id,date_start,date_end,column_ls,save_path)
    # save_dir=r"D:\python_code\LSTM-master\bond_price\real_data\wind_fetch"
    # bondid_dir=r"D:\python_code\LSTM-master\bond_price\real_data\group2year"
    # record_txt=r"D:\python_code\LSTM-master\bond_price\real_data\fail_wd.txt"
    # fetch_batch(bondid_dir,save_dir,date_start,date_end,column_ls,record_txt)
    # wind_data=w.wss("7103.IB", "ptmyear,termnote1,termifexercise","tradeDate=20210624")
    # print(wind_data)
    # df = pd.DataFrame(wind_data.Data,columns=wind_data.Times,index=wind_data.Fields).T
    # # df.fillna(-1,inplace=True)
    # save_path=r"D:\python_code\LSTM-master\bond_price\real_data\feature_get\wws_ts.csv"
    # df.to_csv(save_path,encoding="utf-8-sig")
    # date=int("20210624")
    # fetch_1day(bond_id,date,column_ls,save_path)
    # path1=r"D:\python_code\LSTM-master\bond_price\real_data\group2year_contat\7103.IB_4.csv"
    # df=pd.read_csv(path1,header=0,index_col=0)
    # date_ls=df["日期"].to_list()
    # fetch_id(bond_id,date_ls,column_ls,save_path,
    #          except_txt=r"D:\python_code\LSTM-master\bond_price\real_data\fail_wd1.txt")
    idtime_js=r"D:\python_code\LSTM-master\bond_price\config\bondid_time.json"
    fetchid_batch(idtime_js,
                  column_ls,
                  save_dir=r"D:\python_code\LSTM-master\bond_price\real_data\feature_get",
                  except_txt=r"D:\python_code\LSTM-master\bond_price\real_data\fail_wd0706.txt")