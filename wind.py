import pandas as pd
import numpy as np
from WindPy import w
w.start()
from config.wind_column import column1,column2,column3
from pathlib import Path
from tqdm import tqdm
import json


def fetch_1day(bond_id,date,column_ls):
    column1,column2,column3=column_ls
    wind_data1=w.wss(bond_id,column1,"tradeDate={};unit=1;ratingAgency=104;type=All;industryType=1;date={}".format(date,date))
    wind_data2=w.wss(bond_id,column2,"tradeDate={};type=1;ratingAgency=101;date={}".format(date,date))
    wind_data3=w.wss(bond_id,column3,"tradeDate={};ratingAgency=18;date={}".format(date,date))
    print(wind_data1)
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

def fetchid_batch(idtime_js,column_ls,save_dir,except_txt):
    num=0
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(exist_ok=True,parents=True)
    with open(idtime_js,"r",encoding="utf-8-sig") as fr:
        time_dic=json.load(fr)
    for bondid,date_ls in tqdm(time_dic.items()):
        if num>2:
            break
        save_path=str(Path(save_dir).joinpath(bondid+".csv"))
        fetch_id(bondid,date_ls,column_ls,save_path,except_txt)
        num+=1

if __name__=="__main__":
    column_ls=[column1,column2,column3]
    idtime_js=r"D:\python_code\LSTM-master\bond_price\config\bondid_time.json"
    fetchid_batch(idtime_js,
                  column_ls,
                  save_dir=r"D:\python_code\LSTM-master\bond_price\real_data\feature_get0710",
                  except_txt=r"D:\python_code\LSTM-master\bond_price\real_data\fail_wd0710.txt")