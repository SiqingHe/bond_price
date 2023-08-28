import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import dtUtils
import time
from tqdm import tqdm

def noEnum_get(org_json,org_Enum_json,excel_dir,save_dir,saveEnum_json):
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    with open(org_json,"r",encoding="utf-8-sig") as wr:
        org_ls=json.load(wr)
    noEnum_ls=list(org_ls.keys())
    save_dic=org_ls.copy()
    enum_dic=dtUtils.json_read(org_Enum_json)
    save_enum_dic=enum_dic.copy()
    enum_ls=list(enum_dic.keys())
    for csv_path in tqdm(Path(excel_dir).glob("*.csv")):
        dfs=pd.read_csv(csv_path,header=0,index_col=0)
        for tag_it in noEnum_ls:
            added=list(set(dfs[tag_it].dropna().to_list())-set(save_dic[tag_it]))
            save_dic[tag_it]+=added
        for tage in enum_ls:
            added=list(set(dfs[tage].dropna().to_list())-set(save_enum_dic[tage]))
            if tage=="CLAUSEABBR":
                org_list=dfs[tage].dropna().to_list()
                org_list=[_ for _ in org_list if _!=0]
                trans_list=[",".join(sorted(_.split(","))) for _ in org_list]
                added=list(set(trans_list)-set(save_enum_dic[tage]))
            save_enum_dic[tage]+=added
    # for tag_it in save_dic.keys():
    #     save_dic[tag_it]=list(set(save_dic[tag_it]))
    tic=time.time()
    timeAr=time.localtime(tic)
    save_name="noEnum_{}.json".format(time.strftime("%Y-%m-%d.%H_%M_%S",timeAr))
    dtUtils.jsonSave(str(Path(save_dir).joinpath(save_name)),save_dic)
    dtUtils.jsonSave(saveEnum_json,save_enum_dic)

if __name__=="__main__":
     noEnum_get(org_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\no_Enum\noEnum_2023-07-24.20_34_05.json",
                org_Enum_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\kindEnum_0724.json",
                excel_dir=r"D:\python_code\LSTM-master\bond_price\real_data\excel2year_contat",
                save_dir=r"D:\python_code\LSTM-master\bond_price\dataPy\config\no_Enum",
                saveEnum_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\kindEnum_0726.json")