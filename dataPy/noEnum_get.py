import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import dtUtils
import time

def noEnum_get(org_json,excel_dir,save_dir,noEnum_ls=["ISSUERUPDATED","AGENCY_GUARANTOR"]):
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(exist_ok=True,parents=True)
    with open(org_json,"r",encoding="utf-8-sig") as wr:
        org_ls=json.load(wr)
    save_dic=defaultdict(list)
    for csv_path in Path(excel_dir).glob("*.csv"):
        dfs=pd.read_csv(csv_path,header=0,index_col=0)
        for tag_it in noEnum_ls:
            save_dic[tag_it]+=list(set(org_ls[tag_it]+dfs[tag_it].dropna().to_list()))
    for tag_it in save_dic.keys():
        save_dic[tag_it]=list(set(save_dic[tag_it]))
    tic=time.time()
    timeAr=time.localtime(tic)
    save_name="noEnum_{}.json".format(time.strftime("%Y-%m-%d.%H_%M_%S",timeAr))
    dtUtils.jsonSave(str(Path(save_dir).joinpath(save_name)),save_dic)

if __name__=="__main__":
     noEnum_get(org_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\org_noEnum.json",
                excel_dir=r"D:\python_code\LSTM-master\bond_price\real_data\dlFt_combine",
                save_dir=r"D:\python_code\LSTM-master\bond_price\dataPy\config\no_Enum")