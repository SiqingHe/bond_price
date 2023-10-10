import pandas as pd
from pathlib import Path
import time
from . import dtUtils
# import dtUtils
import re

def getAllOrder(table_dir,save_path):
    Path(save_path).parent.mkdir(exist_ok = True,parents = True)
    pd_ls = []
    for tablePath in Path(table_dir).glob("*.csv"):
        table_pd = pd.read_csv(tablePath,encoding = "utf-8-sig")
        pd_ls.append(table_pd)
    concat_pd = pd.concat(pd_ls)
    concat_pd.sort_values(by = "deal_time",inplace = True)
    concat_pd.to_csv(save_path,encoding = "utf-8-sig")

def saveTradeDay(table_path,save_path):
    Path(save_path).parent.mkdir(exist_ok = True,parents = True)
    table_pd = pd.read_csv(table_path)
    date_column = table_pd.apply(lambda x:timestamp2date(x["date"]),axis = 1)
    date_ls = date_column.to_list()
    distinct_date = sorted(set(date_ls),key = date_ls.index)
    dtUtils.jsonSave(save_path,distinct_date)

def timestamp2date(tic):
    timeArray = time.localtime(tic)
    date = time.strftime("%Y%m%d",timeArray)
    return date

def train_valid(table_pd,valid_date,train_date):
    table_pd["date_org"] = table_pd.apply(lambda x:timestamp2date(x["date"]),axis = 1)
    train_pd = table_pd[table_pd["date_org"].isin(train_date)]
    valid_pd = table_pd[table_pd["date_org"]==str(valid_date)]
    return train_pd,valid_pd

def interest_mean(arr):
    return arr[""]

def isActive(x,deal_count_dict):
    return deal_count_dict[x]>=5

def stat_save(combine_pd,save_path,it_ratiols=[0.1,0.05,0.01], cd_ratio = 0.1,message = "all_deal"):
    stat_group = pd.DataFrame(columns = ["total_mean"])
    stat_group["total_mean"] = combine_pd.groupby("date_org")["|yt-yp|"].mean()
    stat_group["total_count"] = combine_pd.groupby("date_org")["|yt-yp|"].count()
    interest_pd = combine_pd[combine_pd["ISSUERUPDATED"].isin([110,1180,1831,2047])]
    credit_pd = combine_pd[~combine_pd["ISSUERUPDATED"].isin([110,1180,1831,2047])]
                                                          
    stat_group["interest_mean"] = interest_pd.groupby("date_org")["|yt-yp|"].mean()
    stat_group["interest_count"] = interest_pd.groupby("date_org")["|yt-yp|"].count()
    for it_ratio in it_ratiols:
        stat_group["interest >{}".format(it_ratio)] = interest_pd.groupby("date_org")["|yt-yp|"].apply(lambda x:(x>it_ratio).mean())
    
    stat_group["credit_mean"] = credit_pd.groupby("date_org")["|yt-yp|"].mean()
    stat_group["credit_count"] = credit_pd.groupby("date_org")["|yt-yp|"].count()
    stat_group["credit >{}".format(cd_ratio)] = combine_pd.groupby("date_org")["|yt-yp|"].apply(lambda x:(x>cd_ratio).mean())
    if Path(save_path).exists():
        with pd.ExcelWriter(save_path, engine='openpyxl',mode = "a") as writer:
        # 将DataFrame写入Excel文件的一个工作表
            stat_group.to_excel(writer, sheet_name='stat_{}'.format(message))
    else:
        with pd.ExcelWriter(save_path, engine='openpyxl',mode = "w") as writer:
        # 将DataFrame写入Excel文件的一个工作表
            stat_group.to_excel(writer, sheet_name='stat_{}'.format(message))

def valid_combine(table_dir,save_path):
    Path(save_path).parent.mkdir(exist_ok = True,parents = True)
    table_ls = []
    mark = re.compile(r"test_\d{8}")
    for table_path in Path(table_dir).glob("*.csv"):
        match = re.search(mark,table_path.name).group(0)
        if match != table_path.stem:continue
        table_pd = pd.read_csv(str(table_path))
        deal_counts = table_pd["bond_id"].value_counts().to_dict()
        table_pd["is_active"] = table_pd.apply(lambda x:isActive(x["bond_id"],deal_counts),axis=1)
        table_ls.append(table_pd)
    combine_pd = pd.concat(table_ls)
    combine_pd["nidx"] = list(range(combine_pd.shape[0]))
    combine_pd = combine_pd.loc[:,~combine_pd.columns.str.contains("^Unnamed")]
    # combine_reindex = combine_pd.reindex(range(combine_pd.shape[0]))
    combine_pd.set_index(combine_pd["nidx"],inplace = True)
    # stat_pd = combine_pd[["date_org","|yt-yp|"]]
    # df_group = combine_pd.groupby("date_org")["|yt-yp|"].mean()
    combine_old_pd = combine_pd[combine_pd["is_present"]]
    combine_active = combine_pd[combine_pd["is_active"]]
    combine_nactive = combine_pd[~combine_pd["is_active"]]
    combine_nactive_pre = combine_pd[(combine_pd["is_active"]==False) & (combine_pd["is_present"]==True)]
    print(combine_pd.shape[0])
    print(combine_old_pd.shape[0])
    
    save_stat = Path(save_path).parent.joinpath("stat_combine.xlsx")
    
    stat_save(combine_pd,save_stat,it_ratiols=[0.1,0.05,0.01], cd_ratio = 0.1,message = "all_deal")
    stat_save(combine_old_pd,save_stat,it_ratiols=[0.1,0.05,0.01], cd_ratio = 0.1,message = "appeared")
    stat_save(combine_active,save_stat,it_ratiols=[0.1,0.05,0.01], cd_ratio = 0.1,message = "active")
    stat_save(combine_nactive,save_stat,it_ratiols=[0.1,0.05,0.01], cd_ratio = 0.1,message = "unactive")
    stat_save(combine_nactive_pre,save_stat,it_ratiols=[0.1,0.05,0.01], cd_ratio = 0.1,message = "unactive_appeared")
    
    combine_pd.to_csv(save_path,encoding = "utf-8-sig")

def time_revise(table_dir,save_dir):
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    for table_path in Path(table_dir).glob("*.csv"):
        table_pd = pd.read_csv(str(table_path),encoding="gbk")
        table_pd["date"] = table_pd.apply(lambda x:time.mktime(time.strptime(timestamp2date(x["deal_time"]),"%Y%m%d")),axis = 1)
        save_path = Path(save_dir).joinpath(table_path.name)
        table_pd.to_csv(str(save_path),encoding = "utf-8-sig")
if __name__ == "__main__":
    # time_revise(table_dir = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0817",
    #             save_dir = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0818")
    # getAllOrder(table_dir = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0818",
    #             save_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0818to0818\allData.csv")
    # saveTradeDay(table_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0818to0818\allData.csv",
    #              save_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0818to0818\tradeDate_distinct.json")
    # all_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0818to0818\allData.csv"
    # all_pd = pd.read_csv(all_path,encoding = "utf-8")
    # # print(all_pd["sec_name"])
    
    # date_json = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0818to0818\tradeDate_distinct.json"
    # distinct_datels = dtUtils.json_read(date_json)
    # valid_date = distinct_datels[-37]
    # train_date = distinct_datels[0:-37]
    # train_date = [str(int(_)) for _  in distinct_datels if int(_) < int(valid_date)]
    # train_valid(all_pd,valid_date,train_date)
    valid_combine(table_dir = r"D:\python_code\LSTM-master\bond_price\model\xgboost\23.08.30\res3",
                  save_path = r"D:\python_code\LSTM-master\bond_price\model\xgboost\23.08.30\res3_combine\valid_combine.csv")