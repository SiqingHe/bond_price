import pandas as pd
from pathlib import Path
import time
# from . import dtUtils
import dtUtils
import re

def getAllOrder(table_dir,save_path):
    Path(save_path).parent.mkdir(exist_ok = True,parents = True)
    pd_ls = []
    for tablePath in Path(table_dir).glob("*.csv"):
        table_pd = pd.read_csv(tablePath,encoding = "gbk")
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
def valid_combine(table_dir,save_path):
    Path(save_path).parent.mkdir(exist_ok = True,parents = True)
    table_ls = []
    mark = re.compile(r"valid_\d{8}")
    for table_path in Path(table_dir).glob("*.csv"):
        match = re.search(mark,table_path.name).group(0)
        if match != table_path.stem:continue
        table_pd = pd.read_csv(str(table_path))
        table_ls.append(table_pd)
    combine_pd = pd.concat(table_ls)
    combine_pd["nidx"] = list(range(combine_pd.shape[0]))
    combine_pd = combine_pd.loc[:,~combine_pd.columns.str.contains("^Unnamed")]
    # combine_reindex = combine_pd.reindex(range(combine_pd.shape[0]))
    combine_pd.set_index(combine_pd["nidx"],inplace = True)
    # stat_pd = combine_pd[["date_org","|yt-yp|"]]
    # df_group = combine_pd.groupby("date_org")["|yt-yp|"].mean()
    combine_old_pd = combine_pd[combine_pd["is_present"]]
    print(combine_pd.shape[0])
    print(combine_old_pd.shape[0])
    stat_group = pd.DataFrame(columns = ["total_mean"])
    stat_group["total_mean"] = pd.DataFrame(combine_pd.groupby("date_org")["|yt-yp|"].mean())
    stat_group["total_count"] = pd.DataFrame(combine_pd.groupby("date_org")["|yt-yp|"].count())
    stat_group["interest_mean"] = pd.DataFrame(combine_pd[combine_pd["ISSUERUPDATED"].\
                                                          isin([110,1180,1831,2047])].groupby("date_org")["|yt-yp|"].mean())
    stat_group["interest_count"] = pd.DataFrame(combine_pd[combine_pd["ISSUERUPDATED"].\
                                                         isin([110,1180,1831,2047])].groupby("date_org")["|yt-yp|"].count())
    stat_group["credit_mean"] = pd.DataFrame(combine_pd[~combine_pd["ISSUERUPDATED"].\
                                                          isin([110,1180,1831,2047])].groupby("date_org")["|yt-yp|"].mean())
    stat_group["credit_count"] = pd.DataFrame(combine_pd[~combine_pd["ISSUERUPDATED"].\
                                                         isin([110,1180,1831,2047])].groupby("date_org")["|yt-yp|"].count())
    
    stat_group["total_pmean"] = pd.DataFrame(combine_old_pd.groupby("date_org")["|yt-yp|"].mean())
    stat_group["total_pcount"] = pd.DataFrame(combine_old_pd.groupby("date_org")["|yt-yp|"].count())
    stat_group["interest_pmean"] = pd.DataFrame(combine_old_pd[combine_old_pd["ISSUERUPDATED"].\
                                                          isin([110,1180,1831,2047])].groupby("date_org")["|yt-yp|"].mean())
    stat_group["interest_pcount"] = pd.DataFrame(combine_old_pd[combine_old_pd["ISSUERUPDATED"].\
                                                         isin([110,1180,1831,2047])].groupby("date_org")["|yt-yp|"].count())
    stat_group["credit_pmean"] = pd.DataFrame(combine_old_pd[~combine_old_pd["ISSUERUPDATED"].\
                                                          isin([110,1180,1831,2047])].groupby("date_org")["|yt-yp|"].mean())
    stat_group["credit_pcount"] = pd.DataFrame(combine_old_pd[~combine_old_pd["ISSUERUPDATED"].\
                                                         isin([110,1180,1831,2047])].groupby("date_org")["|yt-yp|"].count())
    save_stat = Path(save_path).parent.joinpath("stat_combine.csv")
    # ["|yt-yp|"].mean()
    # .agg({"|yt-yp|":["mean"]})
    # print(df_group,type(df_group))
    stat_group.to_csv(save_stat,encoding = "utf-8-sig")
    combine_pd.to_csv(save_path,encoding = "utf-8-sig")
if __name__ == "__main__":
    # getAllOrder(table_dir = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0808",
    #             save_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0808to0814\allData.csv")
    # saveTradeDay(table_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0808to0814\allData.csv",
    #              save_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0808to0814\tradeDate_distinct.json")
    # all_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0808to0814\allData.csv"
    # all_pd = pd.read_csv(all_path,encoding = "utf-8")
    # # print(all_pd["sec_name"])
    
    # date_json = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0808to0814\tradeDate_distinct.json"
    # distinct_datels = dtUtils.json_read(date_json)
    # valid_date = distinct_datels[-60]
    # train_date = [str(int(_)) for _  in distinct_datels if int(_) < int(valid_date)]
    # train_valid(all_pd,valid_date,train_date)
    valid_combine(table_dir = r"D:\python_code\LSTM-master\bond_price\model\xgboost\23.08.14\res2",
                  save_path = r"D:\python_code\LSTM-master\bond_price\model\xgboost\23.08.14\res2_combine\valid_combine.csv")