import pandas as pd
from pathlib import Path
import numpy as np
import csv
from tqdm import tqdm
import os

def unusual(table_dir,save_dir):
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    save_outliers = str(Path(save_dir).joinpath("outlier.csv"))
    files_num = len(os.listdir(table_dir))
    for excel_path in tqdm(Path(table_dir).glob("*.csv"),total=files_num):
        table_pd = pd.read_csv(excel_path)
        table_pd = table_pd.loc[:,~table_pd.columns.str.contains("^Unnamed")]
        outliers = table_pd[(table_pd["time_diff"]<86400) * (np.abs(table_pd["yield"]-table_pd["yield-1"])>0.15)]
        table_len = table_pd.shape[0]
        if outliers.shape[0]<1:continue
        if not Path(save_outliers).exists():
            with open(save_outliers,"w",newline="") as csv_open:
                csv_write=csv.writer(csv_open)
                csv_write.writerow(outliers.columns.to_list())
        else:
            # data = outliers.values.tolist()
            with open(save_outliers,"a",newline="") as csv_open:
                csv_write = csv.writer(csv_open)
                csv_write.writerow(row.values.tolist())
                # if not isinstance(data[0],list) or len(data)==1:
                #     csv_write.writerow(data[0])
                # else:
                #     csv_write.writerows(outliers.values)
                bool2 = False
                yield_record,time_record = 0,0
                for idx,row in table_pd.iterrows():
                    # inter_l = idx
                    # inter_u = table_len-idx-1
                    # min_inter = min(inter_l,inter_u)
                    # max_inter = max(inter_l,inter_u)
                    begin = max(idx-6,0)
                    end = min(table_len,begin+12)
                    table_roll = table_pd.iloc[begin:end,:]
                    table_roll = table_roll[table_roll["yield"] > 0]
                    if table_roll.shape[0] < 1:
                        csv_write.writerow(row.values.tolist())
                        continue
                    mean_roll = np.mean(table_roll["yield"].values)
                    std_roll = np.std(table_roll["yield"].values)
                    row_dic = dict(row)
                    # bool1 = abs(row_dic["yield"]-row_dic["yield-1"])>0.5
                    # # and row_dic["time_diff"]<86400
                    # bool3 = abs(row_dic["yield"]-yield_record)>0.5
                    # # and row_dic["deal_time"]-time_record<86400
                    # if bool1 and not bool2:
                    #     csv_write.writerow(row.values.tolist())
                    #     yield_record = row_dic["yield-1"]
                    #     time_record = row_dic["deal_time"]-row_dic["time_diff"]
                    #     bool2 = True
                    # elif bool2 and bool3:
                    #     csv_write.writerow(row.values.tolist())
                    # else:
                    #     yield_record = row_dic["yield-1"]
                    #     time_record = row_dic["deal_time"]-row_dic["time_diff"]
                    #     bool2 = False
                    time_bool = row_dic["time_diff"]<259200 or (pd.isna(row_dic["time_diff"]) and table_pd.iloc[idx+1,:]["time_diff"]<259200)
                    if (abs(row_dic["yield"]-mean_roll) > max(std_roll*2,0.15) and time_bool) or row_dic["yield"] <=0:
                        print("*"*20)
                        print(table_roll[["yield","org_date"]])
                        print(row_dic["yield"],mean_roll,row_dic["yield"]-mean_roll,std_roll,std_roll*2)
                        print("*"*20)
                        csv_write.writerow(row.values.tolist())
def table_pd_sift(table_pd,outlier_pd,save_dir):
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    table_pd=table_pd.loc[:,~table_pd.columns.str.contains("^Unnamed")]
    bond_id = table_pd["bond_id"].values.tolist()[0]
    outlier_this = outlier_pd[outlier_pd["bond_id"]==bond_id]
    if outlier_this.shape[0]<1:
        save_name = bond_id+"_{}.csv".format(table_pd.shape[0])
        table_pd.to_csv(str(Path(save_dir).joinpath(save_name)),encoding="utf-8-sig")
        return 
    list_out = outlier_this[["org_date","yield"]].values.tolist()
    drop_ls = []
    for id,row in table_pd.iterrows():
        jd_ls = [row["deal_time"],row["yield"]]
        if jd_ls in list_out or row["yield"]<0:
            drop_ls.append(id)
    copy_pd = table_pd.copy()
    copy_pd.drop(drop_ls,inplace = True)
    save_name = bond_id+"_{}.csv".format(copy_pd.shape[0])
    copy_pd.to_csv(str(Path(save_dir).joinpath(save_name)),encoding="utf-8-sig")

def sift_batch(table_dir,outlier_path,save_dir):
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    dir_len = len(os.listdir(table_dir))
    for table_path in tqdm(Path(table_dir).glob("*.csv"),total = dir_len):
        table_pd = pd.read_csv(table_path)
        outlier_pd = pd.read_csv(outlier_path,encoding="gbk")
        table_pd_sift(table_pd,outlier_pd,save_dir)
if __name__ == "__main__":
    pass
    # test_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\dealed_0729\101900785.IB_32.csv"
    outlier_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\except_files\outlier08031.csv"
    # # table_pd = pd.read_csv(test_path)
    # # outlier_pd = pd.read_csv(outlier_path,encoding="gbk")
    table_dir = r"D:\python_code\LSTM-master\bond_price\real_data\excel2year_contat"
    save_dir = r"D:\python_code\LSTM-master\bond_price\real_data\excel2year_sift2"
    # table_pd_sift(table_pd,outlier_pd,save_dir)
    sift_batch(table_dir,outlier_path,save_dir)
    # unusual(table_dir=r"D:\python_code\LSTM-master\bond_price\dealed_dir\dealed_0729",
    #         save_dir=r"D:\python_code\LSTM-master\bond_price\dealed_dir\except_files")