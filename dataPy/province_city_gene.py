import pandas as pd
import dtUtils
from pathlib import Path
import csv
import dtUtils
import clm_deal
from functools import partial
import os
from tqdm import tqdm

#TODO: this province_city_id.xlsx is unincomplete, need be replaced
def region_gene(table_path,save_json):
    tb_pd=pd.read_excel(table_path,header=None)
    rank_ls=["province","city","town"]
    save_dic=dict()
    for rk in rank_ls:
        save_dic[rk]={}
    save_dic["province"]={}
    save_dic["city"]={}
    save_dic["town"]={}
    for id,item in tb_pd.iterrows():
        item_dic=item.to_dict()
        value=list(item_dic.values())[0]
        value_split=value.split(".")
        region=value_split[1]
        region_id=value_split[0]
        if len(region_id)==2:
            save_dic["province"][region]=region_id
            save_dic["province"][region_id]=region
            save_dic["city"][region]={}
            save_dic["town"][region]={}
        elif len(region_id)==4:
            province_id=region_id[0:2]
            province=save_dic["province"][province_id]
            save_dic["city"][province][region]=region_id
            save_dic["city"][province][region_id]=region
            # save_dic["city"][region]=region_id
            # save_dic["city"][region_id]=region
            # save_dic["town"][province][region]={}
            # save_dic["town"][province][region]={}
        else:
            province_id=region_id[0:2]
            city_id=region_id[0:4]
            province=save_dic["province"][province_id]
            city=save_dic["city"][province][city_id]
            # city=save_dic["city"][city_id]
            if region=="市辖区":
                save_dic["town"][province][city]=region_id
                save_dic["town"][province][region_id]=city
            else:
                save_dic["town"][province][region]=region_id
                save_dic["town"][province][region_id]=region
    dtUtils.jsonSave(save_json,save_dic)

def get2022(table_path,save_path):
    Path(save_path).parent.mkdir(exist_ok=True,parents=True)
    table_pd = pd.read_csv(table_path,encoding="gbk")
    table_pd2022 = table_pd[table_pd["DATA_YEAR"]=="2022年"]
    table_pd2022.to_csv(save_path,encoding="utf-8-sig")
    
def ibms_province_deal(table_path,save_dir):
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    table_pd = pd.read_csv(table_path)
    save_dic,write = {},{}
    save_province = Path(save_dir).joinpath("province.csv")
    save_town = Path(save_dir).joinpath("town.csv")
    sp_province = ["重庆市","北京市","上海市","天津市"]
    for idx,row in table_pd.iterrows():
        if pd.isna(row["CITY"]) and pd.isna(row["COUNTY"]) and row["PROVINCE"] not in sp_province:
            if not Path(save_province).exists():
                with open(save_province,"w",newline="") as save_dic["province"]:
                    write["province"] = csv.writer(save_dic["province"])
                    write["province"].writerow(table_pd.columns.to_list())
            else:
                with open(save_province,"a",newline="") as save_dic["province"]:
                    write["province"] = csv.writer(save_dic["province"])
                    write["province"].writerow(row.values.tolist())
        else:
            if not Path(save_town).exists():
                with open(save_town,"w",newline="") as save_dic["town"]:
                    write["town"] = csv.writer(save_dic["town"])
                    write["town"].writerow(table_pd.columns.to_list())
            else: 
                with open(save_town,"a",newline="") as save_dic["town"]:
                    write["town"]=csv.writer(save_dic["town"])
                    if pd.isna(row["CITY"]) and pd.isna(row["COUNTY"]) and row["PROVINCE"] in sp_province:
                        row["COUNTY"] = row["PROVINCE"]
                    elif pd.isna(row["COUNTY"]):
                        row["COUNTY"] = row["CITY"]
                    write["town"].writerow(row.values.tolist())
def merge_test(test_path,town_path,save_path):
    test_pd = pd.read_csv(test_path,encoding = "gbk")
    # print(test_pd[test_pd["CITY"].isin([-1,-2])])
    # print(test_pd.shape[0])
    town_pd = pd.read_csv(town_path,encoding = "utf-8-sig")
    # town_pd.rename(columns = {"CITY":"CITY_OLD","COUNTY":"CITY"},inplace = 1)
    merge_pd = pd.merge(test_pd,town_pd,how="left",on=["PROVINCE","CITY","MUNICIPALBOND"])
    merge_pd = merge_pd.loc[:,~merge_pd.columns.str.contains("^Unnamed")]
    merge_pd.fillna(-1,inplace=True)
    if test_pd.shape[0]!=merge_pd.shape[0]:
        print(Path(test_path).name,test_pd.shape[0],merge_pd.shape[0])
    # merge_pd.drop_duplicates(keep='first',inplace=True)
    # print(merge_pd.shape[0])
    merge_pd.to_csv(save_path,encoding="utf-8-sig")

def merge_batch(table_dir,town_path,save_dir):
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    dir_num = len(os.listdir(table_dir))
    for table_path in tqdm(Path(table_dir).glob("*.csv"),total = dir_num):
        save_path = str(Path(save_dir).joinpath(table_path.name))
        merge_test(str(table_path),town_path,save_path)

def sp_province_del(x):
    return x.strip("市")
def city_now(province,city,town,region_dict):
    sp_province = ["天津","北京","上海","重庆"]
    # province = province.replace("黔西南","黔西南布依族苗族自治州").\
    #     replace("")
    if province in sp_province:
        province+="市"
    if isinstance(city,str):
        city = city.replace("巿","市")
    if isinstance(town,str):
        town = town.replace("巿","市")
        town = town.replace("泰贤区","奉贤区")
    repeated_town = list(region_dict["repeated"].values())
    try:
        if town in region_dict["province"]:
            return region_dict["province"][province]+"0000"
        elif city in region_dict["province"]:
            return region_dict["town"][city][town]
        elif province in region_dict["city"] and town in region_dict["city"][province]:
            return region_dict["city"][province][town]
        elif city in region_dict["town"][province] and town in region_dict["town"][province][city]:
            return region_dict["town"][province][city][town]
        elif town not in repeated_town:
            return region_dict["town1"][province][town]
        else:
            print(province,city,town)
            return -2
    except Exception as e:
        print(e)
        print(province,city,town)
        return -3
    
def town_deal(town_path,save_path,region_json):
    town_pd = pd.read_csv(town_path,encoding = "gbk")
    town_pd.rename(columns = {"CITY":"CITY_OLD","COUNTY":"CITY"},inplace = 1)
    copy_pd = town_pd.copy()
    region_dict=dtUtils.json_read(region_json)
    
    province_partial=partial(clm_deal.province,region_dict=region_dict)
    city_partial=partial(city_now,region_dict=region_dict)
    town_pd["PROVINCE"] = town_pd.apply(lambda x:sp_province_del(x["PROVINCE"]),axis=1)
    copy_pd["PROVINCE"] = town_pd.apply(lambda x:province_partial(x["PROVINCE"]),axis=1)
    copy_pd["CITY"] = town_pd.apply(lambda x:city_partial(x["PROVINCE"],x["CITY_OLD"],x["CITY"]),axis=1)
    # copy_pd["CITY"] = town_pd.apply(lambda x:region_dict["town"][x[""]])
    keeped_column = ["PROVINCE","CITY","GDP","GENERAL_BUDGET_MONEY","SSSR_RADIO","CZZJL",
                     "ZFXJJ_MONEY","ZFZWYE","QYFZCTYX","QYFZCTCXZ","QYFZCTCXZ_RADIO",
                     "CTYXZWZS","CTYXZWBS"]
    copy_pd = copy_pd[keeped_column]
    copy_pd.to_csv(save_path,encoding = "utf-8-sig")
    
def repeated_region(table_path,save_path):
    table_pd = pd.read_csv(table_path)
    repeated_index = table_pd.duplicated(["PROVINCE","CITY","COUNTY"])
    repeated_pd = table_pd[repeated_index]
    pass

def code_repeated_region():
    pass

def municipalBond_add(table_path,save_path):
    table_pd = pd.read_csv(table_path,index_col=0)
    table_pd["MUNICIPALBOND"] = [1]*table_pd.shape[0]
    table_pd.to_csv(save_path,encoding = "utf-8-sig")
# MUNICIPALBOND
if __name__=="__main__":
    # table_path=r"D:\python_code\LSTM-master\bond_price\real_data\province_city_id.xlsx"
    # region_gene(table_path,
    #             save_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\province_city_new.json")
    
    # get2022(table_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\\DATAD.csv",
    #         save_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\city_data.csv")
    # ibms_province_deal(table_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\city_data.csv",
    #                    save_dir = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add")
    # merge_test(test_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\dealed_0803\220019.IB_29381.csv",
    #            town_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\town_end.csv",
    #            save_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\220019.IB_29381.csv")
    
    # town_deal(town_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\town.csv",
    #          save_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\town_end.csv",
    #          region_json = r"D:\python_code\LSTM-master\bond_price\dataPy\config\region.json")
    
    merge_batch(table_dir = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0808",
                town_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\town_endct.csv",
                save_dir = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0809region")
    # repeated_region(table_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\city_data.csv",
    #                 save_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\city_repeated.csv")
    # municipalBond_add(table_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\town_end.csv",
    #                   save_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\province_add\town_endct.csv")