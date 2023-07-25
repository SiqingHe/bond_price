import pandas as pd
import numpy as np
from pathlib import Path
from utils import *
from collections import defaultdict
from shutil import copyfile

def dynamic_trans(excel_pd):
    pass

def feature_trans(excel_pd):
    save_dic={}
    for idx,item in excel_pd.iterrows():
        item_dic=item.to_dict()
        if idx in save_dic:
            print("repeat feature",idx,item_dic)
        save_dic[idx]=item_dic
    return save_dic

def excel_combine(pdlist:list(),save_path):
    combine1=pd.merge(pdlist[0],pdlist[1],how="outer",on=["债券ID","债券简称"])
    print(combine1.columns.to_list())
    combine2=pd.merge(combine1,pdlist[2],how="outer",on=["债券ID","债券简称","日期"])
    # combine2.to_csv(save_path,encoding="utf_8_sig")
    print(combine2.columns)
    # save_json=r"D:\python_code\LSTM-master\bond_price\real_data\column.json"
    # jsonSave(save_json,combine2.columns.to_list())
    jsonSort=r"D:\python_code\LSTM-master\bond_price\real_data\column_sort.json"
    columns_sort=get_data(jsonSort)
    combine3=combine2.reindex(columns=columns_sort)
    combine3.sort_values(by="成交时间",inplace=True)
    combine3.to_csv(save_path,encoding="utf_8_sig")
    pass

def combine_list(pd_list):
    dy_pd=pd.concat(pd_list,axis=0)
    return dy_pd

def splitById(excel_pd,save_dir):
    group_data=excel_pd.groupby("债券ID")
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    for group,df in group_data:
        times=df.shape[0]
        # if times>50:
        save_path=Path(save_dir).joinpath(group+"_{}_.csv".format(times))
        save_path=increment_path(save_path)
        csv_save(df,save_path)
    pass

def kind2():
    pass

def deal2bondId(deal_excel_dir,save_dir):
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(exist_ok=True,parents=True)
    for excel_pd in read_table_iter(deal_excel_dir,suffix=".xlsx"):
        splitById(excel_pd,save_dir)

def itertt(): 
    for i in range(10):yield i
    
def bondid_path_corr(deal_dir):
    bondid_dic=defaultdict(list)
    for filepath in Path(deal_dir).glob("*.csv"):
        name=filepath.stem
        name_split=name.split("_")
        bond_id=name_split[0]
        if name not in bondid_dic[bond_id]:
            bondid_dic[bond_id].append(name)
    # print(bondid_dic)
    return bondid_dic

def deal_feature_combine():
    path_ls=[r"D:\python_code\LSTM-master\bond_price\real_data\group2year\7103.IB_2_.csv",
                r"D:\python_code\LSTM-master\bond_price\real_data\wind7103.IB_2year.csv"]
    pd_list=[]
    for idx,ph in enumerate(path_ls):
        if idx==0:
            pds=pd.read_csv(ph,header=0,index_col=0)
            pds.日期=pds.apply(lambda x:tm_format_trans(x.日期),axis=1)
            # print(type(pds["日期"].iloc[0]))
        else:
            pds=pd.read_csv(ph,header=0)
            pds.rename(columns={'Unnamed: 0':"日期","SEC_NAME":"债券简称"},inplace=1)
        print(pds.columns)
        pd_list.append(pds)
    combine_save=r"D:\python_code\LSTM-master\bond_price\real_data\cmb_tt.csv"
    com_pd=column_combine(pd_list)
    csv_save(com_pd,combine_save)
    
def bondid_corr_combine(group_dir,save_dir):
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(exist_ok=True,parents=True)
    bondid_dic=bondid_path_corr(group_dir)
    for k in bondid_dic.keys():
        if len(bondid_dic[k])==1:
            name=bondid_dic[k][0].strip("_")
            copyfile(Path(group_dir).joinpath(bondid_dic[k][0]+".csv"),
                     Path(save_dir+"1").joinpath(name+".csv"))
        else:
            excel_dir=[str(Path(group_dir).joinpath(_+".csv")) for _ \
                in bondid_dic[k]]
            pd_list=read_csv_batch(excel_dir)
            combine_pd=combine_list(pd_list)
            times=combine_pd.shape[0]
            save_path=Path(save_dir).joinpath(k+"_{}.csv".format(times))
            combine_pd.sort_values(by="日期", inplace=True, ascending=True)
            csv_save(combine_pd,save_path)

def json_rem(json_origin,save_new,fetched_dir):
    with open(json_origin,"r",encoding="utf-8-sig") as wr:
        jsori=json.load(wr)
    fetched=[_.stem for _ in Path(fetched_dir).glob("*")]
    new_dic={}
    for k,v in jsori.items():
        if k not in fetched:
            new_dic[k]=sorted(set(v),key=v.index)
    print(len(new_dic))
    print(len(jsori))
    jsonSave(save_new,new_dic)


if __name__=="__main__":
    pass
    json_rem(json_origin=r"D:\python_code\LSTM-master\bond_price\config\bondid_time_remain0718.json",
             save_new=r"D:\python_code\LSTM-master\bond_price\config\bondid_time_remain0724.json",
             fetched_dir=r"D:\python_code\LSTM-master\bond_price\feature_data\data_all")
    # feature_path=r"E:\hsq_material\cjhx\data\债券属性数据(1).xlsx"
    # deal_path=r"E:\hsq_material\cjhx\data\成交2023年1月2月.xlsx"
    # dynamic_path1=r"E:\hsq_material\cjhx\data\债券动态数据\20230103-20230110.xlsx"
    # dynamic_path2=r"E:\hsq_material\cjhx\data\债券动态数据\20230111-20230120.xlsx"
    # dynamic_path3=r"E:\hsq_material\cjhx\data\债券动态数据\20230201-20230210.xlsx"
    # dynamic_path4=r"E:\hsq_material\cjhx\data\债券动态数据\20230211-20230228.xlsx"
    # feature_pd=read_excel(feature_path)
    # deal_pd=read_excel(deal_path)
    # dy_pd1=read_excel(dynamic_path1)
    # dy_pd2=read_excel(dynamic_path2)
    # dy_pd3=read_excel(dynamic_path3)
    # dy_pd4=read_excel(dynamic_path4)
    # dy_pd=pd.concat([dy_pd1,dy_pd2,dy_pd3,dy_pd4],axis=0)
    # save_dypath=r"D:\python_code\LSTM-master\bond_price\real_data\dy_combine.csv"
    # csv_save(dy_pd,save_dypath)
    # save_path=r"D:\python_code\LSTM-master\bond_price\real_data\deal_feature_dy_sort.csv"
    # excel_combine([feature_pd,deal_pd,dy_pd],save_path)
    # feature_trans(feature_pd)
    # list_path=r"E:\hsq_material\cjhx\data\dynamic0630"
    # pd_list=read_excel_batch(list_path)
    # combine_pd=combine_list(pd_list)
    # # save_combine=r"D:\python_code\LSTM-master\bond_price\real_data\combine_0703.csv"
    # # csv_save(combine_pd,save_combine)
    # combine_chs=combine_pd[["日期","债券ID","创金内部主体评级"]]
    # path1=r"E:\hsq_material\cjhx\data\成交2023年1月2月copy.xlsx"
    # jiaoyi_pd=read_excel(path1)
    # combine_cj=column_combine([jiaoyi_pd,combine_chs])
    # save_combine=r"D:\python_code\LSTM-master\bond_price\real_data\combine_cj.csv"
    # csv_save(combine_cj,save_combine)
    # path_comb=r"E:\hsq_material\cjhx\data\chengjiao_value.xlsx"
    # excel_pd=read_excel(path_comb)
    # # splitById(excel_pd,
    # #           save_dir=r"D:\python_code\LSTM-master\bond_price\real_data\bondIdGroup50")
    # columns=excel_pd.columns.to_list()
    # save_path=r"D:\python_code\LSTM-master\bond_price\real_data\column2english.json"
    # jsonSave(save_path,columns)
    # deal2bondId(deal_excel_dir=r"E:\hsq_material\cjhx\data\deal_data",
    #             save_dir=r"D:\python_code\LSTM-master\bond_price\real_data\group2year")
    # bondid_path_corr(deal_dir=r"D:\python_code\LSTM-master\bond_price\real_data\group2year")
    # bondid_corr_combine(group_dir=r"D:\python_code\LSTM-master\bond_price\real_data\group2year",
    #                     save_dir=r"D:\python_code\LSTM-master\bond_price\real_data\group2year_contat")