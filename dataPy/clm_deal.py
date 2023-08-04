import pandas as pd
import dtUtils
import time
from functools import partial
from pathlib import Path
from tqdm import tqdm

def province(x,region_dict):
    # if x is None or x=="0":
    if x=="0" or x==-1 or x==0:
        return x
    if x not in region_dict["province"]:
        # print(x)
        x=str(x)+"市"
    return region_dict["province"][x]

def city(province,city,region_dict):
    if city=="0" or city==-1 or city==0:
        return city
    # if province=="北京市":
    #     print(province,city)
    sp_province=["北京","上海","天津","重庆","11","12","31","50"]
    sp_city = ["北京市","上海市","天津市","重庆市"]
    if province in sp_province and city in sp_city:
        return region_dict["province"][province+"市"]+"0000"
    else:
        if province in sp_province:
            province += "市"
        # if city not in region_dict["city"][province]:
        #     return town_city_iter(province,city,region_dict)
        if city not in region_dict["town"][province]:
            city_t=city.replace("市","县")
            if city  in region_dict["city"][province]:
                return region_dict["city"][province][city]+"00"
            elif city_t in region_dict["town"][province]:
                return region_dict["town"][province][city_t]
            print(province,city)
            return -2
        return region_dict["town"][province][city]

def town_city_iter(province,city,region_dict):
    # print(province)
    for  k,v in region_dict["town"][province].items():
        # print(k,v)
        for kk,vv in region_dict["town"][province][k].items():
            # print(kk,vv)
            if kk==city:
                return vv
    print(city)
    return city
def time_clm(tt,deal=False):
    if tt is None or tt=="0" or tt==-1 or tt==0:
        # print(tt)
        return tt
    tt=str(tt)
    if deal:
        if " " in tt:
            s_t=time.strptime(tt,"%Y-%m-%d %H:%M:%S")
        else:
            s_t=time.strptime(tt,"%Y%m%d-%H:%M:%S")
    else:
        tt=tt.strip(" 00:00:00")
        if "/" in str(tt):
            s_t=time.strptime(tt,"%Y/%m/%d")
        elif "-" in tt:
            s_t=time.strptime(tt,"%Y-%m-%d")
        elif "GMT" in tt:
            s_t=time.strptime(tt,"%a, %d %b %Y %H:%M:%S GMT")
        else:
            s_t=time.strptime(tt,"%Y%m%d")
    return time.mktime(s_t)

def termnote1(x):
    x=str(x)
    x_split=x.split("+")
    return x_split[0]

def termnote2(x):
    x=str(x)
    x_split=x.split("+")
    if len(x_split)>1:
        res=x_split[1]
    else:
        return None
    return res if res!="--" else 0.5

def termnote3(x):
    x=str(x)
    x_split=x.split("+")
    if len(x_split)>2:
        res=x_split[2]
    else:
        return None
    return res if res!="N" else 100
def issueUpdated(x,issue_list):
    if x is None or x==0 or x==-1:
        return 0
    else:
        return issue_list.index(x)+1

def enum_series(x,enum_dic,column):
    if x is None or x=="0" or x==-1 or x==0:
        return x
    if column=="CLAUSEABBR":
        x_split_sort=sorted(x.split(","))
        x=",".join(x_split_sort)
    if x not in enum_dic[column]:
        print(column,x,type(x))
        return -2
    return enum_dic[column].index(x)+1

def column_trans(clm,enum_json,noEnum_json,region_json):
    enum_dic=dtUtils.json_read(enum_json)
    enum_partial=partial(enum_series,
                         enum_dic=enum_dic,
                         column=clm)
    noEnum_dict=dtUtils.json_read(noEnum_json)
    issue_list=noEnum_dict["ISSUERUPDATED"]
    guarantor_list=noEnum_dict["AGENCY_GUARANTOR"]
    region_dict=dtUtils.json_read(region_json)
    issue_partial=partial(issueUpdated,issue_list=issue_list)
    guarantor_partial=partial(agency_guarantor,guarantor_list=guarantor_list)
    province_partial=partial(province,region_dict=region_dict)
    city_partial=partial(city,region_dict=region_dict)
    clm_func={
    "date":time_clm,
    "bond_id":None,
    "sec_name":None,
    "deal_time":partial(time_clm,deal=1),
    "net_price":None,
    "full_price":None,
    "yield":None,
    "cnbd4_expire_yield":None,
    "cnbd3_op_yield":None,
    "deal-cnbd_bp":None,
    "PTMYEAR":None,
    "TERMNOTE1":termnote1,## TODO
    "TERMIFEXERCISE":None,
    "ISSUERUPDATED":issue_partial, ##TODO
    "LATESTPAR":None,
    "ISSUEAMOUNT":None,
    "OUTSTANDINGBALANCE":None,
    "LATESTISSURERCREDITRATING":enum_partial,
    "RATINGOUTLOOKS":enum_partial,
    "RATE_RATEBOND":enum_partial,
    "RATE_RATEBOND2":enum_partial,
    "RATE_LATESTMIR_CNBD":enum_partial,
    "INTERESTTYPE":enum_partial,
    "COUPONRATE":None,
    "COUPONRATE2":None,
    "PROVINCE":province_partial,
    "CITY":city_partial,
    "MUNICIPALBOND":enum_partial,
    "WINDL2TYPE":enum_partial,
    "SUBORDINATEORNOT":enum_partial,
    "PERPETUALORNOT":enum_partial,
    "PRORNOT":enum_partial,
    "INDUSTRY_SW":enum_partial,
    "ISSUE_ISSUEMETHOD":enum_partial,
    "EXCH_CITY":enum_partial,
    "TAXFREE":enum_partial,
    "MULTIMKTORNOT":enum_partial,
    "IPO_DATE":time_clm,
    "MATURITYDATE":time_clm,
    "NXOPTIONDATE":time_clm,
    "NATURE1":enum_partial,
    "AGENCY_GRNTTYPE":enum_partial,
    "AGENCY_GUARANTOR":guarantor_partial,
    "RATE_RATEGUARANTOR":enum_partial,
    "LISTINGORNOT1":None,
    "EMBEDDEDOPT":enum_partial,
    "CLAUSEABBR":enum_partial
    }
    return clm_func[clm]
def agency_guarantor(x,guarantor_list):
    if x is None or x=="0" or x==-1 or x==0:
        return x
    return guarantor_list.index(x)+1

def shift_value(df,time,column):
    df[column+"-{}".format(time)]=df[column].shift(time)
    return df

def clauseabbr(x,idx):
    if x is None or x=="0" or x==-1 or x==0:
        return x
    clause_list=["A","C","Etc","P","Pr","S","T"]
    x_split=x.split(",")
    if clause_list[idx] in x_split:
        return 1
    else:
        return 0

def table_trans(csv_path,save_path,enum_json,noEnum_json,region_json):
    csv_pd=pd.read_csv(csv_path)
    csv_pd.fillna(-1,inplace=True)
    # csv_pd=pd.read_excel(csv_path)
    columns=csv_pd.columns.to_list()
    copy_pd=csv_pd.copy()
    ignore_ls=['Unnamed: 0','cjhx_rate','cjhx_quantile',"Unnamed: 0.1","CLAUSEABBR"]
    for column in columns:
        if column in ignore_ls:continue
        trans=column_trans(column,enum_json,noEnum_json,region_json)
        if trans is not None:
            if column!="CITY":
                copy_pd[column]=csv_pd.apply(lambda x:trans(x[column]),axis=1)
            else:
                copy_pd[column]=csv_pd.apply(lambda x:trans(x["PROVINCE"],x[column]),axis=1)
    copy_pd.sort_values(by="deal_time",ascending=True,inplace=True)
    copy_pd["termnote2"]=csv_pd.apply(lambda x:termnote2(x["TERMNOTE1"]),axis=1)
    copy_pd["termnote3"]=csv_pd.apply(lambda x:termnote3(x["TERMNOTE1"]),axis=1)
    for i in range(7):
        copy_pd["CLAUSEABBR_{}".format(i)]=csv_pd.apply(lambda x:clauseabbr(x["CLAUSEABBR"],i),axis=1)
    copy_pd=copy_pd.loc[(copy_pd["yield"]<=50) & (copy_pd["yield"]>-20) & (copy_pd["yield"]!=0)]
    if copy_pd.shape[0]<1:
        return
    copy_pd["org_date"] = csv_pd["deal_time"]
    copy_pd["time_diff"] = copy_pd["deal_time"].diff()
    for yi in range(5):
        copy_pd["yield-{}".format(yi+1)] = copy_pd['yield'].shift(yi+1)
        copy_pd["time_diff-{}".format(yi+1)] = copy_pd['time_diff'].shift(yi+1)
        copy_pd["yd*diff-{}".format(yi+1)] = copy_pd["yield-{}".format(yi+1)]*copy_pd["time_diff-{}".format(yi+1)]
    # copy_pd["yield-1"]=copy_pd['yield'].shift(1)
    # copy_pd["yield-2"]=copy_pd['yield'].shift(2)
    # copy_pd["yield-3"]=copy_pd['yield'].shift(3)
    # copy_pd["yield-4"]=copy_pd['yield'].shift(4)
    # copy_pd["yield-5"]=copy_pd['yield'].shift(5)
    
    # print(copy_pd.dtypes)
    # csv_pd["PROVINCE"]=csv_pd.apply(lambda x:province(x["PROVINCE"],province_dict),axis=1)
    dtUtils.csv_save(copy_pd,save_path)

def trans_batch(csv_dir,save_dir,enum_json,noEnum_json,region_json):
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    import os
    dir_num=len(os.listdir(csv_dir))
    num=0
    for csv_path in tqdm(Path(csv_dir).glob("*.csv"),total=dir_num):
        save_path=str(Path(save_dir).joinpath(csv_path.name))
        if Path(save_path).exists():continue
        # if num>10:break
        table_trans(csv_path,save_path,enum_json,noEnum_json,region_json)
        # try:
        #     table_trans(csv_path,save_path,enum_json,noEnum_json,region_json)
        # except Exception as e:
        #     print(e)
        #     print(csv_path.name)
        #     continue
        num+=1
if __name__=="__main__":
    pass
    # province_dict=dtUtils.json_read(region_json)
    # table_trans(csv_path=r"D:\python_code\LSTM-master\bond_price\real_data\combine_dir\dlFt_combine0714\010216.IB_7.csv",
    #             save_path=r"D:\python_code\LSTM-master\bond_price\real_data\test\010216.IB_7_test.csv",
    #             enum_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\kindEnum.json",
    #             noEnum_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\no_Enum\noEnum_2023-07-14.15_15_57.json",
    #             region_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\province_city_add.json")
    trans_batch(csv_dir=r"D:\python_code\LSTM-master\bond_price\real_data\excel2year_sift2",
                save_dir=r"D:\python_code\LSTM-master\bond_price\dealed_dir\dealed_08032",
                enum_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\kindEnum_0726.json",
                noEnum_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\no_Enum\noEnum_2023-07-26.20_15_14.json",
                region_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\province_city_add.json")