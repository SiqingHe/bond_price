
import dtUtils
import pandas as pd
from pathlib import Path
from config import clm2english

def deal_feature_combine(deal_path,feature_path,save_dir):
    bond_id=Path(feature_path).stem.split("_")[0]
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(exist_ok=True,parents=True)
    deal_pd=pd.read_csv(deal_path,header=0,index_col=0)
    feature_pd=pd.read_csv(feature_path,header=0)
    # deal_pd.日期=deal_pd.apply(lambda x:dtUtils.tm_format_trans(x.日期),axis=1)
    feature_pd.rename(columns={'Unnamed: 0':"日期","SEC_NAME":"债券简称"},inplace=1)
    com_pd=pd.merge(deal_pd,feature_pd,how="left",on=["日期","债券简称"])
    
    com_pd.rename(columns=clm2english.rn_dic,inplace=1)
    pd_len=deal_pd.shape[0]
    save_path=Path(save_dir).joinpath("{}_{}.csv".format(bond_id,pd_len))
    dtUtils.csv_save(com_pd,save_path)

def com_batch(deal_dir,feature_dir,save_dir):
    num=0
    for deal_path in Path(deal_dir).glob("*.csv"):
        bond_id=deal_path.stem.split("_")[0]
        feat_path=Path(feature_dir).joinpath(bond_id+".csv")
        if not feat_path.exists():continue
        # if num>2:
        #     break
        deal_feature_combine(deal_path,feat_path,save_dir)
        num+=1
if __name__=="__main__":
    pass
    com_batch(deal_dir=r"D:\python_code\LSTM-master\bond_price\real_data\group2year_contat",
              feature_dir=r"D:\python_code\LSTM-master\bond_price\real_data\feature_get",
              save_dir=r"D:\python_code\LSTM-master\bond_price\real_data\dlFt_combine")