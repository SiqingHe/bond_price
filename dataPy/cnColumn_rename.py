import pandas as pd
from config import clm2English_old
import dtUtils
from pathlib import Path

rn_dict=clm2English_old.rn_dic


def clm_rename_batch(table_dir,save_dir,rn_dict):
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(exist_ok=True,parents=True)
    for filepath in Path(table_dir).glob("*.csv"):
        df=pd.read_csv(str(filepath),header=0,index_col=0)
        name=filepath.stem
        df.rename(columns=rn_dict,inplace=1)
        tb_len=df.shape[0]
        save_path=str(Path(save_dir).joinpath("{}_{}.csv".format(name,tb_len)))
        dtUtils.csv_save(df,save_path)

if __name__=="__main__":
    pass
    clm_rename_batch(table_dir=r"D:\python_code\LSTM-master\bond_price\real_data\bondIdGroup",
                     save_dir=r"D:\python_code\LSTM-master\bond_price\real_data\bondIdGroup_rn",
                     rn_dict=rn_dict)

