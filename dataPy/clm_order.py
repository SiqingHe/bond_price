import pandas as pd
from pathlib import Path
import dtUtils

def re_order(dfs,new_order):
    dfs=dfs[new_order]
    return dfs

def reOrder_batch(table_dir,save_dir,new_order):
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(exist_ok=True,parents=True)
    for filepath in Path(table_dir).glob("*.csv"):
        csv_pd=pd.read_csv(filepath,header=0,index_col=0)
        csv_order=re_order(csv_pd,new_order)
        save_path=str(Path(save_dir).joinpath(filepath.name))
        dtUtils.csv_save(csv_order,save_path)
        
if __name__=="__main__":
    pass
    clm_path=r"D:\python_code\LSTM-master\bond_price\dataPy\config\column.json"
    clm=dtUtils.json_read(clm_path)
    reOrder_batch(table_dir=r"D:\python_code\LSTM-master\bond_price\real_data\bondIdGroup_rn",
                  save_dir=r"D:\python_code\LSTM-master\bond_price\real_data\bondIdGroup_rn_od",
                  new_order=clm)
        
        