import pandas as pd
import dtUtils

def column_save(table_path,save_json):
    df=pd.read_csv(table_path,header=0,index_col=0)
    dtUtils.jsonSave(save_json,df.columns.to_list())
    
if __name__=="__main__":
    column_save(table_path=r"D:\python_code\LSTM-master\bond_price\real_data\dlFt_combine\010011.IB_2.csv",
                save_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\column.json")