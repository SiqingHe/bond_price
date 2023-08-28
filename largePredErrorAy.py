import json 
import pandas as pd
import pickle
from utils import get_data
import numpy as np

def visualExcelGene(valid_json,columns,idx_pkl,save_path):
    with open(idx_pkl,"rb") as rd:
        bigEId=pickle.load(rd)
    print(bigEId)
    valid_data=get_data(valid_json)
    bigEData=np.array(valid_data)[bigEId]
    valid_pd=pd.DataFrame(np.array(bigEData),columns=columns)
    valid_pd.to_csv(save_path)
    
def get_columns(excel_path):
    excelPd=pd.read_csv(excel_path,header=0,index_col=0)
    column=excelPd.columns.tolist()
    columns=[column[0]]+column[2:]+[column[1]]
    # print(columns)
    return columns

if __name__=="__main__":
    pass
    excelpath=r"D:\python_code\LSTM-master\bond_data\train.csv"
    columns=get_columns(excelpath)
    # valid_json=r"D:\python_code\LSTM-master\bond_price\bond_trdataNonull\valid.json"
    train_json=r"D:\python_code\LSTM-master\bond_price\bond_trdataNonull\train.json"
    idx_pkl=r"D:\python_code\LSTM-master\bond_price\bond_trdataNonull\mr50_trLabel.pkl"
    save_path=r"D:\python_code\LSTM-master\bond_price\bond_trdataNonull\mr50_tr.csv"
    visualExcelGene(train_json,columns,idx_pkl,save_path)
    
    