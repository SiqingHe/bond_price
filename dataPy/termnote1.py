import pandas as pd


def termnote1_look(excel_path):
    excel_pd=pd.read_csv(excel_path,header=0,index_col=0)
    termnote=excel_pd["TERMNOTE1"].to_list()
    termnote_plus=[_ for _ in termnote if "+" in _]
    print(termnote_plus)
    pass


if __name__=="__main__":
    termnote1_look(excel_path=r"D:\python_code\LSTM-master\bond_price\real_data\test\chengjiao_test.csv")