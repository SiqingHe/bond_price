import pandas as pd
from pathlib import Path
import dtUtils

def deal_concat(deal_dir,save_dir):
    # Path(save_dir).mkdir(exist_ok=True,parents=True)
    df_list=[]
    for excel_ph in Path(deal_dir).glob("*"):
        dfs=pd.read_excel(str(excel_ph))
        df_list.append(dfs)
    df_combine=pd.concat(df_list)
    df_combine.to_csv(r"E:\hsq_material\cjhx\data\deal_combined.csv",encoding="utf-8-sig")
    pass

def split_time(table_path,save_dir):
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    tb_pds=pd.read_csv(table_path)
    # print(tb_path["日期"].to_list())
    split_time_ls=[
                    ["20210101","20210331"],
                    ["20210401","20210630"],
                    ["20210701","20210930"],
                    ["20211001","20211231"],
                    ["20220101","20220331"],
                    ["20220401","20220630"],
                    ["20220701","20220930"],
                    ["20221001","20221231"],
                    ["20230101","20230331"]
                   ]
    print(tb_pds.shape[0])
    total=0
    for tm in split_time_ls:
        tb_pd=tb_pds.copy()
        split_pd=tb_pd.loc[(tb_pd["日期"]>=int(tm[0])) & (tb_pd["日期"]<=int(tm[1]))]
        print(split_pd.shape[0])
        if split_pd.shape[0]<1:
            continue
        total+=split_pd.shape[0]
        save_name="deal_{}_{}.csv".format(tm[0],tm[1])
        split_pd.sort_values(by="成交时间",inplace=True)
        split_pd.to_csv(str(Path(save_dir).joinpath(save_name)),encoding="utf-8-sig")
    print("split_total",total)

if __name__=="__main__":
    pass
    # deal_concat(deal_dir=r"E:\hsq_material\cjhx\data\deal_data",
    #             save_dir=r"")
    split_time(table_path=r"E:\hsq_material\cjhx\data\deal_combined.csv",
               save_dir=r"E:\hsq_material\cjhx\data\deal_split")