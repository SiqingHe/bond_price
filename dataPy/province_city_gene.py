import pandas as pd
import dtUtils

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


if __name__=="__main__":
    table_path=r"D:\python_code\LSTM-master\bond_price\real_data\province_city_id.xlsx"
    region_gene(table_path,
                save_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\province_city_new.json")