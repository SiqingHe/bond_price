import dtUtils
from config.clm2English_old import rn_dic

def enum_trans(kind_json,rn_dic,save_json):
    kind_dic=dtUtils.json_read(kind_json)
    save_dic={}
    # print(clm2us_dic)
    for k,v in kind_dic.items():
        if k in rn_dic:
            k=rn_dic[k]
        if isinstance(v,dict):
            v=list(v.keys())
        save_dic[k]=v
    for k,v in rn_dic.items():
        if k[0:2]=="是否":
            save_dic[v]=["否","是"]
    dtUtils.jsonSave(save_json,save_dic)
    # return save_dic



if __name__=="__main__":
    enum_trans(kind_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\datakind_trans.json",
               rn_dic=rn_dic,
               save_json=r"D:\python_code\LSTM-master\bond_price\dataPy\config\kindEnum.json")
