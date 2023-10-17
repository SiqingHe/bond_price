import torch
from siameseNetwork import ContrastiveLoss,TransformerBackbone,SiameseNetwork
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from config import xgboost_cfg
xgb_cfg = xgboost_cfg.cfg
from data_torch import dataInference
from torch.utils.data import DataLoader
import copy
import heapq


def siamese_inference(table_path,input_dim,model_path,batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model = TransformerBackbone(input_dim=input_dim, hidden_dim=128, num_layers=6, num_heads=5)
    model = SiameseNetwork(transformer_model)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    table_pd = pd.read_csv(table_path)
    table_valid = table_pd[table_pd['date']>=1664553600]
    index = table_pd.shape[0] - table_valid.shape[0]
    table_pick = table_pd[xgb_cfg.X_COLUMN+xgb_cfg.Y_COLUMN]
    # table_pd = table_org[table_org['date']>=1664553600]
    # table_pick = table_pd[xgb_cfg.X_COLUMN+xgb_cfg.Y_COLUMN]
    # table_pick_copy = table_pick.copy()
    # table_pick_copy2 = table_pick.copy()
    # nums_clm = [_ for _  in xgb_cfg.X_COLUMN if _ not in xgb_cfg.TYPE_COLUMN]
    # table_pick_copy = table_pick.apply(lambda x: (x - x.mean()) / x.std())
    # table_pick_copy.fillna(-1,inplace = True)
    # for _ in nums_clm:
    #     table_pick_copy2[_] = table_pick_copy[_]
    # table_check = table_pick_copy2.iloc[index+1:,]
    table_check = table_pick.iloc[index+1:,]
    assert batch_size>10
    # print(model)
    
    with torch.no_grad():
        for ii,(_,item) in enumerate(table_check.iterrows()):
            # input_data1 = item.values[0:-1]
            y1 = item.values[-1]
            res = torch.zeros((5,)).cuda()
            infer_pd = table_pick.iloc[0:index+ii,:]
            print(infer_pd.shape)
            dataset = dataInference(infer_pd,item)
            # data_len = table_pick_copy2.shape
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=False,num_workers=20)
            for i, data in enumerate(dataloader):
                try:
                    # left = data_len - i*batch_size
                    input_data1,input_data2 = data
                    # input_len = min(left,batch_size)
                    input_len = input_data1.shape[0]
                    input_data1 = input_data1.reshape(input_len,1,input_dim).float().cuda()
                    input_data2 = input_data2.reshape(input_len,1,input_dim).float().cuda()
                    # print(input_data1.shape,input_data2.shape)
                    output1, output2 = model(input_data1, input_data2)
                    similarity_score = F.cosine_similarity(output1, output2, dim=1)
                    # similarity_score = F.pairwise_distance(output1, output2)
                    if res.equal(torch.zeros((5,)).cuda()):
                        res = copy.deepcopy(similarity_score)
                    else:
                        # print(type(res))
                        # print(type(similarity_score))
                        res = torch.cat((res,similarity_score))
                # print(res.shape)
                # print(similarity_score.cpu().numpy().tolist())
                except Exception as e:
                    print(e)
                    print(input_data1.shape,input_data2.shape)
            res_end = res.cpu().numpy().tolist()
            # arr_min = heapq.nsmallest(10,res_end)
            arr_min = heapq.nsmallest(1,res_end)
            index_min = list(map(res_end.index,arr_min))
            y_pred = infer_pd.iloc[index_min,]["yield"].iloc[0]
            print(y1,y_pred,y1-y_pred)
            
    
                
if __name__ == "__main__":
    pass
    siamese_inference(table_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0818to0818\allData.csv",
                      input_dim = len(xgb_cfg.X_COLUMN),
                      model_path = r"D:\python_code\LSTM-master\bond_price\model\torch\23.10.17\res8\torch_resnet_metric1016_1_0.pth",
                      batch_size = 2000)