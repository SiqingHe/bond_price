import torch
from siameseNetwork import ContrastiveLoss,TransformerBackbone,SiameseNetwork
import pandas as pd
import torch.nn.functional as F
from config import xgboost_cfg
xgb_cfg = xgboost_cfg.cfg
from data_torch import dataInference
from torch.utils.data import DataLoader


def siamese_inference(table_path,input_dim,model_path,batch_size):
    transformer_model = TransformerBackbone(input_dim=input_dim, hidden_dim=256, num_layers=2, num_heads=3)
    model = SiameseNetwork(transformer_model)
    model.load_state_dict(torch.load(model_path))
    table_pd = pd.read_csv(table_path)
    table_pick = table_pd[xgb_cfg.X_COLUMN+xgb_cfg.Y_COLUMN]
    table_pick_copy = table_pick.copy()
    table_pick_copy2 = table_pick.copy()
    nums_clm = [_ for _  in xgb_cfg.X_COLUMN if _ not in xgb_cfg.TYPE_COLUMN]
    table_pick_copy = table_pick.apply(lambda x: (x - x.mean()) / x.std())
    table_pick_copy.fillna(-1,inplace = True)
    for _ in nums_clm:
        table_pick_copy2[_] = table_pick_copy[_]
    assert batch_size>10
    with torch.no_grad():
        for _,item in table_pick_copy2.iterrows():
            # input_data1 = item.values[0:-1]
            # y1 = item.values[-1]
            dataset = dataInference(table_pick_copy2,item)
            dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=False,num_workers=20)
            for i, data in enumerate(dataloader, 0):
                input_data1,input_data2 = data
                output1, output2 = model(input_data1, input_data2)
                similarity_score = F.cosine_similarity(output1, output2, dim=1)
                print(similarity_score)
                
if __name__ == "__main__":
    pass
    siamese_inference(table_path = r"",
                      input_dim = len(xgb_cfg.X_COLUMN),
                      model_path = r"",
                      batch_size = 200)