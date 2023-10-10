from data_torch import metricDataset,metricCacheDataset,dataIter
from model_torch import MetricNet,TripleLoss
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import numpy as np
from torch_loss import Rmse
import h5py
# from xgboost_model import Xy_Value
from config import xgboost_cfg
xgb_cfg = xgboost_cfg.cfg
import pandas as pd
from siameseNetwork import ContrastiveLoss,TransformerBackbone,SiameseNetwork,similarity
import time
import sys
import signal
from pathlib import Path


def metric_train(train_path,input_dim,embedding_dim,batch_size,save_path,
                 num_epochs = 2,
                 margin = 0.1,
                 num_workers = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MetricNet(1, embedding_dim)
    model.to(device)
    # 创建 Triple Loss 损失函数
    triplet_loss = TripleLoss(margin)
    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # train_pd = pd.read_csv(train_path)
    # mtDataset = metricDataset(train_pd,xgb_cfg.X_COLUMN,xgb_cfg.Y_COLUMN)
    file = h5py.File(train_path, "r")
    mtDataset = metricCacheDataset(file)
    mtDataLoader = DataLoader(mtDataset, batch_size=batch_size, shuffle=False )#,num_workers = num_workers
    
    # 训练循环
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        num = 0
        for batch_data in mtDataLoader:  # 假设你有一个用于训练的数据加载器
            # anchor, positive, negative = batch_dataat
            # anchor = batch_data['anchor']
            # positive = batch_data['positive']
            # negative = batch_data['negative']
            anchor,positive,negative = batch_data
            # X = X.reshape(X.shape[0],1,X.shape[1]).float().cuda()
            if num == len(mtDataLoader)-1:
                batch_temp = len(mtDataset)%batch_size
                anchor = anchor[:,0:-1].reshape(batch_temp,1,input_dim).float().cuda()
                positive = positive[:,0:-1].reshape(batch_temp,1,input_dim).float().cuda()
                negative = negative[:,0:-1].reshape(batch_temp,1,input_dim).float().cuda()
            else:
                anchor = anchor[:,0:-1].reshape(batch_size,1,input_dim).float().cuda()
                positive = positive[:,0:-1].reshape(batch_size,1,input_dim).float().cuda()
                negative = negative[:,0:-1].reshape(batch_size,1,input_dim).float().cuda()
            # anchor = torch.cat(anchor).reshape(input_dim,1,batch_size).permute(2,0,1).float().cuda()
            # positive = torch.cat(positive).reshape(input_dim,1,batch_size).permute(2,0,1).float().cuda()
            # negative = torch.cat(negative).reshape(input_dim,1,batch_size).permute(2,0,1).float().cuda()
            
            optimizer.zero_grad()
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)
            loss = triplet_loss(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if num%batch_size==0:
            print(f'data [{num+1}/{len(mtDataLoader)}], Loss: {loss.item():.6f}')
            num += 1
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(mtDataLoader)}')

    print('Finished Training')
    torch.save(model.state_dict(), save_path)
    
def save_training_state(signal, frame):
    print("Received Ctrl+C signal. Saving training state...")
    checkpoint = {
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_except)
    sys.exit(0)  # 退出程序


   
def siameseTrain(input_dim,train_path,batch_size,save_path,margin = 0.5,epochs = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建Siamese数据集
    dfs = pd.read_csv(train_path)
    dfs = dfs[xgb_cfg.X_COLUMN+xgb_cfg.Y_COLUMN]
    dfs_copy = dfs.copy()
    dfs_copy2 = dfs.copy()
    nums_clm = [_ for _  in xgb_cfg.X_COLUMN if _ not in xgb_cfg.TYPE_COLUMN]
    dfs_copy = dfs.apply(lambda x: (x - x.mean()) / x.std())
    dfs_copy.fillna(-1,inplace = True)
    for _ in nums_clm:
        dfs_copy2[_] = dfs_copy[_]
    #     dfs_copy[_] = dfs.apply(lambda x: (x[_] - x[_].mean()) / x[_].std(), axis = 1)
    idx_types = [xgb_cfg.X_COLUMN.index(_) for _ in xgb_cfg.TYPE_COLUMN]
    idx_nums = [xgb_cfg.X_COLUMN.index(_) for _  in xgb_cfg.X_COLUMN if _ not in xgb_cfg.TYPE_COLUMN]
    dataset = dataIter(dfs_copy2,idx_types,idx_nums)
    # batch_size = 3
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=20)
    # 创建Siamese网络和损失函数
    transformer_model = TransformerBackbone(input_dim=input_dim, hidden_dim=256, num_layers=2, num_heads=3)
    net = SiameseNetwork(transformer_model)
    criterion = ContrastiveLoss(margin)
    optimizer = optim.Adam(net.parameters(), lr = 0.005)
    
    # 训练Siamese网络
    try:
        for epoch in range(epochs):
            for i, data in enumerate(dataloader, 0):
                tic = time.time()
                data1,data2,label = data
                data1 = data1.float().cuda()
                data2 = data2.float().cuda()
                label = label.int().cuda()
                data1_x = data1[:,0:-1]
                data1_y = data1[:,-1]
                data2_x = data2[:,0:-1]
                data2_y = data2[:,-1]
                # tic1 = time.time()
                # print("1",tic1-tic)
                # dis = similarity(data1_x,data2_x,idx_types,idx_nums)
                # # tic2 = time.time()
                # # print("2",tic2-tic1)
                # label =  torch.where(((data1_y - data2_y)<0.05) & (dis>0.9)) > 5, 1, torch.where(vector < -5, -1, 0))
                # (((data1_y - data2_y)<0.05) & (dis>0.9)).to(torch.int).cuda()
                optimizer.zero_grad()
                data1_x = data1_x.reshape(batch_size,1,input_dim).float().cuda()
                data2_x = data2_x.reshape(batch_size,1,input_dim).float().cuda()
                output1, output2 = net(data1_x, data2_x)
                # tic3 = time.time()
                # print("3",tic3-tic2)
                loss_contrastive = criterion(output1, output2, label)
                # tic4 = time.time()
                # print("4",tic4-tic3)
                loss_contrastive.backward()
                optimizer.step()
                # tic5 = time.time()
                # print("5",tic5-tic4)
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataloader)}, Loss: {loss_contrastive.item()}")
                if i%10000 == 0 :
                    save_except = str(Path(save_path).parent.joinpath(Path(save_path).stem+"_{}_{}.pth".format(epoch,i)))
                    torch.save(net.state_dict(), save_except)
                # 注册信号处理函数
                # 
                # signal.signal(signal.SIGINT, save_training_state)
    # except Exception as e:
    except KeyboardInterrupt as e:
        print(f"Exception occurred: {str(e)}")
    # 在异常发生时保存模型状态
        checkpoint = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }
        save_except = str(Path(save_path).parent.joinpath(Path(save_path).stem+"_{}_{}.pth".format(epoch,i)))
        torch.save(checkpoint, save_except)
    print("Siamese Network Training Finished!")

    # 保存训练好的Siamese网络模型
    torch.save(net.state_dict(), save_path)


if __name__ == "__main__":
    pass
    # train_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0818\valid.h5"
    # input_dim = len(xgb_cfg.X_COLUMN)
    # embedding_dim = 20
    # save_path = r"D:\python_code\LSTM-master\bond_price\model\torch\torch_resnet_metric0915.pth"
    # metric_train(train_path,input_dim,embedding_dim,
    #              save_path = save_path,
    #              batch_size = 100 , num_epochs = 2, margin = 0.1)
    input_dim = len(xgb_cfg.X_COLUMN)
    train_path =  r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0818\train.csv"
    batch_size = 200
    # 10000
    save_path = r"D:\python_code\LSTM-master\bond_price\model\torch\torch_resnet_metric1009.pth"
    try:
        siameseTrain(input_dim,train_path,batch_size,save_path,margin = 1,epochs = 10)
    except Exception as e:
        print(e)