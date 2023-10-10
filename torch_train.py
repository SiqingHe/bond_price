from data_torch import mydataset
from model_torch import mymodel,ResNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import numpy as np
from utils import get_data,scaler_trans
from torch_loss import Rmse
from xgboost_model import Xy_Value
from config import xgboost_cfg
xgb_cfg = xgboost_cfg.cfg
import pandas as pd


def train(model, X_train, y_train, num_epochs, learning_rate,save_path,batch_size):
    dataset=mydataset(X_train,y_train)
    datald=DataLoader(dataset,batch_size=batch_size,shuffle=False)
    # criterion = nn.MSELoss()
    criterion= Rmse()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 定义学习率衰减策略
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        num=0
        scheduler.step()
        print(scheduler.get_lr()[0])
        # self.optimizer.param_groups[0]["lr"]
        for X,y in datald:
            # print(X,y)
            # print(X)
            # print(y)
            # print(X.shape,y.shape)
            X=X.reshape(X.shape[0],1,X.shape[1]).float().cuda()
            # X=X.float()
            y=y.float().cuda()
            model.train()
            optimizer.zero_grad()
            # print(X.shape)
            outputs = model(X)
            # print(outputs)
            # print(outputs.shape)
            # print("预测结果:", outputs,"真实结果:",y,"预测误差",outputs-y)
            # print("预测误差",outputs-y)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if num%batch_size==0:
                print(f'data [{num+1}/{len(y_train)/batch_size}], Loss: {loss.item():.6f}')
            num+=1
        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
    torch.save(model.state_dict(), save_path)
    
if __name__=="__main__":
    pass
    # train_path=r"D:\python_code\LSTM-master\model_bond\bond_trdataNonull\train.json"
    # valid_path=r"D:\python_code\LSTM-master\model_bond\bond_trdataNonull\valid.json"
    # train_data=np.array(get_data(train_path))
    # X_train,y_train=train_data[:,0:-1],train_data[:,-1]
    # X_train,scalex=scaler_trans(X_train)
    # y_train,scaley=scaler_trans(y_train)
    train_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0818\train.csv"
    train_pd = pd.read_csv(train_path)
    X_train,y_train = Xy_Value(train_pd,xgb_cfg.X_COLUMN,xgb_cfg.Y_COLUMN)
    # model=mymodel()
    model=ResNet(1,1)
    batch_size=1000
    save_path=r"D:\python_code\LSTM-master\bond_price\model\torch\torch_resnet_nlrmse0913.pth"
    num_epochs=5
    learning_rate=1e-2
    X_train,scalex = scaler_trans(X_train.values)
    y_train,scaley = scaler_trans(y_train.values)
    train(model, X_train, y_train, num_epochs, learning_rate,save_path,batch_size)