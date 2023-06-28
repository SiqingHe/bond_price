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


def train(model, X_train, y_train, num_epochs, learning_rate,save_path,batch_size):
    dataset=mydataset(X_train,y_train)
    datald=DataLoader(dataset,batch_size=batch_size,shuffle=False)
    # criterion = nn.MSELoss()
    criterion= Rmse()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 定义学习率衰减策略
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(num_epochs):
        num=0
        scheduler.step()
        print(scheduler.get_lr()[0])
        # self.optimizer.param_groups[0]["lr"]
        for X,y in datald:
            # print(X,y)
            # print(X.shape)
            X=X.reshape(X.shape[0],1,X.shape[1]).float()
            # X=X.float()
            y=y.float()
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
    train_path=r"D:\python_code\LSTM-master\model_bond\bond_trdataNonull\train.json"
    valid_path=r"D:\python_code\LSTM-master\model_bond\bond_trdataNonull\valid.json"
    train_data=np.array(get_data(train_path))
    X_train,y_train=train_data[:,0:-1],train_data[:,-1]
    X_train,scalex=scaler_trans(X_train)
    y_train,scaley=scaler_trans(y_train)
    # model=mymodel()
    model=ResNet(1,1)
    batch_size=100
    save_path=r"D:\python_code\LSTM-master\model_bond\model\torch\torch_resnet_nlrmse.pth"
    num_epochs=20
    learning_rate=1e-4
    train(model, X_train, y_train, num_epochs, learning_rate,save_path,batch_size)