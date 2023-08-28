import torch
from utils import invert_scale,get_data,scaler_trans
import numpy as np
from model_torch import mymodel,ResNet
from torch.utils.data import DataLoader
from data_torch import mydataset
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import pickle
import xgboost_model
import pandas as pd
from config import xgboost_cfg
xgb_cfg = xgboost_cfg.cfg


def test(model,test_data,y_test,scaler,batch_size=30):
    model.eval()
    # test_data = torch.tensor([[6.0, 0.1], [7.0, 0.2]], dtype=torch.float32)
    test_len=len(test_data)
    dataset=mydataset(test_data,y_test)
    datald=DataLoader(dataset,batch_size=batch_size,shuffle=False)
    y_pred=[]
    y_true=[]
    with torch.no_grad():
        for X_test,y in datald:
            X_test=X_test.reshape(X_test.shape[0],1,X_test.shape[1]).float()
            y=y.float()
            # X_test,y=test_data[i:i+batch_size],y_test[i:i+batch_size]
            predicted_labels = model(X_test)
            # yhat=predicted_labels
            yhat = invert_scale(scaler, predicted_labels)
            y_real=invert_scale(scaler,y)
            # print("预测结果:", predicted_labels,"真实结果:",y,"预测误差",predicted_labels-y)
            yhat=yhat.reshape(-1,).tolist()
            y_real=y_real.reshape(-1,).tolist()
            y_pred+=yhat
            y_true+=y_real
            # print(math.sqrt(mean_squared_error(yhat, y_real)))
            print(math.sqrt(mean_absolute_error(yhat, y_real)))
            # print("预测结果:", yhat,"真实结果:",y_real,"预测误差",np.array(yhat)-np.array(y_real))
    rmse = math.sqrt(mean_squared_error(y_pred, y_true))
    mae = math.sqrt(mean_absolute_error(y_pred, y_true))
    y_minus=np.abs(np.array(y_true)-np.array(y_pred))
    y_lg10=np.where(y_minus>50)
    # print(y_lg10,y_lg10[0].shape)
    pkl_save=r"D:\python_code\LSTM-master\bond_price\bond_trdataNonull\mr50_trLabel.pkl"
    with open(pkl_save,"wb") as wr:
        pickle.dump(y_lg10,wr)
    # print(test_data[y_lg10],len(test_data[y_lg10]))
    print('valid RMSE:%.3f' % rmse)
    print('valid MAE:%.3f' % mae)
            
if __name__=="__main__":
    # train_path=r"D:\python_code\LSTM-master\bond_price\bond_trdataNonull\train.json"
    # valid_path=r"D:\python_code\LSTM-master\bond_price\bond_trdataNonull\valid.json"
    # train_data=np.array(get_data(train_path))
    # X_train,y_train=train_data[:,0:-1],train_data[:,-1]
    # X_train,scalex=scaler_trans(X_train)
    # y_train,scaley=scaler_trans(y_train)
    # valid_data=np.array(get_data(valid_path))
    # X_valid,y_valid=valid_data[:,0:-1],valid_data[:,-1]
    # X_valid=scalex.transform(X_valid)
    train_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0818\train.csv"
    train_pd = pd.read_csv(train_path)
    X_train,y_train = xgboost_model.Xy_Value(train_pd,xgb_cfg.X_COLUMN,xgb_cfg.Y_COLUMN)
    X_train,scalex = scaler_trans(X_train.values)
    y_train,scaley = scaler_trans(y_train.values)
    
    valid_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0818\valid.csv"
    valid_pd = pd.read_csv(valid_path)
    X_valid,y_valid =xgboost_model.Xy_Value(valid_pd,xgb_cfg.X_COLUMN,xgb_cfg.Y_COLUMN)
    X_valid=scalex.transform(X_valid.values)
    y_valid,scaley=scaler_trans(y_valid.values)
    y_valid=scaley.transform(np.array(y_valid).reshape(-1,1))
    load_path=r"D:\python_code\LSTM-master\bond_price\model\torch\torch_resnet_nlrmse0828.pth"
    # model=mymodel()
    model=ResNet(1,1)
    model.load_state_dict(torch.load(load_path))
    # test(model,X_train,y_train,scaley)
    test(model,X_valid,y_valid,scaley)