# coding=utf-8
from pandas import read_csv
from pandas import datetime
from pandas import concat
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from keras.models import load_model
from math import sqrt
from matplotlib import pyplot
import numpy
import json
import numpy as np
from pathlib import Path
import pickle
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from data2time import jsonSave
from keras.optimizers import adam_v2

from keras.callbacks import LearningRateScheduler

def schedule(epoch, learning_rate):
    # 自定义学习率衰减函数
    # 这里使用了每10个epoch将学习率减少为原来的一半的策略
    learning_rate=learning_rate/((epoch+1)*2)
    return learning_rate

lr_scheduler = LearningRateScheduler(schedule)

def fit_lstm(X,y, batch_size, nb_epoch, neurons,lr=5):
    # X, y = train[:, 0:-1], train[:, -1]
    # X = X.reshape(X.shape[0], 1, X.shape[1])
    train_rem=len(X)%(batch_size**2)
    real_num=len(X)-train_rem
    X=X[0:real_num]
    y=y[0:real_num]
    adam=adam_v2.Adam(lr=lr)
    model = Sequential()
    # 添加LSTM层
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    # model.add(Dense(59))
    # model.add(Dense(10,input_dim=59,activation="relu"))
    # # model.add(BatchNormalization())
    # model.add(Dense(8,activation="softmax"))
    # model.add(Dense(4))
    # model.add(BatchNormalization())
    model.add(Dense(1))
    # model.add(Dense(1,activation="linear"))  # 输出层1个node
    # model.add(Dropout(0.2))
    # 编译，损失函数mse+优化算法adam
    model.compile(loss='mean_squared_error', optimizer=adam)
    print("---------train start------------")
    model.fit(X, y, epochs=5, batch_size=batch_size, verbose=1, shuffle=False,callbacks=[lr_scheduler])
    # for i in range(nb_epoch):
    #     print("-------epoch {} start--------".format(i))
    #     # 按照batch_size，一次读取batch_size个数据
    #     model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False,callbacks=[lr_scheduler])
    #     print(model.optimizer.lr.numpy())
    #     model.reset_states()
    #     print("当前计算次数："+str(i))
    return model

def invert_scale(scaler, X):
    # print(X)
    # print(X.shape,type(X))
    # new_row = [x for x in X] + [value]
    # new_row=np.hstack((X,value))
    X_ar = np.array(X)
    # print(len(X_ar.shape))
    if len(X_ar.shape)<2:
        X_ar=X_ar.reshape(-1,1)
    # array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(X_ar)
    # print(inverted.shape)
    return inverted

# 1步长预测
def forcast_lstm(model, batch_size, X):
    # X = X.reshape(batch_size, 1, 59)
    yhat = model.predict(X, batch_size=batch_size)
    # print(yhat)
    return yhat

# print(train_scaled.shape)
# model_path=r"D:\python_code\LSTM-master\model_bond\model\bond_dense4_bt1_lin1_e5.h5"

# train_scaled=train_scaled[0:610500,:]
# train_data=train_data[0:610500,:]
# batch_size=1
def train_lstm(model_path,X_train,y_train,batch_size,nb_epoch, neurons):
    if not Path(model_path).exists():
        lstm_model = fit_lstm(X_train,y_train, batch_size, nb_epoch, neurons)  # 训练数据，batch_size，epoche次数, 神经元个数
        lstm_model.save(model_path)
    else:
        lstm_model=load_model(model_path)
    return lstm_model

def get_data(json_path):
    with open(json_path,"r") as wr:
        data=json.load(wr)
    return data

def pred_lstm(lstm_model,X_test,y_test,batch_size,scaler):
    predictions = list()
    valid_rem=len(X_test)%(batch_size**2)
    # print(len(X_test)-valid_rem)
    # len(X_test)-valid_rem
    for i in range(0,len(X_test)-valid_rem,batch_size):#根据测试数据进行预测，取测试数据的一个数值作为输入，计算出下一个预测值，以此类推
        # 1步长预测
        X, y = X_test[i:i+batch_size], y_test[i:i+batch_size]
        # X, y = valid_data[i:i+10, 0:-1], valid_data[i:i+10, -1]
        yhat = forcast_lstm(lstm_model, batch_size, X)
        # print("the value yhat-----------",yhat)
        # 逆缩放
        # print(mean_squared_error(yhat,y))
        yhat = invert_scale(scaler, yhat)
        # 逆差分
        # yhat = inverse_difference(raw_values, yhat, len(valid_scaled) + 1 - i)
        # print(yhat)
        # print(X.shape)
        # print(X[0],y[0])
        predictions+=np.array(yhat).tolist()
        expected = invert_scale(scaler,y)
        print('Moth={}, Predicted={}, Expected={}'.format(i + 1, yhat, expected))
    savePath=r"D:\python_code\LSTM-master\model_bond\timedata\y_pred.json"
    jsonSave(savePath,predictions)
    # 性能报告
    # valid_num=len(X_test)-valid_rem
    valid_num=len(X_test)-valid_rem
    # print(valid_data.shape,type(valid_data))
    # print(predictions.shape,type(valid_data))
    y_invscale=invert_scale(scaler,y_test)
    rmse = sqrt(mean_squared_error(y_invscale[0:valid_num], predictions))
    print('valid RMSE:%.3f' % rmse)
    
if __name__=="__main__":
    model_path=r"D:\python_code\LSTM-master\model_bond\model\bond_tm_lstm_lr1_yorg1.h5"
    X_train_path=r"D:\python_code\LSTM-master\model_bond\timedata\x_train.json"
    y_train_path=r"D:\python_code\LSTM-master\model_bond\timedata\y_train.json"
    X_test_path=r"D:\python_code\LSTM-master\model_bond\timedata\x_test.json"
    y_test_path=r"D:\python_code\LSTM-master\model_bond\timedata\y_test.json"
    batch_size=100
    # X_test=np.array(get_data(X_train_path))
    # y_test=np.array(get_data(y_train_path))
    X_train=np.array(get_data(X_train_path))
    y_train=np.array(get_data(y_train_path))
    
    # print(X_train.shape)
    # print(X_train[0:10,:,:])
    # print(y_train.shape)
    # print(y_train[0:10])
    
    # train_lstm(model_path,X_train,y_train,batch_size,nb_epoch=1, neurons=5)
    pkl_path=r"D:\python_code\LSTM-master\model_bond\timedata\scale_y.pkl"
    with open(pkl_path,"rb") as rd:
        scaley=pickle.load(rd)
    y_train=invert_scale(scaley,y_train)
    train_lstm(model_path,X_train,y_train,batch_size,nb_epoch=5, neurons=5)
    # lstm_model=load_model(model_path)
    
    # # X_test=np.array(get_data(X_test_path))
    # # y_test=np.array(get_data(y_test_path))
    # print(X_test.shape)
    # pred_lstm(lstm_model,X_test,y_test,batch_size,scaley)