from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path
import joblib
from utils import get_data,invert_scale,scaler_trans

def model_train(model,X_train,y_train,save_path):
    if not Path(save_path).exists():
        # 训练模型
        model.fit(X_train, y_train)
        Path(save_path).parent.mkdir(exist_ok=True,parents=True)
        joblib.dump(model, save_path)
    else:
        model=joblib.load(save_path)
    return model

def model_predict(model,X_test,y_test,scaley,save_txt):
    # X_test = [[x1, x2, ...], [x1, x2, ...], ...]  # 待预测的特征向量列表
    y_pred = model.predict(X_test)
    y_pred=y_pred.reshape(-1,1)
    if scaley is not None:
        y_pred=invert_scale(scaley,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    print("rmse",rmse)
    print(1/len(y_pred)*np.sum(np.abs(y_test-y_pred)))
    output_write(y_test,y_pred,save_txt)
    return rmse

def output_write(y_test,y_pred,save_txt):
    if not Path(save_txt).exists():
        Path(save_txt).parent.mkdir(exist_ok=True,parents=True)
    with open(save_txt,"w") as wr:
        for i in range(len(y_test)):
            if i==0:
                wr.write("y_pred y_test\n")
                wr.write(str(y_pred[i])+" "+str(y_test[i])+"\n")
            else:
                wr.write(str(y_pred[i])+" "+str(y_test[i])+"\n")
                
if __name__=="__main__":
    pass
    # train_path=r"D:\python_code\LSTM-master\bond_price\bond_trdataNonull\train.json"
    # valid_path=r"D:\python_code\LSTM-master\bond_price\bond_trdataNonull\valid.json"
    # train_data=np.array(get_data(train_path))
    # X_train,y_train=train_data[:,1:-1],train_data[:,-1]
    # X_train,scalex=scaler_trans(X_train)
    # y_train,scaley=scaler_trans(y_train)
    # valid_data=np.array(get_data(valid_path))
    # X_valid,y_valid=valid_data[:,1:-1],valid_data[:,-1]
    # X_valid=scalex.transform(X_valid)
    # y_valid,scaley=scaler_trans(y_valid)
    # y_valid=scaley.transform(np.array(y_valid).reshape(-1,1))
    train_path=r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split\train.xlsx"
    valid_path=r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split\valid.xlsx"
    train_pd=pd.read_excel(train_path)
    valid_pd=pd.read_excel(valid_path)
    # print(train_pd.dtypes)
    from dataPy.data_split import X_column,y_column
    # train_value=
    X_train,y_train=train_pd[X_column].values,train_pd[y_column].values
    X_valid,y_valid=valid_pd[X_column].values,valid_pd[y_column].values
    model=RandomForestRegressor(n_estimators=100,criterion='squared_error',n_jobs=10)
    save_path=r"D:\python_code\LSTM-master\bond_price\model\random_forest\forest0716_100.pkl"
    model=model_train(model,X_train,y_train,save_path)
    save_txt=r"D:\python_code\LSTM-master\bond_price\result\randForest\res_compare0716_100.txt"
    model_predict(model,X_valid,y_valid,None,save_txt)


