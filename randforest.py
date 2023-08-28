from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
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
    mae=mean_absolute_error(y_test,y_pred)
    print("rmse",rmse)
    print(1/len(y_pred)*np.sum(np.abs(y_test-y_pred)))
    print("mae",mae)
    output_write(y_test,y_pred,save_txt)
    return rmse,y_pred

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

def test_analysis(y_pred,test_pd,y_column,save_path):
    y_test=test_pd[y_column].values
    test_pd["y_pred"]=y_pred
    test_pd["yt-yp"]=y_test-y_pred
    test_pd.describe()["yt-yp"]
    test_pd.to_csv(save_path,encoding="utf-8-sig")
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
    train_path=r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0808\train.csv"
    valid_path=r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0808\valid.csv"
    test_path=r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0808\test.csv"
    # train_pd=pd.read_excel(train_path)
    # valid_pd=pd.read_excel(valid_path)
    # test_pd=pd.read_excel(test_path)
    train_pd=pd.read_csv(train_path,encoding="gbk")
    valid_pd=pd.read_csv(valid_path,encoding="gbk")
    test_pd=pd.read_csv(test_path,encoding="gbk")
    
    # train_pd=train_pd.loc[(train_pd["yield"]<=50) & (train_pd["yield"]>-20) & (train_pd["yield"]!=0)]
    # valid_pd=valid_pd.loc[(valid_pd["yield"]<=50) & (valid_pd["yield"]>-20) & (valid_pd["yield"]!=0)]
    # test_pd=test_pd.loc[(test_pd["yield"]<=50) & (test_pd["yield"]>-20) & (test_pd["yield"]!=0)]
    # print(train_pd.describe()["yield"])
    # from dataPy.data_split import X_column,y_column
    from config import xgboost_cfg
    cfg = xgboost_cfg.cfg
    X_column = cfg.X_COLUMN
    y_column = cfg.Y_COLUMN
    # train_value=
    X_train,y_train=train_pd[X_column].values,train_pd[y_column].values
    X_valid,y_valid=valid_pd[X_column].values,valid_pd[y_column].values
    X_test,y_test=test_pd[X_column].values,test_pd[y_column].values
    model=RandomForestRegressor(n_estimators=90,criterion='squared_error',n_jobs=10)
    save_path=r"D:\python_code\LSTM-master\bond_price\model\random_forest\forest0809_90_term_p0_no0.pkl"
    model=model_train(model,X_train,y_train,save_path)
    save_txt=r"D:\python_code\LSTM-master\bond_price\result\randForest\res_compare_ts0809_90_term_p0_no0_train.txt"
    rmse,y_pred=model_predict(model,X_train,y_train,None,save_txt)
    test_analysis(y_pred,train_pd,y_column,
                  save_path=r"D:\python_code\LSTM-master\bond_price\result\randForest\res_compare_ts0809_90_term_p0_no0_train.csv")
    save_txt=r"D:\python_code\LSTM-master\bond_price\result\randForest\res_compare_ts0809_90_term_p0_no0_val.txt"
    rmse,y_pred=model_predict(model,X_valid,y_valid,None,save_txt)
    test_analysis(y_pred,valid_pd,y_column,
                  save_path=r"D:\python_code\LSTM-master\bond_price\result\randForest\res_compare_ts0809_90_term_p0_no0_val.csv")


