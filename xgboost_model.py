import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn import metrics
from xgboost import plot_importance
from matplotlib import pyplot as plt
# from dataPy.data_split import X_column,y_column
from pathlib import Path
import joblib
import time
from config import xgboost_cfg
from dataPy import dtUtils
import os
import numpy as np
import logging

cfg = xgboost_cfg.cfg
cfg_param = xgboost_cfg.cfg.PARAM



def setsDtGet(train_path,valid_path,test_path):
    # train_pd = pd.read_excel(train_path)
    # valid_pd = pd.read_excel(valid_path)
    # test_pd = pd.read_excel(test_path)
    train_pd = pd.read_csv(train_path,encoding = "gbk")
    valid_pd = pd.read_csv(valid_path,encoding = "gbk")
    test_pd = pd.read_csv(test_path,encoding = "gbk")

    train_pd = train_pd.loc[(train_pd["yield"]<= 50) & (train_pd["yield"]>-20) & (train_pd["yield"]!= 0)]
    valid_pd = valid_pd.loc[(valid_pd["yield"]<= 50) & (valid_pd["yield"]>-20) & (valid_pd["yield"]!= 0)]
    test_pd = test_pd.loc[(test_pd["yield"]<= 50) & (test_pd["yield"]>-20) & (test_pd["yield"]!= 0)]
    # print(train_pd.describe()["yield"])

    # X_train,y_train = train_pd[X_column].values,train_pd[y_column].values
    # X_valid,y_valid = valid_pd[X_column].values,valid_pd[y_column].values
    # X_test,y_test = test_pd[X_column].values,test_pd[y_column].values
    return train_pd,valid_pd,test_pd

def xgbValue_get(train_pd,valid_pd,test_pd,X_column,y_column):
    X_train,y_train = train_pd[X_column],train_pd[y_column]
    X_valid,y_valid = valid_pd[X_column],valid_pd[y_column]
    X_test,y_test = test_pd[X_column],test_pd[y_column]
    return X_train,y_train,X_valid,y_valid,X_test,y_test

def param_xgbTrain(param,saveDir,num_round,X_train,y_train,X_valid,y_valid):
    # num_round  =  3500 迭代次数
    Path(saveDir).mkdir(exist_ok = True,parents = True)
    dtrain = xgb.DMatrix(data = X_train, label = y_train)
    dval = xgb.DMatrix(data = X_valid, label = y_valid)
    if cfg.EARLY_STOPPING:
        xgb_model = xgb.train(params = param, 
                            dtrain = dtrain, 
                            num_boost_round = num_round,
                            evals = [(dval, cfg.EVAL_NAME)], 
                            early_stopping_rounds = cfg.EARLY_STOP_ROUNDS)
    else:
        xgb_model = xgb.train(params = param, 
                            dtrain = dtrain, 
                            num_boost_round = num_round
                            )
    actual_num_rounds = xgb_model.best_iteration + 1
    print("actual num rounds",actual_num_rounds)
    logging.info("actual num rounds : {}".format(actual_num_rounds))
    model_path = Path(saveDir).joinpath("model.pkl")
    joblib.dump(xgb_model, model_path)
    return xgb_model

def param_xgbPredict(model,X_test,y_test ,message="train"):
    # model = joblib.load(model_path)
    dtest = xgb.DMatrix(data = X_test)
    y_pred = model.predict(dtest)
    # print(X_test)
    # print(y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    logging.info("{} mae:{}".format(message, str(mae)))
    print(message,"mae",mae)
    return y_pred

def residule_record(y_pred,test_pd,y_column,save_path,message="train"):
    y_test = test_pd[y_column].values
    y_pred = y_pred.reshape(-1,1)
    test_pd["y_pred"] = y_pred
    test_pd["yt-yp"] = y_test-y_pred
    test_pd["|yt-yp|"] = np.abs(y_test-y_pred)
    logging.info("{} 总债券数量 : {}".format(message,test_pd.shape[0]))
    interest_pd = test_pd[test_pd["ISSUERUPDATED"].isin([110,1180,1831,2047])]
    credit_pd = test_pd[~test_pd["ISSUERUPDATED"].isin([110,1180,1831,2047])]
    logging.info("{} 利率债数量 : {}".format(message,interest_pd.shape[0]))
    logging.info("{} 利率债mae : {}".format(message, np.mean(interest_pd["|yt-yp|"])))
    logging.info("{} 利率债mae大于0.1数量 : {}".format(message, interest_pd[interest_pd["|yt-yp|"]>0.1].shape[0]))
    interest_time_pd = interest_pd[(interest_pd["|yt-yp|"]>0.1) & (interest_pd["time_diff"]<3600) \
                                   & (interest_pd["time_diff"]!=-1)]
    logging.info("{} 利率债mae大于0.1 且时间差小于3600 数量 : {}".\
                 format(message, interest_time_pd.shape[0]))
    logging.info("{} 信用债数量 : {}".format(message, credit_pd.shape[0]))
    logging.info("{} 信用债mae : {}".format(message, np.mean(credit_pd["|yt-yp|"])))
    logging.info("{} 信用债mae大于0.1数量 : {}".format(message, credit_pd[credit_pd["|yt-yp|"]>0.1].shape[0]))
    credit_time_pd = credit_pd[(credit_pd["|yt-yp|"]>0.1) & (credit_pd["time_diff"]<3600) \
                               & (credit_pd["time_diff"]!=-1)]
    logging.info("{} 信用债mae大于0.1 且时间差小于3600 数量 : {}".\
                 format(message, credit_time_pd.shape[0]))
    test_pd.to_csv(save_path,encoding = "utf-8-sig")

def xgboost_analyse(model,save_path):
    plt.rcParams['font.size']  =  5
    plot_importance(model)#打印重要程度结果
    plt.savefig(save_path,dpi = 300,bbox_inches = 'tight')
    # plt.show()
    
def get_param(cfgPARAM):
    param = {
        'max_depth':cfgPARAM.MAX_DEPTH,
        'eval_metric': cfgPARAM.EVAL_METRIC,
        'eta':cfgPARAM.ETA, 
        'objective':cfgPARAM.OBJECTIVE,
        "tree_method":cfgPARAM.TREE_METHOD,
        "device":cfgPARAM.DEVICE,
        "subsample":cfgPARAM.SUBSAMPLE,
        "alpha":cfgPARAM.ALPHA,
        "verbosity":cfgPARAM.VERBOSITY
         }
    return param

def main():
    # TODO: add others xgboost
    current_path = os.path.abspath(os.path.dirname(__file__))
    model_save = dtUtils.increment_path(str(Path(current_path).joinpath(cfg.MODEL_SAVE)))
    Path(model_save).mkdir(exist_ok=True,parents=True)
    log_save = str(Path(model_save).joinpath("xgboost.log"))
    dtUtils.log_set(log_save,log_level = logging.INFO)
    param = get_param(cfg_param)
    train_path = str(Path(current_path).joinpath(cfg.TRAIN_PATH))
    valid_path = str(Path(current_path).joinpath(cfg.VALID_PATH))
    test_path = str(Path(current_path).joinpath(cfg.TEST_PATH))
    X_column = cfg.X_COLUMN
    y_column = cfg.Y_COLUMN
    train_pd,valid_pd,test_pd = setsDtGet(train_path,valid_path,test_path)
    X_train,y_train,X_valid,y_valid,X_test,y_test  =  xgbValue_get(train_pd,valid_pd,test_pd,X_column,y_column)
    dtUtils.configSave(cfg,model_save)
    model = param_xgbTrain(param,model_save,cfg.NUM_ROUND,X_train,y_train,X_valid,y_valid)
    train_pred = param_xgbPredict(model,X_train,y_train,message = "训练集")
    valid_pred = param_xgbPredict(model,X_valid,y_valid,message = "验证集")
    test_pred = param_xgbPredict(model,X_test,y_test,message = "测试集")
    save_train = str(Path(current_path).joinpath(model_save).joinpath("train_pred.csv"))
    save_valid = str(Path(current_path).joinpath(model_save).joinpath("valid_pred.csv"))
    save_test = str(Path(current_path).joinpath(model_save).joinpath("test_pred.csv"))
    residule_record(train_pred,train_pd,y_column,save_train,message = "训练集")
    residule_record(valid_pred,valid_pd,y_column,save_valid,message = "验证集")
    residule_record(test_pred,test_pd,y_column,save_test,message = "测试集")
    save_importance = str(Path(current_path).joinpath(model_save).joinpath("importance.png"))
    xgboost_analyse(model,save_importance)
if __name__ == "__main__":
    pass
    main()