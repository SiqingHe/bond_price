# -*- coding: utf-8 -*-
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from xgboost import plot_importance
from matplotlib import pyplot as plt
# from dataPy.data_split import X_column,y_column
from pathlib import Path
import joblib
import time
import datetime
from config import xgboost_cfg
from dataPy import dtUtils
import os
import numpy as np
import logging
from dataPy import updateRecycle

cfg = xgboost_cfg.cfg
cfg_param = xgboost_cfg.cfg.PARAM

pdtype =[
           ("deal_time",int),
           ("date",int),
           ("PTMYEAR",float),
           ("TERMNOTE1",float),
           ("TERMIFEXERCISE",float),
           ("ISSUERUPDATED",int),
         ("LATESTPAR",float),
         ("ISSUEAMOUNT",float),
         ("OUTSTANDINGBALANCE",float),
         ("LATESTISSURERCREDITRATING",int),
         ("RATINGOUTLOOKS",int),
         ("RATE_RATEBOND",int),
         ("RATE_RATEBOND2",int),
         ("RATE_LATESTMIR_CNBD",int),
         ("INTERESTTYPE",int),
         ("COUPONRATE",float),
         ("COUPONRATE2",float),
         ("PROVINCE",int),
         ("CITY",int),
         ("MUNICIPALBOND",int),
         ("WINDL2TYPE",int),
         ("SUBORDINATEORNOT",int),
         ("PERPETUALORNOT",int),
         ("PRORNOT",int),
         ("INDUSTRY_SW",int),
         ("ISSUE_ISSUEMETHOD",int),
         ("EXCH_CITY",int),
         ("TAXFREE",int),
         ("MULTIMKTORNOT",int),
         ("IPO_DATE",int),
         ("MATURITYDATE",float),
         ("NXOPTIONDATE",float),
         ("NATURE1",int),
         ("AGENCY_GRNTTYPE",int),
         ("AGENCY_GUARANTOR",int),
         ("RATE_RATEGUARANTOR",int),
         ("LISTINGORNOT1",int),
         ("EMBEDDEDOPT",int),
         ("CLAUSEABBR_0",int),
         ("CLAUSEABBR_1",int),
         ("CLAUSEABBR_2",int),
         ("CLAUSEABBR_3",int),
         ("CLAUSEABBR_4",int),
         ("CLAUSEABBR_5",int),
         ("CLAUSEABBR_6",int),
         ("termnote2",float),
         ("termnote3",int),
         ("time_diff",float),
         ("time_diff-1",float),
         ("time_diff-2",float),
         ("time_diff-3",float),
         ("time_diff-4",float),
         ("time_diff-5",float),
         ("yd*diff-1",float),
         ("yield-1",float),
         ("yield-2",float),
         ("yield-3",float),
         ("yield-4",float),
         ("yield-5",float),
         ("GDP",float),
         ("GENERAL_BUDGET_MONEY",float),
         ("SSSR_RADIO",float),
         ("CZZJL",float),
           ("ZFXJJ_MONEY",float),
           ("ZFZWYE",float),
           ("QYFZCTYX",float),
           ("QYFZCTCXZ",float),
           ("QYFZCTCXZ_RADIO",float),
           ("CTYXZWZS",float),
           ("CTYXZWBS",float)
]

def setsDtGet(train_path,valid_path,test_path):
    # train_pd = pd.read_excel(train_path)
    # valid_pd = pd.read_excel(valid_path)
    # test_pd = pd.read_excel(test_path)
    type = dict(pdtype)
    
    encoding_tp = "utf-8-sig"
    train_pd = pd.read_csv(train_path,encoding = encoding_tp,dtype=type)
    valid_pd = pd.read_csv(valid_path,encoding = encoding_tp,dtype=type)
    test_pd = pd.read_csv(test_path,encoding = encoding_tp,dtype=type)
    
    # print(set(train_pd["GDP"].values.tolist()))

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

def Xy_Value(train_pd,X_column,y_column):
    X_train,y_train = train_pd[X_column],train_pd[y_column]
    return X_train,y_train

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
    interest_err_pd = interest_pd[interest_pd["|yt-yp|"]>0.1]
    logging.info("{} 利率债mae大于0.1数量 : {}".format(message, interest_err_pd.shape[0]))
    
    interest_err_pd5bp = interest_pd[interest_pd["|yt-yp|"]>0.05]
    logging.info("{} 利率债mae大于0.05数量: {},比例为: {}".format(message, 
                                                        interest_err_pd5bp.shape[0],
                                                        interest_err_pd5bp.shape[0]/interest_pd.shape[0]))
    interest_err_pd1bp = interest_pd[interest_pd["|yt-yp|"]>0.01]
    logging.info("{} 利率债mae大于0.01数量: {},比例为: {}".format(message, 
                                                        interest_err_pd1bp.shape[0],
                                                        interest_err_pd1bp.shape[0]/interest_pd.shape[0]))
    
    save_interest_err = str(Path(save_path).parent.joinpath(Path(save_path).stem+"_berr_interset.csv"))
    interest_err_pd.to_csv(save_interest_err,encoding="utf-8-sig")
    
    interest_time_pd = interest_pd[(interest_pd["|yt-yp|"]>0.1) & (interest_pd["time_diff"]<3600) \
                                   & (interest_pd["time_diff"]!=-1)]
    logging.info("{} 利率债mae大于0.1 且时间差小于3600 数量 : {}".\
                 format(message, interest_time_pd.shape[0]))
    logging.info("{} 信用债数量 : {}".format(message, credit_pd.shape[0]))
    logging.info("{} 信用债mae : {}".format(message, np.mean(credit_pd["|yt-yp|"])))
    
    credit_err_pd = credit_pd[credit_pd["|yt-yp|"]>0.1]
    logging.info("{} 信用债mae大于0.1数量 : {}".format(message, credit_err_pd.shape[0]))
    
    credit_time_pd = credit_pd[(credit_pd["|yt-yp|"]>0.1) & (credit_pd["time_diff"]<3600) \
                               & (credit_pd["time_diff"]!=-1)]
    logging.info("{} 信用债mae大于0.1 且时间差小于3600 数量 : {}".\
                 format(message, credit_time_pd.shape[0]))
    
    credit_city_pd = credit_pd[credit_pd["MUNICIPALBOND"]==1]
    logging.info("{} 城投债数量 : {}".format(message,credit_city_pd.shape[0]))
    logging.info("{} 城投债mae : {}".format(message, np.mean(credit_city_pd["|yt-yp|"])))

    save_credit_err = str(Path(save_path).parent.joinpath(Path(save_path).stem+"_berr_credit.csv"))
    credit_err_pd.to_csv(save_credit_err,encoding="utf-8-sig")
    
    
    test_pd.to_csv(save_path,encoding = "utf-8-sig")

def xgboost_analyse(model,save_path,importance_type="weight"):
    plt.rcParams['font.size']  =  4
    plot_importance(model,importance_type=importance_type)#打印重要程度结果
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
        "colsample_bytree":cfgPARAM.COLSAMPLE_BYTREE,
        "verbosity":cfgPARAM.VERBOSITY
         }
    return param

def randForest(X_train,y_train,save_dir):
    model = RandomForestRegressor(n_estimators = cfg.RF.N_ESTIMATORS,max_depth = cfg.RF.MAX_DEPTH,
                                #   min_samples_leaf = 8,min_samples_split=8,
                                  criterion = 'squared_error',n_jobs = cfg.RF.N_JOBS)
    save_path = str(Path(save_dir).joinpath("rf_model.pkl"))
    if not Path(save_path).exists():
        # 训练模型
        model.fit(X_train, y_train)
        Path(save_path).parent.mkdir(exist_ok=True,parents=True)
        joblib.dump(model, save_path)
    else:
        model=joblib.load(save_path)
    return model
def main():
    # TODO: add others xgboost
    current_path = os.path.abspath(os.path.dirname(__file__))
    today = datetime.datetime.now().strftime('%y.%m.%d')
    model_save = dtUtils.increment_path(str(Path(current_path).joinpath(cfg.MODEL_TAG).joinpath(today).joinpath(cfg.MODEL_SAVE)))
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
    # rf_model = randForest(X_train,y_train,model_save)
    train_pred = param_xgbPredict(model,X_train,y_train,message = "训练集")
    valid_pred = param_xgbPredict(model,X_valid,y_valid,message = "验证集")
    test_pred = param_xgbPredict(model,X_test,y_test,message = "测试集")
    
    # rf_train_pred = rf_model.predict(X_train)
    # rf_valid_pred = rf_model.predict(X_valid)
    # rf_test_pred = rf_model.predict(X_test)
    
    # train_pred = train_pred/2 + rf_train_pred/2
    # valid_pred = valid_pred/2 + rf_valid_pred/2
    # test_pred = test_pred/2 + rf_test_pred/2
    
    # logging.info("randforest train mae:{}".format(mean_absolute_error(y_train,rf_train_pred)))
    # logging.info("randforest valid mae:{}".format(mean_absolute_error(y_valid,rf_valid_pred)))
    # logging.info("randforest test mae:{}".format(mean_absolute_error(y_test,rf_test_pred)))
    
    # logging.info("ensemble train mae:{}".format(mean_absolute_error(y_train,train_pred)))
    # logging.info("ensemble valid mae:{}".format(mean_absolute_error(y_valid,valid_pred)))
    # logging.info("ensemble test mae:{}".format(mean_absolute_error(y_test,test_pred)))
    
    save_train = str(Path(current_path).joinpath(model_save).joinpath("train_pred.csv"))
    save_valid = str(Path(current_path).joinpath(model_save).joinpath("valid_pred.csv"))
    save_test = str(Path(current_path).joinpath(model_save).joinpath("test_pred.csv"))
    residule_record(train_pred,train_pd,y_column,save_train,message = "训练集")
    residule_record(valid_pred,valid_pd,y_column,save_valid,message = "验证集")
    residule_record(test_pred,test_pd,y_column,save_test,message = "测试集")
    save_importance = str(Path(current_path).joinpath(model_save).joinpath("importance.png"))
    save_gain = str(Path(current_path).joinpath(model_save).joinpath("importance_gain.png"))
    xgboost_analyse(model,save_importance)
    xgboost_analyse(model,save_gain,importance_type="gain")

def modelSave(log_name):
    current_path = os.path.abspath(os.path.dirname(__file__))
    today = datetime.datetime.now().strftime('%y.%m.%d')
    model_save = dtUtils.increment_path(str(Path(current_path).joinpath(cfg.MODEL_TAG).joinpath(today).joinpath(cfg.MODEL_SAVE)))
    Path(model_save).mkdir(exist_ok=True,parents=True)
    log_save = str(Path(model_save).joinpath(Path(log_name).stem+"_{}.log".format(today)))
    dtUtils.log_set(log_save,log_level = logging.INFO)
    return model_save

# def IsNew(x,bond):
#     return x in 
def updateByDayTest(table_path,distinct_json,test_days = 60):
    table_pd = pd.read_csv(table_path,encoding = "utf-8")
    distinct_date = dtUtils.json_read(distinct_json)
    model_save = modelSave(cfg.LOG_NAME)
    dtUtils.configSave(cfg,model_save)
    param = get_param(cfg_param)
    for i in range(-1*test_days,0,1):
        test_date = distinct_date[i]
        train_datels = distinct_date[0:i]
        valid_date = distinct_date[i-1]
        train_pd,test_pd = updateRecycle.train_valid(table_pd,test_date,train_datels)
        valid_pd = table_pd[table_pd["date_org"]==str(valid_date)]
        X_column = cfg.X_COLUMN
        y_column = cfg.Y_COLUMN
        X_train,y_train = Xy_Value(train_pd,X_column,y_column)
        X_valid,y_valid = Xy_Value(valid_pd,X_column,y_column)
        X_test,y_test = Xy_Value(test_pd,X_column,y_column)
        model = param_xgbTrain(param,model_save,cfg.NUM_ROUND,X_train,y_train,X_valid,y_valid)
        message = "test_{}".format(test_date)
        test_pred = param_xgbPredict(model,X_test,y_test,message = message)
        save_path = str(Path(model_save).joinpath("test_{}.csv".format(test_date)))
        existed_bondls = train_pd["bond_id"].to_list()
        test_pd["is_present"] = test_pd.apply(lambda x:x["bond_id"] in existed_bondls,axis = 1)
        residule_record(test_pred,test_pd,y_column,save_path,message = message)
if __name__ == "__main__":
    pass
    # main()
    updateByDayTest(table_path = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0818to0818\allData.csv",
                    distinct_json = r"D:\python_code\LSTM-master\bond_price\dealed_dir\combine0818to0818\tradeDate_distinct.json",
                    test_days = 80)