import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn import metrics
from xgboost import plot_importance
from matplotlib import pyplot

train_path=r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0724\train.xlsx"
valid_path=r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0724\valid.xlsx"
test_path=r"D:\python_code\LSTM-master\bond_price\dealed_dir\sets_split0724\test.xlsx"
train_pd=pd.read_excel(train_path)
valid_pd=pd.read_excel(valid_path)
test_pd=pd.read_excel(test_path)

train_pd=train_pd.loc[(train_pd["yield"]<=50) & (train_pd["yield"]>-20) & (train_pd["yield"]!=0)]
valid_pd=valid_pd.loc[(valid_pd["yield"]<=50) & (valid_pd["yield"]>-20) & (train_pd["yield"]!=0)]
test_pd=valid_pd.loc[(test_pd["yield"]<=50) & (test_pd["yield"]>-20) & (train_pd["yield"]!=0)]
# print(train_pd.describe()["yield"])
from dataPy.data_split import X_column,y_column
# train_value=
# X_train,y_train=train_pd[X_column].values,train_pd[y_column].values
# X_valid,y_valid=valid_pd[X_column].values,valid_pd[y_column].values
# X_test,y_test=test_pd[X_column].values,test_pd[y_column].values
X_train,y_train=train_pd[X_column],train_pd[y_column]
X_valid,y_valid=valid_pd[X_column],valid_pd[y_column]
X_test,y_test=test_pd[X_column],test_pd[y_column]

dtrain = xgb.DMatrix(data = X_train, label = y_train)
dtest = xgb.DMatrix(data = X_test, label = y_test)

param = {'max_depth':10, 'eta':0.08, 'objective':'reg:squarederror',"tree_method":"hist","device":"cuda",
         "subsample":0.8,"alpha":0.5}
# xgb_model = xgb.XGBRegressor(
#     objective='reg:squarederror',  # 指定损失函数为平方误差
#     n_estimators=1200,             # 决策树的数量
#     learning_rate=0.005,            # 学习率
#     max_depth=20,                  # 决策树的最大深度
#     random_state=42             # 随机种子，用于重现结果
# )
# tree_method="gpu_hist",
#     gpu_id=0
# num_round = 2
# xgb_md = xgb.train(param,dtrain, num_round)
num_round = 3500  # 迭代次数
xgb_model = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round)

# 进行预测
dtest = xgb.DMatrix(data=X_test)
y_pred = xgb_model.predict(dtest)
try_pred=xgb_model.predict(dtrain)
# test_preds = xgb.predict(dtest)
# xgb_res = xgb_model.fit(param,dtrain, num_round)
# xgb_model.fit(X_train,y_train)
# try_pred = xgb_model.predict(X_train)

trmae=mean_absolute_error(y_train,try_pred)
print("train_mae",trmae)
# y_pred = xgb_model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
print("test_mae",mae)
# print(test_preds)
# test_predictions = [round(value) for value in test_preds] #变成0、1
 #显示特征重要性
plot_importance(xgb_model)#打印重要程度结果
pyplot.show()