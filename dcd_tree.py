from sklearn.tree import DecisionTreeRegressor
from data_a import get_data,scale,invert_scale
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from pathlib import Path

# 准备训练数据
# X_train = [[x1, x2, ...], [x1, x2, ...], ...]  # 特征向量列表
# y_train = [y1, y2, ...]  # 目标值列表
def data_pre(train_path,valid_path):
    train_data=get_data(train_path)
    valid_data=get_data(valid_path)
    train_data,valid_data=np.array(train_data),np.array(valid_data)
    scaler, train_scaled, valid_scaled = scale(train_data, valid_data)
    return scaler,train_scaled,valid_scaled

# 创建决策树回归模型
def model_train(X_train,y_train,save_path):
    if not Path(save_path).exists():
        model = DecisionTreeRegressor()
        # 训练模型
        model.fit(X_train, y_train)
        Path(save_path).parent.mkdir(exist_ok=True,parents=True)
        joblib.dump(model, save_path)
    else:
        model=joblib.load(save_path)
    print(model.get_params())
    print(model.get_depth())
    print(model.get_n_leaves())
    return model
# 进行预测
def model_predict(model,X_test,y_test,scaled,save_txt):
    # X_test = [[x1, x2, ...], [x1, x2, ...], ...]  # 待预测的特征向量列表
    y_pred = model.predict(X_test)
    y_pred=y_pred.reshape(-1,1)
    y_pred=invert_scale(scaled,X_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
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
# 输出预测结果
# print(y_pred)
if __name__=="__main__":
    train_path=r"D:\python_code\LSTM-master\model_bond\bond_trdata\train.json"
    valid_path=r"D:\python_code\LSTM-master\model_bond\bond_trdata\valid.json"
    save_path=r"D:\python_code\LSTM-master\model_bond\model\dcd_tree\regtree_n.pkl"
    save_txt=r"D:\python_code\LSTM-master\model_bond\result\dcd_tree\res_compare_n.txt"
    valid_data=np.array(get_data(valid_path))
    scaled,train_sc,valid_sc=data_pre(train_path,valid_path)
    model=model_train(train_sc[:,0:-1],train_sc[:,-1],save_path)
    rmse=model_predict(model,valid_sc[:,0:-1],valid_data[:,-1],scaled,save_txt)
    print(rmse)
    