from sklearn.svm import SVR
from sklearn import metrics
from utils import get_data,scaler_trans,invert_scale
import numpy as np
import pickle
import joblib
from pathlib import Path
import math
from sklearn.metrics import mean_squared_error


train_path=r"D:\python_code\LSTM-master\bond_price\bond_trdataNonull\train.json"
valid_path=r"D:\python_code\LSTM-master\bond_price\bond_trdataNonull\valid.json"
train_data=np.array(get_data(train_path))
X_train,y_train=train_data[:,0:-1],train_data[:,-1]
X_train,scalex=scaler_trans(X_train)
y_train,scaley=scaler_trans(y_train)

valid_data=np.array(get_data(valid_path))
X_valid,y_valid=valid_data[:,0:-1],valid_data[:,-1]
X_valid=scalex.transform(X_valid)
# y_valid,scaley=scaler_trans(y_valid)
y_valid=scaley.transform(np.array(y_valid).reshape(-1,1))

svc_model=SVR()

svc_model.fit(X_train,y_train)

save_path=r"D:\python_code\LSTM-master\bond_price\model\svm\svm_ts.pkl"
if not Path(save_path).exists():
    Path(save_path).parent.mkdir(exist_ok=True,parents=True)
joblib.dump(svc_model, save_path)
pred=svc_model.predict(X_valid)
rmse=math.sqrt(mean_squared_error(pred,y_valid))
print(metrics.accuracy_score(pred,y_valid))
print("rmse:",rmse)