import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_a import get_data
from lstm_keras import invert_scale
import pickle

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        # print(lstm_out)
        output = self.fc(lstm_out[-1])  # 只使用最后一个时间步的输出
        return output

# 定义训练函数
def train(model, train_data, train_labels, num_epochs, learning_rate,save_path):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i in range(0,len(train_labels),2):
            X,y=train_data[i:i+2],train_labels[i:i+2]
            model.train()
            optimizer.zero_grad()
            # print(X.shape)
            outputs = model(X)
            # print(outputs)
            # print(outputs.shape)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if i%1000==0:
                print(f'data [{i+1}/{len(train_labels)}], Loss: {loss.item():.8f}')
        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
    torch.save(model.state_dict(), save_path)

def test(model,test_data,y_test,scaler):
    model.eval()
    # test_data = torch.tensor([[6.0, 0.1], [7.0, 0.2]], dtype=torch.float32)
    test_len=len(test_data)
    with torch.no_grad():
        for i in range(0,test_len,2):
            X_test,y=test_data[i:i+2],y_test[i:i+2]
            predicted_labels = model(X_test)
            yhat = invert_scale(scaler, predicted_labels)
            y_real=invert_scale(scaler,y)
            print("预测结果:", yhat,"真实结果:",y_real,"预测误差",yhat-y_real)
if __name__=="__main__":
    pass
    input_size=9
    hidden_size=2
    output_size=1
    model = LSTMModel(input_size, hidden_size, output_size)
    X_train_path=r"D:\python_code\LSTM-master\model_bond\timedata\x_train.json"
    y_train_path=r"D:\python_code\LSTM-master\model_bond\timedata\y_train.json"
    # # 训练模型
    # num_epochs = 3
    # learning_rate = 0.001
    X_test=torch.from_numpy(np.array(get_data(X_train_path))).to(torch.float32)
    y_test=torch.from_numpy(np.array(get_data(y_train_path))).to(torch.float32)
    # save_path=r"D:\python_code\LSTM-master\model_bond\model\torch\train_test.pth"
    # train(model, X_train, y_train, num_epochs, learning_rate,save_path)
    X_test_path=r"D:\python_code\LSTM-master\model_bond\timedata\x_test.json"
    y_test_path=r"D:\python_code\LSTM-master\model_bond\timedata\y_test.json"
    pkl_path=r"D:\python_code\LSTM-master\model_bond\timedata\scale_y.pkl"
    with open(pkl_path,"rb") as rd:
        scaley=pickle.load(rd)
    # X_test=torch.from_numpy(np.array(get_data(X_test_path))).to(torch.float32)
    # y_test=torch.from_numpy(np.array(get_data(y_test_path))).to(torch.float32)
    load_path=r"D:\python_code\LSTM-master\model_bond\model\torch\train_test.pth"
    model.load_state_dict(torch.load(load_path))
    test(model,X_test,y_test,scaley)