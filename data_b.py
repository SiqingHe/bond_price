import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 准备训练数据
# 假设有以下样本数据，其中 x1 和 x2 是自变量，y 是因变量，t 是时间
# 时间序列长度为 5
data = [
    [1.0, 0.5, 2.0, '2023-01-01'],
    [2.0, 0.4, 4.0, '2023-01-02'],
    [3.0, 0.3, 6.0, '2023-01-03'],
    [4.0, 0.2, 8.0, '2023-01-04'],
    [5.0, 0.1, 10.0, '2023-01-05']
]

# 转换为LSTM输入形式
# 定义窗口大小和特征维度
window_size = 3
input_size = 2

inputs = []
outputs = []

for i in range(len(data) - window_size):
    input_seq = []
    for j in range(window_size):
        # 提取自变量和时间
        input_seq.append(data[i+j][:2])
    inputs.append(input_seq)
    
    # 提取因变量
    outputs.append([data[i+window_size][2]])

print(inputs)
print(outputs)
# 将时间转换为时间差特征（以天为单位）
# 可以使用datetime库进行时间的解析和计算
import datetime

def parse_date(date_str):
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')

def time_diff_in_days(date1, date2):
    return (date2 - date1).days

# 获取第一个样本的时间
start_date = parse_date(data[0][3])

# 转换时间为时间差特征
for i in range(len(inputs)):
    for j in range(window_size):
        current_date = parse_date(data[i+j][3])
        inputs[i][j].append(time_diff_in_days(start_date, current_date))
print(inputs)
# 数据归一化
scaler = MinMaxScaler()
inputs = scaler.fit_transform(inputs)
outputs = scaler.fit_transform(outputs)

# 数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, shuffle=False)

# 转换为张量形式
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 打印训练集和测试集的形状
print("训练集输入形状:", X_train.shape)
print("训练集输出形状:", y_train.shape)
print("测试集输入形状:", X_test.shape)
print("测试集输出形状:", y_test.shape)
