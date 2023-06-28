import torch.nn as nn


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

class mymodel(nn.Module):
    def __init__(self) -> None:
        super(mymodel, self).__init__()
        self.model1=nn.Sequential(
            nn.Conv1d(1,16,2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,32,2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Flatten(),
        )
        self.model2=nn.Sequential(
            nn.Linear(in_features=224,out_features=1,bias=True)
        )
    def forward(self,input):
        x = self.model1(input)
        # print(x.shape)
        x = self.model2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # 残差连接
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.res_block1 = ResidualBlock(16, 16)
        self.res_block2 = ResidualBlock(16, 16)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# # 创建模型实例
# in_channels = 1  # 输入通道数
# num_classes = 10  # 类别数
# model = ResNet(in_channels, num_classes)
