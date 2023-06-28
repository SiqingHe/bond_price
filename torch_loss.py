import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.mean((predictions - targets) ** 2)
        return loss
    
class Rmse(nn.Module):
    def __init__(self):
        super(Rmse, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
        return loss

if __name__=="__main__":
    # 使用自定义损失函数
    criterion = CustomLoss()

