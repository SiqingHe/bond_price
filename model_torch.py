import torch.nn as nn
import torch
import torch.nn.functional as F


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

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = queries.view(N, query_len, self.heads, self.head_dim)
        
        values = values.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        num_layers,
        forward_expansion,
        dropout,
        num_classes,
        max_length
    ):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.max_length = max_length
        
        self.embedding = nn.Embedding(max_length, embed_size)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        
        self.fc_out = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
        x = self.embedding(positions) + self.embedding(x)
        
        for transformer in self.transformer_blocks:
            x = transformer(x, x, x, mask)
            
        x = self.fc_out(x)
        return x

# 定义度量学习损失函数（这里使用Triplet Loss）
class TripleLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(TripleLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_distance = torch.norm(anchor - positive, p=2, dim=1)
        neg_distance = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.relu(pos_distance - neg_distance + self.margin).mean()
        return loss

# 构建度量学习网络
class MetricNet(nn.Module):
    def __init__(self, input_dim, embeding_dim):
        super(MetricNet, self).__init__()
        self.fc = ResNet(input_dim,embeding_dim)
        
    def forward(self, x):
        # 前向传播逻辑
        embed = self.fc(x)
        embed = F.normalize(embed, p=2, dim=1)
        return embed