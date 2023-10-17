import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class SiameseNetwork(nn.Module):
    def __init__(self, backbone):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone  # 单独的模块，可以是预训练的模型

        # 最后的全连接层用于计算相似度
        self.fc = nn.Sequential(
            nn.Linear(in_features=45, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=48),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=128, out_features=64)
        )
        # .to("cuda")

    def forward_one(self, x):
        # 前向传播一个样本
        x = self.backbone(x)
        # print(x.shape)
        x = x.view(x.size()[0], -1)
        # print(x.shape)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        # 分别前向传播两个样本
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
    
# class TransformerBackbone(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
#         super(TransformerBackbone, self).__init__()
        
#         # 定义 Transformer 层
#         self.transformer = nn.Transformer(
#             d_model=input_dim,
#             nhead=num_heads,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=num_layers,
#             dim_feedforward=hidden_dim,
#             dropout=dropout
#         )
        
#     def forward(self, x):
#         # 输入 x 的尺寸：(sequence_length, batch_size, input_dim)
#         # 将输入转置为 (sequence_length, batch_size, input_dim)，以满足 Transformer 的输入格式要求
#         print(x.shape)
#         x = x.permute(1, 0, 2)
#         print(x.shape)
#         # 前向传播通过 Transformer 层
#         x = self.transformer(x)
        
#         # 将输出转置回原始形状
#         x = x.permute(1, 0, 2)
        
#         return x

class TransformerBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(TransformerBackbone, self).__init__()

        # 位置编码
        self.position_encoder = PositionalEncoding(input_dim, dropout)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        # .to("cuda")
        
    def forward(self, x):
        # 输入数据的形状为 (sequence_length, batch_size, input_dim)
        
        # 添加位置编码
        # x = x.float().cuda()
        x = self.position_encoder(x)
        
        # 传递数据通过Transformer编码器
        # print(x.device)
        # print("tranformer",x.shape)
        x = self.transformer_encoder(x)
        # print(x.device)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        # .float().cuda()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # .float().cuda()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # .float().cuda()
        # print(d_model)
        # print(div_term)
        # print((position * div_term).shape)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos((position * div_term)[:,0:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.device)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 使用示例
# input_dim = 512
# hidden_dim = 1024
# num_layers = 6
# num_heads = 8

# transformer_backbone = TransformerBackbone(input_dim, hidden_dim, num_layers, num_heads)

# 定义Contrastive Loss损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim = True)
        # print(euclidean_distance.shape,label.shape)
        # print(torch.cat((euclidean_distance,label.reshape(label.shape[0],1)),dim=1))
        euclidean_distance = 1- nn.functional.cosine_similarity(output1, output2,dim=1)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # TODO: 如果label为不确定是否相似，则不计算损失
        return loss_contrastive
class Similarity(nn.Module):
    def __init__(self, margin=2.0):
        super(Similarity, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim = True)
        print(euclidean_distance)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # TODO: 如果label为不确定是否相似，则不计算损失
        return loss_contrastive
def similarity(array1,array2,idx_types,idx_nums):
    if array1.ndim==1:
        array1 = torch.tensor(array1).view(1,-1)
        array2 = torch.tensor(array2).view(1,-1)
    # print(array1.shape)
    tp1 = array1[:,idx_types]
    tp2 = array2[:,idx_types]
    total = len(idx_nums)+len(idx_types)
    dis1 = 1 - torch.sum(tp1 == tp2,dim=1)/len(idx_types)
    # numls = set(range(len(array1)))-set(type_idls)
    num1 = array1[:,idx_nums]
    num2 = array2[:,idx_nums]
    # dot_product = torch.matmul(num1, num2.transpose(0, 1))
    # norm_vector1 = torch.norm(num1,p=2,dim=1)
    # norm_vector2 = torch.norm(num2,p=2,dim=1)
    dis2 = (1 - F.cosine_similarity(num1, num2, dim=1))/2
    # if torch.all(norm_vector1== 0) or torch.all(norm_vector2== 0):
    #     dis2 = torch.zeros(norm_vector1.shape)
    # else:
    #     cosine_similarity = torch.diag(dot_product) / (norm_vector1 * norm_vector2)
    #     dis2 = 1 - cosine_similarity
    dis = dis1*len(idx_types)/total + dis2*len(idx_nums)/total
    return dis

siamese_network = SiameseNetwork(TransformerBackbone)