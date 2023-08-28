paper code
# 模型训练
torch_train.py
# 模型推断
torch_pred.py
# 损失函数定义
torch_loss.py
# 数据加载器
data_torch.py
# 模型定义
model_torch.py
# 非时间序列化数据处理，划分
data_a.py
# 时间序列化数据处理，划分
data2time.py
# 决策树
dcd_tree.py
# 支持向量回归
svg_reg.py

# bond_trdataNonull 丢弃含nan数据，划分的非时间序列三集数据
# timedata 划分的时间序列数据
# model 保存的训练模型
# result 决策树推断和真实比较

mkl-fft==1.3.6
requests==2.29.0
certifi==2022.12.7
six==1.16.0
idna 2.8
urllib3 1.25.8