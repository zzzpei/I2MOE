import torch
import torch.nn as nn
import torch.nn.functional as F

# 通道路由器，用于动态调整各模态的权重
# 调整的方式是通过一个两层的全连接网络，输入是拼接后的多模态特征，输出是各模态的权重分布
# t是温度参数，用于控制权重分布的平滑程度, 温度越高，分布越平滑
class router(nn.Module):
    def __init__(self, dim, channel_num, t):
        super().__init__()
        self.l1 = nn.Linear(dim, 128)
        self.l2 = nn.Linear(128, channel_num)
        self.t = t
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # 通过两层全连接网络计算各模态的权重分布
        # normalize用于对特征进行L2归一化，避免特征尺度过大影响权重计算
        # relu用于增加非线性，使得模型能够学习更复杂的权重
        x = self.l2(F.relu(F.normalize(self.l1(x), p=2, dim=1)))/self.t
        output = torch.softmax(x, dim=1)
        return output