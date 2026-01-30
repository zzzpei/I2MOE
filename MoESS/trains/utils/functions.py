import torch
import numpy as np
import random
import pynvml
import logging
import torch.nn.functional as F

logger = logging.getLogger('EMOE')


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def assign_gpu(gpu_ids, memory_limit=1e16):
    if len(gpu_ids) == 0 and torch.cuda.is_available():
        # find most free gpu
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        dst_gpu_id, min_mem_used = 0, memory_limit
        for g_id in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        logger.info(f'Found gpu {dst_gpu_id}, used memory {min_mem_used}.')
        gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()
    # logger.info("Let's use %d GPUs!" % len(gpu_ids))
    device = torch.device('cuda:%d' % int(gpu_ids[0]) if using_cuda else 'cpu')
    return device

def count_parameters(model):
    res = 0
    for p in model.parameters():
        if p.requires_grad:
            res += p.numel()
            # print(p)
    return res

# 计算每个模态的重要性分数
def eva_imp(y_pred, y_true):
    """
    回归逻辑适配分类：将labels转为one-hot编码
    y_pred: [batch_size, num_classes]
    y_true: [batch_size]（类别索引）
    """
    num_classes = y_pred.shape[-1]
    y_true_onehot = F.one_hot(y_true, num_classes=num_classes).float()  # [batch_size, 5]
    res = (y_pred - y_true_onehot) ** 2  # 平方差
    res = res.mean(dim=-1)  # 每个样本的平均平方差，[batch_size]
    return res

def uni_distill(logits1, logits2):
    prob1 = torch.softmax(logits1, dim=-1)
    prob2 = torch.softmax(logits2, dim=-1)
    # mse可以看作是两个概率分布之间的距离
    mse = torch.mean((prob1 - prob2) ** 2, dim=-1)
    return torch.mean(mse)

# 计算熵值损失，鼓励模型分配更均匀的权重
# 鼓励分配更均匀的权重可以防止模型过度依赖某个模态，从而提升模型的泛化能力
def entropy_balance(probs):
    # clamp函数用于将输入张量的值限制在指定范围内，防止出现log(0)的情况，min=1e-9表示张量中的最小值为1e-9
    probs = torch.clamp(probs, min=1e-9)
    # N是模态的数量
    N = probs.size(1)
    # entropy.size = (batch_size,)
    entropy = N * torch.sum(probs * torch.log(probs), dim=1)
    return torch.mean(entropy)