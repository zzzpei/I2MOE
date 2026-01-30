import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == "Classification":
            self.metrics_dict = {
                'SLEEPEDF-20': self.__eval_sleepedf20_classification
            }

    def __eval_sleepedf20_classification(self, y_pred, y_true):
        """
        睡眠分期五分类评估
        类别映射:
        0: Wake (清醒)
        1: N1 (睡眠阶段1)
        2: N2 (睡眠阶段2) 
        3: N3 (睡眠阶段3/慢波睡眠)
        4: REM (快速眼动睡眠)
        """
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()

        # three classes
        y_pred_5 = np.argmax(y_pred, axis=1)
        acc_5 = accuracy_score(y_pred_5, y_true)
        f1_5 = f1_score(y_true, y_pred_5, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred_5, average='macro', zero_division=0)  # 宏平均F1
        precision_5 = precision_score(y_true, y_pred_5, average='weighted', zero_division=0)
        recall_5 = recall_score(y_true, y_pred_5, average='weighted', zero_division=0)
        
        # 计算每个类别的F1分数
        f1_per_class = f1_score(y_true, y_pred_5, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred_5, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred_5, average=None, zero_division=0)
        
        # 计算混淆矩阵相关指标
        from sklearn.metrics import confusion_matrix, cohen_kappa_score
        cm = confusion_matrix(y_true, y_pred_5)
        kappa = cohen_kappa_score(y_true, y_pred_5)

        eval_results = {
            # 五分类主要指标
            "Acc_5": round(acc_5, 4),
            "F1_weighted": round(f1_5, 4),
            "F1_macro": round(f1_macro, 4),
            "Precision": round(precision_5, 4),
            "Recall": round(recall_5, 4),
            "Kappa": round(kappa, 4),
            
            # 各类别详细指标
            "F1_Wake": round(f1_per_class[0], 4),
            "F1_N1": round(f1_per_class[1], 4),
            "F1_N2": round(f1_per_class[2], 4),
            "F1_N3": round(f1_per_class[3], 4),
            "F1_REM": round(f1_per_class[4], 4),
            
            "Precision_Wake": round(precision_per_class[0], 4),
            "Precision_N1": round(precision_per_class[1], 4),
            "Precision_N2": round(precision_per_class[2], 4),
            "Precision_N3": round(precision_per_class[3], 4),
            "Precision_REM": round(precision_per_class[4], 4),
            
            "Recall_Wake": round(recall_per_class[0], 4),
            "Recall_N1": round(recall_per_class[1], 4),
            "Recall_N2": round(recall_per_class[2], 4),
            "Recall_N3": round(recall_per_class[3], 4),
            "Recall_REM": round(recall_per_class[4], 4),
        }
        return eval_results
    
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]