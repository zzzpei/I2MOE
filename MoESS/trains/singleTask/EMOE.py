import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from ..utils import MetricsTop, dict_to_str, eva_imp, uni_distill, entropy_balance

logger = logging.getLogger('EMOE')


class EMOE():
    """
    EMOE Trainer
    - 保持 do_train / do_test 的签名不变
    - 你要求：只修改 scheduler 部分 + 相关最小改动
    """

    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

        # 训练超参默认值（如果 args 没提供就用这些）
        self.total_epochs = getattr(args, "total_epochs", 30)
        self.accum_steps = max(int(getattr(args, "update_epochs", 1)), 1)
        self.max_grad_norm = float(getattr(args, "max_grad_norm", 2.0))
        self.warmup_epochs = int(getattr(args, "warmup_epochs", 3))

        # loss 权重（可在 args 里覆写）
        self.lambda_router = float(getattr(args, "lambda_router", 0.1))
        self.lambda_ud = float(getattr(args, "lambda_ud", 0.1))
        self.lambda_entropy = float(getattr(args, "lambda_entropy", 0.1))
        self.lambda_sim = float(getattr(args, "lambda_sim", 0.01))

    def _build_optimizer(self, model):
        lr = float(self.args.learning_rate)

        groups = {"eeg": [], "eog": [], "data": [], "router": [], "classifier": [], "others": []}

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            nl = n.lower()
            if "router" in nl:
                groups["router"].append(p)
            elif "classifier" in nl or "out_layer" in nl:
                groups["classifier"].append(p)
            elif "eeg" in nl:
                groups["eeg"].append(p)
            elif "eog" in nl:
                groups["eog"].append(p)
            elif "data" in nl:
                groups["data"].append(p)
            else:
                groups["others"].append(p)

        params = [
            {"params": groups["eeg"], "lr": lr,        "weight_decay": 1e-4},
            {"params": groups["eog"], "lr": lr,        "weight_decay": 1e-4},
            {"params": groups["data"], "lr": lr * 1.2, "weight_decay": 5e-5},
            {"params": groups["router"], "lr": lr * 0.5, "weight_decay": 1e-3},
            {"params": groups["classifier"], "lr": lr, "weight_decay": 1e-4},
            {"params": groups["others"], "lr": lr,     "weight_decay": 1e-4},
        ]

        optimizer = optim.AdamW(params, lr=lr, weight_decay=5e-4)
        return optimizer

    def _apply_warmup(self, optimizer, epoch_idx):
        """前 warmup_epochs 线性升到 base lr"""
        if self.warmup_epochs <= 0:
            return
        if epoch_idx >= self.warmup_epochs:
            return

        scale = float(epoch_idx + 1) / float(self.warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = g.get("initial_lr", g["lr"]) * scale

    def _build_dist_from_imp(self, eeg_dist, eog_dist, data_dist, eps=0.1):
        imp = torch.stack([eeg_dist, eog_dist, data_dist], dim=1)  # [B,3]
        score = 1.0 / (imp + eps)
        dist = score / (score.sum(dim=1, keepdim=True) + 1e-8)
        return dist

    def do_train(self, model, dataloader, return_epoch_results=False, fold_id=0):
        optimizer = self._build_optimizer(model)

        # ====== Scheduler：ReduceLROnPlateau（关键修改点）======
        # 注意：它必须在拿到验证指标后 step(metric)，不能像 cosine 那样 step()
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",          # 监控 Acc_5：越大越好
            factor=0.7,
            patience=2,
            threshold=1e-4,
            min_lr=1e-6,
            verbose=True
        )
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.total_epochs, eta_min=1e-6)

        best_valid = 0.0
        best_epoch = 0

        if return_epoch_results:
            epoch_results = {"train": [], "valid": [], "test": []}

        # 记录 initial_lr 供 warmup 使用（plateau 也兼容）
        for g in optimizer.param_groups:
            g["initial_lr"] = g["lr"]

        for epoch in range(self.total_epochs):
            # warmup（手动缩放 lr）
            self._apply_warmup(optimizer, epoch)

            model.train()
            # 收集本 epoch 的路由权重
            w_all = []
            y_pred, y_true = [], []
            y_pred_eeg, y_pred_eog, y_pred_data = [], [], []

            loss_total_meter = 0.0
            loss_eeg_meter = 0.0
            loss_eog_meter = 0.0
            loss_data_meter = 0.0

            optimizer.zero_grad(set_to_none=True)

            with tqdm(dataloader["train"]) as td:
                for step, batch_data in enumerate(td):
                    eeg = batch_data["eeg"].to(self.args.device)
                    eog = batch_data["eog"].to(self.args.device)
                    labels = batch_data["labels"]["M"].to(self.args.device).view(-1)

                    output = model(eeg, eog)
                    w = output["channel_weight"]  # (B,3)
                    w_all.append(w.detach().cpu())

                    y_pred.append(output["logits_c"].detach().cpu())
                    y_pred_eeg.append(output["logits_eeg"].detach().cpu())
                    y_pred_eog.append(output["logits_eog"].detach().cpu())
                    y_pred_data.append(output["logits_data"].detach().cpu())
                    y_true.append(labels.detach().cpu())

                    loss_task_eeg = self.criterion(output["logits_eeg"], labels)
                    loss_task_eog = self.criterion(output["logits_eog"], labels)
                    loss_task_data = self.criterion(output["logits_data"], labels)
                    loss_task_m = self.criterion(output["logits_c"], labels)

                    loss_eeg_meter += float(loss_task_eeg.item())
                    loss_eog_meter += float(loss_task_eog.item())
                    loss_data_meter += float(loss_task_data.item())

                    eeg_dist = eva_imp(output["logits_eeg"], labels)    # [B]
                    eog_dist = eva_imp(output["logits_eog"], labels)    # [B]
                    data_dist = eva_imp(output["logits_data"], labels)  # [B]
                    dist = self._build_dist_from_imp(eeg_dist, eog_dist, data_dist, eps=0.1)  # [B,3]

                    loss_sim = torch.mean(torch.sum((dist.detach() - w) ** 2, dim=1))

                    lam_ent = self.lambda_entropy * max(0.0, 1.0 - epoch / 10.0)
                    loss_ety = entropy_balance(w)

                    if self.args.fusion_method == "sum":
                        fused_features = (output["eeg_proj"] * w[:, 0:1] +
                                          output["eog_proj"] * w[:, 1:2] +
                                          output["data_proj"] * w[:, 2:3])
                    else:
                        fused_features = torch.cat([
                            output["eeg_proj"] * w[:, 0:1],
                            output["eog_proj"] * w[:, 1:2],
                            output["data_proj"] * w[:, 2:3]
                        ], dim=1)

                    loss_ud = uni_distill(output["c_proj"], fused_features.detach())

                    data_weight = 1.2 if epoch < (self.total_epochs // 2) else 1.0

                    loss = (
                        loss_task_m
                        + (loss_task_eeg + loss_task_eog + data_weight * loss_task_data) / (2.0 + data_weight)
                        + self.lambda_router * (lam_ent * loss_ety + self.lambda_sim * loss_sim)
                        + self.lambda_ud * loss_ud
                    )

                    loss = loss / float(self.accum_steps)
                    loss.backward()

                    loss_total_meter += float(loss.item()) * float(self.accum_steps)

                    if (step + 1) % self.accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    if epoch == 0 and step == 0:
                        logger.info(f"[Debug] w shape: {tuple(w.shape)}")
                        logger.info(f"[Debug] w mean: {w.mean(dim=0).detach().cpu().numpy()}")
                        logger.info(f"[Debug] w std : {w.std(dim=0).detach().cpu().numpy()}")
                        logger.info(f"[Debug] w first5:\n{w[:5].detach().cpu().numpy().round(4)}")

            if (len(dataloader["train"]) % self.accum_steps) != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # --- epoch metrics ---
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            pred_eeg = torch.cat(y_pred_eeg)
            pred_eog = torch.cat(y_pred_eog)
            pred_data = torch.cat(y_pred_data)

            train_results = self.metrics(pred, true)
            train_results_eeg = self.metrics(pred_eeg, true)
            train_results_eog = self.metrics(pred_eog, true)
            train_results_data = self.metrics(pred_data, true)

            train_results["EEG_Acc"] = train_results_eeg.get("Acc", train_results_eeg.get("Acc_5", 0))
            train_results["EOG_Acc"] = train_results_eog.get("Acc", train_results_eog.get("Acc_5", 0))
            train_results["Data_Acc"] = train_results_data.get("Acc", train_results_data.get("Acc_5", 0))

            train_results["Loss"] = round(loss_total_meter / max(len(dataloader["train"]), 1), 4)
            train_results["EEG_Loss"] = round(loss_eeg_meter / max(len(dataloader["train"]), 1), 4)
            train_results["EOG_Loss"] = round(loss_eog_meter / max(len(dataloader["train"]), 1), 4)
            train_results["Data_Loss"] = round(loss_data_meter / max(len(dataloader["train"]), 1), 4)

            # ====== 每个 epoch 输出 w_mean / w_std ======
            if len(w_all) > 0:
                w_cat = torch.cat(w_all, dim=0)  # [N,3]
                w_mean = w_cat.mean(dim=0).numpy()
                w_std = w_cat.std(dim=0, unbiased=False).numpy()

            logger.info(
                f">> Epoch: {epoch + 1}/{self.total_epochs} "
                f"TRAIN-({self.args.model_name}) "
                f">> {dict_to_str(train_results)} "
                f"EEG_Acc: {train_results['EEG_Acc']:.4f} "
                f"EOG_Acc: {train_results['EOG_Acc']:.4f} "
                f"Data_Acc: {train_results['Data_Acc']:.4f}"
                f"[RouterStats][Epoch {epoch+1}] w_mean={np.round(w_mean, 4)} w_std={np.round(w_std, 4)}"   
            )

            # --- validation（先算 metric，再 scheduler.step(metric)）---
            val_results = self.do_test(model, dataloader["valid"], mode="VAL")
            cur_valid = val_results.get("Acc_5", val_results.get("Acc", 0.0))

            # ====== 关键修改：ReduceLROnPlateau 在这里 step(cur_valid) ======
            # warmup 期间你手动改 lr，所以建议 warmup 结束后再让 plateau 接管
            if epoch >= self.warmup_epochs:
                scheduler.step(cur_valid)

            # best model (maximize acc) ——你说你已有最小门槛 + 平滑，这里保留你原版
            isBetter = cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid = cur_valid
                best_epoch = epoch + 1
                model_save_path = f'./pt/emoe_fold{fold_id}.pth'
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), model_save_path)
                else:
                    torch.save(model.state_dict(), model_save_path)
                logger.info(f"New best model saved with Acc: {best_valid:.4f} at epoch {best_epoch}")

            if (epoch + 1) % 10 == 0:
                checkpoint_path = f'./pt/emoe_fold{fold_id}_epoch{epoch + 1}.pth'
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Checkpoint saved at epoch {epoch + 1}")

            if return_epoch_results:
                epoch_results["train"].append(train_results)
                epoch_results["valid"].append(val_results)

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1} completed. Best valid Acc: {best_valid:.4f}, Best epoch: {best_epoch}, LR: {current_lr:.6f}")

        return best_valid

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False, f=0):
        model.eval()
        y_pred, y_true = [], []
        y_pred_eeg, y_pred_eog, y_pred_data = [], [], []

        eval_loss = 0.0
        eval_loss_eeg = 0.0
        eval_loss_eog = 0.0
        eval_loss_data = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    eeg = batch_data["eeg"].to(self.args.device)
                    eog = batch_data["eog"].to(self.args.device)
                    labels = batch_data["labels"]["M"].to(self.args.device).view(-1)

                    output = model(eeg, eog)

                    loss = self.criterion(output["logits_c"], labels)
                    loss_eeg = self.criterion(output["logits_eeg"], labels)
                    loss_eog = self.criterion(output["logits_eog"], labels)
                    loss_data = self.criterion(output["logits_data"], labels)

                    eval_loss += float(loss.item())
                    eval_loss_eeg += float(loss_eeg.item())
                    eval_loss_eog += float(loss_eog.item())
                    eval_loss_data += float(loss_data.item())

                    y_pred.append(output["logits_c"].cpu())
                    y_pred_eeg.append(output["logits_eeg"].cpu())
                    y_pred_eog.append(output["logits_eog"].cpu())
                    y_pred_data.append(output["logits_data"].cpu())
                    y_true.append(labels.cpu())

        n = max(len(dataloader), 1)
        eval_loss /= n
        eval_loss_eeg /= n
        eval_loss_eog /= n
        eval_loss_data /= n

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        pred_eeg = torch.cat(y_pred_eeg)
        pred_eog = torch.cat(y_pred_eog)
        pred_data = torch.cat(y_pred_data)

        eval_results = self.metrics(pred, true)
        eval_results_eeg = self.metrics(pred_eeg, true)
        eval_results_eog = self.metrics(pred_eog, true)
        eval_results_data = self.metrics(pred_data, true)

        eval_results["EEG_Acc"] = eval_results_eeg.get("Acc", eval_results_eeg.get("Acc_5", 0))
        eval_results["EOG_Acc"] = eval_results_eog.get("Acc", eval_results_eog.get("Acc_5", 0))
        eval_results["Data_Acc"] = eval_results_data.get("Acc", eval_results_data.get("Acc_5", 0))

        eval_results["Loss"] = round(eval_loss, 4)
        eval_results["EEG_Loss"] = round(eval_loss_eeg, 4)
        eval_results["EOG_Loss"] = round(eval_loss_eog, 4)
        eval_results["Data_Loss"] = round(eval_loss_data, 4)

        logger.info(
            f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)} "
            f"EEG_Acc: {eval_results['EEG_Acc']:.4f} "
            f"EOG_Acc: {eval_results['EOG_Acc']:.4f} "
            f"Data_Acc: {eval_results['Data_Acc']:.4f}"
        )
        return eval_results
