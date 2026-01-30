import logging

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ..utils import MetricsTop

logger = logging.getLogger("EMOE")


class EMOEI2MOE:
    """Trainer for EMOE experts under the I2MOE framework."""

    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.total_epochs = int(getattr(args, "total_epochs", 30))
        self.interaction_loss_weight = float(
            getattr(args, "interaction_loss_weight", 1.0)
        )
        self.max_grad_norm = float(getattr(args, "max_grad_norm", 2.0))

    def _build_optimizer(self, model):
        lr = float(self.args.learning_rate)
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    def _compute_loss(self, logits, interaction_losses, labels):
        task_loss = self.criterion(logits, labels)
        interaction_loss = sum(interaction_losses) / float(len(interaction_losses))
        total_loss = task_loss + self.interaction_loss_weight * interaction_loss
        return total_loss, task_loss, interaction_loss

    def do_train(self, model, dataloader, return_epoch_results=False, fold_id=0):
        optimizer = self._build_optimizer(model)
        best_valid = 0.0
        best_epoch = 0

        if return_epoch_results:
            epoch_results = {"train": [], "valid": [], "test": []}

        for epoch in range(self.total_epochs):
            model.train()
            y_pred, y_true = [], []
            loss_total_meter = 0.0

            with tqdm(dataloader["train"]) as td:
                for batch_data in td:
                    eeg = batch_data["eeg"].to(self.args.device)
                    eog = batch_data["eog"].to(self.args.device)
                    labels = batch_data["labels"]["M"].to(self.args.device).view(-1)

                    output = model(eeg, eog)
                    logits = output["logits"]
                    interaction_losses = output["interaction_losses"]

                    loss, _, _ = self._compute_loss(
                        logits, interaction_losses, labels
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.max_grad_norm
                    )
                    optimizer.step()

                    loss_total_meter += float(loss.item())
                    y_pred.append(logits.detach().cpu())
                    y_true.append(labels.detach().cpu())

            train_results = self.metrics(torch.cat(y_pred), torch.cat(y_true))
            train_results["Loss"] = round(loss_total_meter / len(dataloader["train"]), 4)

            valid_results = self.do_test(model, dataloader["valid"], mode="VALID")

            if return_epoch_results:
                epoch_results["train"].append(train_results)
                epoch_results["valid"].append(valid_results)

            cur_valid = valid_results["Acc_5"]
            if cur_valid > best_valid:
                best_valid = cur_valid
                best_epoch = epoch
                save_target = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(
                    save_target.state_dict(), f"pt/emoe_i2moe_fold{fold_id}.pth"
                )

            logger.info(
                f"[Fold {fold_id}] Epoch {epoch + 1}/{self.total_epochs} "
                f"Train Acc: {train_results['Acc_5']:.4f}, "
                f"Valid Acc: {valid_results['Acc_5']:.4f}"
            )

        logger.info(f"[Fold {fold_id}] Best Valid Acc: {best_valid} @ epoch {best_epoch}")

        if return_epoch_results:
            return epoch_results
        return None

    @torch.no_grad()
    def do_test(self, model, dataloader, mode="TEST"):
        model.eval()
        y_pred, y_true = [], []

        for batch_data in dataloader:
            eeg = batch_data["eeg"].to(self.args.device)
            eog = batch_data["eog"].to(self.args.device)
            labels = batch_data["labels"]["M"].to(self.args.device).view(-1)

            inference_model = model.module if isinstance(model, nn.DataParallel) else model
            output = inference_model.inference(eeg, eog)
            logits = output["logits"]

            y_pred.append(logits.detach().cpu())
            y_true.append(labels.detach().cpu())

        results = self.metrics(torch.cat(y_pred), torch.cat(y_true))
        logger.info(f"{mode} results: {results}")
        return results
