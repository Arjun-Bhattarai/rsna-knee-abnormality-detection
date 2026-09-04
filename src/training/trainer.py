import torch
from pathlib import Path

from ..utils.metrics import macro_roc_auc


class Trainer:
    """
    Training and validation manager for the knee abnormality model.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        scheduler=None,
        scaler=None,
        checkpoint_dir="outputs/checkpoints",
        grad_clip=1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.scaler = scaler
        self.grad_clip = grad_clip

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_auc = -float("inf")

    def train_epoch(self, loader):
        self.model.train()

        # DINOv2 is frozen, so keep it in evaluation mode.
        if hasattr(self.model.backbone, "model"):
            self.model.backbone.model.eval()

        total_loss = 0.0

        for sagittal, coronal, axial, targets in loader:

            sagittal = sagittal.to(self.device, non_blocking=True)
            coronal = coronal.to(self.device, non_blocking=True)
            axial = axial.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                ):
                    logits = self.model(
                        sagittal,
                        coronal,
                        axial,
                    )

                    loss = self.criterion(
                        logits,
                        targets,
                    )

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                logits = self.model(
                    sagittal,
                    coronal,
                    axial,
                )

                loss = self.criterion(
                    logits,
                    targets,
                )

                loss.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip,
                    )

                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []

        for sagittal, coronal, axial, targets in loader:

            sagittal = sagittal.to(self.device, non_blocking=True)
            coronal = coronal.to(self.device, non_blocking=True)
            axial = axial.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(
                sagittal,
                coronal,
                axial,
            )

            loss = self.criterion(
                logits,
                targets,
            )

            total_loss += loss.item()

            predictions = torch.sigmoid(logits)

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                targets.cpu()
            )

        predictions = torch.cat(
            all_predictions,
            dim=0,
        ).numpy()

        targets = torch.cat(
            all_targets,
            dim=0,
        ).numpy()

        auc = macro_roc_auc(
            targets,
            predictions,
        )

        return {
            "loss": total_loss / max(len(loader), 1),
            "auc": auc,
        }

    def save_checkpoint(
        self,
        epoch,
        val_loss,
        val_auc,
        filename="best.pt",
    ):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_auc": val_auc,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = (
                self.scheduler.state_dict()
            )

        path = self.checkpoint_dir / filename

        torch.save(
            checkpoint,
            path,
        )

        return path

    def fit(self, train_loader, val_loader, epochs):
        history = []

        for epoch in range(1, epochs + 1):

            train_loss = self.train_epoch(
                train_loader
            )

            validation = self.validate(
                val_loader
            )

            val_loss = validation["loss"]
            val_auc = validation["auc"]

            if self.scheduler is not None:
                self.scheduler.step()

            if val_auc > self.best_auc:
                self.best_auc = val_auc

                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_loss,
                    val_auc=val_auc,
                )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                }
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f}"
            )

        return history