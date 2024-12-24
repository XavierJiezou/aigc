from typing import Any, Dict, Tuple, List

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from src.models.var import VAR, VQVAE, VectorQuantizer2


class VARLitModule(LightningModule):

    def __init__(
        self,
        var: VAR,
        patch_nums: Tuple[int, ...],
        label_smooth: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        iters_train=1563,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=-1,
        vae_config_path="checkpoints/var/vae_ch160v4096z32.pth",
        compile: bool = False,
    ) -> None:
        """Initialize a `VARLitModule`."""
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["var"])
        self.var = var
        self.var.init_weights(
            init_adaln=init_adaln,
            init_adaln_gamma=init_adaln_gamma,
            init_head=init_head,
            init_std=init_std,
        )
        self.var.vae_proxy[0].load_state_dict(torch.load(vae_config_path),strict=True)
        self.vae_local = self.var.vae_proxy[0]
        self.quantize_local = self.vae_local.quantize

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=label_smooth, reduction="none"
        )
        self.val_criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=0.0, reduction="mean"
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_mean_loss = MeanMetric()
        self.val_tail_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_mean_acc = MeanMetric()
        self.val_tail_acc = MeanMetric()

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.begin_ends = []
        self.pg0 = 1
        self.pg = 0.8
        self.ep: int = 250
        self.wp = self.ep * 1 / 50
        self.pgwp = self.ep * 1 / 300

        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]

        self.loss_weight = torch.ones(1, self.L) / self.L
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_mean_loss.reset()
        self.val_tail_loss.reset()
        self.val_mean_acc.reset()
        self.val_tail_acc.reset()
        self.test_loss.reset()
        self.train_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], prog_si: int, prog_wp_it: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :param prog_si: The current training progression stage.
        :param prog_wp_it: The total iterations for progressive warm-up.

        :return: A tuple containing:
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # Extract the input image and label batch
        
        inp_B3HW, label_B = batch
        if self.loss_weight.device != inp_B3HW.device:
            self.loss_weight = self.loss_weight.to(inp_B3HW.device)
        B, V = label_B.shape[0], self.vae_local.vocab_size

        # Progressive training logic
        self.var.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1:
                self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog:
            prog_wp = 1  # No prog warmup at first stage
        if prog_si == len(self.hparams.patch_nums) - 1:
            prog_si = -1  # Final stage, no prog

        # Forward pass through the model
        gt_idx_Bl: List[torch.Tensor] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: torch.Tensor = self.quantize_local.idxBl_to_var_input(
            gt_idx_Bl
        )

        # Model forward pass
        logits_BLV = self.var(label_B, x_BLCv_wo_first_l)

        # Loss calculation
        loss = self.criterion(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)

        if prog_si >= 0:  # In progressive training
            bg, ed = self.begin_ends[prog_si]
            lw = self.loss_weight[:, :ed].clone()
            lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
        else:  # Not in progressive training
            lw = self.loss_weight

        # Apply loss weight and compute final loss
        loss = loss.mul(lw).sum(dim=-1).mean()

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        g_it = self.global_step
        wp_it = self.wp * self.hparams.iters_train
        max_it = self.ep * self.hparams.iters_train
        if (
            self.pg
        ):  # default: args.pg == 0.0, means no progressive training, won't get into this
            if g_it <= wp_it:
                prog_si = self.pg0
            elif g_it >= max_it * self.pg:
                prog_si = len(self.hparams.patch_nums) - 1
            else:
                delta = len(self.hparams.patch_nums) - 1 - self.pg0
                progress = min(
                    max((g_it - wp_it) / (max_it * self.pg - wp_it), 0), 1
                )  # from 0 to 1
                prog_si = self.pg0 + round(
                    progress * delta
                )  # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1

        loss = self.model_step(
            batch, prog_si, prog_wp_it=self.pgwp * self.hparams.iters_train
        )

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        inp_B3HW, label_B = batch
        B, V = label_B.shape[0], self.vae_local.vocab_size
        gt_idx_Bl: List[torch.Tensor] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: torch.Tensor = self.quantize_local.idxBl_to_var_input(
            gt_idx_Bl
        )
        logits_BLV = self.var(label_B, x_BLCv_wo_first_l)
        mean_loss = self.val_criterion(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
        tail_loss = (
            self.val_criterion(
                logits_BLV.data[:, -self.last_l :].reshape(-1, V),
                gt_BL[:, -self.last_l :].reshape(-1),
            )
            * B
        )

        mean_acc = (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (
            100 / gt_BL.shape[1]
        )
        tail_acc = (
            logits_BLV.data[:, -self.last_l :].argmax(dim=-1)
            == gt_BL[:, -self.last_l :]
        ).sum() * (100 / self.last_l)

        # update and log metrics
        self.val_mean_loss(mean_loss)
        self.val_tail_loss(tail_loss)
        self.val_mean_acc(mean_acc)
        self.val_tail_acc(tail_acc)
        self.log(
            "val_loss", self.val_mean_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_tail_loss",
            self.val_tail_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_mean_acc",
            self.val_mean_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_tail_acc",
            self.val_tail_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.var = torch.compile(self.var)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    pass
