import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
import numpy as np
import math
from src.models.components.mask_encoder import MaskEncoder
import copy
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import clip


# This implementation is adapted from the CLIP model training code in the following repository:
# Refer to https://github.com/Zasder3/train-CLIP/blob/main/models/wrapper.py
# The original code provides a PyTorch Lightning wrapper for the CLIP model, enabling efficient training
# and validation with features such as mini-batching, multi-GPU support, and custom optimization.
# Modifications have been made to suit specific use cases or improve functionality.
class CLIPWrapper(LightningModule):
    def __init__(
        self,
        clip_config: dict,
        minibatch_size: int,
        num_training_steps:int,
        mask_encoder: MaskEncoder = None,
    ):
        """A lightning wrapper for a CLIP model as specified in the paper.

        Args:
            model_name (str): A case sensitive visual model name.
            config (dict): A dictionary containing the CLIP instantiation parameters.
        """
        super().__init__()
        self.clip_config = clip_config
        self.clip, _ = clip.load(**clip_config)
        # farl_state=torch.load(face_state_dict)
        # self.clip.load_state_dict(farl_state["state_dict"],strict=False)
        self.minibatch_size = minibatch_size
        self.mask_encoder = mask_encoder
        self.mask_encoder.init_visual(copy.deepcopy(self.clip))
        self.num_training_steps = num_training_steps

        self.automatic_optimization = False

    def froze(self):
        self.clip.eval()
        self.clip.requires_grad_(False)
        for p in self.clip.parameters():
            p.requires_grad = False

    def on_train_start(self) -> None:
        self.froze()

    # Training loss: https://github.com/openai/CLIP/issues/83
    # Mini-batching thanks to https://github.com/crowsonkb / https://twitter.com/RiversHaveWings
    # Multi-GPU support: https://github.com/MicPie/clasp
    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()

        image = train_batch["face"]
        mask = train_batch["mask"].long()

        n = math.ceil(len(image) // self.minibatch_size)
        n = max(n,1)
        image_mbs = torch.chunk(image, n)
        mask_mbs = torch.chunk(mask, n)

        # calculate original statistics
        with torch.no_grad():
            ims = [F.normalize(self.clip.encode_image(im), dim=1) for im in image_mbs]
            mak = [F.normalize(self.mask_encoder.forward(t), dim=1) for t in mask_mbs]
            # gather from all GPUs
            ims = self.all_gather(torch.cat(ims))
            mak = self.all_gather(torch.cat(mak))

            if len(ims.shape) == 3:
                ims = list(ims)
                mak = list(mak)
            else:
                ims = [ims]
                mak = [mak]

            image_logits = (
                torch.cat(ims).to(mak[0].dtype) @ torch.cat(mak).t() * self.clip.logit_scale.exp()
            )
            ground_truth = (
                torch.arange(len(image_logits)).long().to(image_logits.device)
            )
            loss = (
                F.cross_entropy(image_logits, ground_truth)
                + F.cross_entropy(image_logits.t(), ground_truth)
            ).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            self.log_dict(
                {
                    "train_loss": loss / len(ims),
                    "train_acc": (acc_i + acc_t) / 2 / len(image) / len(ims),
                },
                prog_bar=True,
                sync_dist=True,
            )

        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        optimizer.zero_grad()

        # # image loss
        # for j, mb in enumerate(image_mbs):
        #     images_tmp = copy.deepcopy(ims)
        #     images_tmp[self.global_rank][
        #         j * self.minibatch_size : (j + 1) * self.minibatch_size
        #     ] = F.normalize(self.clip.encode_image(mb), dim=1)
        #     image_logits = (
        #         torch.cat(images_tmp).to(mak[0].dtype) @ torch.cat(mak).t() * self.clip.logit_scale.exp()
        #     )
        #     ground_truth = (
        #         torch.arange(len(image_logits)).long().to(image_logits.device)
        #     )
        #     loss = (
        #         F.cross_entropy(image_logits, ground_truth)
        #         + F.cross_entropy(image_logits.t(), ground_truth)
        #     ) / 2
        #     self.manual_backward(loss)

        # mask loss
        for j, mb in enumerate(mask_mbs):
            text_tmp = copy.deepcopy(mak)
            text_tmp[self.global_rank][
                j * self.minibatch_size : (j + 1) * self.minibatch_size
            ] = F.normalize(self.mask_encoder.forward(mb), dim=1)
            image_logits = (
                torch.cat(ims).to(text_tmp[0].dtype) @ torch.cat(text_tmp).t() * self.clip.logit_scale.exp()
            )
            loss = (
                F.cross_entropy(image_logits, ground_truth)
                + F.cross_entropy(image_logits.t(), ground_truth)
            ) / 2
            self.manual_backward(loss)

        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.clip.logit_scale.data.clamp_(-np.log(100), np.log(100))

    def validation_step(self, train_batch, idx):

        image = train_batch["face"]
        mask = train_batch["mask"].long()

        n = math.ceil(len(image) // self.minibatch_size)
        n = max(n,1)
        image_mbs = torch.chunk(image, n)
        mask_mbs = torch.chunk(mask, n)

        # calculate original statistics
        with torch.no_grad():
            ims = [F.normalize(self.clip.encode_image(im), dim=1) for im in image_mbs]
            mak = [F.normalize(self.mask_encoder.forward(t), dim=1) for t in mask_mbs]
            # gather from all GPUs
            ims = self.all_gather(torch.cat(ims))
            mak = self.all_gather(torch.cat(mak))

            if len(ims.shape) == 3:
                ims = list(ims)
                mak = list(mak)
            else:
                ims = [ims]
                mak = [mak]
            # print(ims[0].dtype,mak[0].dtype,self.clip.logit_scale.exp().dtype)
            image_logits = (
                torch.cat(ims).to(mak[0].dtype) @ torch.cat(mak).t() * self.clip.logit_scale.exp()
            )
            ground_truth = (
                torch.arange(len(image_logits)).long().to(image_logits.device)
            )
            loss = (
                F.cross_entropy(image_logits, ground_truth)
                + F.cross_entropy(image_logits.t(), ground_truth)
            ).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            self.log_dict(
                {
                    "val_loss": loss / len(ims),
                    "val_acc": (acc_i + acc_t) / 2 / len(image) / len(ims),
                },
                prog_bar=True,
                sync_dist=True
            )

    def configure_optimizers(self):
        lr = {
            "RN50": 5e-4,
            "RN101": 5e-4,
            "RN50x4": 5e-4,
            "RN50x16": 4e-4,
            "RN50x64": 3.6e-4,
            "ViT-B/32": 5e-4,
            "ViT-B/16": 5e-4,
            "ViT-L/14": 4e-4,
            "ViT-L/14-336px": 2e-5,
        }[self.clip_config["name"]]
        
        self.froze()
        trainable_params = [
            p for p in self.trainer.model.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.2,
        )

        # Source: https://github.com/openai/CLIP/issues/107
        # Use pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.num_training_steps,
            cycle_mult=1.0,
            max_lr=lr,
            min_lr=0,
            warmup_steps=1000,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
