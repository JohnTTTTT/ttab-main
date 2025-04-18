import os
from datetime import datetime
from dataclasses import dataclass, field
import tyro
from typing import Optional
import sys
sys.path.append("..")
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import ttab.model_adaptation.utils as adaptation_utils
from ttab.loads.define_model import SelfSupervisedModel
from ttab.configs.datasets import dataset_defaults
from third_party.datasets import get_train_dataset
from third_party.utils import build_model, get_train_params
import faulthandler
faulthandler.enable()

@dataclass
class TrainingParameters:
    data_name: str
    model_name: str
    data_path: str = "./home/johnt/scratch/AffectNet/extracted_files/train_set/separated_images"
    """Parent folder of the target dataset."""
    ckpt_path: str = "./pretrain/checkpoint"
    """Path to save the checkpoint."""
    seed: int = 2022
    device: str = "cuda"
    entry_of_shared_layers: str = "layer4"
    """The position of network where the main model and auxiliary branch split."""
    dim_out: int = 4
    """The dimension of the auxiliary branch output."""
    use_ls: bool = False
    """Whether to use label smoothing."""
    rotation_type: str = "expand"
    """The type of rotation augmentation used in self-supervised auxiliary task."""
    group_norm: Optional[int] = None
    """The number of groups used in group normalization."""
    use_iabn: bool = False
    """Whether to use instance-aware batch normalization (IABN) like note does."""
    iabn_k: int = 4
    """The hyperparameter in IABN that determines the confidence level of the BN statistics."""
    threshold_note: int = 1
    """Skip threshold to discard adjustment in note."""
    batch_size: int = 32
    lr: float = 0.0001
    momentum: float = 0.9
    weight_decay: float = 5e-4
    smooth: float = 0.1
    """"""
    maxEpochs: int = 100
    milestones: list = field(default_factory=lambda: [75, 125])
    """The epochs to decay the learning rate used in lr scheduler."""
    save_epoch: int = 10
    num_workers: int = 32

    @property
    def base_data_name(self):
        return self.data_name.split("_")[0]


DATE_FORMAT = "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
TIME_NOW = datetime.now().strftime(DATE_FORMAT)


def train(config: TrainingParameters) -> None:
    start = time.time()
    model.main_model.train()
    model.ssh.train()
    for _, _, batch in train_loader.iterator(
        batch_size=config.batch_size,
        shuffle=True,
        repeat=False,
        ref_num_data=None,
        num_workers=config.num_workers if hasattr(config, "num_workers") else 32,
        pin_memory=True,
        drop_last=False,
    ):

        optimizer.zero_grad()
        inputs_cls, targets_cls = batch._x, batch._y
        targets_cls_hat = model.main_model(inputs_cls)
        loss = loss_function(targets_cls_hat, targets_cls)

        inputs_ssh, targets_ssh = adaptation_utils.rotate_batch(
            batch._x, config.rotation_type, config.device
        )
        targets_ssh_hat = model.ssh(inputs_ssh)
        loss_ssh = loss_function(targets_ssh_hat, targets_ssh)
        loss += loss_ssh

        loss.backward()
        optimizer.step()

    finish = time.time()
    print("epoch {} training time consumed: {:.2f}s".format(epoch, finish - start))


@torch.no_grad()
def eval_training(config: TrainingParameters, epoch: int) -> float:

    model.main_model.eval()
    model.ssh.eval()

    test_loss = 0.0  # classification error
    ssh_loss = 0.0  # auxiliary task error
    correct = 0.0
    correct_ssh = 0.0

    for _, _, batch in val_loader.iterator(
        batch_size=config.batch_size,
        shuffle=True,
        repeat=False,
        ref_num_data=None,
        num_workers=config.num_workers if hasattr(config, "num_workers") else 32,
        pin_memory=True,
        drop_last=False,
    ):
        # main task.
        with torch.no_grad():
            outputs = model.main_model(batch._x)
            loss = loss_function(outputs, batch._y)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(batch._y).sum()

        # auxiliary task
        with torch.no_grad():
            inputs_ssh, targets_ssh = adaptation_utils.rotate_batch(
                batch._x, config.rotation_type, config.device
            )
            outputs_ssh = model.ssh(inputs_ssh)
            loss_ssh = loss_function(outputs_ssh, targets_ssh)

        ssh_loss += loss_ssh.item()
        _, preds_ssh = outputs_ssh.max(1)
        correct_ssh += preds_ssh.eq(targets_ssh).sum()

    print(
        ("Epoch %d/%d:" % (epoch, config.maxEpochs)).ljust(24)
        + "%.2f\t\t%.2f"
        % (
            correct.float() / len(val_loader.dataset) * 100,
            correct_ssh.float() / (4 * len(val_loader.dataset)) * 100,
        )
    )

    return correct.float() / len(val_loader.dataset)


if __name__ == "__main__":
    cudnn.benchmark = True  # faster runtime
    config = tyro.cli(TrainingParameters)

    # configure model.
    init_model = build_model(config)
    model = SelfSupervisedModel(init_model, config)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model.main_model = nn.DataParallel(model.main_model)
        model.ssh = nn.DataParallel(model.ssh)

    model.main_model.to("cuda")
    model.ssh.to("cuda")
    params = get_train_params(model, config)

    # get dataset.
    train_loader, val_loader = get_train_dataset(config)

    # configure optimization things.
    if config.use_ls:
        loss_function = adaptation_utils.CrossEntropyLabelSmooth(
            num_classes=dataset_defaults[config.base_data_name]["statistics"]["n_classes"], device=config.device, epsilon=config.smooth
        )
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.maxEpochs)

    model.main_model.requires_grad_(True)
    model.ssh.requires_grad_(True)
    model.main_model.train()
    model.ssh.train()

    # create checkpoint folder to save model
    checkpoint_path = os.path.join(
        config.ckpt_path,
        config.model_name + "_with_head",
        config.data_name,
        TIME_NOW,
    )

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if config.group_norm is not None:
        checkpoint_path = os.path.join(
            checkpoint_path, "{model_name}-{epoch}-{type}-gn-{acc}.pth"
        )
    else:
        checkpoint_path = os.path.join(
            checkpoint_path, "{model_name}-{epoch}-{type}-{acc}.pth"
        )

    best_acc = 0.0

    print("Running...")
    print("Error (%)\t\ttest\t\tself-supervised")
    for epoch in range(1, config.maxEpochs + 1):

        train(config)
        acc = eval_training(config, epoch)

        if epoch > config.milestones[1] and best_acc < acc:
            weights_path = checkpoint_path.format(
                model_name=config.model_name + "_with_head",
                epoch=epoch,
                type="best",
                acc=acc,
            )
            print("saving weights file to {}".format(weights_path))
            states = {
                "model": model.main_model.state_dict(),
                "head": model.head.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(states, weights_path)
            best_acc = acc
            continue

        if not epoch % config.save_epoch:
            weights_path = checkpoint_path.format(
                model_name=config.model_name + "_with_head",
                epoch=epoch,
                type="regular",
                acc=acc,
            )
            print("saving weights file to {}".format(weights_path))
            states = {
                "model": model.main_model.state_dict(),
                "head": model.head.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(states, weights_path)

        train_scheduler.step()
