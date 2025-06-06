# utils functions.
import copy
import numpy as np
import sys
sys.path.append("..")
import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import ttab.model_adaptation.utils as adaptation_utils
from ttab.loads.models import resnet, WideResNet
from ttab.configs.datasets import dataset_defaults


def convert_iabn(module: nn.Module, config, **kwargs) -> nn.Module:
    module_output = module
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        IABN = (
            adaptation_utils.InstanceAwareBatchNorm2d
            if isinstance(module, nn.BatchNorm2d)
            else adaptation_utils.InstanceAwareBatchNorm1d
        )
        module_output = IABN(
            num_channels=module.num_features,
            k=config.iabn_k,
            eps=module.eps,
            momentum=module.momentum,
            threshold=config.threshold_note,
            affine=module.affine,
        )

        module_output._bn = copy.deepcopy(module)

    for name, child in module.named_children():
        module_output.add_module(name, convert_iabn(child, **kwargs))
    del module
    return module_output

def update_pytorch_model_key(state_dict: dict) -> dict:
    """This function is used to modify the state dict key of pretrained model from pytorch."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if "downsample" in key:
            name_split = key.split(".")
            if name_split[-2] == "0":
                name_split[-2] = "conv" 
                new_key = ".".join(name_split)
                new_state_dict[new_key] = value
            elif name_split[-2] == "1":
                name_split[-2] = "bn" 
                new_key = ".".join(name_split)
                new_state_dict[new_key] = value
        elif "fc" in key:
            name_split = key.split(".")
            if name_split[0] == "fc":
                name_split[0] = "classifier"
                new_key = ".".join(name_split)
                new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict

def build_model(config) -> nn.Module:
    """Build model from `config`"""
    num_classes = dataset_defaults[config.base_data_name]["statistics"]["n_classes"]
    if config.base_data_name in ["officehome", "pacs", "waterbirds"]:
        pretrained_model = models.resnet50(pretrained=True)
        pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, out_features=num_classes, bias=False)
        pretrained_model.fc.weight.data.normal_(mean=0, std=0.01)

        model = resnet(config.base_data_name, depth=50, split_point=config.entry_of_shared_layers, grad_checkpoint=True).to(config.device)
        model_dict = model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        new_pretrained_dict = update_pytorch_model_key(pretrained_dict)
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        model.load_state_dict(new_pretrained_dict)
        model.to(config.device)
        del pretrained_dict, pretrained_model, new_pretrained_dict
        if config.use_iabn:
            model = convert_iabn(model, config)
    else:
        if "wideresnet" in config.model_name:
            components = config.model_name.split("_")
            depth = int(components[0].replace("wideresnet", ""))
            widen_factor = int(components[1])

            model = WideResNet(
                depth,
                widen_factor,
                num_classes,
                split_point=config.entry_of_shared_layers,
                dropout_rate=0.3,
            )
        elif "resnet" in config.model_name:
            depth = int(config.model_name.replace("resnet", ""))
            model = resnet(
                config.base_data_name,
                depth,
                split_point=config.entry_of_shared_layers,
                group_norm_num_groups=config.group_norm,
            ).to(config.device)
            if config.use_iabn:
                assert config.group_norm is None, "IABN cannot be used with group norm."
                model = convert_iabn(model, config)
        elif "vit" in config.model_name.lower():
            # choose the large variant if user gave “vit”
            full_model_name = (
                "vit_large_patch16_224"
                if config.model_name.lower() == "vit"
                else config.model_name
            )

            # 1) load your finetuned ViT bundle from config.ckpt_path
            ckpt = torch.load(config.ckpt_path, map_location=config.device)
            state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

            # 2) try to find the head.weight entry
            head_keys = [k for k in state_dict if k.endswith("head.weight")]
            if head_keys:
                head_w = state_dict[head_keys[0]]
                num_classes_ckpt = head_w.shape[0]
            else:
                # fallback: look up AffectNet’s class count
                num_classes_ckpt = dataset_defaults["affectnet"]["statistics"]["n_classes"]
                print(
                    f"⚠️ head.weight not found in checkpoint; "
                    f"defaulting to {num_classes_ckpt} classes"
                )

            # 3) instantiate a ViT whose head matches that size
            model = timm.create_model(
                full_model_name,
                pretrained=False,
                num_classes=num_classes_ckpt,
            )

            # 4) remap any fc_norm.* → norm.* keys
            new_state = {}
            for k, v in state_dict.items():
                key = k[len("model."):] if k.startswith("model.") else k
                if key.startswith("fc_norm."):
                    key = "norm." + key.split(".", 1)[1]
                new_state[key] = v

            # 5) load weights (strict=False lets you get a fresh head if none existed)
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        else:
            raise ValueError(f"Unsupported model name: {config.model_name}")
    return model

def get_train_params(model: nn.Module, config) -> list:
    """Define the trainable parameters for a model using `config`"""
    main_model = model.main_model.module if isinstance(model.main_model, torch.nn.DataParallel) else model.main_model
    ssh_model = model.ssh.module if isinstance(model.ssh, torch.nn.DataParallel) else model.ssh

    if config.base_data_name in ["officehome", "pacs"]:
        params = []
        learning_rate = config.lr

        # Use the unwrapped main_model
        for name_module, module in main_model.named_children():
            if name_module != "classifier":
                for _, param in module.named_parameters():
                    params += [{"params": param, "lr": learning_rate * 0.1}]
            else:
                for _, param in module.named_parameters():
                    params += [{"params": param, "lr": learning_rate}]
        
        # Use the unwrapped ssh_model for accessing head
        for name_module, module in ssh_model.head.named_children():
            if isinstance(module, nn.Linear):
                for _, param in module.named_parameters():
                    params += [{"params": param, "lr": learning_rate}]
            else:
                for _, param in module.named_parameters():
                    params += [{"params": param, "lr": learning_rate * 0.1}]
    else:
        # Again, use the unwrapped models here
        params = list(main_model.parameters()) + list(ssh_model.head.parameters())
    return params

def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def mixup_data(x, y, alpha=1.0, device=None):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device is None:
        index = torch.randperm(batch_size)
    else:
        index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
