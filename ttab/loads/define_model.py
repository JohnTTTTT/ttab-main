# -*- coding: utf-8 -*-
"""
define_model.py

This module defines model constructors and pretrained-loading logic for TTAB.
"""
import os
import timm
import torch
from torch import nn
from collections import OrderedDict
import ttab.model_adaptation.utils as adaptation_utils
from ttab.loads.models import WideResNet, cct_7_3x1_32, resnet
from ttab.loads.models.resnet import (
    ResNetCifar,
    ResNetImagenet,
    ResNetMNIST,
    ResNetAffectNet,
)


def strip_module_prefix(state_dict: dict) -> OrderedDict:
    """
    Remove 'module.' prefix from state_dict keys (for DataParallel compatibility).
    """
    new_state = OrderedDict()
    for k, v in state_dict.items():
        new_state[k.replace("module.", "")] = v
    return new_state


class SelfSupervisedModel(nn.Module):
    """
    Wrapper for self-supervised TTT adaptation, adding an auxiliary head.
    """
    def __init__(self, model: nn.Module, config):
        super(SelfSupervisedModel, self).__init__()
        self._config = config
        self.main_model = model
        self.ext, self.head = self._define_head()
        self.ssh = adaptation_utils.ExtractorHead(self.ext, self.head)

    # ... existing _define_resnet_head, _define_vit_head, _define_head methods ...

    def load_pretrained_parameters(self, ckpt_path: str):
        """
        Load and merge backbone and head weights from a checkpoint into the main_model.
        """
        # 1) load checkpoint dict
        ckpt = torch.load(ckpt_path, map_location=self._config.device)
        backbone_sd = ckpt.get("model", {})
        head_sd = ckpt.get("head", {})

        # 2) strip DataParallel prefixes
        clean_backbone = strip_module_prefix(backbone_sd)
        clean_head = strip_module_prefix(head_sd)

        # 3) merge both dicts
        merged_sd = OrderedDict(**clean_backbone, **clean_head)

        # 4) load into the wrapped model (main_model)
        missing, unexpected = self.main_model.load_state_dict(merged_sd, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)


def define_model(config):
    """
    Instantiate a model based on config.data_names and config.model_name.
    """
    # Imagenet-pretrained via timm
    if 'imagenet' in config.data_names:
        if config.group_norm_num_groups is not None:
            assert config.model_name == 'resnet50'
            return timm.create_model(config.model_name + '_gn', pretrained=True)
        return timm.create_model(config.model_name, pretrained=True)

    # Local models (ResNet, WideResNet, etc.)
    if 'wideresnet' in config.model_name:
        depth, widen_factor = map(int, config.model_name.replace('wideresnet', '').split('_'))
        return WideResNet(
            depth, widen_factor, config.statistics['n_classes'],
            split_point=config.entry_of_shared_layers, dropout_rate=0.0
        )
    elif 'resnet' in config.model_name:
        depth = int(config.model_name.replace('resnet', ''))
        return resnet(
            config.base_data_name,
            depth,
            split_point=config.entry_of_shared_layers,
            group_norm_num_groups=config.group_norm_num_groups,
            grad_checkpoint=config.grad_checkpoint,
        )
    elif 'vit_large_patch16_224' in config.model_name:
        # just build the ViT-Large classifier head—no weight loading here
        model = timm.create_model(
            config.model_name,
            pretrained=False,
            num_classes=config.statistics['n_classes']
        )
        if config.grad_checkpoint:
            model.set_grad_checkpointing()
        return model
    elif 'cct' in config.model_name:
        return cct_7_3x1_32(pretrained=False)
    else:
        raise NotImplementedError(f"invalid model_name={config.model_name}.")


def load_pretrained_model(config, model: nn.Module):
    """
    Load a checkpoint into model (can be plain or SelfSupervisedModel).
    """
    # sanity check
    assert os.path.exists(config.ckpt_path), (
        'Checkpoint path does not exist: %s' % config.ckpt_path
    )

    # skip imagenet-pretrained branches
    if 'imagenet' in config.data_names:
        return

    # handle self-supervised wrapper
    if isinstance(model, SelfSupervisedModel):
        model.load_pretrained_parameters(config.ckpt_path)
        return

    # 1) load checkpoint
    ckpt = torch.load(config.ckpt_path, map_location=config.device)
    backbone_sd = ckpt.get('model', {})
    head_sd = ckpt.get('head', {})

    # 2) strip DataParallel prefixes
    clean_backbone = strip_module_prefix(backbone_sd)
    
    remapped = OrderedDict()
    for k, v in clean_backbone.items():
        if k.startswith("fc_norm."):
            # -> norm.weight or norm.bias
            new_k = k.replace("fc_norm.", "norm.")
        else:
            new_k = k
        remapped[new_k] = v

    # 3) now merge remapped backbone + head (if any)
    clean_head = strip_module_prefix(head_sd)
    merged_sd = OrderedDict(**remapped, **clean_head)

    # 4) load into model
    missing, unexpected = model.load_state_dict(merged_sd, strict=False)
    print('Missing keys:', missing)
    print('Unexpected keys:', unexpected)
