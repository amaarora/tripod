# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# and UP-DETR (https://github.com/dddzg/up-detr)
# Copyright 2021 Aman Arora
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn

from tripod.util import box_ops
from tripod.util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (
    DETRsegm,
    PostProcessPanoptic,
    PostProcessSegm,
    dice_loss,
    sigmoid_focal_loss,
)
from .transformer import build_transformer
from .detr import DETR, PostProcess, SetCriterion, MLP
from .updetr import UPDETR

def build_model(args):
    num_classes = 20 if args.dataset_file != "coco" else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    if args.mode == "pretrain":
        model = UPDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            num_patches=args.num_patches,
            feature_recon=args.feature_recon,
            query_shuffle=args.query_shuffle,
        )
    else:
        model = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
        )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    return model


def build_criterion(args):
    num_classes = 20 if args.dataset_file != "coco" else 91
    device = torch.device(args.device)

    matcher = build_matcher(args)
    weight_dict = {"loss_ce": 1, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
    )
    criterion.to(device)
    return criterion


def build_postprocessors(args):
    postprocessors = {"bbox": PostProcess()}
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85
            )
    return postprocessors
