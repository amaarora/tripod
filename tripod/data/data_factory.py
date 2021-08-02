# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# and UP-DETR (https://github.com/dddzg/up-detr)
# Copyright 2021 Aman Arora
# ------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader, DistributedSampler
from tripod.util.misc import default_collate_fn, updetr_collate_fn


def build_loader(
    dataset,
    is_training,
    is_distributed,
    shuffle=False,
    batch_size=2,
    drop_last=True,
    use_updetr_collate=False,
    num_workers=4,
):
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        SamplerClass = (
            torch.utils.data.RandomSampler
            if is_training and shuffle
            else torch.utils.data.SequentialSampler
        )
        sampler = SamplerClass(dataset)

    batch_sampler = torch.utils.data.BatchSampler(
        sampler, batch_size, drop_last=drop_last
    )

    collate_fn = default_collate_fn if not use_updetr_collate else updetr_collate_fn

    
    data_loader = DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=num_workers
    )
    return data_loader
