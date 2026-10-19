"""Data loading utilities for Chini cluster datasets."""

import os

import torch


def load_chini_dataset(data_dir: str):
    """Load all .pt files from data_dir, return list of Data objects."""
    data_list = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".pt"):
            continue
        data = torch.load(os.path.join(data_dir, fname), map_location="cpu",
                          weights_only=False)
        data.filename = fname
        data_list.append(data)
    return data_list


def collate_fn(batch):
    """Collate list of Data objects into batched tensors."""
    z_list, pos_list, y_list, batch_idx = [], [], [], []
    for i, data in enumerate(batch):
        z_list.append(data.z)
        pos_list.append(data.pos)
        y_list.append(float(data.y))
        batch_idx.append(torch.full((data.z.size(0),), i, dtype=torch.long))
    return (torch.cat(z_list), torch.cat(pos_list),
            torch.cat(batch_idx), torch.tensor(y_list, dtype=torch.float))
