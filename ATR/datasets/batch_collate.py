from typing import List, Dict, Any

import torch
from aac_datasets.utils.collections import list_dict_to_dict_list
from torch import Tensor


class CustomCollate:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, batch_lst: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_dic: Dict[str, Any] = list_dict_to_dict_list(batch_lst)
        keys = list(batch_dic.keys())

        for key in keys:
            values = batch_dic[key]

            if len(values) == 0:
                batch_dic[key] = values
                continue

            are_tensors = [isinstance(value, Tensor) for value in values]
            if not all(are_tensors):
                batch_dic[key] = values
                continue

            if can_be_stacked(values):
                values = torch.stack(values)
                batch_dic[key] = values
                continue

            batch_dic[key] = values
        return batch_dic


def can_be_stacked(tensors: List[Tensor]) -> bool:
    """Returns true if a list of tensors can be stacked with torch.stack function."""
    if len(tensors) == 0:
        return False
    shape0 = tensors[0].shape
    are_stackables = [tensor.shape == shape0 for tensor in tensors]
    return all(are_stackables)
