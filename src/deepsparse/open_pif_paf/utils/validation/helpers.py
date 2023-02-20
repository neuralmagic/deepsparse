import torch
from openpifpaf import transforms
import numpy

__all__ = ["apply_deepsparse_preprocessing", "deepsparse_fields_to_torch"]

def deepsparse_fields_to_torch(fields_batch, device='cpu'):
    result = []
    fields = fields_batch.fields
    for idx, (cif, caf) in enumerate(zip(*fields)):
        result.append([torch.from_numpy(cif).to(device), torch.from_numpy(caf).to(device)])
    return

def apply_deepsparse_preprocessing(data_loader: torch.utils.data.DataLoader, img_size: int) -> torch.utils.data.DataLoader:
    data_loader.dataset.preprocess.preprocess_list[2] = transforms.CenterPad(img_size)