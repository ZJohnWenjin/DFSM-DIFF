import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib


def get_all_nii_paths_FMM(root_dir, modality_list):
    all_cases = []

    for subdir in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        files = os.listdir(subdir_path)

        found = {}
        for modality in modality_list:
            modality_lower = modality.lower()
            for fname in files:
                fname_lower = fname.lower()
                if modality_lower in fname_lower:
                    found[modality] = os.path.join(subdir_path, fname)
                    break 

        if len(found) == len(modality_list):
            case_paths = [found[m] for m in modality_list]
            all_cases.append(case_paths)

    return all_cases


def get_all_nii_paths_Diff(root_dir, modalities_name, modality_target):
    all_cases = []

    all_modalities = modalities_name + [modality_target]

    for subdir in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        files = os.listdir(subdir_path)
        found = {}

        for modality in all_modalities:
            modality_lower = modality.lower()

            for fname in files:
                if modality_lower in fname.lower():
                    found[modality] = os.path.join(subdir_path, fname)
                    break

        if len(found) == len(all_modalities):
            name_group = [found[m] for m in modalities_name]    
            target_path = found[modality_target]              

            all_cases.append([name_group, target_path])

    return all_cases

class NiiDataset_FMM(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        modalities_list = self.file_list[idx]
        m_list_tensor = []
        for i in range(len(modalities_list)):
            labels_nii = nib.load(modalities_list[i])
            m_numpy = self.min_max_normalize(labels_nii.get_fdata())
            m_list_tensor.append(m_numpy)

        m_list_tensor = torch.cat(m_list_tensor, dim=0)
        # shape: (num_modality, H, W, Z）
        return m_list_tensor

    def min_max_normalize(self,arr):

        min_val = np.min(arr)
        max_val = np.max(arr)

        normalized_arr = (arr - min_val) / (max_val - min_val)

        if max_val == 0:
          return np.zeros_like(arr)

        normalized_arr = np.transpose(normalized_arr, (2, 0, 1))
        tensor_nii = torch.from_numpy(normalized_arr).float().unsqueeze(1)  # shape: (Z, 1, W, H)

        tensor_nii = F.interpolate(tensor_nii, size=(224, 224), mode='bilinear', align_corners=False)

        return tensor_nii.permute(1,0,2,3) # shape: (1, Z, W, H)

class NiiDataset_Diff(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        modalities_list = self.file_list[idx][0]
        m_list_tensor = []
        for i in range(len(modalities_list)):
            inputs_nii = nib.load(modalities_list[i])
            m_numpy = self.min_max_normalize(inputs_nii.get_fdata())
            m_list_tensor.append(m_numpy)
        m_tensors = torch.cat(m_list_tensor, dim=0)

        modalities_label = self.file_list[idx][1]
        labels_nii = nib.load(modalities_label)
        label_tensor= self.min_max_normalize(labels_nii.get_fdata())
        # shape: (M Z H W) (1 Z H W)
        return m_tensors,label_tensor.squeeze(0)

    def min_max_normalize(self,arr):

        min_val = np.min(arr)
        max_val = np.max(arr)

        normalized_arr = (arr - min_val) / (max_val - min_val)

        if max_val == 0:
          return np.zeros_like(arr)

        normalized_arr = np.transpose(normalized_arr, (2, 0, 1))
        tensor_nii = torch.from_numpy(normalized_arr).float().unsqueeze(1)  # shape: (Z, 1, W, H)

        tensor_nii = F.interpolate(tensor_nii, size=(224, 224), mode='bilinear', align_corners=False)

        return tensor_nii.permute(1,0,2,3) # shape: (1, Z, W, H)


def build_loader_for_FMM(data_path,m_list,train=True):
    nii_files = get_all_nii_paths_FMM(data_path,m_list)
    nii_len = int(len(nii_files)*0.8)
    if train:
        nii_files = nii_files[:nii_len]
    else:
        nii_files = nii_files[nii_len:]
    dataset = NiiDataset_FMM(nii_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12,
                            pin_memory=True, drop_last=False)

    return dataloader

def build_loader_for_Diff(data_path,m_list,m_target,train=True):
    nii_files = get_all_nii_paths_Diff(data_path,m_list,m_target)
    nii_len = int(len(nii_files)*0.8)
    if train:
        nii_files = nii_files[:nii_len]
    else:
        nii_files = nii_files[nii_len:]
    dataset = NiiDataset_Diff(nii_files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=12,
                            pin_memory=True, drop_last=False)
    return dataloader


