import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class BrainDataset(Dataset):
    def __init__(self, data_dir, mode='pretrain', target_pair=None):
        """
        Args:
            data_dir (str): h5文件存放的文件夹路径
            mode (str): 'pretrain' (加载所有数据) 或 'finetune' (加载特定两类)
            target_pair (list/tuple): 仅在 finetune 模式下有效，例如 [0, 3]。
                                      列表第一个元素将作为类别0，第二个元素作为类别1。
        """
        self.data_dir = data_dir
        self.mode = mode
        self.target_pair = target_pair
        
        # 获取所有h5文件路径
        self.all_files = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.h5')
        ]
        
        self.valid_files = []
        self.labels = []
        
        print(f"[{mode.upper()}] Scanning {len(self.all_files)} files in {data_dir}...")
        
        # 预扫描：根据模式筛选文件
        for file_path in self.all_files:
            try:
                # 只需要快速读取label，不读取大数据
                with h5py.File(file_path, 'r') as f:
                    # 假设 label 存储为标量或数组
                    label = int(f['label'][()]) 
                #label
                if mode == 'pretrain':
                    # 预训练模式：加载所有数据，不关心具体是哪一类
                    self.valid_files.append(file_path)
                    self.labels.append(label)
                    
                elif mode == 'finetune':
                    # 微调模式：只保留 target_pair 中的类别
                    if target_pair is not None and label in target_pair:
                        self.valid_files.append(file_path)
                        # 将原始标签 (e.g., 0, 3) 映射为二分类标签 (0, 1)
                        binary_label = 0 if label == target_pair[0] else 1
                        self.labels.append(binary_label)
                        
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        print(f"[{mode.upper()}] Loaded {len(self.valid_files)} valid samples.")
        if mode == 'finetune':
            print(f"Target Pair: {target_pair} (Mapped to 0 and 1)")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file_path = self.valid_files[idx]
        label = self.labels[idx]
        
        with h5py.File(file_path, 'r') as f:
            # 1. 读取原始数据
            # 注意：根据你的描述，fMRI是 (68, 370)，EEG是 (68, 250)
            fmri_data = f['fMRI-DK68-node-mat'][:]  # (68, 370)
            eeg_data = f['sLORETA_mean_func'][:]    # (68, 250)
            
            # 2. 转换为 Float Tensor
            fmri_tensor = torch.tensor(fmri_data, dtype=torch.float32)
            eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
            
            # 3. 维度转置 (Transpose)
            # 模型通常需要 (Sequence_Length, Channels)
            # 原始 (68, T) -> 转置为 (T, 68)
            fmri_tensor = fmri_tensor.permute(1, 0) # (370, 68)
            eeg_tensor = eeg_tensor.permute(1, 0)   # (250, 68)
            
            # 4. 数据标准化 (Z-Score Normalization) - 这一步对收敛非常重要！
            # 对每个样本的每个通道独立归一化
            eeg_tensor = (eeg_tensor - eeg_tensor.mean(dim=0)) / (eeg_tensor.std(dim=0) + 1e-6)
            fmri_tensor = (fmri_tensor - fmri_tensor.mean(dim=0)) / (fmri_tensor.std(dim=0) + 1e-6)

        return eeg_tensor, fmri_tensor, label

# ==========================================
# Mixup 辅助工具 (将在训练脚本中调用)
# ==========================================
def mixup_data(eeg, fmri, y, alpha=1.0, device='cuda'):
    '''
    返回: 混合后的EEG, 混合后的fMRI, 标签A, 标签B, 混合比例lambda
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = eeg.size(0)
    index = torch.randperm(batch_size).to(device)

    # 对 EEG 和 fMRI 同时进行混合，保持对应关系
    mixed_eeg = lam * eeg + (1 - lam) * eeg[index, :]
    mixed_fmri = lam * fmri + (1 - lam) * fmri[index, :]
    
    y_a, y_b = y, y[index]
    return mixed_eeg, mixed_fmri, y_a, y_b, lam