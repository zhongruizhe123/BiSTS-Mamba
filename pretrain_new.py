import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm 
from dataset import BrainDataset
from model_new import OAFMamba

# ==========================================
# 配置参数
# ==========================================
CONFIG = {
    "data_dir": r"E:\Project\Mamba-pretrain\dataset_h5", 
    "batch_size": 32,
    "epochs": 100,
    "lr": 1e-3,             
    "temperature": 0.07,    
    "save_dir": "./checkpoints",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# InfoNCE Loss (保持不变，支持复用)
# ==========================================
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_a, z_b):
        """
        计算两个视图 z_a 和 z_b 之间的对比损失
        """
        batch_size = z_a.shape[0]
        # Cosine Similarity
        logits = torch.matmul(z_a, z_b.T) / self.temperature
        # Labels (Diagonal)
        labels = torch.arange(batch_size).to(z_a.device)
        return self.criterion(logits, labels)

# ==========================================
# 训练主循环
# ==========================================
def main():
    # 1. 环境设置
    if not os.path.exists(CONFIG["save_dir"]):
        os.makedirs(CONFIG["save_dir"])
    
    print(f"Running Pre-training on: {CONFIG['device']}")
    print(f"Loading data from: {CONFIG['data_dir']}")

    # 2. 数据加载
    dataset = BrainDataset(data_dir=CONFIG["data_dir"], mode='pretrain')
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=0,
        drop_last=True 
    )

    # 3. 模型初始化 (适配新的 OAFMamba)
    model = OAFMamba(num_classes=2, d_model=68, n_layers=1).to(CONFIG["device"])
    
    # 4. 优化器与损失
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    criterion = InfoNCELoss(temperature=CONFIG["temperature"])

    # 5. 开始训练
    model.train()
    min_loss = float('inf')
    
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for eeg, fmri, _ in loop: 
            eeg, fmri = eeg.to(CONFIG["device"]), fmri.to(CONFIG["device"])
            
            optimizer.zero_grad()
            
            # ==========================================
            # 🔥 [修改] Forward: 获取4个特征向量
            # ==========================================
            # z_time_a: 时域 Tower A (EEG first)
            # z_time_b: 时域 Tower B (fMRI first)
            # z_freq_a: 频域 Tower A (EEG first)
            # z_freq_b: 频域 Tower B (fMRI first)
            z_time_a, z_time_b, z_freq_a, z_freq_b = model.forward_contrastive(eeg, fmri)
            
            # ==========================================
            # 🔥 [新增] Loss Calculation: 三重对比机制
            # ==========================================
            
            # 1. 时域对比 (Time Domain Contrastive)
            # 目标：学习序列顺序的不变性 (Original OAF logic)
            loss_time = criterion(z_time_a, z_time_b)
            
            # 2. 频域对比 (Frequency Domain Contrastive)
            # 目标：在频域特征上学习序列顺序的不变性
            loss_freq = criterion(z_freq_a, z_freq_b)
            
            # 3. 时频一致性对比 (Time-Frequency Consistency)
            # 目标：强制同一数据的时域表征和频域表征在语义上对齐
            # 计算 Tower A 的时频一致性 + Tower B 的时频一致性
            loss_tf_a = criterion(z_time_a, z_freq_a)
            loss_tf_b = criterion(z_time_b, z_freq_b)
            loss_consistency = (loss_tf_a + loss_tf_b) / 2
            
            # 总损失：简单加权求和 (可以根据需要调整权重，例如 1.0, 1.0, 0.5)
            loss = loss_time + loss_freq + loss_consistency
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 更新进度条显示，增加详细信息以便监控
            loop.set_postfix({
                "L_tot": f"{loss.item():.2f}",
                "L_tm": f"{loss_time.item():.2f}", 
                "L_fr": f"{loss_freq.item():.2f}"
            })
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < min_loss:
            min_loss = avg_loss
            best_path = os.path.join(CONFIG["save_dir"], "oaf_mamba_pretrained_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"🌟 New Best Model Saved! (Loss: {min_loss:.4f})")
            
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(CONFIG["save_dir"], f"oaf_mamba_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            
    final_path = os.path.join(CONFIG["save_dir"], "oaf_mamba_pretrained_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n>>> Pre-training Complete! Model saved to: {final_path}")
    
if __name__ == "__main__":
    main()