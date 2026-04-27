import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft  # [新增] 引入FFT模块
import math

try:
    from mamba_ssm import MambaSE
except ImportError:
    print("Error: 'mamba_ssm' not found. Please install via: pip install mamba-ssm")
    exit()

# ==========================================
# 🔧 [Mamba-3 组件] RMSNorm
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

# ==========================================
# 🔧 [Mamba-3 核心] Rotary Positional Embedding (RoPE)
# ==========================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_seq_len = 0
        self.register_buffer("cos_cached", None)
        self.register_buffer("sin_cached", None)

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len > self.cached_seq_len:
            self.cached_seq_len = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, :, :])
            self.register_buffer("sin_cached", emb.sin()[None, :, :])
        return self.cos_cached[:, :seq_len, :], self.sin_cached[:, :seq_len, :]

def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat([x1 * cos[..., :d] - x2 * sin[..., :d],
                      x1 * sin[..., :d] + x2 * cos[..., :d]], dim=-1)

# ==========================================
# 🌟 RoPE-Mamba Block
# ==========================================
class RoPEMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mixer = MambaSE(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.norm = RMSNorm(d_model)
        self.rope_emb = RotaryEmbedding(d_model)

    def forward(self, x):
        z = self.norm(x)
        cos, sin = self.rope_emb(z)
        z_rotated = apply_rope(z, cos, sin)
        out = self.mixer(z_rotated)
        return x + out

# ==========================================
# 🚀 整体架构: OAF-Mamba (Time-Frequency Enhanced)
# ==========================================
class OAFMamba(nn.Module):
    def __init__(self, num_classes=2, d_model=68, n_layers=2, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        
        # ---------------------------------------------------------
        # 1. Neuro-Tokenization (Time Domain) - 保持原有
        # ---------------------------------------------------------
        self.eeg_proj = nn.Sequential(
            nn.Linear(68, d_model), RMSNorm(d_model), nn.Dropout(0.2)
        )
        self.fmri_proj = nn.Sequential(
            nn.Linear(68, d_model), RMSNorm(d_model), nn.Dropout(0.2)
        )
        
        # ---------------------------------------------------------
        # 🔥 [新增] 2. Neuro-Tokenization (Frequency Domain)
        # 频域分布与时域不同，需要独立的投影层来对齐到 d_model 空间
        # ---------------------------------------------------------
        self.eeg_freq_proj = nn.Sequential(
            nn.Linear(68, d_model), RMSNorm(d_model), nn.Dropout(0.2)
        )
        self.fmri_freq_proj = nn.Sequential(
            nn.Linear(68, d_model), RMSNorm(d_model), nn.Dropout(0.2)
        )

        self.modality_embed = nn.Embedding(2, d_model)

        # ---------------------------------------------------------
        # 3. Dual-Tower Backbone (Shared for Time & Freq)
        # 我们复用同一个 Tower，让 Mamba 学习跨域的通用表征
        # ---------------------------------------------------------
        # Tower A: EEG-First Sequence
        self.tower_a = nn.Sequential(*[RoPEMambaBlock(d_model) for _ in range(n_layers)])
        self.norm_a = RMSNorm(d_model)

        # Tower B: fMRI-First Sequence
        self.tower_b = nn.Sequential(*[RoPEMambaBlock(d_model) for _ in range(n_layers)])
        self.norm_b = RMSNorm(d_model)
        
        # ---------------------------------------------------------
        # 4. Projection Heads (Contrastive Learning)
        # ---------------------------------------------------------
        # Time Domain Heads (原有)
        self.proj_head_a = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 32))
        self.proj_head_b = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 32))
        
        # 🔥 [新增] Frequency Domain Heads
        self.proj_head_a_freq = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 32))
        self.proj_head_b_freq = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 32))
        
        # ---------------------------------------------------------
        # 5. Classifier (Fine-tuning)
        # 输入维度变大：(Time_Feat + Freq_Feat) * 2 (Towers) = d_model * 4
        # ---------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(136, 64), # 修改维度以融合时频特征
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def compute_fft_magnitude(self, x):
        """
        输入 x: [Batch, SeqLen, Channels] (假设 Channel=68 在最后)
        输出: 同维度的频域幅值特征
        """
        # 对 SeqLen 维度 (dim=1) 进行 FFT
        # 使用 torch.fft.fft 而不是 rfft 以保持序列长度一致 (符合你要求的 "same dimension")
        x_freq = torch.fft.fft(x, dim=1)
        # 取模 (Magnitude) 得到实数
        x_mag = torch.abs(x_freq)
        # 归一化 (可选，防止频域数值过大)
        x_mag = F.normalize(x_mag, dim=1)
        return x_mag

    def get_embeddings(self, eeg, fmri, use_freq=False):
        """
        统一的 Embedding 获取函数
        """
        # 1. 决定使用哪个投影层
        if use_freq:
            # 如果是频域，先提取 FFT 特征
            eeg_in = self.compute_fft_magnitude(eeg)
            fmri_in = self.compute_fft_magnitude(fmri)
            proj_eeg = self.eeg_freq_proj
            proj_fmri = self.fmri_freq_proj
        else:
            # 如果是时域，直接使用原始数据
            eeg_in = eeg
            fmri_in = fmri
            proj_eeg = self.eeg_proj
            proj_fmri = self.fmri_proj

        # 2. 投影
        x_eeg = proj_eeg(eeg_in)
        x_fmri = proj_fmri(fmri_in)
        
        # 3. 添加 Modality Embedding
        batch_size = eeg.shape[0]
        idx_0 = torch.zeros(batch_size, dtype=torch.long, device=eeg.device)
        idx_1 = torch.ones(batch_size, dtype=torch.long, device=fmri.device)
        
        x_eeg = x_eeg + self.modality_embed(idx_0).unsqueeze(1)
        x_fmri = x_fmri + self.modality_embed(idx_1).unsqueeze(1)
        
        return x_eeg, x_fmri

    def _process_towers(self, x_eeg, x_fmri):
        """
        内部函数：处理拼接和通过 Mamba Towers
        """
        # 拼接策略
        seq_a = torch.cat([x_eeg, x_fmri], dim=1) 
        seq_b = torch.cat([x_fmri, x_eeg], dim=1) 
        
        # Tower A
        out_a = self.tower_a(seq_a)
        out_a = self.norm_a(out_a)
        feat_a = out_a.mean(dim=1) 
        
        # Tower B
        out_b = self.tower_b(seq_b)
        out_b = self.norm_b(out_b)
        feat_b = out_b.mean(dim=1)
        
        return feat_a, feat_b

    def forward_features_all(self, eeg, fmri):
        """
        同时计算时域和频域的所有特征
        """
        # --- Time Domain Path ---
        t_x_eeg, t_x_fmri = self.get_embeddings(eeg, fmri, use_freq=False)
        feat_time_a, feat_time_b = self._process_towers(t_x_eeg, t_x_fmri)
        
        # --- Frequency Domain Path ---
        f_x_eeg, f_x_fmri = self.get_embeddings(eeg, fmri, use_freq=True)
        feat_freq_a, feat_freq_b = self._process_towers(f_x_eeg, f_x_fmri)
        
        return feat_time_a, feat_time_b, feat_freq_a, feat_freq_b

    def forward_contrastive(self, eeg, fmri):
        """
        预训练用：返回4个投影后的向量，供外部 Loss 计算多重对比
        """
        feat_time_a, feat_time_b, feat_freq_a, feat_freq_b = self.forward_features_all(eeg, fmri)
        
        # 投影到对比空间
        z_time_a = F.normalize(self.proj_head_a(feat_time_a), dim=1)
        z_time_b = F.normalize(self.proj_head_b(feat_time_b), dim=1)
        
        z_freq_a = F.normalize(self.proj_head_a_freq(feat_freq_a), dim=1)
        z_freq_b = F.normalize(self.proj_head_b_freq(feat_freq_b), dim=1)
        
        # 返回4个向量，训练脚本里可以做 Time-Time, Freq-Freq, Time-Freq 对比
        return z_time_a, z_time_b, z_freq_a, z_freq_b

    # def forward(self, eeg, fmri):
    #     """
    #     微调用：融合时域和频域特征进行分类
    #     """
    #     feat_time_a, feat_time_b, feat_freq_a, feat_freq_b = self.forward_features_all(eeg, fmri)
        
    #     # 融合所有特征: [Batch, d_model * 4]
    #     feat_combined = torch.cat([feat_time_a, feat_time_b, feat_freq_a, feat_freq_b], dim=1)
    #     print(feat_combined.shape)
    #     logits = self.classifier(feat_combined)
    #     return logits
    
    def forward(self, eeg, fmri):
        feat_time_a, feat_time_b, feat_freq_a, feat_freq_b = self.forward_features_all(eeg, fmri)
        
        # 融合策略：直接相加 (假设预训练已经把它们对齐了)
        feat_a = feat_time_a + feat_freq_a
        feat_b = feat_time_b + feat_freq_b
        
        # 现在的维度是 d_model * 2，和旧模型 model.py 一样！
        feat_combined = torch.cat([feat_a, feat_b], dim=1)
        # print(feat_combined.shape)
        logits = self.classifier(feat_combined)
        return logits