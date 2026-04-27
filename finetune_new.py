import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import time
import copy

# 导入你的模块
from dataset import BrainDataset, mixup_data
from model_new import OAFMamba

# ==========================================
# ⚙️ 实验配置
# ==========================================
CONFIG = {
    "data_dir": r"E:\Project\Mamba-pretrain\dataset_h5_hand", 
    "pretrained_path": "./checkpoints/oaf_mamba_pretrained_final.pth",
    "save_dir": "./kfold_results",
    
    # ⚠️ 务必确保这里和您旧结果的配置一致
    "target_pair": [0, 1, 2], 
    
    "num_runs": 10,       
    "k_folds": 5,          
    
    "batch_size": 16,
    "epochs": 40,
    "lr": 5e-4, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 🔥 [新增] 显式指定层数，确保与预训练(n_layers=1)结构一致，否则权重无法对齐
    "n_layers": 2
}

# ==========================================
# 工具函数
# ==========================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_pretrained_model(model, path, device):
    """
    修改版：增加了 Shape 检查，防止因模型结构微调导致权重形状不匹配而报错。
    """
    if not os.path.exists(path): 
        print(f"⚠️ Warning: Pretrained model not found at {path}")
        return model
        
    print(f"Loading pretrained weights from: {path}")
    checkpoint = torch.load(path, map_location=device)
    model_dict = model.state_dict()
    
    # 🔥 [修改] 智能过滤：不仅检查 key 是否存在，还检查 shape 是否一致
    # 新模型的 classifier 输入是 d_model * 4，旧权重可能是 d_model * 2，
    # 这一步会自动跳过形状不匹配的层（如 classifier），仅加载匹配的骨干网络（Tower/Projections）。
    pretrained_dict = {
        k: v for k, v in checkpoint.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    
    # 打印加载情况反馈
    full_keys = len(checkpoint.keys())
    loaded_keys = len(pretrained_dict)
    print(f"--> Loaded {loaded_keys}/{full_keys} layers. (Skipped mismatched/missing layers)")

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def calculate_metrics(y_true, y_pred, y_probs):
    y_true, y_pred, y_probs = np.array(y_true), np.array(y_pred), np.array(y_probs)
    acc = (y_true == y_pred).mean() * 100
    p = precision_score(y_true, y_pred, zero_division=0) * 100
    r = recall_score(y_true, y_pred, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100
    try: auc_pr = average_precision_score(y_true, y_probs) * 100
    except: auc_pr = 0.0
    return {"Acc": acc, "Precision": p, "Recall": r, "F1": f1, "AUC-PR": auc_pr}

# ==========================================
# 单折训练函数
# ==========================================
def train_one_fold(train_idx, val_idx, dataset, run_id, fold_id):
    # 1. 构建 Subset
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # 2. 模型初始化
    # 🔥 [修改] 传入 n_layers 参数，确保结构与预训练一致
    model = OAFMamba(
        num_classes=2, 
        d_model=68, 
        n_layers=CONFIG["n_layers"]
    ).to(CONFIG["device"])
    
    model = load_pretrained_model(model, CONFIG["pretrained_path"], CONFIG["device"])
    
    # 3. 动态计算类别权重
    train_labels = [dataset.labels[i] for i in train_idx]
    counts = torch.bincount(torch.tensor(train_labels))
    class_weights = len(train_labels) / (2.0 * counts.float())
    class_weights = class_weights.to(CONFIG["device"])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_metrics = None
    best_f1 = 0.0
    best_model_state_dict = None  
    best_outputs = None           
    
    epoch_history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }
    
    # --- 4. 训练循环 ---
    for epoch in range(CONFIG["epochs"]):
        # Training
        model.train()
        train_loss_accum = 0.0
        train_targets = []
        train_preds = []
        
        for eeg, fmri, labels in train_loader:
            eeg, fmri, labels = eeg.to(CONFIG["device"]), fmri.to(CONFIG["device"]), labels.to(CONFIG["device"])
            optimizer.zero_grad()
            
            # Mixup 逻辑保持不变：
            # Mixup 在时域进行是合法的。FFT 是线性变换，FFT(a*x1 + b*x2) = a*FFT(x1) + b*FFT(x2)。
            # 所以模型内部计算 Mixup 数据的 FFT 依然符合 Mixup 的语义。
            eeg_mix, fmri_mix, y_a, y_b, lam = mixup_data(eeg, fmri, labels, alpha=1.0, device=CONFIG["device"])
            
            outputs = model(eeg_mix, fmri_mix) # 模型内部会自动处理 FFT 和特征拼接
            
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item() * eeg.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            train_targets.extend(labels.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())
            
        avg_train_loss = train_loss_accum / len(train_subset)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0) * 100
            
        # Validation
        model.eval()
        val_loss_accum = 0.0 
        all_targets, all_preds, all_probs = [], [], []
        
        with torch.no_grad():
            for eeg, fmri, labels in val_loader:
                eeg, fmri, labels = eeg.to(CONFIG["device"]), fmri.to(CONFIG["device"]), labels.to(CONFIG["device"])
                outputs = model(eeg, fmri)
                loss = criterion(outputs, labels)
                val_loss_accum += loss.item() * eeg.size(0)
                probs = F.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs.data, 1)
                all_targets.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_val_loss = val_loss_accum / len(val_subset)
        metrics = calculate_metrics(all_targets, all_preds, all_probs)
        
        epoch_history['train_loss'].append(avg_train_loss)
        epoch_history['val_loss'].append(avg_val_loss)
        epoch_history['train_f1'].append(train_f1)
        epoch_history['val_f1'].append(metrics['F1'])
        
        if metrics['F1'] > best_f1:
            best_f1 = metrics['F1']
            best_metrics = metrics
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_outputs = {
                "y_true": np.array(all_targets),
                "y_pred": np.array(all_preds),
                "y_probs": np.array(all_probs)
            }
            
    if best_metrics is None:
        best_metrics = metrics
        best_model_state_dict = copy.deepcopy(model.state_dict())
        best_outputs = {
            "y_true": np.array(all_targets),
            "y_pred": np.array(all_preds),
            "y_probs": np.array(all_probs)
        }
        
    print(f"  [Run {run_id+1} | Fold {fold_id+1}] Best F1: {best_metrics['F1']:.2f}% | AUC-PR: {best_metrics['AUC-PR']:.2f}%")
    
    # ==========================================
    # RNG 状态保护区 (保持不变)
    # ==========================================
    rng_state_torch = torch.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rng_state_np = np.random.get_state()
    
    try:
        if best_outputs is not None:
            pred_filename = f"predictions_run{run_id+1}_fold{fold_id+1}.npz"
            np.savez(os.path.join(CONFIG["save_dir"], pred_filename), 
                     y_true=best_outputs["y_true"], 
                     y_pred=best_outputs["y_pred"], 
                     y_probs=best_outputs["y_probs"])

        history_filename = f"history_run{run_id+1}_fold{fold_id+1}.npz"
        np.savez(os.path.join(CONFIG["save_dir"], history_filename),
                 train_loss=epoch_history['train_loss'],
                 val_loss=epoch_history['val_loss'],
                 train_f1=epoch_history['train_f1'],
                 val_f1=epoch_history['val_f1'])

        # t-SNE 特征提取
        # 虽然模型输出变了，但 linear_layers[-1] 依然指向分类器的输出层
        # 我们这里依然提取倒数第二层的特征用于可视化
        model.load_state_dict(best_model_state_dict)
        model.eval()
        
        features_list = []
        def get_features_hook(module, input, output):
            # input[0] 是进入最后一层全连接层的数据，即融合了时频特征的向量 (Size: d_model * 4)
            features_list.append(input[0].cpu().detach())
            
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if len(linear_layers) > 0:
            handle = linear_layers[-1].register_forward_hook(get_features_hook)
            tsne_inputs_raw = []
            tsne_labels = []
            
            with torch.no_grad():
                for eeg, fmri, labels in val_loader:
                    eeg, fmri, labels = eeg.to(CONFIG["device"]), fmri.to(CONFIG["device"]), labels.to(CONFIG["device"])
                    _ = model(eeg, fmri)
                    
                    batch_size = eeg.shape[0]
                    # 依然保存时域的原始数据用于对比
                    raw_data = torch.cat([eeg.view(batch_size, -1), fmri.view(batch_size, -1)], dim=1)
                    tsne_inputs_raw.append(raw_data.cpu().numpy())
                    tsne_labels.extend(labels.cpu().numpy())
            
            handle.remove()
            features_extracted = torch.cat(features_list, dim=0).numpy()
            raw_inputs_extracted = np.concatenate(tsne_inputs_raw, axis=0)
            tsne_labels = np.array(tsne_labels)
            
            tsne_filename = f"tsne_data_run{run_id+1}_fold{fold_id+1}.npz"
            np.savez(os.path.join(CONFIG["save_dir"], tsne_filename),
                     features=features_extracted,
                     raw_data=raw_inputs_extracted,
                     labels=tsne_labels)
                     
    except Exception as e:
        print(f"⚠️ Warning: Data saving failed for Run {run_id+1} Fold {fold_id+1}: {e}")

    torch.set_rng_state(rng_state_torch)
    if rng_state_cuda is not None:
        torch.cuda.set_rng_state_all(rng_state_cuda)
    np.random.set_state(rng_state_np)

    return best_metrics

# ==========================================
# 主程序 (保持不变)
# ==========================================
def main():
    if not os.path.exists(CONFIG["save_dir"]): os.makedirs(CONFIG["save_dir"])
    start_time = time.time()
    
    print("Loading Full Dataset...")
    full_dataset = BrainDataset(data_dir=CONFIG["data_dir"], mode='finetune', target_pair=CONFIG['target_pair'])
    
    X_dummy = np.zeros(len(full_dataset)) 
    y_labels = np.array(full_dataset.labels)
    
    all_results = [] 

    print(f"\n🚀 Starting {CONFIG['num_runs']} Runs x {CONFIG['k_folds']} Folds Cross-Validation...")
    
    for run in range(CONFIG["num_runs"]):
        run_seed = 42 + run 
        set_seed(run_seed)
        
        print(f"\n{'='*40}")
        print(f"Run {run+1}/{CONFIG['num_runs']} (Seed={run_seed})")
        print(f"{'='*40}")
        
        skf = StratifiedKFold(n_splits=CONFIG["k_folds"], shuffle=True, random_state=run_seed)
        fold_results_buffer = [] 
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_dummy, y_labels)):
            metrics = train_one_fold(train_idx, val_idx, full_dataset, run, fold)
            
            record = {
                "Run": run + 1,
                "Fold": fold + 1,
                "Acc": metrics["Acc"],
                "F1": metrics["F1"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "AUC-PR": metrics["AUC-PR"]
            }
            all_results.append(record)
            fold_results_buffer.append(record)
        
        run_df = pd.DataFrame(fold_results_buffer)
        print(f"--> Run {run+1} Average F1: {run_df['F1'].mean():.2f}% ± {run_df['F1'].std():.2f}")

    final_df = pd.DataFrame(all_results)
    csv_path = os.path.join(CONFIG["save_dir"], f"full_results_10x5_{CONFIG['target_pair']}.csv")
    final_df.to_csv(csv_path, index=False)
    
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS (Average of 10 Runs)")
    print("="*60)
    
    run_averages = final_df.groupby("Run").mean()
    
    metrics_list = ["Acc", "F1", "Precision", "Recall", "AUC-PR"]
    print(f"{'Metric':<10} | {'Mean':<10} | {'Std Dev':<10} | {'Format for Paper'}")
    print("-" * 60)
    
    summary_text = []
    
    for m in metrics_list:
        mean_val = run_averages[m].mean()
        std_val = run_averages[m].std()
        paper_format = f"{mean_val:.2f} ± {std_val:.2f}"
        print(f"{m:<10} | {mean_val:<10.2f} | {std_val:<10.2f} | {paper_format}")
        summary_text.append(f"{m}: {paper_format}")

    with open(os.path.join(CONFIG["save_dir"], f"summary_report{CONFIG['target_pair']}.txt"), "w") as f:
        f.write("\n".join(summary_text))
        
    print("="*60)
    print(f"Total time: {(time.time()-start_time)/60:.1f} mins")
    print(f"Detailed CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()