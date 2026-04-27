# BiSTS-Mamba: Bidirectional Spatial-Temporal-Spectral Mamba for Multimodal Brain Decoding

Official implementation of **BiSTS-Mamba**, a lightweight yet powerful deep learning framework designed for the fusion of multimodal neuroimaging data (EEG and fMRI). By leveraging the linear complexity of State Space Models (Mamba) and a dual-domain causal modeling approach, BiSTS-Mamba achieves state-of-the-art performance in various brain disease diagnostic tasks. More details, including pre-trained weights and comprehensive experimental logs, will be released upon the formal acceptance of our manuscript.

## 🌟 Data Format

Data is stored in h5 files with the following keys:

* `fMRI-DK68-node-mat`: MRI node data, dimensions (68, 370)
* `fMRI-DK68-edge-mat`: MRI edge data, dimensions (68, 68)
* `sLORETA_mean_func`: EEG node data, dimensions (68, 250)
* `sLORETA_mean_CorrMatrix`: EEG edge data, dimensions (68, 68)
* `label`: Labels, values (0,1,2,3,4,5) // 0 represents the control group, while 1 (depression), 2 (anxiety disorder), 3 (neurodevelopmental disorder (specific learning disorder with dyslexia)), 4 (autism), and 5 (attention deficit hyperactivity disorder) represent the experimental groups.

All data is stored in , where represents data with labels 0 and 1, represents data with labels 0 and 2, and so on.`New-EEG+fMRI_h5``01``02`

## 📁 Repository Structure

* `model_new.py`: Core architecture implementation, including the `OAFMamba` (Order-Aware Fusion) engine, RoPE (Rotary Positional Embedding), and RMSNorm.
* `dataset.py`: Data pipeline for `.h5` neuroimaging files, including Z-Score normalization and Mixup augmentation.
* `pretrain_new.py`: Script for self-supervised pre-training using contrastive learning.
* `finetune_new.py`: Downstream task adaptation script with 10-run 5-fold cross-validation support.
