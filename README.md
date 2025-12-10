# TokenAlign: Rethinking Text-Audio Retrieval with Fine-grained Token-level Correspondence
## ARIN-5203 Project
### BAI Yu, CHEN Chuxi

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/yourusername/TokenAlign)

This repository contains the official implementation of **TokenAlign**, a token-level, temporally aware interaction framework for text-audio retrieval. TokenAlign addresses the limitation of global pooling in existing methods by preserving all audio segments and text tokens end-to-end, enabling fine-grained alignment with temporal awareness.

## 🎯 Overview

Text-Audio Retrieval (TAR) datasets provide only clip-level pairs, which pushes most systems to pool audio frames and text tokens into global vectors and ignore order. This weak supervision leaves models insensitive to phrases like "A then B" and unable to localize which audio segment supports which phrase.

**TokenAlign** introduces:
- **Token-level alignment** with learnable Gaussian relative positional bias for temporal consistency
- **Hierarchical alignment framework** with three granularities: Local (Token-Level), Regional (Event-Level), and Global (Clip-Level)
- **Max-Sum token similarity** with online hard negative mining and hybrid objective

## ✨ Key Features

- 🔹 **Temporal-Aware Cross-Modal Interaction**: Learnable Gaussian relative positional bias enforces temporal consistency without timestamp supervision
- 🔹 **Hierarchical Alignment**: Three-level alignment framework (Local, Regional, Global) for comprehensive audio-text correspondence
- 🔹 **Max-Sum Token Similarity**: Fine-grained similarity computation that focuses on salient alignments
- 🔹 **Hybrid Training Objective**: Combines hardest-triplet separation and intra-modal topology consistency
- 🔹 **Online Hard Negative Mining**: Dynamically identifies hardest negatives for improved discriminative power

## 📊 Results

### Performance on AudioCaps
| Method | T2A R@10 | A2T R@10 | mAP@10 |
|--------|----------|----------|--------|
| CLAP* | 83.4 | 84.0 | 50.2 |
| T-CLAP | 84.9 | 85.1 | 51.9 |
| **TokenAlign (Ours)** | **86.60** | **87.23** | **53.41** |

### Performance on Clotho V2.1
| Method | T2A R@10 | A2T R@10 | mAP@10 |
|--------|----------|----------|--------|
| CLAP* | 52.1 | 55.1 | 27.2 |
| **TokenAlign (Ours)** | **58.32** | **61.91** | **32.17** |

*Improvements: +6.2% T2A R@10 and +6.8% A2T R@10 over CLAP on Clotho; +3.8% on AudioCaps*

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/TokenAlign.git
cd TokenAlign
```

2. **Set up environment** (see [Environment Setup](#-environment-setup) section below for details)

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Data Preparation

The code automatically downloads datasets when first used. Supported datasets:
- **Clotho V2.1**: Audio captioning dataset (15-30 seconds clips)
- **AudioCaps**: Large-scale dataset (~50k clips, 10 seconds each)
- **WavCaps**: Large-scale weakly labeled dataset (optional)

### Training

**Standard TokenAlign configuration** (as in paper):
```bash
python -m ATR.train \
    --enable_token_align \
    --enable_hierarchical_alignment \
    --enable_token_align_loss \
    --token_align_sigma_initial 1.0 \
    --token_align_sigma_trainable \
    --n_events 4 \
    --sinkhorn_reg 0.1 \
    --local_weight 1.0 \
    --regional_weight 0.5 \
    --global_weight 0.3 \
    --local_tau 1.0 \
    --token_align_margin 0.2 \
    --token_align_lambda 0.1 \
    --token_align_sigma_margin 0.0 \
    --batch_size 32 \
    --max_epochs 20 \
    --encoder_lr 5e-5 \
    --projection_lr 1e-4 \
    --clotho \
    --exp_name "tokenalign_clotho"
```

**Training on AudioCaps**:
```bash
python -m ATR.train \
    --enable_token_align \
    --enable_hierarchical_alignment \
    --enable_token_align_loss \
    --batch_size 32 \
    --max_epochs 20 \
    --encoder_lr 5e-5 \
    --projection_lr 1e-4 \
    --audiocaps \
    --exp_name "tokenalign_audiocaps"
```

## 🏗️ Architecture

TokenAlign consists of three main components:

1. **Unimodal Feature Encoding**
   - Audio: PaSST encoder with 10-second segment-level encoding
   - Text: RoBERTa-large with full token sequence preservation

2. **Temporal-Aware Cross-Modal Interaction**
   - TokenAlign module with learnable Gaussian relative positional bias
   - Cross-attention: Q=text, K=V=audio

3. **Hierarchical Alignment Framework**
   - **Local Alignment**: Max-Sum token similarity
   - **Regional Alignment**: Optimal Transport for event-phrase alignment
   - **Global Alignment**: Clip-level semantic consistency

4. **Training Objectives**
   - Online hard negative mining
   - Hardest-triplet loss
   - Intra-modal consistency loss

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tokenalign2026,
  title={TokenAlign: Rethinking Text-Audio Retrieval with Fine-grained Token-level Correspondence},
  author={Bai, Yu and Chen, Chuxi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
