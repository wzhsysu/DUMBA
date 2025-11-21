# Deep Opinion-Unaware Blind Image Quality Assessment Learning and Adapting from Multiple Annotators

**Zhihua Wang\***, **Xuelin Liu\***, Jiebin Yan, Jie Wen, Wei Wang, Chao Huang  
*\* Co-first authors*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-%F0%9F%A7%AA-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Stars](https://img.shields.io/github/stars/USERNAME/REPO_NAME?style=social)]()

---

## 🌑 Overview

DUBMA is an **opinion-unaware** blind IQA framework that:

⚫ Learns image quality **without any human MOS labels**  
⚫ Uses **synthetic distorted image pairs** and **pseudo rankings** from multiple FR-IQA models  
⚫ Learns **annotator reliability weights** via maximum likelihood estimation  
⚫ Bridges synthetic → real domain shift using **Unbalanced Joint Optimal Transport (OT)**  
⚫ Achieves strong results on **LIVE / CSIQ / KADID-10k** and **KonIQ-10k / SPAQ**

---

## 🔥 Features

⚫ Fully opinion-unaware BIQA (no subjective ratings needed)  
⚫ Multi-annotator reliability modeling  
⚫ Learn-to-Rank pseudo supervision  
⚫ Robust UDA via unbalanced OT  
⚫ ResNet-18 backbone  
⚫ Strong generalization on authentic distortions  

---
## 🛰 Framework Diagram

<p align="center">
  <img src="pipeline.png" alt="DUBMA Framework" width="78%">
  <br>
  <em>Figure: Framework of DUBMA. The first part illustrates the process of learning from synthetically-distorted images and evaluating their performance, while the entire framework encompasses learning from simulated distortions and adapting to the quality evaluation of authentically distorted images.</em>
</p>

## ⚙️ Installation

```bash
git clone https://github.com/USERNAME/DUBMA.git
cd DUBMA

conda create -n dubma python=3.9 -y
conda activate dubma

pip install -r requirements.txt
```
---

## 🗂 Dataset Setup

### **1. Distortion Simulation**

- Based on **[Waterloo Exploration Database](https://kedema.org/project/exploration/index.html)** 
- [25 distortion types × 5 levels](https://dmail-my.sharepoint.com/:u:/g/personal/hlin001_dundee_ac_uk/EbKjpdH7-U9DmY2C7eKCWMUBiAW2M9bg3bvy-hKPAstZkg?e=bEbF2w)
- ~90,000 image pairs
- Pseudo-annotated by **10 FR-IQA models** ( 
[SSIM](https://ece.uwaterloo.ca/~z70wang/research/ssim/), [MS-SSIM](https://ece.uwaterloo.ca/~z70wang/research/ssim/msssim.zip), [FSIM](https://web.comp.polyu.edu.hk/cslzhang/IQA/FSIM/FSIM.htm), [VIF](https://github.com/abhinaukumar/vif), [VSI](https://github.com/ideal-iqa/iqa-eval/tree/main/IQA%20metrics%20matlab/VSI), [SR-SIM](https://ieeexplore.ieee.org/document/6467149), [GMSD](https://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm), [MDSI](https://github.com/ideal-iqa/iqa-eval/tree/main/IQA%20metrics%20matlab/MDSI), [NLPD](https://www.cns.nyu.edu/~lcv/NLPyr/), [A-DISTS](https://github.com/dingkeyan93/A-DISTS))

### **2. Synthetic IQA Datasets for Testing**
- **[LIVE](https://live.ece.utexas.edu/research/quality/subjective.htm)**
- **[CSIQ](http://vision.eng.shizuoka.ac.jp/mod/page/view.php?id=23)**
- **[KADID](http://database.mmsp-kn.de/kadid-10k-database.html)**
### **3. Authentic IQA Datasets for Testing**
- **[KonIQ-10k](https://database.mmsp-kn.de/koniq-10k-database.html)**
- **[SPAQ](https://github.com/h4nwei/SPAQ)**

### **Directory Structure**

```
data/
  waterloo/
    images/
    pairs/
    annotations/
  koniq/
  spaq/
```

---

## 🚀 Training

### **1. Synthetic-only Training**

```bash
python train_synthetic.py   --config configs/synthetic_resnet18.yaml
```

### **2. Synthetic → Real Domain Adaptation**

```bash
python train_da.py   --config configs/uda_koniq.yaml
```

---

## 📈 Evaluation

```bash
python eval.py   --checkpoint checkpoints/dubma_koniq.pth   --dataset koniq
```

---

## 📊 Results

### **1. Synthetic Distortions (LIVE / CSIQ / KADID-10k)**

| Method      | SRCC LIVE | SRCC CSIQ | SRCC KADID | PLCC LIVE | PLCC CSIQ | PLCC KADID |
|-------------|-----------|-----------|------------|-----------|-----------|------------|
| PieAPP      | 0.919     | 0.892     | 0.836      | 0.908     | 0.896     | 0.839      |
| LPIPS       | 0.932     | 0.876     | 0.843      | 0.934     | 0.896     | 0.839      |
| DISTS       | 0.954     | 0.929     | 0.887      | 0.954     | 0.928     | 0.886      |
| QAC         | 0.868     | 0.490     | 0.239      | 0.863     | 0.708     | 0.390      |
| PIQE        | 0.840     | 0.512     | 0.541      | 0.839     | 0.677     | 0.306      |
| LPSI        | 0.818     | 0.522     | 0.148      | 0.826     | 0.718     | 0.443      |
| NIQE        | 0.906     | 0.627     | 0.374      | 0.904     | 0.716     | 0.428      |
| ILNIQE      | 0.898     | 0.815     | 0.531      | 0.903     | 0.854     | 0.573      |
| SISBLIM     | 0.774     | 0.660     | 0.209      | 0.807     | 0.737     | 0.388      |
| SNP-NIQE    | 0.907     | 0.609     | 0.371      | 0.766     | 0.731     | 0.422      |
| NPQI        | 0.912     | 0.634     | 0.391      | 0.904     | 0.805     | 0.400      |
| RankIQA     | 0.897     | 0.808     | 0.569      | 0.891     | 0.832     | 0.569      |
| dipIQ       | **0.938** | 0.527     | 0.304      | **0.935** | 0.779     | 0.402      |
| CaHDC       | 0.928     | 0.778     | 0.540      | 0.918     | 0.827     | 0.574      |
| EONSS       | 0.927     | 0.677     | 0.413      | 0.918     | 0.766     | 0.453      |
| Ma19        | 0.919     | **0.915** | 0.466      | 0.917     | **0.926** | 0.501      |
| Zhu21       | 0.135     | 0.391     | 0.185      | 0.517     | 0.481     | 0.342      |
| ContentSep  | 0.748     | 0.587     | 0.506      | 0.700     | 0.589     | 0.486      |
| MDFS        | **0.936** | 0.777     | 0.598      | **0.933** | 0.827     | **0.625**  |
| **DUBMA (Ours)** | **0.930** | **0.840** | **0.863** | **0.918** | **0.859** | **0.864** |

---

### **2. Authentic Distortions (KonIQ-10k / SPAQ)**

| Method      | SRCC LIVE | SRCC KonIQ | SRCC SPAQ | PLCC LIVE | PLCC KonIQ | PLCC SPAQ |
|-------------|-----------|------------|-----------|-----------|------------|-----------|
| QAC         | 0.868     | 0.092      | 0.340     | 0.863     | 0.244      | 0.371     |
| PIQE        | 0.840     | 0.245      | 0.232     | 0.839     | 0.210      | 0.251     |
| LPSI        | 0.818     | 0.224      | 0.001     | 0.826     | 0.107      | 0.276     |
| NIQE        | 0.906     | 0.530      | 0.703     | 0.904     | 0.538      | 0.712     |
| ILNIQE      | 0.898     | 0.506      | 0.714     | 0.903     | 0.531      | 0.721     |
| SISBLIM     | 0.774     | 0.616      | 0.701     | 0.807     | 0.619      | 0.718     |
| SNP-NIQE    | 0.907     | 0.628      | 0.540     | 0.766     | 0.618      | 0.727     |
| NPQI        | 0.912     | 0.613      | 0.600     | 0.904     | 0.556      | 0.627     |
| RankIQA     | 0.897     | 0.483      | 0.584     | 0.891     | 0.482      | 0.587     |
| dipIQ       | **0.938** | 0.236      | 0.385     | **0.935** | 0.435      | 0.497     |
| CaHDC       | 0.928     | 0.423      | 0.562     | 0.918     | 0.441      | 0.594     |
| EONSS       | 0.927     | 0.191      | 0.348     | 0.918     | 0.206      | 0.376     |
| Ma19        | 0.919     | 0.456      | 0.379     | 0.917     | 0.462      | 0.391     |
| Zhu21       | 0.135     | 0.636      | 0.683     | 0.517     | 0.641      | 0.688     |
| ContentSep  | 0.748     | 0.640      | 0.708     | 0.700     | 0.645      | 0.706     |
| MDFS        | 0.936     | **0.733**  | 0.741     | **0.933** | **0.737**  | **0.754** |
| **DUBMA (Ours)** | **0.928** | **0.703** | **0.834** | **0.924** | **0.740** | **0.841** |

---

## 🧪 Pretrained Models

| Model  | Download |
|-------|----------|
| DUBMA-Synthetic  | [BaiduYunpan] [GoogleDrive] |
| DUBMA-Authentic (adapt to KonIQ)  | [BaiduYunpan] [GoogleDrive]|
| DUBMA-Authentic (adapt to SPAQ)  | [BaiduYunpan] [GoogleDrive] |

---

## 📚 Citation

```bibtex
@inproceedings{wang2025dubma,
  title   = {Deep Opinion-Unaware Blind Image Quality Assessment by Learning and Adapting from Multiple Annotators},
  author  = {Wang, Zhihua and Liu, Xuelin and Yan, Jiebin and Wen, Jie and Wang, Wei and Huang, Chao},
  booktitle = {Proceedings of International Joint Conferences on Artificial Intelligence},
  year    = {2025}
}
```

---

## 📬 Contact

**Zhihua Wang** (<a href="mailto:zhihua.wang@my.cityu.edu.hk">zhihua.wang@my.cityu.edu.hk</a>) and **Xuelin Liu**(<a href="mailto:xuelinliu-bill@foxmail.com">xuelinliu-bill@foxmail.com</a> )



