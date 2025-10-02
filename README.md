# DeepGreen AI 🌱

This repository contains the replication package for the paper:

> **Deep Green AI: Energy Efficiency of Deep Learning across Programming Languages and Frameworks**  
> Leonardo Pampaloni, Marco Pagliocca, Enrico Vicario, Roberto Verdecchia  
> University of Florence, Italy

📄 [Preprint PDF](./DeepGreenAI.pdf)

---

## 📌 Overview
DeepGreen AI is an empirical study investigating how **programming languages and frameworks** influence the **energy efficiency of deep learning (DL) workloads**.  
We benchmarked two canonical CNN architectures (**ResNet-18** and **VGG-16**) across **six programming languages** (Python, C++, Java, R, MATLAB, Rust), multiple frameworks (PyTorch, TensorFlow, JAX, LibTorch, Burn, Deeplearning4j, etc.), and three datasets of increasing complexity (**Fashion-MNIST, CIFAR-100, Tiny ImageNet**).

Experiments were executed on a dedicated **NVIDIA L40S GPU server**, with energy usage measured via the **CodeCarbon toolkit**.

---

## 🔬 Research Questions
- **RQ1.1:** How does programming language choice affect the energy efficiency of DL training?  
- **RQ1.2:** How does programming language choice affect the energy efficiency of DL inference?  

---

## 📂 Repository Structure
```text
cpp/            # C++ implementations (LibTorch)
java/           # Java implementations (Deeplearning4j)
julia/          # Julia implementations (Flux, Lux)
matlab/         # MATLAB scripts (TF/Keras wrappers)
python/         # Python (PyTorch, TensorFlow, JAX)
r/              # R (TensorFlow wrapper)
rust/           # Rust (Burn)
dataloader/     # Unified data loading utilities
data/           # Dataset links and preprocessing scripts
results/        # Experimental results (CSV, logs, figures)
requirements.txt
README.md       # This file
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/DeepGreen.git
cd DeepGreen
```

### 2. Python environment
```bash
conda env create -f environment.yml
conda activate deepgreen
```

### 3. Datasets
Download datasets (Fashion-MNIST, CIFAR-100, Tiny ImageNet) following instructions in [`dataloader`].  

---

## 📊 Replication Package
This replication package includes:
1. Source code for all implementations (Python, C++, Java, R, MATLAB, Rust).  
2. Scripts for automated training and inference runs.  
3. Environment specifications for each ecosystem.  
4. Raw energy logs and aggregated CSV data.  
5. Plotting scripts to reproduce all figures and tables from the paper.  

---

## 📈 Key Findings
- **Compiled languages (Rust, C++)** are consistently more energy-efficient during training.  
- **Python (PyTorch, JAX)** achieves competitive efficiency despite interpretation overhead.  
- **Wrapper-based languages (MATLAB, R, Java)** incur substantial overheads.  
- **Inference vs training efficiency diverge**: C++ and PyTorch dominate inference, Rust dominates training.  
- **Faster ≠ Greener**: execution time is not a reliable proxy for energy usage.  

---

## 📖 Citation
If you use this package, please cite:

```bibtex
@article{pampaloni2025deepgreen,
  title   = {Deep Green AI: Energy Efficiency of Deep Learning across Programming Languages and Frameworks},
  author  = {Pampaloni, Leonardo and Pagliocca, Marco and Vicario, Enrico and Verdecchia, Roberto},
  journal = {Preprint},
  year    = {2025},
  doi     = {10.5281/zenodo.xxxxxxx}
}
```

---

## 📜 License
This project is released under the [MIT License](./LICENSE).
