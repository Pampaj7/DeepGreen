# DeepGreen AI :seedling:

This repository contains the replication package for the paper:

> **Deep Green AI: Energy Efficiency of Deep Learning across Programming Languages and Frameworks**  
> Leonardo Pampaloni, Marco Pagliocca, Enrico Vicario, Roberto Verdecchia  
> University of Florence, Italy

📄 [Preprint PDF](./DeepGreenAI.pdf)

---

## :pushpin: Overview
DeepGreen AI is an empirical study investigating how **programming languages and frameworks** influence the **energy efficiency of deep learning (DL) workloads**.  
We benchmarked two canonical CNN architectures ([**ResNet-18**](https://arxiv.org/abs/1512.03385 "ResNet18 paper: Deep Residual Learning for Image Recognition, He et al") and [**VGG-16**](https://arxiv.org/abs/1409.1556 "VGG16 paper: Very Deep Convolutional Networks for Large-Scale Image Recognition, Simonyan et al")) across **six programming languages** (Python, C++, Java, R, MATLAB, Rust), multiple frameworks (PyTorch, TensorFlow, JAX, LibTorch, Burn, Deeplearning4j, etc.), and three datasets of increasing complexity ([**Fashion-MNIST**](https://github.com/zalandoresearch/fashion-mnist "Fashion MNIST Github repository"), [**CIFAR-100**](https://www.cs.toronto.edu/~kriz/cifar.html "CIFAR-100 official webpage"), [**Tiny ImageNet**](https://github.com/rmccorm4/Tiny-Imagenet-200 "Tiny ImageNet Github repository")).

Experiments were executed on a dedicated **NVIDIA L40S GPU server**, with energy usage measured via the [**CodeCarbon**](https://codecarbon.io/ "CodeCarbon official webpage") toolkit.

---

## :microscope: Research Questions
- **RQ1.1:** How does programming language choice affect the energy efficiency of DL *training*?  
- **RQ1.2:** How does programming language choice affect the energy efficiency of DL *inference*?  

---

## :flashlight: Highlights
- **Machine-code compiled languages (Rust, C++)** are consistently more energy-efficient during training.  
- **Mature Python frameworks (PyTorch, JAX)** achieve competitive efficiency despite interpretation overhead.  
- **High-level languages (TensorFlow, Java, R)** incur substantial overheads if they are unable to exploit the available hardware resources.
- **Inference vs training efficiency diverge**: C++ and PyTorch dominate inference, Rust dominates training.  
- **Faster $\neq$ Greener**: execution time is not a reliable proxy for energy usage.  

---

## :open_file_folder: Repository Structure
```text
Java/deepgreen-dl4j/    # Java implementations (Deeplearning4j)
cpp/                    # C++ implementations (LibTorch)
julia/                  # Julia implementations (Flux, Lux)
matlab/                 # MATLAB scripts (TF/Keras wrappers)
python/                 # Python (PyTorch, TensorFlow, JAX)
R/                      # R (TensorFlow wrapper)
rust/                   # Rust (Burn)
dataloader/             # Unified data loading utilities
data/                   # Dataset links and preprocessing scripts
results/                # Experimental results (CSV, logs, figures)
README.md               # This file
```

---

## :gear: Setup

### 1. Clone the repository
```bash
git clone https://github.com/Pampaj7/DeepGreen.git
cd DeepGreen
```

### 2. Python environment
```bash
conda env create -f environment.yml
conda activate deepgreen
```

### 3. Datasets
Download datasets (Fashion-MNIST, CIFAR-100, Tiny ImageNet) using the Python scripts provided in [`dataloader`](./dataloader "Dataloader folder").

---

## :bar_chart: Replication Package
This replication package includes:
1. Source code for all implementations (Python, C++, Java, R, MATLAB, Rust).  
2. Scripts for automated training and inference runs.  
3. Environment specifications for each ecosystem.  
4. Raw energy logs and aggregated CSV data.  
5. Plotting scripts to reproduce all figures and tables from the paper.  

---

## :open_book: Citation
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

## :scroll: License
This project is released under the [MIT License](./LICENSE).
