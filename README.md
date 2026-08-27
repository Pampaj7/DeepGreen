# DeepGreen AI :seedling:

This repository contains the replication package for the paper:

> **Deep Green AI: Energy Efficiency of Deep Learning across Language-Framework Ecosystems**  
> Leonardo Pampaloni, Marco Pagliocca, Enrico Vicario, Roberto Verdecchia  
> University of Florence, Italy

📄 [Preprint PDF](./DeepGreenAI.pdf)

> [!IMPORTANT]
> **Revision in progress (JSS, August 2026).** The first submission was rejected;
> the reviews identified real defects in this package and the re-analysis found
> more. See **[REVIEWERS_RESPONSE.md](./REVIEWERS_RESPONSE.md)** for the
> point-by-point response and **[results/analysis/](./results/analysis)** for the
> corrected pipeline. In short:
>
> * energy was reported in **kWh under a Joule label** (a factor of 3.6 × 10⁶);
> * the campaign has **2,880** tracked measurement blocks, not 7,200, from
>   **one** run per configuration — the 30 epochs of a run are not independent;
> * **the instrument was not the same across ecosystems** (two CodeCarbon major
>   versions, two tracking modes, two sampling intervals), and for four stacks
>   roughly two thirds of the reported energy is a constant host power times
>   duration;
> * the four **LibTorch** stacks alone span 98–100% of the reported spread, so
>   the effect is host-side, not language-intrinsic;
> * under a common measurement boundary the energy and time rankings agree
>   almost perfectly, which does **not** support "faster is not greener";
> * **the eight stacks did not run the same experiment** — two learning rates,
>   four data-loader settings, one stack normalising its inputs, and Rust
>   training Fashion-MNIST at 28×28×1 and evaluating one image at a time.
>
> The seven in-scope ecosystems are now held to a single written specification
> (`results/analysis/experiment_spec.md`), enforced by
> `scripts/check_consistency.py` (47 checks) and by runtime smoke tests for the
> shared model, the measurement harness, and every non-Python build.
>
> Nothing in `results/scripts/`, `results/plots/` or `results/tables/` should be
> used; those files are kept for provenance and are marked deprecated.
>
> The revised manuscript is in **[paper/](./paper)**. All seven in-scope
> ecosystems now run on the GPU under one shared measurement contract
> (`tools/deepgreen_tracker.py`) and record per-epoch accuracy; conformance is
> enforced by `scripts/check_consistency.py` (56 checks).

---

## :pushpin: Overview
DeepGreen AI is an empirical study investigating how **programming languages and frameworks** influence the **energy efficiency of deep learning (DL) workloads**.  
We benchmarked eight **language-framework ecosystems** (Python/PyTorch, Python/TensorFlow, Python/JAX, C++/LibTorch, Java/DL4J, R/torch, Rust/tch, MATLAB/DLT) over six languages. The unit of comparison is the ecosystem, not the language: the stacks delegate computation to compiled backends, and four of the eight share LibTorch.

We benchmarked two canonical CNN architectures ([**ResNet-18**](https://arxiv.org/abs/1512.03385 "ResNet18 paper: Deep Residual Learning for Image Recognition, He et al") and [**VGG-16**](https://arxiv.org/abs/1409.1556 "VGG16 paper: Very Deep Convolutional Networks for Large-Scale Image Recognition, Simonyan et al")) across **six programming languages** (Python, C++, Java, R, MATLAB, Rust), multiple frameworks (PyTorch, TensorFlow, JAX, LibTorch, tch, R torch, Deeplearning4j, MATLAB Deep Learning Toolbox), and three datasets of increasing complexity ([**Fashion-MNIST**](https://github.com/zalandoresearch/fashion-mnist "Fashion MNIST Github repository"), [**CIFAR-100**](https://www.cs.toronto.edu/~kriz/cifar.html "CIFAR-100 official webpage"), [**Tiny ImageNet**](https://github.com/rmccorm4/Tiny-Imagenet-200 "Tiny ImageNet Github repository")).

Experiments were executed on a dedicated **NVIDIA L40S GPU server**, with energy usage measured via the [**CodeCarbon**](https://codecarbon.io/ "CodeCarbon official webpage") toolkit.

---

## :microscope: Research Questions
- **RQ1.1:** How does programming language choice affect the energy efficiency of DL *training*?  
- **RQ1.2:** How does programming language choice affect the energy efficiency of DL *inference*?  

---

## :flashlight: Highlights (revised)
> The highlights below replace those of the submitted version. The originals
> are superseded by the audit in [`results/analysis/`](./results/analysis).

- **Rust/tch** has the lowest training energy and **C++/LibTorch** the lowest inference energy, on every energy definition tested.
- **The effect is host-side, not language-intrinsic.** The four stacks that share the LibTorch backend span 98–100% of the total spread.
- **The GPU is lightly loaded** (24–56% of the L40S board limit), and 30–69% of the measured energy is host-side, so the study measures data loading and dispatch more than deep-learning computation.
- **Execution time is a good proxy for energy here.** Within an ecosystem the median R² of energy on duration is 0.98; across ecosystems, under a common measurement boundary, the energy and time rankings agree perfectly for inference and differ by one pair for training. The submitted "faster ≠ greener" claim does not survive the correction.
- **Rankings are not robust to the measurement boundary**: best and worst change in three of the four phase × definition combinations.
- **No claim of statistical significance between ecosystems is supported** by the current data: each configuration was run once.

---

## :open_file_folder: Repository Structure
```text
REVIEWERS_RESPONSE.md   # Point-by-point response to the JSS reviews
tools/                  # Shared measurement harness (pinned CodeCarbon config)
scripts/run_campaign.py # Campaign driver with independent run-level repetitions
results/analysis/       # Corrected analysis pipeline (run_all.sh)
results/revision/       # Corrected tables and figures
Java/deepgreen-dl4j/    # Java implementations (Deeplearning4j)
cpp/                    # C++ implementations (LibTorch)
matlab/                 # MATLAB (Deep Learning Toolbox)
python/                 # Python (PyTorch, TensorFlow, JAX)
R/                      # R (torch + torchvision, i.e. LibTorch bindings)
rust/                   # Rust (tch, i.e. LibTorch bindings)
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
6. The corrected re-analysis pipeline (`results/analysis/`) and the measurement
   protocol for the replicated campaign (`results/analysis/repetition_protocol.md`).

### Reproducing the corrected analysis
```bash
pip install pandas numpy scipy matplotlib seaborn tabulate
./results/analysis/run_all.sh          # -> results/revision/{tables,figures}
```

### Running the replicated campaign
```bash
python3 scripts/run_campaign.py --repetitions 5 --print-plan
python3 scripts/run_campaign.py --repetitions 5
python3 results/analysis/09_campaign_v2.py
```
Requires CodeCarbon >= 3.0; the harness refuses to start under 2.x.

---

## :scroll: License
This project is released under the [MIT License](./LICENSE).
