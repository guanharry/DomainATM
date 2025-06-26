# DomainATM: Domain Adaptation Toolbox for Medical Data Analysis

**License:** MIT  
**Language:** MATLAB  

More details can be found in the file "**manual.pdf**". Note, for Mac, please use the DomainATM_Mac.zip

---

## Overview

**DomainATM** is a MATLAB-based toolbox that provides an intuitive GUI and ready-to-use implementations for **feature-level** and **image-level domain adaptation** methods, specifically tailored for medical data analysis. The toolbox supports both real-world and synthetic datasets and enables easy evaluation, visualization, and extension with user-defined algorithms.

---

## Installation

### Requirements

- MATLAB (2020 or later versions are recommended)
- Image Processing Toolbox
- Statistics & Machine Learning Toolbox

### Installation Steps

1. Download the `DomainATM.mlappinstall` file.
2. On **Windows/macOS**: Double-click the `.mlappinstall` file.  
   On **Linux**: Open MATLAB → `Apps` tab → Click `Install App`.
3. Set the **MATLAB current directory** to the folder containing the toolbox.  
   Example: `E:/DomainATM`

---

## Folder Structure

- `data/`: Stores real or synthetic datasets for fast verification.
- `algorithms_feat/`: Feature-level adaptation methods.
- `algorithms_img/`: Image-level adaptation methods.
- `evaluation/`: Stores the output of each experiment.
- `tools/`: Utility functions used by the GUI.

---

## Feature-Level Domain Adaptation

### Supported Methods

- Subspace Alignment (SA)
- Correlation Alignment (CORAL)
- Transfer Component Analysis (TCA)
- Optimal Transport (OT)
- Joint Distribution Adaptation (JDA)
- Transfer Joint Matching (TJM)
- Geodesic Flow Kernel (GFK)
- Scatter Component Analysis (SCA)
- Information-Theoretic Learning (ITL)

### Usage

1. Click `Create Dataset` to generate synthetic data or place your `.mat` dataset in `data/`.
2. Click `Load Data` and select your dataset.
3. Click `Feature-Level Adaptation` and choose an algorithm to apply.
4. Click `Feature-Level Metrics` to evaluate the results (accuracy, visualization, etc.).

---

## Image-Level Domain Adaptation

### Supported Methods

- Histogram Matching (HM)
- Spectrum Swapping-based Image-Level Harmonization (SSIMH)

### Usage

1. Click `Image-Level Adaptation`.
2. Select source and target `.nii` images.
3. Choose an algorithm and click `Run`.
4. Use `Image-Level Metrics` to evaluate harmonization quality.


---

## Custom Algorithms

To add your own domain adaptation method:

- For **feature-level**: Place your `.m` script in `algorithms_feat/` using:
```
  X_adapted = FeatureDA(X, domain_label, Y, param);
```
- For **image-level**: Place your `.m` script in `algorithms_img/` using:
```
  S_adapted = ImageDA(source, target, param);
```

Use "Add Your Algorithm" in the GUI and click "Refresh".

---
## Citation

If you find this toolbox useful in your research, please cite:

```bibtex
@article{guan2023domainatm,
  title={DomainATM: Domain adaptation toolbox for medical data analysis},
  author={Guan, Hao and Liu, Mingxia},
  journal={NeuroImage},
  volume={268},
  pages={119863},
  year={2023},
  publisher={Elsevier}
}
```

