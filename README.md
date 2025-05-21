# 📚 Project: Predicting brain activity from arousal measurements

## 👤 Overview

This project accompanies the Bachelor's thesis:

> **Federico Sbarbati**  
> _Predicting brain activity from arousal measurements_  
> University of Padua, 2025  
> [Thesis link](https://thesis.unipd.it/handle/20.500.12608/84641)

The work investigates how variational autoencoders (VAEs) can be used to extract meaningful low-dimensional representations from fMRI and eye-tracking time series. The pipeline integrates signal processing, delay-coordinate embeddings, PCA, and deep learning methods for unsupervised modeling of latent neural dynamics.

---

## 🧠 Features

### 1. **Data Processing and Compression**
- Signal filtering using Butterworth filters.
- Z-score normalization and dimensionality reduction via PCA.
- Scripts for compressing high-dimensional time series into latent representations.

### 2. **Model Training and Evaluation**
- Encoder-decoder architectures implemented in PyTorch.
- Support for different latent dimensionalities and β-VAE annealing.
- Model evaluation using R², MSE, and MaxSE metrics.

### 3. **Interactive Notebooks**
- Modular notebooks for preprocessing, training, validating and visualizing results.
- Reproducible analysis and results export.

---

## 📁 Repository Structure

```plaintext
├── Notebook/              # Jupyter notebooks for preprocessing, training and analysis
├── Compressed Data/       # Latent compressed representations from compress_data.py
├── Uncompressed Data/     # Folder to place the raw data from Zenodo
├── Def Models/            # Trained neural network weights and configurations
├── compress_data.py       # Script to compress unprocessed time series
└── README.md              # Project documentation

```

To use the project, first download the original dataset from:  
[https://zenodo.org/records/4670277](https://zenodo.org/records/4670277)  
and place the files inside the `Uncompressed Data/` folder.

## Usage Instructions

### 1. Preprocess and Compress Data
- Run `compress_data.py` to:
  - Reduce data size and dimensionality
  - Filter and normalize time series

### 2. Train VAE Models
- Open `fmri-pc-recon.ipynb` in `Notebook/` to:
  - Load compressed data.
  - Train a VAE with desired configuration.
  - Save the model weights in `Def Models/`.

### 3. Evaluate and Visualize
- Use the validation notebook `fmri-pc-recon-analysis.ipynb` to:
  - Evaluate reconstruction performance.
  - Plot PCA variance, latent trajectories, and time-series reconstructions.

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- nibabel
- PyTorch

You can install all dependencies with:

```bash
pip install numpy scipy matplotlib scikit-learn nibabel torch

```

## Citation

If you use this code or ideas from this project, please cite the original thesis:

```bibtex
@misc{sbarbati2025vae,
  author = {Federico Sbarbati},
  title = {Predicting brain activity from arousal measurements},
  year = {2025},
  note = {Bachelor's Thesis, University of Padua},
  url = {https://thesis.unipd.it/handle/20.500.12608/84641}
}
```
## License
This repository is released under the MIT License.

### Contact
For questions or feedback, feel free to open an issue on GitHub or contact the repository maintainer.

