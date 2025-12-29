
# Self-Supervised Physics-Guided Quantitative Mapping from Conventional MRI

This repository contains the official implementation of the paper: **"Quantitative Mapping from Conventional MRI Using Self-Supervised Physics-Guided Deep Learning: Applications to a Large-Scale, Heterogeneous Clinical Dataset"**.

## 📋 Abstract

insert abstract of arxiv paper


## 🛠️ Installation

This project is managed using **[uv](https://github.com/astral-sh/uv)** for fast and reliable dependency management.

### Prerequisites

  * Python 3.11+
  * [uv](https://github.com/astral-sh/uv) installed

### Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/qmap-physics-learning.git
    cd qmap-physics-learning
    ```

2.  **Initialize environment and install dependencies:**

    ```bash
    # Create a virtual environment and install dependencies from pyproject.toml 
    uv sync
    ```

-----

## 📂 Project Structure

```text
.
├── checkpoints/             # Directory where trained models/weights are saved
├── configs/                 # YAML configuration files for training and testing
│   ├── test_umcu_paper1.yaml
│   └── train_umcu_paper1.yaml
├── example_data/            # Sample data for testing the pipeline 
│   ├── images/              # Folder containing NIfTI images
│   ├── masks/               # Folder containing brain masks (if available)
│   └── test_example.csv     # CSV file defining the dataset and sequence parameters
├── qmap/                    # Main source code package
│   ├── data/
│   │   └── conventional_dataset.py   # Dataset loading logic (TorchIO)
│   ├── models/
│   │   ├── base_model.py
│   │   ├── losses.py                 # Physics-guided loss functions
│   │   ├── networks.py               # Neural network architectures 
│   │   └── qmap_synth_model.py       # Core physics synthesis model and backward function
│   ├── options/
│   │   ├── base_options.py
│   │   ├── test_options.py
│   │   └── train_options.py
│   └── util/
│       ├── lipari.csv
│       ├── navia.csv
│       └── util.py                   # Helper functions
├── .gitignore
├── .python-version
├── pyproject.toml           # Project configuration and dependencies
├── README.md
├── test.py                  # Main entry point for inference/testing
├── train.py                 # Main entry point for training
└── uv.lock                  # UV lockfile for reproducible environments
```

-----

## 🏃 Usage

### 1\. Data Preparation

The model expects preprocessed NIfTI files. Per the paper's methodology:

1.  **Convert** DICOM to NIfTI (e.g., using `dcm2niix`).
2.  **Register** T1w and FLAIR to the T2w space (rigid registration).
3.  **Resample** to $1 \times 1 \text{ mm}^2$ in-plane resolution.
4.  **Bias Correction** N4 bias field correction.

#### CSV Configuration

To run the code, you must provide a CSV file (like `example_data/test_example.csv`) containing the specific sequence parameters for each subject for each contrast. These parameters are required for the Bloch equation signal modeling.

**Required CSV Columns:**

  * **Subject ID / Paths:** `subject_id`, 
  * **Sequence Parameters:**
      * **T1w:** `T1w_TR` (Repetition Time), `T1w_TE` (Echo Time), `T1w_TI` (Inversion Time), `T1w_FA` (Flip Angle).
      * **T2w:** `T2w_TR`, `T2w_TE`, `T2w_FA` (TSE often assumes effective TE; the model applies internal corrections for TSE).
      * **FLAIR:** `FLAIR_TR`, `FLAIR_TE`, `FLAIR_TI`, `FLAIR_FA`.
     
#### File structure

images
masks
csv file

### 2\. Training

> **Note:** We cannot make the original clinical training dataset public due to privacy regulations. However, you can train the model on your own dataset using the command below.

To train the model from scratch using `uv`:

```bash
uv run train.py --config configs/train_umcu_paper1.yaml
```

### 3\. Inference / Testing

We provide a pre-processed example of a healthy volunteer in the `example_data/` folder to verify the installation.

1.  **Download Weights:** Download the pre-trained model weights from the **Releases** page.
2.  **Place Weights:** Move the downloaded model file into the following directory:
    `checkpoints/trained_model_paper/qstar_paper_PD0p1_TV0p01_lq20`
3.  **Run Inference:**

<!-- end list -->

```bash
uv run test.py --config configs/test_umcu_paper1.yaml
```

The results (Quantitative Maps and metrics) will be saved in the `results/test_T1wT2wFLAIR_example/` directory.

**Example Output:**


![Example output slice](checkpoints/trained_model_paper/qstar_paper_PD0p1_TV0p01_lq20/test_T1wT2wFLAIR_example/output_figures/example_001_slice_plot.png)
-----

## 🧠 Methodology Details

### The Physics-Guided Loop

The model (`qmap_synth_model.py`) operates on a self-supervised cycle:

1.  **Generator (CNN):** Takes conventional input images (T1w, T2w, FLAIR) $\rightarrow$ Predicts **T1**, **T2**, and **PD** maps.
2.  [cite_start]**Physics Layer:** Uses the predicted quantitative maps + the known acquisition parameters (TR, TE, TI, FA) from the CSV to synthesize *fake* conventional images using differentiable Bloch equations[cite: 147].
3.  **Loss Calculation:**
      * [cite_start]$\mathcal{L}_{L1}$: The pixel-wise difference between the *Real* input images and the *Synthesized* images[cite: 171].
      * [cite_start]$\mathcal{L}_{TV}$: Isotropic Total Variation regularization to suppress noise in the generated maps[cite: 179].
      * [cite_start]$\mathcal{L}_{PD}$: A soft constraint to enforce realistic proton density bounds ($>0.6$)[cite: 184].

-----


## 📄 License

This project is licensed under a **Creative Commons Attribution-NonCommercial (CC BY-NC)** license.
