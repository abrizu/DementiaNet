# Predicting the Onset of Dementia Using Neural Networks
### A Deep Learning Approach to Neuroimaging and Multimodal Analysis

Research Report: *(insert link)*  
Abstract: *(insert)*

---

## Overview

**DementiaNet** is a multimodal deep-learning system that predicts Alzheimer's Disease / Dementia risk across three independent data modalities, each backed by its own trained neural network:

| Modality | Model Type | Classes / Output |
|---|---|---|
| **MRI** | CNN (`_MRI_CNN`) | Non-Demented · Very Mild · Mild · Moderate |
| **Biomarkers** | Feedforward NN (`_Biomarker_NN`) | No Dementia · Dementia (probability) |
| **Speech** | *(in progress)* | Low Risk · High Risk |

Results from individual models are fused via a weighted ensemble to produce a combined dementia risk score.

---

## Features

- **Full in-app training** — no terminal required. Train the MRI and Biomarker models directly from the Streamlit UI with configurable epochs and learning rate.
- **Named checkpoint management** — save multiple trained weights with custom names and notes. Switch between checkpoints at any time without restarting the app.
- **Checkpoint registry** — a `checkpoints/registry.json` file tracks every saved model's accuracy, hyperparameters, date, and notes, persisting across sessions.
- **Demo-ready loading** — on demo day, open the app, go to the Model Manager, and load your best checkpoint in one click.
- **Weighted ensemble** — the Ensemble tab combines MRI (45%), Biomarker (25%), and Speech (30%) predictions into a single risk gauge.
- **Live training feedback** — per-epoch loss and validation accuracy stream into the UI in real time during training.

---

## Project Structure

```
480-NN-Project-MRI/
├── app.py                        # Streamlit application (main entry point)
├── run.sh                        # One-command launcher (venv + deps + app)
├── requirements.txt              # Python dependencies
│
├── functions/
│   ├── mri.py                    # MRI CNN model definition and training script
│   ├── biomarker.py              # Biomarker NN model definition and training script
│   ├── speech.py                 # Speech model (in progress)
│   ├── multimodal.py             # Ensemble / multimodal utilities
│   └── main.py                   # CLI entry point
│
├── checkpoints/
│   ├── registry.json             # Named checkpoint registry (auto-generated)
│   ├── mri_best.pt               # Active MRI weights (auto-generated)
│   └── biomarker_best.pt         # Active Biomarker weights (auto-generated)
│
└── Data/
    ├── MRIData/
    │   └── Data/
    │       ├── Mild Dementia/
    │       ├── Very mild Dementia/
    │       ├── Moderate Dementia/
    │       └── Non Demented/
    └── BMData/
        └── main.csv
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/abrizu/480-NN-Project-MRI.git
cd 480-NN-Project-MRI
```

### 2. Download datasets

Datasets are not included due to size. Download and extract them manually:

| Dataset | Source | Extract to |
|---|---|---|
| MRI Images (4-class) | [Kaggle – Alzheimer MRI Dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis) | `Data/MRIData/Data/` |
| Biomarker CSV | [Kaggle – Comprehensive Health & Brain Imaging](https://www.kaggle.com/datasets/snmahsa/comprehensive-health-and-brain-imaging-dataset) | `Data/BMData/main.csv` |
| Speech Dataset | *(insert link)* | `Data/SPData/` |

After extraction, your `Data/MRIData/Data/` folder should contain exactly four class subfolders:  
`Mild Dementia/`, `Very mild Dementia/`, `Moderate Dementia/`, `Non Demented/`

### 3. Launch the app

```bash
bash run.sh
```

`run.sh` automatically:
- Detects your Python version (3.9+ required)
- Creates a `.venv` virtual environment if one doesn't exist
- Installs all dependencies
- Starts the Streamlit app at `http://localhost:8501`

---

## Using the App

### Training a model

1. Open `http://localhost:8501` in your browser.
2. Navigate to the **MRI** or **Biomarkers** tab.
3. Expand the **⚙️ Model Manager** panel.
4. Click the **🏋️ Train New Model** sub-tab.
5. Adjust epochs and learning rate using the sliders.
6. Click **🚀 Start Training** — live loss/accuracy updates stream into the UI.
7. When training completes, enter a **name** and optional **notes** for the checkpoint (e.g. `demo_run`, `high_lr_v2`).
8. Click **💾 Save & Activate** — the checkpoint is saved, registered, and immediately loaded.

### Loading a saved checkpoint (e.g. on demo day)

1. Open the **⚙️ Model Manager** panel in the relevant tab.
2. Click **📁 Saved Checkpoints**.
3. Select the checkpoint from the dropdown and click **⬆️ Load & Activate**.

### Running inference

- **MRI**: Upload a `.jpg` / `.png` brain MRI scan → click **Analyze MRI** → view class probabilities.
- **Biomarkers**: Fill in the patient form (age, comorbidities, imaging scores) → click **Analyze Biomarkers** → view the risk gauge.
- **Ensemble**: After running at least one modality, visit the **📊 Ensemble** tab for a combined weighted risk score.

---

## Training from the CLI (optional)

```bash
# MRI model
python functions/mri.py

# Biomarker model
python functions/biomarker.py
```

Manually save weights to `checkpoints/mri_best.pt` or `checkpoints/biomarker_best.pt` after CLI training to use them in the app.

---

## Dependencies

All installed automatically by `run.sh`:

```
torch · torchvision · torchaudio
numpy · pandas · scikit-learn · scipy · matplotlib
streamlit · plotly
librosa · soundfile
Pillow · nibabel · kaggle · tqdm · requests
```

---

## Disclaimer

> ⚠️ **Research use only.**  
> DementiaNet is an academic research project and is **not** a validated clinical diagnostic instrument. Do not use it for actual medical decision-making.
