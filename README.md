# TF-RR: Time-Frequency Joint Deep Learning Model for Respiratory Rate Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0-red">
  <img src="https://img.shields.io/badge/License-MIT-green">
</p>

## Table of Contents

**Setup & Data**
- [Requirements](#requirements)

**Training & Inference**
- [Training](#training)
- [Inference](#inference)

**Others**
- [Project Structure](#project-structure)
- [Citation](#citation)


### Changelog
Coming soon.



## Setup

### Requirements

- Python 3.9+
- PyTorch 2.6.0+
- Other packages: see [requirements.txt](./requirements.txt)


```bash
pip install -r requirements.txt
```
### Included Dependencies

S4 and FEDformer's embedding module are not on PyPI. We have included them in this repo:
- `s4.py` from [state-spaces/s4](https://github.com/state-spaces/s4)
- `Embed.py` from [FEDformer](https://github.com/MAZiqing/FEDformer)

Everything is ready to use.




## Training

The training script uses [PyTorch Lightning](https://www.pytorchlightning.ai/) for training loop management.

### Data

Prepare `.npy` files for each fold with the following naming:
- `trainx_{i}.npy`, `trainy_{i}.npy`
- `validx_{i}.npy`, `validy_{i}.npy`
- `testx_{i}.npy`, `testy_{i}.npy`

Update the paths in `train.py` (lines 257).

**Dataset sizes:**
- BIDMC: 49 folds
- Capnobase: 41 folds

> 📌 Our dataset will be open-sourced soon.

### Hyperparameters

Edit `train.py`:
- Model settings → `PLModel.__init__` line 82

### Run

```bash
python train.py -is_train yes
```

### Outputs
Update the paths in `train.py` (lines 257, 330 and 360).


## Inference

To run inference using a trained checkpoint:

```bash
python train.py -is_train no
```



## Project Structure

TF-RR/
  train.py             # Main training & inference script
  model_freq_time.py   # TF-RR model definition
  dataset.py           # Data loader
  callbacks.py         # Lightning callbacks
  evaluations.py       # Evaluation metrics
  result_rr.py         # RR prediction post-processing
  result_rs.py         # RS prediction post-processing
  s4.py                # S4 module (external)
  Embed.py             # Token embedding (external)



## Citation

Our work is currently under review. A preprint will be available soon.

If you find this code useful, please star the repository and cite our upcoming paper.
