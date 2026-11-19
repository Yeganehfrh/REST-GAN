# REST-GAN: A Deep Generative Model for Synthesizing and Decoding Resting-State EEG
This repository contains the official implementation of REST-GAN -- a generative model for synthesizing multi-channel resting-state EEG of individual participants using Wasserstein GAN with Gradient Penalty (WGAN-GP), whose critic is also used as an unsupervised feature extractor.
An earlier version of this work (YARE-GAN) is available as a preprint on [arXiv](https://arxiv.org/abs/2503.02636v3).


## 📁 Repository Structure

```bash
├── .vscode/                      # Editor settings
├── jobs/                         # Job scripts (Slurm, Modal)
├── notebooks/                    # Experiment notebooks
├── splits/                       # Held-out participants
├── src/EEGModalNet/              # Main source code
│   ├── data/                     # Data loading & handling
│   ├── evaluation/               # Features, measures & decoding
│   ├── models/                   # GAN models definitions
│   ├── pipeline/                 # End-to-end execution logic
│   ├── preprocessing/            # EEG preprocessing tools
│   ├── utils/                    # Utility functions
│   ├── __init__.py
│   └── training.py               # Training loop
├── .gitattributes
├── .gitignore
├── README.md
└── pyproject.toml                # Project metadata & dependencies
```

The model of the paper is in `models/FiLMGAN.py`, `models/layers.py` and `models/spectral.py`. The other models in `models/` (WGAN, cWGAN, ...) are earlier Keras experiments.


## Setup

Install the required packages using the following command:

```bash
pixi shell
```

## Training

The model is trained on the LEMON recordings (`data/LEMON_DATA/EC_ch-8_sf-128.nc5` and `EO_ch-8_sf-128.nc5`), without the 40 held-out participants listed in `splits/lemon_heldout40.json`. The defaults are the model of the paper, which was trained three times, with seeds 1, 2 and 3:

```bash
python -m src.EEGModalNet.pipeline.train_gan_con --seed 1 --run-name main_s1 --heldout-split splits/lemon_heldout40.json
```

Each ablation variant (Table 1 of the paper) adds one option to this command:

| Variant | Change from the full model | Option |
| :- | :- | :- |
| A1 | No spectral head in the critic | `--no-spectral-head` |
| A2 | No spectral loss | `--spectral-weight 0` |
| A3 | No participant conditioning | `--no-subject-cond` |
| A4 | No eye-state conditioning | `--no-state-cond` |
| A5 | No second-difference input to the critic | `--no-second-diff` |
| A6 | No minibatch standard deviation layer | `--no-mbsd` |
| A7 | Random starting values of the participant embeddings | `--embedding-init random` |
| A8 | No reconstruction branch | `--no-recon` |
| G1 | No slow noise | `--no-slow-noise` |
| G2 | No dilation | `--no-dilation` |

The variant with FiLM conditioning at the critic's input (Supplementary Section S4) is trained with `--critic-cond film`.

## Evaluating the generated EEG

```bash
sh jobs/run_evaluation.sh logs/main_s1_epoch_100.pt logs/main_s2_epoch_100.pt logs/main_s3_epoch_100.pt
```

This runs `pipeline/evaluate.py` for every checkpoint and combines the results into tables (`pipeline/aggregate_eval.py`). The features (band power, coherence, Hjorth parameters, log-correlation and the non-Gaussian properties) are in `evaluation/features.py`, the measures (Fréchet distance, precision and recall, memorization, identification and similarity) in `evaluation/metrics.py`, and the fitting of an embedding to a held-out participant in `evaluation/enrolment.py`.

## Downstream decoding

```bash
sh jobs/run_decoding.sh                # the three models, in the 607 participants of ds005385
sh jobs/run_decoding_ablations.sh      # the ablation variants
```

`jobs/run_decoding.sh` computes the features of every segment (the readouts of the critic, band power, filter-bank covariance, the 1-Hz cross-spectrum and CBraMod), runs the cross-validation with the readouts chosen inside the training folds, and writes the scores and the comparisons with Holm's correction (`pipeline/decode.py`). It also fits the model to the same participants (`pipeline/enrol_external.py`). The CBraMod features (`pipeline/cbramod_features.py`) need the code and the pretrained weights of CBraMod. The header of each script lists its settings.

## Citation
If you find this work useful, please cite our preprint:

```bash
@article{farahzadi2025yare,
  title={YARE-GAN: Yet Another Resting State EEG-GAN},
  author={Farahzadi, Yeganeh and Ansarinia, Morteza and Kekecs, Zoltan},
  journal={arXiv preprint arXiv:2503.02636},
  year={2025}
}
```
