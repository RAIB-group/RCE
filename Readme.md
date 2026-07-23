## Beyond Error-vs-Discard Characteristic: Toward Stable and Reliable Evaluation for Face Image Quality Assessment

_Accepted at the IEEE/IAPR International Joint Conference on Biometrics (IJCB 2026)._

* [Research Paper](#) *(Link coming soon)*

## Table of Contents

- [Abstract](#abstract)
- [Compute RCE Score](#compute-rce-score)
- [Input Structure](#input-structure)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Abstract

Face Image Quality Assessment (FIQA) aims to estimate the utility of facial images for reliable recognition. The evaluation of FIQA methods is predominantly based on the Error-versus-Discard Characteristic (EDC), which evaluates performance by progressively discarding low-quality samples and measuring recognition error on the retained subset. In this work, we demonstrate that the widely used EDC protocol has fundamental limitations: Test-Set Divergence and Threshold Drift, which together limit the reliability and comparability of FIQA methods. To address this, we propose discard-based EDC variants and a rank-based Rank Consistency Evaluation (RCE) metric that operates on the entire test set without discarding samples, using a fixed decision threshold. Extensive experiments on five datasets, four face recognition models, and 15 state-of-the-art FIQA methods demonstrate both the limitations of EDC and the effectiveness of the proposed approaches in enabling a more reliable and comparable evaluation.Despite evaluated on face images only, the limitations arise from the EDC protocol rather than the biometric modality, suggesting a broader applicability to biometric quality assessment in general.

<p align="center">
<img src="assets/limitation.jpg" style="max-height: 375px;" alt="Limitations of the EDC Protocol." />
</p>

Figure: **Limitations of the EDC Protocol.** With increasing discard rates the FIQA methods (a) are evaluated on progressively different test sets (Test-Set Divergence) and (b) operate on completely different opertation points (Threshold Drift), leading to limited comparability of different FIQA methods.

<p align="center">
<img src="assets/solution.jpg" width="85%" alt="Visualization of the RCE framework." />
</p>

Figure: **Visualization of the RCE framework.** In Rank Consistency Evaluation (RCE), the reference ranking is constructed from recognition errors in the test set, with ties resolved based on the maximum imposter-minimum genuine margin. The final RCE score is computed as the weighted rank correlation between the reference ranking and the ranking based on the FIQA predictions.



## Compute RCE Score

Run the following command to compute the Rank Consistency Evaluation (RCE) scores:

Install dependencies:

```bash
pip install numpy pandas scipy
```

Replace `/home/bw/FIQA` with the path to your local repository.

```bash
python rce.py \
        --fr-features-root /home/bw/FIQA/fr_features \
        --quality-dir /home/bw/FIQA/quality_scores \
        --ca-fiqa-data-root /home/bw/FIQA/ca-fiqa/data \
        --out-root output/sol4_rank_sample \
        --threshold-method roc \
        --weight-alphas 0,3
```

## Input Structure

### Face Recognition Embeddings

```text
fr_features/
└── {dataset_name}/
    └── {fr_model}-embeddings.pkl
```

Example:

```text
fr_features/
└── lfw/
    ├── adaface-embeddings.pkl
    └── arcface_o-embeddings.pkl
```

#### Embedding PKL Format

```text
{
    "image_001.jpg": np.array([0.1, 0.2, 0.3]),
    "image_002.jpg": np.array([0.4, 0.5, 0.6])
}
```

---

### FIQA Quality Scores

```text
quality_scores/
└── {fiqa_method}/
    └── {dataset_name}-quality.pkl
```

Example:

```text
quality_scores/
└── cr-fiqa/
    └── lfw-quality.pkl
```

#### Quality Score PKL Format

```text
{
    "image_001.jpg": 0.92,
    "image_002.jpg": 0.71
}
```

---

### Verification Pairs

```text
data/
└── {dataset_name}/
    └── pairs/
        └── pairs.csv
```

Example:

```text
data/
└── lfw/
    └── pairs/
        └── pairs.csv
```

#### Pairs CSV Format

```text
img1_id,img2_id,label
image_001.jpg,image_002.jpg,1
image_001.jpg,image_003.jpg,0
```

Labels:

- `1`: genuine pair
- `0`: impostor pair


## Citation

If you use this code in your work, please cite the following paper:

```bibtex
@inproceedings{wani2026rce,
    author    = {Wani, Bhavesh and Babnik, {\v{Z}}iga and {\v{S}}truc, Vitomir and Terh{\"o}rst, Philipp},
    title     = {Beyond Error-vs-Discard Characteristic: Toward Stable and Reliable Evaluation for Face Image Quality Assessment},
    booktitle = {Proceedings of the IEEE/IAPR International Joint Conference on Biometrics (IJCB)},
    year      = {2026}
}
```


## Acknowledgement

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Grant 544631027.

## License

This project is licensed under the terms of the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. Copyright (c) 2026 Johannes Gutenberg University Mainz (JGU). You are free to use, modify, and redistribute this software for non-commercial research purposes, provided appropriate attribution is given. Commercial use requires prior permission from the copyright holder.
