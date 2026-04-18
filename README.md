# ComAD + PatchCore for MVTec LOCO

This repository combines [ComAD](https://github.com/liutongkun/ComAD) and [PatchCore](https://github.com/amazon-science/patchcore-inspection) for anomaly detection on the [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco) dataset.

---

## Changes from Original

This repository was used as a baseline in our paper: Additional
<!--**"Heterogeneous Multi-Score Integration for Structural and Logical Anomaly Detection"** (IEEE Access, 2026) -->

As noted in the paper, directly summing ComAD and PatchCore scores leads to instability due to scale mismatch. To address this, score normalization was added in `fusion_runner.py`: anomaly scores from each method are independently normalized using mean and standard deviation computed from the normal validation set. To improve robustness against outliers, only scores within the 20th–80th percentile range are used for statistics estimation. The normalized scores are then fused via z-score summation.

---

## Dataset

Download the [MVTec LOCO AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-loco) and place it under:

```
mvtec_loco_anomaly_detection/
├── breakfast_box/
├── juice_bottle/
├── pushpins/
├── screw_bag/
└── splicing_connectors/
```

---

## Usage

### Step 1. Extract ComAD Features

```bash
python seg_image.py
# --datasetpath ./mvtec_loco_anomaly_detection/  (optional, modify if needed)
```

### Step 2. Run Fused Evaluation (ComAD + PatchCore)

```bash
python fusion_runner.py
```

This will run ComAD and PatchCore on all categories, normalize scores using validation-set statistics, and report logical and structural AUROC.

---

### Optional: Run Each Method Standalone

```bash
# PatchCore only
python run_patchcore.py

# ComAD only (logical anomaly detection)
python test_comad_baseline.py
```

---

## Results on MVTec LOCO

Results below are from our own reproduction. For full comparison with other methods, please refer to **Table 1** in our paper.

| Category | Logical AUROC | Structural AUROC |
|---|---|---|
| breakfast_box | 92.4 | 80.6 |
| juice_bottle | 88.1 | 91.6 |
| pushpins | 77.9 | 83.5 |
| screw_bag | 86.9 | 88.2 |
| splicing_connectors | 87.8 | 70.0 |
| **Mean** | **86.6** | **82.8** |

---

## Acknowledgements

- **ComAD**: [liutongkun/ComAD](https://github.com/liutongkun/ComAD)  
  Liu et al., *"Component-aware Anomaly Detection Framework for Adjustable and Logical Industrial Visual Inspection"*, Advanced Engineering Informatics, 2023.

- **PatchCore**: [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)  
  Roth et al., *"Towards Total Recall in Industrial Anomaly Detection"*, CVPR 2022.
