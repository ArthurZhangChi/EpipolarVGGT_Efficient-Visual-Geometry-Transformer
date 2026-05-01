# Efficient Visual Geometry Transformer

This repository contains the implementation and experimental materials for **Efficient Visual Geometry Transformer**, a project that improves the efficiency of the global attention module in VGGT through **epipolar-guided staged sparse attention**.

The project focuses on reducing redundant dense cross-view token interactions in multi-view geometry transformers while preserving competitive camera pose estimation and 3D reconstruction performance.

## Overview

Visual Geometry Grounded Transformer (VGGT) is a feed-forward visual geometry model that directly predicts camera parameters, depth maps, point maps, and point tracks from one or multiple input images. It provides a strong unified framework for multi-view geometry modeling, but its global attention module introduces high computational and memory cost when tokens from all views are densely connected.

Given `S` input views and `P` tokens per view, dense global attention has a complexity approximately proportional to:

```text
O((S × P)^2)
```

## Main Contributions

The main contributions of this project are:

1. **Systematic VGGT global attention analysis**

   The attention behavior of different token types is analyzed across layers. The analysis shows that patch-to-patch attention dominates the attention matrix and becomes the major computational bottleneck.

2. **Epipolar consistency observation**

   Cross-view patch attention is visualized and compared with epipolar geometry. The results show that middle-stage and late-stage attention gradually align with epipolar-consistent regions.

3. **Epipolar-guided staged sparse attention**

   A sparse attention mechanism is introduced to reduce unnecessary cross-view patch interactions. The method keeps early global attention dense and applies epipolar-guided sparse attention in later layers.

4. **Efficiency and performance trade-off**

   The proposed method reduces cross-view token interactions to about 53% of the original dense computation and achieves around 1.2×–1.3× speedup in the global attention module benchmark while maintaining competitive geometry performance.


### Datasets

#### DTU

DTU is used for:

- Multi-view geometry analysis
- Cross-view attention visualization
- Epipolar consistency analysis
- Pose estimation evaluation

#### 7Scenes

7Scenes is used for:

- 3D reconstruction evaluation
- Qualitative point cloud comparison
- Reconstruction stability analysis

### Evaluation Metrics

For pose estimation on DTU:

- AUC@1
- AUC@3
- AUC@5
- AUC@15
- AUC@30
- Mean rotation error
- Mean translation error
- Mean combined pose error

For 3D reconstruction on 7Scenes:

- Accuracy
- Completeness
- Normal Consistency
- Overall reconstruction score

For efficiency:

- Dense keys per patch query
- Average kept keys
- Average kept ratio
- Average reduction
- Global attention module runtime speedup

## Results

### Pose Estimation on DTU

The proposed staged sparse model maintains competitive pose estimation performance compared with the dense VGGT baseline.

| Method | AUC@1 | AUC@3 | AUC@5 | AUC@15 | AUC@30 |
|---|---:|---:|---:|---:|---:|
| Baseline VGGT | 0.9524 | 0.9841 | 0.9905 | 0.9968 | 0.9984 |
| Epipolar Constraint (L9–14) | 0.7143 | 0.9048 | 0.9429 | 0.9810 | 0.9905 |
| Epipolar Constraint (L12–17) | 0.1429 | 0.2698 | 0.5143 | 0.8381 | 0.9190 |
| Epipolar Constraint (L15–20) | 0.0000 | 0.3810 | 0.6286 | 0.8762 | 0.9381 |
| Epipolar Constraint (L18–23) | 0.7619 | 0.9206 | 0.9524 | 0.9841 | 0.9921 |
| Staged Sparse Model | 0.8571 | 0.9365 | 0.9619 | 0.9873 | 0.9937 |

The results show that directly applying epipolar restriction to middle layers can significantly damage pose estimation performance. In contrast, the staged sparse model applies the constraint mainly in later layers and obtains a better trade-off.

### Pose Error on DTU

Lower values are better.

| Method | Rotation Mean ↓ | Translation Mean ↓ | Error Mean ↓ |
|---|---:|---:|---:|
| Baseline VGGT | 0.3426 | 0.4182 | 0.4866 |
| Epipolar Constraint (L9–14) | 0.6154 | 0.7737 | 0.9226 |
| Epipolar Constraint (L12–17) | 2.3081 | 2.1121 | 3.0085 |
| Epipolar Constraint (L15–20) | 1.7809 | 1.8176 | 2.3151 |
| Epipolar Constraint (L18–23) | 0.5896 | 0.7217 | 0.8094 |
| Staged Sparse Model | 0.5366 | 0.5988 | 0.7578 |

The staged sparse model has a moderate performance drop compared with the dense baseline, but it performs much better than aggressive middle-layer sparse variants.

### 3D Reconstruction on 7Scenes

The proposed sparse model remains close to the dense baseline on reconstruction metrics.

| Method | Accuracy ↓ | Completeness ↓ | Normal Consistency ↑ | Overall ↓ |
|---|---:|---:|---:|---:|
| Baseline VGGT | 0.02809 | 0.02986 | 0.66712 | 0.02897 |
| Staged Sparse Model | 0.02812 | 0.02987 | 0.66560 | 0.02899 |

The results indicate that the proposed sparse attention design does not significantly damage the final reconstructed scene geometry.

### Efficiency Analysis

The proposed method reduces the amount of cross-view patch interaction in the optimized global attention layers.

| Metric | Value |
|---|---:|
| Recorded Layers | 16–23 |
| Dense Keys per Patch Query | 6216 |
| Average Kept Keys | 3276.67 |
| Average Kept Ratio | 52.713% |
| Average Reduction | 47.287% |
| Global Attention Module Speedup | 1.2×–1.3× |

The speedup is measured at the **global attention module level**, not the full end-to-end VGGT pipeline.

## Qualitative Results

The project includes several types of qualitative visualizations:

- Cross-view attention heatmaps
- Epipolar line and epipolar band alignment
- Top-k attention hit ratio inside the epipolar band
- 3D reconstruction results on 7Scenes office and redkitchen scenes

These visualizations show that VGGT attention is not completely random. In the middle and later layers, high-response cross-view attention regions become more consistent with epipolar geometry, which supports the proposed geometry-guided sparse design.

## Project Structure

The repository can be organized as follows:

```bash
Efficient-VGGT/
├── configs/
│   ├── baseline.yaml
│   ├── epipolar_sparse.yaml
│   └── benchmark.yaml
│
├── datasets/
│   └── README.md
│
├── scripts/
│   ├── run_dtu_pose_eval.py
│   ├── run_7scenes_recon_eval.py
│   ├── visualize_attention.py
│   ├── analyze_epipolar_consistency.py
│   └── benchmark_global_attention.py
│
├── vggt/
│   ├── models/
│   ├── layers/
│   ├── heads/
│   └── utils/
│
├── epipolar_sparse/
│   ├── epipolar_geometry.py
│   ├── epipolar_band.py
│   ├── sparse_attention.py
│   ├── sparse_config.py
│   └── staged_schedule.py
│
├── outputs/
│   ├── attention_visualization/
│   ├── epipolar_analysis/
│   ├── pose_results/
│   ├── reconstruction_results/
│   └── benchmark_results/
│
├── requirements.txt
├── README.md
└── LICENSE
```

## License

This repository is intended for academic and research purposes only. Please refer to the license of the original VGGT implementation for any inherited components.
