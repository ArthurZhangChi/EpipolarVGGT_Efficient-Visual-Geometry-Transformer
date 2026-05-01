# save_sparse_metrics.py
# ============================================================
# Save / summarize sparse metrics for staged epipolar sparse attention
#
# Main goal:
#   Record layer-wise and overall sparsity statistics for layers >= min_layer_to_record
#
# What is measured:
#   For each sparse layer:
#       - number of patch queries
#       - dense cross-view patch keys per patch query (100% reference)
#       - avg kept cross-view keys
#       - keep ratio
#       - reduction ratio
#       - min / median / max kept keys
#
# Assumptions:
#   1) Global token order is by view block:
#        [view0 special, view0 patch, view1 special, view1 patch, ...]
#   2) runtime_sparse_dict[layer_id]["allowed_crossview_indices"]
#      is a list of length N_total
#   3) We only evaluate PATCH queries, not special queries
# ============================================================

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch


class SparseMetricsRecorder:
    def __init__(
        self,
        proc_h: int,
        proc_w: int,
        patch_size: int,
        patch_start_idx: int,
        num_views: int,
        min_layer_to_record: int = 16,   # 记录 15 层之后，默认从 16 开始
        strict_check: bool = True,
    ):
        """
        Args:
            proc_h, proc_w:
                预处理后输入模型的图像尺寸
            patch_size:
                patch size, e.g. 14
            patch_start_idx:
                每个 view 内 special token 数（camera + registers）
            num_views:
                当前输入视图数 S
            min_layer_to_record:
                最小需要记录的 global layer，默认 16
            strict_check:
                是否做严格检查
        """
        self.proc_h = int(proc_h)
        self.proc_w = int(proc_w)
        self.patch_size = int(patch_size)
        self.patch_start_idx = int(patch_start_idx)
        self.num_views = int(num_views)
        self.min_layer_to_record = int(min_layer_to_record)
        self.strict_check = bool(strict_check)

        if self.proc_h % self.patch_size != 0 or self.proc_w % self.patch_size != 0:
            raise ValueError(
                f"proc_h/proc_w must be divisible by patch_size, "
                f"got ({self.proc_h}, {self.proc_w}) vs patch_size={self.patch_size}"
            )

        self.grid_h = self.proc_h // self.patch_size
        self.grid_w = self.proc_w // self.patch_size
        self.p_patch = self.grid_h * self.grid_w               # 每个 view 的 patch token 数
        self.p_total = self.patch_start_idx + self.p_patch     # 每个 view 的总 token 数
        self.n_total = self.num_views * self.p_total           # 全局总 token 数

        # dense 参考：每个 patch query 原本 cross-view patch key 总数
        self.dense_crossview_patch_keys_per_query = (self.num_views - 1) * self.p_patch

        # 记录结构
        self.layer_metrics: Dict[int, Dict[str, Any]] = {}
        self.record_history: List[Dict[str, Any]] = []

    # ========================================================
    # internal helpers
    # ========================================================
    def _build_patch_query_indices(self) -> List[int]:
        """
        Return all global token indices corresponding to patch queries.
        """
        q_idx = []
        for v in range(self.num_views):
            start = v * self.p_total + self.patch_start_idx
            end = (v + 1) * self.p_total
            q_idx.extend(range(start, end))
        return q_idx

    def _to_1d_long_tensor(self, x, device=None) -> torch.Tensor:
        if torch.is_tensor(x):
            t = x.detach()
            if device is not None:
                t = t.to(device=device)
            return t.reshape(-1).long()
        return torch.tensor(x, dtype=torch.long, device=device).reshape(-1)

    # ========================================================
    # layer metrics
    # ========================================================
    def compute_layer_metrics(
        self,
        allowed_crossview_indices: List[Any],
        layer_id: int,
        anchor_global_idx: Optional[int] = None,
        bandwidth_patches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute sparse stats for one layer.

        Args:
            allowed_crossview_indices:
                list of length N_total
                each entry is allowed cross-view patch global indices for one query token
            layer_id:
                current sparse layer id
            anchor_global_idx:
                which anchor produced this layer's sparse meta
            bandwidth_patches:
                optional, for record only
        """
        if self.strict_check:
            if len(allowed_crossview_indices) != self.n_total:
                raise ValueError(
                    f"allowed_crossview_indices length mismatch: "
                    f"expect {self.n_total}, got {len(allowed_crossview_indices)}"
                )

        patch_query_indices = self._build_patch_query_indices()

        kept_counts = []
        for q_idx in patch_query_indices:
            allowed = self._to_1d_long_tensor(allowed_crossview_indices[q_idx])
            kept_counts.append(int(allowed.numel()))

        kept_counts_np = np.array(kept_counts, dtype=np.float64)

        avg_kept = float(kept_counts_np.mean()) if kept_counts_np.size > 0 else 0.0
        min_kept = int(kept_counts_np.min()) if kept_counts_np.size > 0 else 0
        max_kept = int(kept_counts_np.max()) if kept_counts_np.size > 0 else 0
        median_kept = float(np.median(kept_counts_np)) if kept_counts_np.size > 0 else 0.0

        dense_ref = float(self.dense_crossview_patch_keys_per_query)

        keep_ratio = avg_kept / dense_ref if dense_ref > 0 else 0.0
        reduction_ratio = 1.0 - keep_ratio

        out = {
            "layer_id": int(layer_id),
            "anchor_global_idx": None if anchor_global_idx is None else int(anchor_global_idx),
            "bandwidth_patches": None if bandwidth_patches is None else int(bandwidth_patches),

            "num_views": int(self.num_views),
            "patch_size": int(self.patch_size),
            "proc_h": int(self.proc_h),
            "proc_w": int(self.proc_w),
            "grid_h": int(self.grid_h),
            "grid_w": int(self.grid_w),

            "num_patch_queries": int(len(patch_query_indices)),
            "dense_crossview_patch_keys_per_query": int(self.dense_crossview_patch_keys_per_query),

            "avg_kept_crossview_keys_per_query": avg_kept,
            "median_kept_crossview_keys_per_query": median_kept,
            "min_kept_crossview_keys_per_query": min_kept,
            "max_kept_crossview_keys_per_query": max_kept,

            # 100% 表示 dense 原始 cross-view patch keys
            "keep_ratio": keep_ratio,
            "keep_percent": keep_ratio * 100.0,
            "reduction_ratio": reduction_ratio,
            "reduction_percent": reduction_ratio * 100.0,
        }
        return out

    # ========================================================
    # main record API
    # ========================================================
    def record_runtime_sparse_dict(
        self,
        runtime_sparse_dict: Dict[int, Dict[str, Any]],
        anchor_global_idx: Optional[int] = None,
        bandwidth_patches: Optional[int] = None,
    ):
        """
        Record sparse metrics for all layers in runtime_sparse_dict,
        but only keep layers >= min_layer_to_record.
        """
        if runtime_sparse_dict is None:
            return

        for layer_id, meta in runtime_sparse_dict.items():
            if int(layer_id) < self.min_layer_to_record:
                continue

            if "allowed_crossview_indices" not in meta:
                raise ValueError(
                    f"runtime_sparse_dict[{layer_id}] missing key: 'allowed_crossview_indices'"
                )

            layer_metrics = self.compute_layer_metrics(
                allowed_crossview_indices=meta["allowed_crossview_indices"],
                layer_id=int(layer_id),
                anchor_global_idx=anchor_global_idx,
                bandwidth_patches=bandwidth_patches,
            )

            # 当前 staged sparse 里每个 layer 正常只会被记录一次
            # 如果以后有重复更新同一层，这里采取覆盖
            self.layer_metrics[int(layer_id)] = layer_metrics

            self.record_history.append({
                "layer_id": int(layer_id),
                "anchor_global_idx": None if anchor_global_idx is None else int(anchor_global_idx),
                "bandwidth_patches": None if bandwidth_patches is None else int(bandwidth_patches),
            })

    # ========================================================
    # overall summary
    # ========================================================
    def get_overall_metrics(self) -> Dict[str, Any]:
        """
        Summarize overall sparse metrics across all recorded layers.
        """
        if len(self.layer_metrics) == 0:
            return {
                "num_recorded_layers": 0,
                "recorded_layer_ids": [],
            }

        layer_ids = sorted(self.layer_metrics.keys())

        avg_kept_list = [self.layer_metrics[L]["avg_kept_crossview_keys_per_query"] for L in layer_ids]
        keep_ratio_list = [self.layer_metrics[L]["keep_ratio"] for L in layer_ids]
        reduction_ratio_list = [self.layer_metrics[L]["reduction_ratio"] for L in layer_ids]

        out = {
            "num_recorded_layers": int(len(layer_ids)),
            "recorded_layer_ids": layer_ids,

            "dense_crossview_patch_keys_per_query": int(self.dense_crossview_patch_keys_per_query),

            "overall_avg_kept_crossview_keys_per_query": float(np.mean(avg_kept_list)),
            "overall_avg_keep_ratio": float(np.mean(keep_ratio_list)),
            "overall_avg_keep_percent": float(np.mean(keep_ratio_list) * 100.0),
            "overall_avg_reduction_ratio": float(np.mean(reduction_ratio_list)),
            "overall_avg_reduction_percent": float(np.mean(reduction_ratio_list) * 100.0),
        }
        return out

    # ========================================================
    # package
    # ========================================================
    def export(self) -> Dict[str, Any]:
        """
        Export all sparse metrics.
        """
        return {
            "meta": {
                "proc_h": int(self.proc_h),
                "proc_w": int(self.proc_w),
                "patch_size": int(self.patch_size),
                "patch_start_idx": int(self.patch_start_idx),
                "num_views": int(self.num_views),
                "grid_h": int(self.grid_h),
                "grid_w": int(self.grid_w),
                "p_patch": int(self.p_patch),
                "p_total": int(self.p_total),
                "n_total": int(self.n_total),
                "min_layer_to_record": int(self.min_layer_to_record),
                "dense_crossview_patch_keys_per_query": int(self.dense_crossview_patch_keys_per_query),
            },
            "layer_metrics": deepcopy(self.layer_metrics),
            "overall_metrics": self.get_overall_metrics(),
            "record_history": deepcopy(self.record_history),
        }

    # ========================================================
    # save
    # ========================================================
    def save_json(self, save_path: str):
        """
        Save metrics as json.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = self.export()

        # 保证 json 可序列化
        def to_jsonable(obj):
            if isinstance(obj, dict):
                return {str(k): to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_jsonable(v) for v in obj]
            if isinstance(obj, tuple):
                return [to_jsonable(v) for v in obj]
            if torch.is_tensor(obj):
                return obj.detach().cpu().tolist()
            return obj

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(export_data), f, ensure_ascii=False, indent=2)

    def save_pt(self, save_path: str):
        """
        Save metrics as torch .pt.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.export(), save_path)