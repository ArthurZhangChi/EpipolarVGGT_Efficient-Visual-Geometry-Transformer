# epipolar_update_function.py
# ============================================================
# Runtime sparse update function for staged sparse VGGT
#
# Role:
#   Given:
#       - anchor_global_idx
#       - pose_enc from current anchor layer
#       - current aggregator state
#   build runtime_sparse_dict for subsequent layers
#
# Intended usage:
#   updater = EpipolarUpdateFunction(cfg, builder)
#   runtime_sparse_dict = updater(
#       anchor_global_idx=15,
#       pose_enc=pose_15,
#       state=state,
#       model=model,
#   )
# ============================================================

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Any, Optional

import torch

from sparse_config import (
    validate_sparse_cfg,
    is_update_anchor_layer,
    get_bandwidth_for_anchor,
    get_apply_layers_for_anchor,
    get_update_mode,
    smooth_pose_if_needed,
)


class EpipolarUpdateFunction:
    """
    Stateful callable update function.

    Why stateful:
        - pose smoothing (EMA) needs previous smoothed pose
        - union_prev mode may need previous runtime_sparse_dict
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        builder,
    ):
        """
        Args:
            cfg:
                sparse config dict from sparse_config.py
            builder:
                EpipolarSparseBuilder instance
        """
        self.cfg = deepcopy(cfg)
        validate_sparse_cfg(self.cfg)

        self.builder = builder

        # 用于 pose smoothing
        self.prev_pose_enc_smooth: Optional[torch.Tensor] = None

        # 用于 union_prev 等策略
        self.prev_runtime_sparse_dict: Optional[Dict[int, Dict[str, Any]]] = None

    # ========================================================
    # merge helpers
    # ========================================================
    def _union_sparse_meta(
        self,
        prev_meta: Dict[str, Any],
        new_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Union sparse meta for one layer.

        Only unions:
            - allowed_crossview_indices
        Keeps:
            - same_view_patch_indices from new_meta if present else prev_meta
            - special_token_indices from new_meta if present else prev_meta
        """
        prev_allowed = prev_meta["allowed_crossview_indices"]
        new_allowed = new_meta["allowed_crossview_indices"]

        if len(prev_allowed) != len(new_allowed):
            raise ValueError(
                f"allowed_crossview_indices length mismatch: "
                f"{len(prev_allowed)} vs {len(new_allowed)}"
            )

        merged_allowed = []
        for a_prev, a_new in zip(prev_allowed, new_allowed):
            if not torch.is_tensor(a_prev):
                a_prev = torch.tensor(a_prev, dtype=torch.long)
            if not torch.is_tensor(a_new):
                a_new = torch.tensor(a_new, dtype=torch.long)

            if a_prev.numel() == 0 and a_new.numel() == 0:
                merged = torch.empty(0, dtype=torch.long, device=a_new.device)
            elif a_prev.numel() == 0:
                merged = a_new
            elif a_new.numel() == 0:
                merged = a_prev
            else:
                merged = torch.unique(torch.cat([a_prev, a_new], dim=0), sorted=True)

            merged_allowed.append(merged)

        out = {
            "allowed_crossview_indices": merged_allowed,
            "same_view_patch_indices": new_meta.get(
                "same_view_patch_indices",
                prev_meta.get("same_view_patch_indices", None),
            ),
            "special_token_indices": new_meta.get(
                "special_token_indices",
                prev_meta.get("special_token_indices", None),
            ),
        }
        return out

    def _merge_runtime_sparse_dict(
        self,
        prev_runtime_sparse_dict: Optional[Dict[int, Dict[str, Any]]],
        new_runtime_sparse_dict: Dict[int, Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Apply update_mode to merge new sparse dict with previous one.
        """
        update_mode = get_update_mode(self.cfg)

        if update_mode == "replace" or prev_runtime_sparse_dict is None:
            return new_runtime_sparse_dict

        if update_mode == "union_prev":
            merged = deepcopy(prev_runtime_sparse_dict)

            for layer_id, new_meta in new_runtime_sparse_dict.items():
                if layer_id not in merged:
                    merged[layer_id] = new_meta
                else:
                    merged[layer_id] = self._union_sparse_meta(merged[layer_id], new_meta)

            return merged

        raise ValueError(f"Unsupported update_mode: {update_mode}")

    # ========================================================
    # main callable
    # ========================================================
    def __call__(
        self,
        anchor_global_idx: int,
        pose_enc: torch.Tensor,
        state: Dict[str, Any],
        model=None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Args:
            anchor_global_idx:
                current anchor layer id, e.g. 15 / 17 / 19 / 21
            pose_enc:
                pose output from camera_head at this anchor layer
                expected shape [B, S, 9], current builder assumes B=1
            state:
                current aggregator state
            model:
                VGGT model instance (optional, not strictly required for now)

        Returns:
            runtime_sparse_dict:
                {
                    future_layer_id: {
                        "allowed_crossview_indices": ...,
                        "same_view_patch_indices": ...,
                        "special_token_indices": ...,
                    },
                    ...
                }
        """
        debug = bool(self.cfg.get("debug_print", False))

        # ----------------------------------------------------
        # 是否为更新点
        # ----------------------------------------------------
        if not is_update_anchor_layer(self.cfg, anchor_global_idx):
            if debug:
                print(f"[SparseUpdate] anchor {anchor_global_idx} is not an update anchor -> return empty dict")
            return {}

        # ----------------------------------------------------
        # pose smoothing（如果启用）
        # 推荐主线先用 none；EMA 是可选后续版本
        # ----------------------------------------------------
        pose_to_use = smooth_pose_if_needed(
            self.cfg,
            prev_pose_enc=self.prev_pose_enc_smooth,
            current_pose_enc=pose_enc,
        )
        self.prev_pose_enc_smooth = pose_to_use.detach()

        # ----------------------------------------------------
        # bandwidth + apply layers
        # ----------------------------------------------------
        bandwidth_patches = get_bandwidth_for_anchor(self.cfg, anchor_global_idx)
        apply_global_layers = get_apply_layers_for_anchor(self.cfg, anchor_global_idx)

        # ----------------------------------------------------
        # 调 builder 生成这次更新对应的 sparse dict
        # ----------------------------------------------------
        new_runtime_sparse_dict = self.builder.build_runtime_sparse_dict_for_anchor(
            anchor_global_idx=anchor_global_idx,
            pose_enc=pose_to_use,
            apply_global_layers=apply_global_layers,
            bandwidth_patches=bandwidth_patches,
        )

        # ----------------------------------------------------
        # 按 update_mode 合并
        # ----------------------------------------------------
        runtime_sparse_dict = self._merge_runtime_sparse_dict(
            prev_runtime_sparse_dict=self.prev_runtime_sparse_dict,
            new_runtime_sparse_dict=new_runtime_sparse_dict,
        )

        self.prev_runtime_sparse_dict = runtime_sparse_dict

        if debug:
            update_mode = get_update_mode(self.cfg)
            print(
                f"[SparseUpdate] anchor={anchor_global_idx} | "
                f"apply_layers={apply_global_layers} | "
                f"bw={bandwidth_patches} patches | "
                f"mode={update_mode}"
            )

        return runtime_sparse_dict