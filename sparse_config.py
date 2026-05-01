# sparse_config.py
# ============================================================
# Staged Sparse / Epipolar Sparse Configuration
#
# 适用于当前方案 A：
#   - global 0~14: dense
#   - global 15  : dense，输出作为第一次可用 pose
#   - global 16,17 使用 pose_15 生成的 sparse meta
#   - global 18,19 使用 pose_17 生成的 sparse meta
#   - global 20,21 使用 pose_19 生成的 sparse meta
#   - global 22,23 使用 pose_21 生成的 sparse meta
#
# 主线推荐：
#   - update_mode = "replace"
#   - pose_smoothing = "none"
#   - true sparse 主线下不使用 alpha
#   - alpha 仅为 soft-band 对照实验预留
# ============================================================

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional


# ============================================================
# 主配置字典
# ============================================================
STAGED_SPARSE_CFG: Dict = {
    # ========================================================
    # 基本开关
    # ========================================================
    "use_staged_sparse": True,          # 是否启用方案 A 的在线 staged sparse
    "use_flash_dense": True,            # 前半段 dense 继续走 fused SDPA / flash 风格路径
    "use_attn_bias_baseline": False,    # 是否启用 dense soft-band 对照
    "save_attention": False,            # 正常推理中不保存 attention map

    # ========================================================
    # Backbone / global layer 调度
    # ========================================================
    "num_global_layers": 24,            # VGGT 默认 global layer 数
    "warmup_dense_global_end": 15,      # 0~15 的 global layer 均为 dense
    "sparse_apply_start": 16,           # 从 16 层开始真正应用 sparse
    "sparse_update_interval": 2,        # 每隔 2 层更新一次 sparse meta

    # ========================================================
    # staged sparse 更新机制
    # ========================================================
    # update_mode:
    #   "replace"      : 直接用新 sparse meta 替换旧的（推荐第一版）
    #   "union_prev"   : 新旧 allowed indices 取并集（更稳但更宽，稀疏性更差）
    "update_mode": "replace",

    # pose_smoothing:
    #   "none"         : 不做平滑，直接用当前 pose
    #   "ema"          : 对 pose 做 EMA，再生成 sparse meta
    "pose_smoothing": "none",
    "pose_ema_beta": 0.8,               # 仅当 pose_smoothing="ema" 时使用

    # ========================================================
    # 稀疏结构范围
    # ========================================================
    "keep_special_dense": True,         # special query / key 相关通路保持 dense
    "keep_same_view_dense": True,       # 当前 view 内 patch-patch 保持 dense
    "sparsify_cross_view_patch_patch_only": True,  # 只稀疏 cross-view patch↔patch

    # ========================================================
    # camera head
    # ========================================================
    "camera_num_iterations": 4,         # CameraHead refinement 次数

    # ========================================================
    # 带宽 schedule（单位：patch）
    #
    # anchor layer -> 用它生成“下一段层”的 sparse meta
    # 例如：
    #   15 -> 给 16,17 用
    #   17 -> 给 18,19 用
    # ========================================================
    "bandwidth_by_anchor_layer": {
        15: 30,
        17: 30,
        19: 2,
        21: 1,
    },

    # ========================================================
    # 显式写出“哪个 anchor 管哪些 layer”
    # 这样 pipeline / sparse_update_fn 会更容易实现
    # ========================================================
    "apply_layers_by_anchor": {
        15: [16, 17],
        17: [18, 19],
        19: [20, 21],
        21: [22, 23],
    },

    # ========================================================
    # 可选：如果你还想做 dense soft-band 对照实验
    # true sparse 主线下通常不需要 alpha
    # 这里仅预留
    # ========================================================
    "soft_band_alpha_by_anchor_layer": {
        15: 1.0,
        17: 1.5,
        19: 2.0,
        21: 3.0,
    },

    # ========================================================
    # 其他运行参数
    # ========================================================
    "debug_print": True,
    "strict_check": True,               # 是否对 layer / anchor / schedule 做严格检查
    "allow_missing_sparse_meta": False, # staged sparse 区间内若缺 sparse meta 是否报错
}


# ============================================================
# 辅助函数：返回默认配置副本
# ============================================================
def get_default_sparse_cfg() -> Dict:
    return deepcopy(STAGED_SPARSE_CFG)


# ============================================================
# 辅助函数：检查配置合法性
# ============================================================
def validate_sparse_cfg(cfg: Dict) -> None:
    num_global_layers = int(cfg["num_global_layers"])
    warmup_dense_global_end = int(cfg["warmup_dense_global_end"])
    sparse_apply_start = int(cfg["sparse_apply_start"])
    sparse_update_interval = int(cfg["sparse_update_interval"])

    if num_global_layers <= 0:
        raise ValueError("num_global_layers must be > 0")

    if not (0 <= warmup_dense_global_end < num_global_layers):
        raise ValueError(
            f"warmup_dense_global_end must be in [0, {num_global_layers - 1}], "
            f"got {warmup_dense_global_end}"
        )

    if not (0 <= sparse_apply_start <= num_global_layers):
        raise ValueError(
            f"sparse_apply_start must be in [0, {num_global_layers}], got {sparse_apply_start}"
        )

    if sparse_update_interval <= 0:
        raise ValueError("sparse_update_interval must be >= 1")

    if cfg["update_mode"] not in ("replace", "union_prev"):
        raise ValueError(f"Unsupported update_mode: {cfg['update_mode']}")

    if cfg["pose_smoothing"] not in ("none", "ema"):
        raise ValueError(f"Unsupported pose_smoothing: {cfg['pose_smoothing']}")

    bandwidth_by_anchor = cfg["bandwidth_by_anchor_layer"]
    apply_layers_by_anchor = cfg["apply_layers_by_anchor"]

    if not isinstance(bandwidth_by_anchor, dict) or len(bandwidth_by_anchor) == 0:
        raise ValueError("bandwidth_by_anchor_layer must be a non-empty dict")

    if not isinstance(apply_layers_by_anchor, dict) or len(apply_layers_by_anchor) == 0:
        raise ValueError("apply_layers_by_anchor must be a non-empty dict")

    for anchor, bw in bandwidth_by_anchor.items():
        if not isinstance(anchor, int):
            raise ValueError(f"anchor layer must be int, got {type(anchor)}")
        if not (0 <= anchor < num_global_layers):
            raise ValueError(f"anchor layer {anchor} out of range [0, {num_global_layers - 1}]")
        if int(bw) <= 0:
            raise ValueError(f"bandwidth for anchor {anchor} must be > 0, got {bw}")

    for anchor, layers in apply_layers_by_anchor.items():
        if not isinstance(anchor, int):
            raise ValueError(f"anchor layer must be int, got {type(anchor)}")
        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError(f"apply_layers_by_anchor[{anchor}] must be a non-empty list")

        for g in layers:
            if not isinstance(g, int):
                raise ValueError(f"global layer id must be int, got {type(g)}")
            if not (0 <= g < num_global_layers):
                raise ValueError(f"global layer {g} out of range [0, {num_global_layers - 1}]")

    # 严格检查：所有 apply layers 最好都在 sparse 区间内
    if cfg.get("strict_check", True):
        for anchor, layers in apply_layers_by_anchor.items():
            for g in layers:
                if g < sparse_apply_start:
                    raise ValueError(
                        f"apply layer {g} (from anchor {anchor}) is earlier than sparse_apply_start={sparse_apply_start}"
                    )


# ============================================================
# 辅助函数：给定 anchor layer，返回该 anchor 对应的 bandwidth
# ============================================================
def get_bandwidth_for_anchor(cfg: Dict, anchor_global_idx: int) -> int:
    bandwidth_by_anchor = cfg["bandwidth_by_anchor_layer"]
    if anchor_global_idx not in bandwidth_by_anchor:
        raise KeyError(f"Anchor layer {anchor_global_idx} not found in bandwidth_by_anchor_layer")
    return int(bandwidth_by_anchor[anchor_global_idx])


# ============================================================
# 辅助函数：给定 anchor layer，返回它负责更新哪些 global layers
# 例如：15 -> [16, 17]
# ============================================================
def get_apply_layers_for_anchor(cfg: Dict, anchor_global_idx: int) -> List[int]:
    apply_layers_by_anchor = cfg["apply_layers_by_anchor"]
    if anchor_global_idx not in apply_layers_by_anchor:
        raise KeyError(f"Anchor layer {anchor_global_idx} not found in apply_layers_by_anchor")
    return list(apply_layers_by_anchor[anchor_global_idx])


# ============================================================
# 辅助函数：给定当前 global layer，反查它由哪个 anchor 控制
# 例如：16 -> 15, 17 -> 15, 18 -> 17
# 如果没找到，返回 None
# ============================================================
def get_anchor_for_global_layer(cfg: Dict, global_idx: int) -> Optional[int]:
    apply_layers_by_anchor = cfg["apply_layers_by_anchor"]
    for anchor, layers in apply_layers_by_anchor.items():
        if global_idx in layers:
            return anchor
    return None


# ============================================================
# 辅助函数：判断某个 global layer 是否应当走 sparse
# ============================================================
def should_use_sparse_for_global_layer(cfg: Dict, global_idx: int) -> bool:
    sparse_apply_start = int(cfg["sparse_apply_start"])
    num_global_layers = int(cfg["num_global_layers"])
    if global_idx < 0 or global_idx >= num_global_layers:
        return False
    return global_idx >= sparse_apply_start


# ============================================================
# 辅助函数：判断某个 global layer 是否是“更新点”
# 即该层跑完之后，要拿它的输出去算 pose，并更新后续 sparse meta
#
# 在当前方案中，anchor layer 就是更新点
# ============================================================
def is_update_anchor_layer(cfg: Dict, global_idx: int) -> bool:
    return global_idx in cfg["apply_layers_by_anchor"]


# ============================================================
# 辅助函数：给定当前 anchor layer，得到下一次更新的 anchor layer
# 比如：15 -> 17 -> 19 -> 21
# 如果没有下一次，返回 None
# ============================================================
def get_next_anchor_layer(cfg: Dict, anchor_global_idx: int) -> Optional[int]:
    anchors = sorted(cfg["apply_layers_by_anchor"].keys())
    for i, a in enumerate(anchors):
        if a == anchor_global_idx:
            if i + 1 < len(anchors):
                return anchors[i + 1]
            return None
    return None


# ============================================================
# 辅助函数：获取某个 anchor 对应的 soft-band alpha
# 仅供 dense soft-band 对照实验使用
# true sparse 主线下通常不用
# ============================================================
def get_soft_alpha_for_anchor(cfg: Dict, anchor_global_idx: int) -> Optional[float]:
    alpha_dict = cfg.get("soft_band_alpha_by_anchor_layer", {})
    if anchor_global_idx not in alpha_dict:
        return None
    return float(alpha_dict[anchor_global_idx])


# ============================================================
# 辅助函数：更新 pose（如果启用 EMA）
#
# 注意：
# - 建议对 pose 做平滑，而不是对 allowed indices 做加权
# - 第一版主线默认 pose_smoothing = "none"
# ============================================================
def smooth_pose_if_needed(
    cfg: Dict,
    prev_pose_enc,
    current_pose_enc,
):
    """
    Args:
        prev_pose_enc: 上一轮平滑后的 pose_enc，或 None
        current_pose_enc: 当前新输出的 pose_enc，shape 通常为 [B, S, 9]

    Returns:
        pose_enc_to_use
    """
    mode = cfg.get("pose_smoothing", "none")

    if mode == "none" or prev_pose_enc is None:
        return current_pose_enc

    if mode == "ema":
        beta = float(cfg.get("pose_ema_beta", 0.8))
        return beta * prev_pose_enc + (1.0 - beta) * current_pose_enc

    raise ValueError(f"Unsupported pose_smoothing mode: {mode}")


# ============================================================
# 辅助函数：更新 sparse meta 的策略
#
# update_mode:
#   - replace   : 用新的 sparse meta 替换旧的（推荐第一版）
#   - union_prev: 新旧 allowed indices 取并集（更稳但更宽）
#
# 这里只定义“策略类型”的接口；
# 真正的 union 逻辑通常在你构造 allowed indices 的代码里做更合适。
# ============================================================
def get_update_mode(cfg: Dict) -> str:
    return str(cfg.get("update_mode", "replace"))


# ============================================================
# 打印配置摘要，方便调试
# ============================================================
def summarize_sparse_cfg(cfg: Dict) -> str:
    lines = []
    lines.append("=== STAGED SPARSE CONFIG SUMMARY ===")
    lines.append(f"use_staged_sparse        : {cfg['use_staged_sparse']}")
    lines.append(f"use_flash_dense          : {cfg['use_flash_dense']}")
    lines.append(f"use_attn_bias_baseline   : {cfg['use_attn_bias_baseline']}")
    lines.append(f"warmup_dense_global_end  : {cfg['warmup_dense_global_end']}")
    lines.append(f"sparse_apply_start       : {cfg['sparse_apply_start']}")
    lines.append(f"sparse_update_interval   : {cfg['sparse_update_interval']}")
    lines.append(f"update_mode              : {cfg['update_mode']}")
    lines.append(f"pose_smoothing           : {cfg['pose_smoothing']}")
    lines.append(f"pose_ema_beta            : {cfg['pose_ema_beta']}")
    lines.append(f"keep_special_dense       : {cfg['keep_special_dense']}")
    lines.append(f"keep_same_view_dense     : {cfg['keep_same_view_dense']}")
    lines.append(f"sparsify_cross_view_only : {cfg['sparsify_cross_view_patch_patch_only']}")
    lines.append(f"camera_num_iterations    : {cfg['camera_num_iterations']}")
    lines.append(f"bandwidth_by_anchor      : {cfg['bandwidth_by_anchor_layer']}")
    lines.append(f"apply_layers_by_anchor   : {cfg['apply_layers_by_anchor']}")
    return "\n".join(lines)


# ============================================================
# 模块导入时可选做一次默认配置检查
# ============================================================
validate_sparse_cfg(STAGED_SPARSE_CFG)


if __name__ == "__main__":
    cfg = get_default_sparse_cfg()
    print(summarize_sparse_cfg(cfg))