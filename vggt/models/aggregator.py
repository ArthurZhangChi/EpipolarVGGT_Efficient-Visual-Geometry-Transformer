# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    （修改部分）
    这版 Aggregator 支持 staged / segmented execution：
        - prepare_state(...)
        - run_until_global_layer(...)
        - run_to_end(...)

    这样可以支持你的方案 A：
        0~14 dense
        15 dense 跑完 -> 立刻送 camera_head -> 得到 pose_15
        用 pose_15 给 16/17 构造 sparse meta
        17 跑完 -> 再更新 -> 给 18/19
        ...
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens (per view)
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False  # hardcoded to False

        # ============================================================
        # （修改部分）
        # Hybrid sparse 调度配置
        # ============================================================
        self.enable_partial_sparse: bool = False
        self.partial_sparse_start_layer: int = 15
        self.partial_sparse_end_layer: int = depth - 1

        # ============================================================
        # （修改部分）
        # 可选 dense-bias 对照实验接口
        # ============================================================
        self.enable_attn_bias: bool = False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            )
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    # ============================================================
    # （修改部分）
    # 构造全局 special token 索引
    # 注意：全局 token 排列是按 view 分块的：
    #   [view0 special, view0 patch, view1 special, view1 patch, ...]
    # 因此 special token 不是“全局前若干个”。
    # ============================================================
    def _build_special_token_indices(self, S: int, P: int, device: torch.device) -> torch.Tensor:
        all_special = []
        for v in range(S):
            v0 = v * P
            all_special.append(torch.arange(v0, v0 + self.patch_start_idx, dtype=torch.long, device=device))
        return torch.cat(all_special, dim=0)

    # ============================================================
    # （修改部分）
    # 构造全局 token 元信息，供后半段 sparse 使用
    # ============================================================
    def _build_global_token_meta(
        self,
        S: int,
        P: int,
        device: torch.device,
    ) -> Dict[str, Any]:
        N = S * P

        # token_to_view: [N]
        token_to_view = torch.arange(S, device=device).repeat_interleave(P)

        # token_is_patch: [N]
        token_is_patch = torch.zeros(N, device=device, dtype=torch.bool)
        for v in range(S):
            start = v * P + self.patch_start_idx
            end = (v + 1) * P
            token_is_patch[start:end] = True

        special_token_indices = self._build_special_token_indices(S=S, P=P, device=device)

        meta = {
            "token_to_view": token_to_view,
            "token_is_patch": token_is_patch,
            "num_special_tokens": int(special_token_indices.numel()),
            "special_token_indices": special_token_indices,
        }
        return meta

    # ============================================================
    # （修改部分）
    # 从外部 runtime_sparse_dict 中取当前层 sparse_meta
    #
    # runtime_sparse_dict 约定：
    # {
    #   global_layer_id: {
    #       "allowed_crossview_indices": ...,
    #       "same_view_patch_indices": ... (optional),
    #       "special_token_indices": ... (optional, if absent use base_token_meta)
    #   }
    # }
    # ============================================================
    def _get_sparse_meta_for_layer(
        self,
        global_idx: int,
        base_token_meta: Dict[str, Any],
        runtime_sparse_dict: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if runtime_sparse_dict is None:
            return None
        if global_idx not in runtime_sparse_dict:
            return None

        layer_sparse = runtime_sparse_dict[global_idx]

        if "allowed_crossview_indices" not in layer_sparse:
            raise ValueError(
                f"runtime_sparse_dict[{global_idx}] missing key: 'allowed_crossview_indices'"
            )

        sparse_meta = {
            "num_special_tokens": base_token_meta["num_special_tokens"],
            "token_to_view": base_token_meta["token_to_view"],
            "token_is_patch": base_token_meta["token_is_patch"],
            "special_token_indices": layer_sparse.get(
                "special_token_indices",
                base_token_meta["special_token_indices"],
            ),
            "allowed_crossview_indices": layer_sparse["allowed_crossview_indices"],
        }

        if "same_view_patch_indices" in layer_sparse:
            sparse_meta["same_view_patch_indices"] = layer_sparse["same_view_patch_indices"]

        return sparse_meta

    # ============================================================
    # （修改部分）
    # 取当前层 attn_bias（可选对照实验）
    # ============================================================
    def _get_attn_bias_for_layer(
        self,
        global_idx: int,
        runtime_attn_bias_dict: Optional[Dict[int, torch.Tensor]] = None,
    ):
        if runtime_attn_bias_dict is None:
            return None
        return runtime_attn_bias_dict.get(global_idx, None)

    # ============================================================
    # （修改部分）
    # 判断当前 global layer 是否应该启用 partial sparse
    # ============================================================
    def _use_partial_sparse_for_layer(self, global_idx: int) -> bool:
        if not self.enable_partial_sparse:
            return False
        return self.partial_sparse_start_layer <= global_idx <= self.partial_sparse_end_layer

    # ============================================================
    # （修改部分）
    # 把输入图像变成 backbone 初始 state
    # ============================================================
    def prepare_state(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        Prepare internal execution state for staged backbone running.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # normalize
        images = (images - self._resnet_mean) / self._resnet_std

        # [B*S, C, H, W]
        images_flat = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images_flat)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P_patch_only, C = patch_tokens.shape

        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # [B*S, P, C]
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S,
                H // self.patch_size,
                W // self.patch_size,
                device=images_flat.device,
            )

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2,
                device=images_flat.device,
                dtype=pos.dtype,
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because special tokens were added
        _, P, C = tokens.shape

        base_token_meta = self._build_global_token_meta(
            S=S,
            P=P,
            device=tokens.device,
        )

        state = {
            "tokens": tokens,                       # current tokens
            "pos": pos,                            # current pos
            "B": B,
            "S": S,
            "P": P,
            "C": C,
            "frame_idx": 0,
            "global_idx": 0,
            "aa_outer_idx": 0,
            "aa_order_pos": 0,                     # pointer inside aa_order
            "output_list": [],                     # same semantics as old aggregated_tokens_list
            "base_token_meta": base_token_meta,
            "last_frame_intermediates": None,
            "last_global_intermediates": None,
            "images_normalized": images,
        }
        return state

    # ============================================================
    # （修改部分）
    # 运行一个 frame-attention step
    # ============================================================
    def _run_frame_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        tokens = state["tokens"]
        pos = state["pos"]
        B, S, P, C = state["B"], state["S"], state["P"], state["C"]
        frame_idx = state["frame_idx"]

        # keep tokens in shape (B*S, P, C)
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(
                    self.frame_blocks[frame_idx],
                    tokens,
                    pos,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)

            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        state["tokens"] = tokens
        state["frame_idx"] = frame_idx
        state["last_frame_intermediates"] = intermediates
        return state

    # ============================================================
    # （修改部分）
    # 运行一个 global-attention step
    # 支持：
    #   - dense
    #   - dense + attn_bias
    #   - true partial sparse
    # ============================================================
    def _run_global_step(
        self,
        state: Dict[str, Any],
        runtime_sparse_dict: Optional[Dict[int, Dict[str, Any]]] = None,
        runtime_attn_bias_dict: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        tokens = state["tokens"]
        pos = state["pos"]
        B, S, P, C = state["B"], state["S"], state["P"], state["C"]
        global_idx = state["global_idx"]
        base_token_meta = state["base_token_meta"]

        # keep tokens in shape (B, S*P, C)
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        N = S * P

        for _ in range(self.aa_block_size):
            # 当前层是否在默认 sparse 区间
            want_sparse = self._use_partial_sparse_for_layer(global_idx)

            # 当前层 sparse_meta（如果外部提供了）
            sparse_meta = None
            if want_sparse:
                sparse_meta = self._get_sparse_meta_for_layer(
                    global_idx=global_idx,
                    base_token_meta=base_token_meta,
                    runtime_sparse_dict=runtime_sparse_dict,
                )

            # 真正是否启用 partial sparse：
            # 既要在 sparse 区间内，又要当前层有 sparse_meta
            use_partial_sparse = want_sparse and (sparse_meta is not None)

            # 当前层 attn_bias（可选）
            cur_attn_bias = self._get_attn_bias_for_layer(
                global_idx=global_idx,
                runtime_attn_bias_dict=runtime_attn_bias_dict,
            )

            if cur_attn_bias is not None:
                if cur_attn_bias.ndim == 2:
                    cur_attn_bias = cur_attn_bias[None, None, :, :]
                elif cur_attn_bias.ndim == 3:
                    cur_attn_bias = cur_attn_bias[:, None, :, :]

                if cur_attn_bias.ndim != 4:
                    raise ValueError(f"attn_bias must be [*,*,N,N], got={tuple(cur_attn_bias.shape)}")

                if cur_attn_bias.shape[-2:] != (N, N):
                    raise ValueError(
                        f"attn_bias shape mismatch at global layer {global_idx}: "
                        f"expect last2=({N},{N}), got={tuple(cur_attn_bias.shape[-2:])}"
                    )

            # debug：每层只打印一次
            if not hasattr(self, "_stage_mode_printed"):
                self._stage_mode_printed = set()

            if global_idx not in self._stage_mode_printed:
                self._stage_mode_printed.add(global_idx)
                mode_str = "PARTIAL_SPARSE" if use_partial_sparse else "DENSE"
                logger.info(
                    f"[Aggregator] global layer {global_idx}: "
                    f"mode={mode_str}, attn_bias={'YES' if cur_attn_bias is not None else 'NO'}"
                )

            # 调 block
            # Block.forward 接口：
            #   block(x, pos=None, attn_bias=None, use_partial_sparse=False, sparse_meta=None)
            if self.training:
                tokens = checkpoint(
                    self.global_blocks[global_idx],
                    tokens,
                    pos,
                    cur_attn_bias,
                    use_partial_sparse,
                    sparse_meta,
                    use_reentrant=self.use_reentrant,
                )
            else:
                tokens = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    attn_bias=cur_attn_bias,
                    use_partial_sparse=use_partial_sparse,
                    sparse_meta=sparse_meta,
                )

            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        state["tokens"] = tokens
        state["global_idx"] = global_idx
        state["last_global_intermediates"] = intermediates
        return state

    # ============================================================
    # （修改部分）
    # 当一个 frame+global cycle 完成后，把 concat 特征写入 output_list
    # output_list 的语义与原版 aggregated_tokens_list 保持一致：
    # 每个元素都是 [B, S, P, 2C]
    # ============================================================
    def _flush_cycle_outputs_if_ready(self, state: Dict[str, Any]) -> Dict[str, Any]:
        frame_inter = state["last_frame_intermediates"]
        global_inter = state["last_global_intermediates"]

        if frame_inter is None or global_inter is None:
            return state

        assert len(frame_inter) == len(global_inter), "frame/global intermediates length mismatch"

        for i in range(len(frame_inter)):
            concat_inter = torch.cat([frame_inter[i], global_inter[i]], dim=-1)
            state["output_list"].append(concat_inter)

        state["last_frame_intermediates"] = None
        state["last_global_intermediates"] = None
        return state

    # ============================================================
    # （修改部分）
    # 运行 aa_order 中的一个 step
    # 例如默认 aa_order=["frame", "global"]、aa_block_size=1 时：
    #   第一次调用 -> frame
    #   第二次调用 -> global，并 flush output
    # ============================================================
    def _run_next_aa_step(
        self,
        state: Dict[str, Any],
        runtime_sparse_dict: Optional[Dict[int, Dict[str, Any]]] = None,
        runtime_attn_bias_dict: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        if state["aa_outer_idx"] >= self.aa_block_num:
            return state

        attn_type = self.aa_order[state["aa_order_pos"]]

        if attn_type == "frame":
            state = self._run_frame_step(state)

        elif attn_type == "global":
            state = self._run_global_step(
                state,
                runtime_sparse_dict=runtime_sparse_dict,
                runtime_attn_bias_dict=runtime_attn_bias_dict,
            )
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

        # move pointer
        state["aa_order_pos"] += 1

        # one full cycle finished
        if state["aa_order_pos"] >= len(self.aa_order):
            state["aa_order_pos"] = 0
            state["aa_outer_idx"] += 1
            state = self._flush_cycle_outputs_if_ready(state)

        return state

    # ============================================================
    # （修改部分）
    # 跑到指定 global layer（inclusive）
    #
    # 用法：
    #   state = prepare_state(images)
    #   state = run_until_global_layer(state, target_global_idx=15, ...)
    #
    # 跑完后：
    #   state["output_list"][-1]
    # 就是 global layer 15 对应的 backbone 输出
    # ============================================================
    def run_until_global_layer(
        self,
        state: Dict[str, Any],
        target_global_idx: int,
        runtime_sparse_dict: Optional[Dict[int, Dict[str, Any]]] = None,
        runtime_attn_bias_dict: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        if target_global_idx < 0:
            raise ValueError("target_global_idx must be >= 0")

        while state["global_idx"] <= target_global_idx and state["global_idx"] < self.depth:
            state = self._run_next_aa_step(
                state,
                runtime_sparse_dict=runtime_sparse_dict,
                runtime_attn_bias_dict=runtime_attn_bias_dict,
            )

        return state

    # ============================================================
    # （修改部分）
    # 从当前 state 一直跑到 backbone 结束
    # ============================================================
    def run_to_end(
        self,
        state: Dict[str, Any],
        runtime_sparse_dict: Optional[Dict[int, Dict[str, Any]]] = None,
        runtime_attn_bias_dict: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        while state["global_idx"] < self.depth:
            state = self._run_next_aa_step(
                state,
                runtime_sparse_dict=runtime_sparse_dict,
                runtime_attn_bias_dict=runtime_attn_bias_dict,
            )
        return state

    # ============================================================
    # （修改部分）
    # 保留一个兼容旧接口的 forward：
    # 如果你还想一次性跑完整个 backbone，也能继续工作
    #
    # 但方案 A 的主线建议不要用这个，
    # 而是用：
    #   prepare_state -> run_until_global_layer -> camera_head -> update sparse -> continue
    # ============================================================
    def forward(
        self,
        images: torch.Tensor,
        runtime_sparse_dict: Optional[Dict[int, Dict[str, Any]]] = None,
        runtime_attn_bias_dict: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        state = self.prepare_state(images)
        state = self.run_to_end(
            state,
            runtime_sparse_dict=runtime_sparse_dict,
            runtime_attn_bias_dict=runtime_attn_bias_dict,
        )
        return state["output_list"], self.patch_start_idx


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined