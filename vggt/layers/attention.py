# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
from typing import Optional, Dict, Any

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    # =========================================================
    # （修改部分）
    # 对原版 forward 进行扩展：
    # 1) 继续支持原有 dense attention
    # 2) 支持 token_mask / attn_bias（用于 hard/soft band）
    # 3) 支持 use_partial_sparse + sparse_meta（真正的 partial sparse）
    # =========================================================
    def forward(
        self,
        x: Tensor,
        pos=None,
        token_mask: Tensor = None,
        attn_bias: Tensor = None,
        use_partial_sparse: bool = False,
        sparse_meta: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Dense path:
            - use_partial_sparse = False
            - 支持 token_mask / attn_bias
            - fused_attn=True 时可走 PyTorch fused SDPA 路径

        Partial sparse path:
            - use_partial_sparse = True
            - special query -> 对所有 key 做 dense
            - patch query   -> 只对 special keys + same-view patch keys
                               + allowed cross-view patch keys 做 attention
            - 这里是真正 sparse，不再依赖 dense 的 N x N mask
        """
        if use_partial_sparse:
            return self._forward_partial_sparse(x, pos=pos, sparse_meta=sparse_meta)
        else:
            return self._forward_dense(x, pos=pos, token_mask=token_mask, attn_bias=attn_bias)

    # =========================================================
    # （修改部分）
    # 把 qkv 的公共逻辑抽出来，供 dense / sparse 两条路径共用
    # =========================================================
    def _compute_qkv(self, x: Tensor, pos=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # [3, B, H, N, D]

        q, k, v = qkv.unbind(0)  # [B, H, N, D]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        return q, k, v

    # =========================================================
    # （修改部分）
    # dense path：基本保留你当前已改好的逻辑
    # - 支持 token_mask / attn_bias
    # - fused_attn=True 时走 F.scaled_dot_product_attention
    # =========================================================
    def _forward_dense(
        self,
        x: Tensor,
        pos=None,
        token_mask: Tensor = None,
        attn_bias: Tensor = None,
    ) -> Tensor:
        B, N, C = x.shape
        q, k, v = self._compute_qkv(x, pos=pos)

        attn_mask = None

        # -----------------------------------------------------
        # （修改部分）
        # soft bias 优先：attn_bias 是加到 logits 上的浮点 bias
        # 支持 [N,N] / [B,N,N] / [1,1,N,N] / [B,1,N,N] / [B,H,N,N]
        # -----------------------------------------------------
        if attn_bias is not None:
            if not torch.is_tensor(attn_bias):
                raise TypeError("attn_bias must be a torch.Tensor")

            if not attn_bias.is_floating_point():
                attn_bias = attn_bias.float()

            if attn_bias.ndim == 2:
                # [N,N] -> [1,1,N,N]
                attn_bias = attn_bias[None, None, :, :]
            elif attn_bias.ndim == 3:
                # [B,N,N] -> [B,1,N,N]
                attn_bias = attn_bias[:, None, :, :]

            if attn_bias.ndim != 4:
                raise ValueError(
                    f"attn_bias must be [N,N]/[B,N,N]/[1,1,N,N]/[B,1,N,N]/[B,H,N,N], got={tuple(attn_bias.shape)}"
                )
            if attn_bias.shape[-2:] != (N, N):
                raise ValueError(
                    f"attn_bias last two dims must be (N,N)=({N},{N}), got={tuple(attn_bias.shape[-2:])}"
                )

            attn_mask = attn_bias

        # -----------------------------------------------------
        # （修改部分）
        # hard mask：True=允许，False=屏蔽
        # 支持 [B,N,N] / [B,1,N,N] / [B,H,N,N]
        # -----------------------------------------------------
        elif token_mask is not None:
            if token_mask.dtype != torch.bool:
                token_mask = token_mask.to(dtype=torch.bool)

            if token_mask.ndim == 3:
                token_mask = token_mask[:, None, :, :]

            if token_mask.ndim != 4:
                raise ValueError(
                    f"token_mask must be [B,N,N]/[B,1,N,N]/[B,H,N,N], got={tuple(token_mask.shape)}"
                )
            if token_mask.shape[-2:] != (N, N):
                raise ValueError(
                    f"token_mask last two dims must be (N,N)=({N},{N}), got={tuple(token_mask.shape[-2:])}"
                )

            attn_mask = token_mask

        # -----------------------------------------------------
        # 原版 fused_attn 路径保留
        # 如果环境满足条件，PyTorch 会自动选择高效 backend
        # -----------------------------------------------------
        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q_scaled = q * self.scale
            attn = q_scaled @ k.transpose(-2, -1)  # [B,H,N,N]

            # -------------------------------------------------
            # （修改部分）
            # soft bias：直接加到 logits 上
            # -------------------------------------------------
            if attn_bias is not None:
                bias = attn_bias
                if bias.ndim == 2:
                    bias = bias[None, None, :, :]
                elif bias.ndim == 3:
                    bias = bias[:, None, :, :]
                attn = attn + bias.to(dtype=attn.dtype, device=attn.device)

            # -------------------------------------------------
            # （修改部分）
            # hard mask：带外位置置为 -inf
            # -------------------------------------------------
            if (attn_bias is None) and (token_mask is not None):
                m = token_mask
                if m.dtype != torch.bool:
                    m = m.to(dtype=torch.bool)
                if m.ndim == 3:
                    m = m[:, None, :, :]
                neg = torch.finfo(attn.dtype).min
                attn = attn.masked_fill(~m, neg)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    # =========================================================
    # （修改部分）
    # 真正的 partial sparse 路径
    #
    # 设计：
    # 1) special query -> 对所有 key dense
    # 2) patch query -> 只对：
    #       - special keys
    #       - same-view patch keys
    #       - allowed cross-view patch keys
    #    做 attention
    #
    # 注意：
    # - 这里不再使用 dense N x N mask
    # - allowed indices 从外部传入
    # - 第一版为了逻辑清晰，patch query 逐 query 执行
    # =========================================================
    def _forward_partial_sparse(
        self,
        x: Tensor,
        pos=None,
        sparse_meta: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        sparse_meta 必须至少包含：
            num_special_tokens: int
            token_to_view: LongTensor [N]
                每个 token 属于哪个 view
            token_is_patch: BoolTensor [N]
                True 表示 patch token，False 表示 special token
            allowed_crossview_indices:
                长度为 N 的 list
                第 q_idx 个元素表示该 query token 允许访问的
                cross-view patch 全局索引（只放稀疏那部分）
        可选：
            same_view_patch_indices:
                长度为 N 的 list，用于缓存每个 query 对应的 same-view patch 索引
        """
        if sparse_meta is None:
            raise ValueError("sparse_meta must be provided when use_partial_sparse=True")

        required_keys = [
            "num_special_tokens",
            "token_to_view",
            "token_is_patch",
            "allowed_crossview_indices",
        ]
        for k_req in required_keys:
            if k_req not in sparse_meta:
                raise ValueError(f"sparse_meta missing required key: {k_req}")

        B, N, C = x.shape
        q, k, v = self._compute_qkv(x, pos=pos)  # [B,H,N,D]

        device = x.device

        token_to_view = sparse_meta["token_to_view"].to(device=device)
        token_is_patch = sparse_meta["token_is_patch"].to(device=device)
        allowed_crossview_indices = sparse_meta["allowed_crossview_indices"]
        num_special_tokens = int(sparse_meta["num_special_tokens"])

        # -----------------------------------------------------
        # （修改部分）
        # 全局 special key 索引
        # -----------------------------------------------------
        special_idx = sparse_meta["special_token_indices"]

        # 可选缓存：如果外面已经预先算好了 same-view patch 索引，就直接复用
        same_view_patch_indices_cache = sparse_meta.get("same_view_patch_indices", None)

        # -----------------------------------------------------
        # （修改部分）
        # query 先分成两类：
        # 1) special query
        # 2) patch query
        # -----------------------------------------------------
        special_q_idx = torch.nonzero(~token_is_patch, as_tuple=False).squeeze(-1)
        patch_q_idx = torch.nonzero(token_is_patch, as_tuple=False).squeeze(-1)

        # 输出先按 [B,H,N,D] 存，最后再 reshape 回 [B,N,C]
        out = torch.zeros(B, self.num_heads, N, self.head_dim, device=device, dtype=q.dtype)

        # -----------------------------------------------------
        # 1) special query -> 对所有 key dense
        # -----------------------------------------------------
        if special_q_idx.numel() > 0:
            q_special = q.index_select(dim=2, index=special_q_idx)  # [B,H,Qs,D]

            if self.fused_attn:
                out_special = F.scaled_dot_product_attention(
                    q_special,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                q_special_scaled = q_special * self.scale
                attn = q_special_scaled @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                out_special = attn @ v

            out[:, :, special_q_idx, :] = out_special.to(dtype=out.dtype)

        # -----------------------------------------------------
        # 2) patch query -> 真正 sparse
        #
        # 每个 patch query 只 gather：
        #   special keys + same-view patch keys + allowed cross-view patch keys
        #
        # 注意：
        # - 这里虽然逐 query gather，但已经不再计算完整 N x N
        # - band 外的 cross-view patch 根本不进入计算
        # -----------------------------------------------------
        if patch_q_idx.numel() > 0:
            for q_idx in patch_q_idx.tolist():
                key_idx = self._build_allowed_key_index_for_query(
                    q_idx=q_idx,
                    token_to_view=token_to_view,
                    token_is_patch=token_is_patch,
                    special_idx=special_idx,
                    allowed_crossview_indices=allowed_crossview_indices,
                    same_view_patch_indices_cache=same_view_patch_indices_cache,
                    device=device,
                )  # [M]

                # 当前单个 query
                q_i = q[:, :, q_idx:q_idx + 1, :]               # [B,H,1,D]
                # 只 gather 允许的 key/value
                k_i = k.index_select(dim=2, index=key_idx)     # [B,H,M,D]
                v_i = v.index_select(dim=2, index=key_idx)     # [B,H,M,D]

                # -------------------------------------------------
                # （修改部分）
                # 对单个 query 的局部 key 集合做 attention
                # 这里虽然也是规则张量，但不再是全局 dense attention
                # -------------------------------------------------
                if self.fused_attn:
                    out_i = F.scaled_dot_product_attention(
                        q_i,
                        k_i,
                        v_i,
                        attn_mask=None,
                        dropout_p=self.attn_drop.p if self.training else 0.0,
                    )  # [B,H,1,D]
                else:
                    q_i_scaled = q_i * self.scale
                    attn_i = q_i_scaled @ k_i.transpose(-2, -1)  # [B,H,1,M]
                    attn_i = attn_i.softmax(dim=-1)
                    attn_i = self.attn_drop(attn_i)
                    out_i = attn_i @ v_i

                out[:, :, q_idx:q_idx + 1, :] = out_i.to(dtype=out.dtype)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    # =========================================================
    # （修改部分）
    # 给单个 patch query 构造允许访问的 key 索引
    #
    # key 集合 = special keys
    #        + same-view patch keys
    #        + allowed cross-view patch keys
    # =========================================================
    def _build_allowed_key_index_for_query(
        self,
        q_idx: int,
        token_to_view: torch.Tensor,
        token_is_patch: torch.Tensor,
        special_idx: torch.Tensor,
        allowed_crossview_indices,
        same_view_patch_indices_cache=None,
        device=None,
    ) -> torch.Tensor:
        if device is None:
            device = token_to_view.device

        q_view = int(token_to_view[q_idx].item())

        # -----------------------------------------------------
        # （修改部分）
        # same-view patch keys：
        # 如果外面预先缓存了，就直接拿；
        # 否则这里按 token_to_view 和 token_is_patch 现算
        # -----------------------------------------------------
        if same_view_patch_indices_cache is not None:
            same_view_patch_idx = same_view_patch_indices_cache[q_idx].to(device=device, dtype=torch.long)
        else:
            same_view_patch_mask = (token_to_view == q_view) & token_is_patch
            same_view_patch_idx = torch.nonzero(
                same_view_patch_mask, as_tuple=False
            ).squeeze(-1).to(device)

        # -----------------------------------------------------
        # （修改部分）
        # allowed cross-view patch keys：由外部传入
        # 这里只负责转成 tensor
        # -----------------------------------------------------
        allowed_cross_idx = allowed_crossview_indices[q_idx]
        if not torch.is_tensor(allowed_cross_idx):
            allowed_cross_idx = torch.tensor(allowed_cross_idx, device=device, dtype=torch.long)
        else:
            allowed_cross_idx = allowed_cross_idx.to(device=device, dtype=torch.long)

        # 拼接三部分 key
        key_idx = torch.cat([special_idx, same_view_patch_idx, allowed_cross_idx], dim=0)

        # 去重 + 排序，保证 gather 稳定
        key_idx = torch.unique(key_idx, sorted=True)

        return key_idx


class MemEffAttention(Attention):

    # =========================================================
    # （修改部分）
    # 在 MemEffAttention 中也加入：
    # - token_mask
    # - use_partial_sparse
    # - sparse_meta
    #
    # 逻辑：
    # 1) 若启用 partial sparse，直接回退到父类自定义实现
    # 2) 若无 xFormers，也回退到父类
    # 3) 只有纯 dense 且 xFormers 可用时，才走原来的 memory efficient 分支
    # =========================================================
    def forward(
        self,
        x: Tensor,
        attn_bias=None,
        pos=None,
        token_mask: Tensor = None,
        use_partial_sparse: bool = False,
        sparse_meta: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        assert pos is None

        # -----------------------------------------------------
        # （修改部分）
        # partial sparse 不走 xFormers 黑盒，直接用父类实现
        # -----------------------------------------------------
        if use_partial_sparse:
            return super().forward(
                x,
                pos=None,
                token_mask=token_mask,
                attn_bias=attn_bias,
                use_partial_sparse=True,
                sparse_meta=sparse_meta,
            )

        # -----------------------------------------------------
        # 无 xFormers：直接回退到父类 dense 实现
        # -----------------------------------------------------
        if not XFORMERS_AVAILABLE:
            return super().forward(
                x,
                pos=None,
                token_mask=token_mask,
                attn_bias=attn_bias,
                use_partial_sparse=False,
                sparse_meta=None,
            )

        # -----------------------------------------------------
        # （修改部分）
        # 当前 xFormers 路径不支持 token_mask
        # 如果你后面想让它支持，需要自己实现 mask/bias 转换
        # -----------------------------------------------------
        if token_mask is not None:
            raise NotImplementedError(
                "Current MemEffAttention(xFormers) does not support token_mask."
            )

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # -----------------------------------------------------
        # 原版逻辑尽量保留
        # 如果你环境里没有 unbind / memory_efficient_attention，
        # 就自动 fallback 回父类 dense 实现
        # -----------------------------------------------------
        try:
            q, k, v = unbind(qkv, 2)
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            x = x.reshape([B, N, C])
        except NameError:
            return super().forward(
                x,
                pos=None,
                token_mask=None,
                attn_bias=attn_bias,
                use_partial_sparse=False,
                sparse_meta=None,
            )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
