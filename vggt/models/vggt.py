# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Callable, Optional, Dict, Any, List
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = (
            DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
            if enable_point else None
        )
        self.depth_head = (
            DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
            if enable_depth else None
        )
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    # =========================================================
    # （修改部分）
    # 小工具：对某一层 tokens 单独跑一次 camera_head
    # 用于 staged sparse 流程中：
    #   layer 15 输出 -> camera_head -> pose_15
    #   layer 17 输出 -> camera_head -> pose_17
    # =========================================================
    def _run_camera_head_on_single_layer_tokens(
        self,
        tokens_L: torch.Tensor,
        cam_num_iterations: int = 4,
    ):
        if self.camera_head is None:
            raise RuntimeError("camera_head is None, but staged sparse requires camera prediction.")

        pose_enc_list_L = self.camera_head(
            [tokens_L],   # 关键：包成 list，让 camera_head 把它当“最后一层”
            num_iterations=cam_num_iterations,
        )
        return pose_enc_list_L

    # =========================================================
    # （修改部分）
    # staged sparse 主流程：
    #
    # 目标：
    #   0~(warmup-1)    : dense
    #   warmup 层       : dense，跑完立刻算 pose
    #   后续每隔 interval 层更新一次 sparse meta
    #
    # sparse_update_fn 的约定：
    #   输入：
    #       anchor_global_idx: 当前刚跑完、可用于更新几何的 global layer id
    #       pose_enc:          该层 camera_head 输出的 pose_enc，shape [B, S, 9]
    #       state:             当前 aggregator state
    #       model:             self (VGGT)
    #   输出：
    #       一个 dict，格式与 aggregator.run_until_global_layer 接口一致：
    #       {
    #           target_global_layer_id: {
    #               "allowed_crossview_indices": ...,
    #               "same_view_patch_indices": ... (optional),
    #           },
    #           ...
    #       }
    # =========================================================
    def _forward_staged_sparse(
        self,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        sparse_warmup_global_layer: int = 15,
        sparse_update_interval: int = 2,
        sparse_update_fn: Optional[Callable[..., Dict[int, Dict[str, Any]]]] = None,
        runtime_attn_bias_dict: Optional[Dict[int, torch.Tensor]] = None,
        cam_num_iterations: int = 4,
        return_pose_per_layer: bool = False,
        pose_layer_ids: Optional[List[int]] = None,
    ):
        if self.camera_head is None:
            raise RuntimeError("staged sparse requires camera_head, but camera_head is None.")

        if sparse_update_fn is None:
            raise ValueError("staged_sparse=True requires sparse_update_fn to build runtime sparse meta.")

        if sparse_warmup_global_layer < 0 or sparse_warmup_global_layer >= self.aggregator.depth:
            raise ValueError(
                f"sparse_warmup_global_layer must be in [0, {self.aggregator.depth - 1}], "
                f"got {sparse_warmup_global_layer}"
            )

        if sparse_update_interval <= 0:
            raise ValueError("sparse_update_interval must be >= 1.")

        # -----------------------------------------------------
        # （修改部分）
        # 准备 backbone state，但不一次性跑完
        # -----------------------------------------------------
        state = self.aggregator.prepare_state(images)

        # -----------------------------------------------------
        # 第一阶段：先 dense 跑到 warmup global layer（例如 15）
        # 此阶段不需要 pose，不需要 sparse meta
        # -----------------------------------------------------
        state = self.aggregator.run_until_global_layer(
            state,
            target_global_idx=sparse_warmup_global_layer,
            runtime_sparse_dict=None,
            runtime_attn_bias_dict=runtime_attn_bias_dict,
        )

        # 记录在线更新过程中的 pose（可选，方便调试/分析）
        online_pose_updates = {}

        # -----------------------------------------------------
        # 第二阶段开始：每隔 interval 层更新一次 sparse meta
        # -----------------------------------------------------
        while state["global_idx"] < self.aggregator.depth:
            # 当前刚跑完的最新 global layer id
            anchor_global_idx = state["global_idx"] - 1

            if anchor_global_idx < 0:
                raise RuntimeError("Unexpected anchor_global_idx < 0 during staged sparse execution.")

            # 拿到该层 backbone 输出（与 aggregated_tokens_list 的语义一致）
            tokens_anchor = state["output_list"][anchor_global_idx]

            # 用该层 tokens 立刻跑 camera_head，得到 pose
            pose_enc_list_anchor = self._run_camera_head_on_single_layer_tokens(
                tokens_anchor,
                cam_num_iterations=cam_num_iterations,
            )
            pose_enc_anchor = pose_enc_list_anchor[-1]

            online_pose_updates[anchor_global_idx] = {
                "pose_enc": pose_enc_anchor,
                "pose_enc_list": pose_enc_list_anchor,
            }

            # -------------------------------------------------
            # 调外部回调，构造接下来一段 layers 要用的 sparse meta
            # 例如：
            #   anchor=15 -> 返回给 16,17 用
            #   anchor=17 -> 返回给 18,19 用
            # -------------------------------------------------
            runtime_sparse_dict = sparse_update_fn(
                anchor_global_idx=anchor_global_idx,
                pose_enc=pose_enc_anchor,
                state=state,
                model=self,
            )

            if runtime_sparse_dict is None:
                runtime_sparse_dict = {}

            # 下一段跑到哪里为止
            next_target = min(anchor_global_idx + sparse_update_interval, self.aggregator.depth - 1)

            # 继续往后跑，期间使用这次更新出来的 sparse meta
            state = self.aggregator.run_until_global_layer(
                state,
                target_global_idx=next_target,
                runtime_sparse_dict=runtime_sparse_dict,
                runtime_attn_bias_dict=runtime_attn_bias_dict,
            )

        # -----------------------------------------------------
        # backbone 全部完成
        # output_list 与原 aggregated_tokens_list 语义一致
        # -----------------------------------------------------
        aggregated_tokens_list = state["output_list"]
        patch_start_idx = self.aggregator.patch_start_idx

        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):

            # =================================================
            # Camera Head（最终输出）
            # =================================================
            if self.camera_head is not None:
                pose_enc_list_last = self.camera_head(
                    aggregated_tokens_list,
                    num_iterations=cam_num_iterations,
                )
                predictions["pose_enc"] = pose_enc_list_last[-1]
                predictions["pose_enc_list"] = pose_enc_list_last

                # ---------------------------------------------
                # （修改部分）
                # 记录 staged sparse 在线更新时用到的中间 pose
                # ---------------------------------------------
                predictions["online_pose_updates"] = online_pose_updates

                # ---------------------------------------------
                # （修改部分）
                # 可选分析接口：逐层跑 camera_head
                # 这不是主线必须项，默认关闭
                # ---------------------------------------------
                if return_pose_per_layer:
                    num_layers = len(aggregated_tokens_list)

                    if pose_layer_ids is None:
                        pose_layer_ids = list(range(num_layers))

                    pose_by_layer = {}
                    for L in pose_layer_ids:
                        if L < 0 or L >= num_layers:
                            raise ValueError(
                                f"Invalid layer id {L}, must be in [0, {num_layers - 1}]."
                            )

                        tokens_L = aggregated_tokens_list[L]
                        pose_enc_list_L = self._run_camera_head_on_single_layer_tokens(
                            tokens_L,
                            cam_num_iterations=cam_num_iterations,
                        )

                        pose_by_layer[L] = {
                            "pose_enc": pose_enc_list_L[-1],
                            "pose_enc_list": pose_enc_list_L,
                        }

                    predictions["pose_enc_by_layer"] = pose_by_layer
                    predictions["pose_layer_ids"] = pose_layer_ids
                    predictions["num_backbone_layers"] = num_layers

            # =================================================
            # Depth Head
            # =================================================
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            # =================================================
            # Point Head
            # =================================================
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        # =====================================================
        # Track Head
        # =====================================================
        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images

        return predictions

    # =========================================================
    # （修改部分）
    # 兼容旧路径 / 静态路径：
    # 1) baseline
    # 2) dense + flash
    # 3) 静态 sparse
    # 4) hybrid static sparse
    #
    # 如果 staged_sparse=False，就走这里
    # =========================================================
    def _forward_non_staged(
        self,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        runtime_sparse_dict: Optional[Dict[int, Dict[str, Any]]] = None,
        runtime_attn_bias_dict: Optional[Dict[int, torch.Tensor]] = None,
        return_pose_per_layer: bool = False,
        pose_layer_ids: Optional[List[int]] = None,
        cam_num_iterations: int = 4,
    ):
        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images,
            runtime_sparse_dict=runtime_sparse_dict,
            runtime_attn_bias_dict=runtime_attn_bias_dict,
        )

        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list_last = self.camera_head(
                    aggregated_tokens_list,
                    num_iterations=cam_num_iterations,
                )
                predictions["pose_enc"] = pose_enc_list_last[-1]
                predictions["pose_enc_list"] = pose_enc_list_last

                if return_pose_per_layer:
                    num_layers = len(aggregated_tokens_list)

                    if pose_layer_ids is None:
                        pose_layer_ids = list(range(num_layers))

                    pose_by_layer = {}
                    for L in pose_layer_ids:
                        if L < 0 or L >= num_layers:
                            raise ValueError(
                                f"Invalid layer id {L}, must be in [0, {num_layers - 1}]."
                            )

                        tokens_L = aggregated_tokens_list[L]
                        pose_enc_list_L = self._run_camera_head_on_single_layer_tokens(
                            tokens_L,
                            cam_num_iterations=cam_num_iterations,
                        )

                        pose_by_layer[L] = {
                            "pose_enc": pose_enc_list_L[-1],
                            "pose_enc_list": pose_enc_list_L,
                        }

                    predictions["pose_enc_by_layer"] = pose_by_layer
                    predictions["pose_layer_ids"] = pose_layer_ids
                    predictions["num_backbone_layers"] = num_layers

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images

        return predictions

    # =========================================================
    # （修改部分）
    # 总入口：
    # - staged_sparse=False: 兼容旧式一次性 backbone forward
    # - staged_sparse=True : 按方案 A 分段执行
    # =========================================================
    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,

        # =========================（修改部分）=========================
        # 非 staged 路径下的静态 sparse / bias 输入
        runtime_sparse_dict: Optional[Dict[int, Dict[str, Any]]] = None,
        runtime_attn_bias_dict: Optional[Dict[int, torch.Tensor]] = None,

        # 是否启用方案 A：在线 staged sparse
        staged_sparse: bool = False,

        # staged sparse 起始 warmup 层（例如 15）
        sparse_warmup_global_layer: int = 15,

        # 每隔多少层更新一次极线 / sparse meta（例如 2）
        sparse_update_interval: int = 2,

        # 根据当前 pose 生成后续 sparse meta 的回调
        sparse_update_fn: Optional[Callable[..., Dict[int, Dict[str, Any]]]] = None,

        # =========================（修改部分）=========================
        # 可选分析接口
        return_pose_per_layer: bool = False,
        pose_layer_ids: Optional[List[int]] = None,
        cam_num_iterations: int = 4,
        # ============================================================
    ):
        """
        主入口。

        staged_sparse=False:
            走普通一次性 backbone forward，
            适合 baseline / dense+flash / 静态 sparse 对照实验。

        staged_sparse=True:
            走方案 A：
                0~(warmup-1) dense
                warmup 层 dense 后立刻算 pose
                后续每隔 sparse_update_interval 层更新一次 sparse meta
        """
        # -----------------------------------------------------
        # If without batch dimension, add it
        # -----------------------------------------------------
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if staged_sparse:
            return self._forward_staged_sparse(
                images=images,
                query_points=query_points,
                sparse_warmup_global_layer=sparse_warmup_global_layer,
                sparse_update_interval=sparse_update_interval,
                sparse_update_fn=sparse_update_fn,
                runtime_attn_bias_dict=runtime_attn_bias_dict,
                cam_num_iterations=cam_num_iterations,
                return_pose_per_layer=return_pose_per_layer,
                pose_layer_ids=pose_layer_ids,
            )
        else:
            return self._forward_non_staged(
                images=images,
                query_points=query_points,
                runtime_sparse_dict=runtime_sparse_dict,
                runtime_attn_bias_dict=runtime_attn_bias_dict,
                return_pose_per_layer=return_pose_per_layer,
                pose_layer_ids=pose_layer_ids,
                cam_num_iterations=cam_num_iterations,
            )