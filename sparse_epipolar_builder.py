# Build cross-view patch sparse indices from pose_enc
#
# Main purpose:
#   Given pose_enc from one anchor layer, build sparse meta for
#   subsequent global layers in staged sparse execution.
#
# Assumptions:
#   1) pose_enc shape is [B, S, 9]
#   2) current sparse implementation assumes B == 1
#   3) pose_enc layout:
#        [tx, ty, tz, qx, qy, qz, qw, last2_a, last2_b]
#   4) R, t are interpreted as world-to-camera (w2c)
#
# Output sparse meta:
#   {
#       "allowed_crossview_indices": list[LongTensor], length = N_total,
#       "same_view_patch_indices":   list[LongTensor], length = N_total,
#       "special_token_indices":     LongTensor,
#   }
#
# Notes:
#   - special/query dense and same-view dense are not encoded into
#     allowed_crossview_indices; they are returned separately.
#   - bandwidth is defined in PATCH units, then converted to pixels by:
#         bandwidth_px = bandwidth_patches * patch_size
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import math
import torch


# ============================================================
# Helper math
# ============================================================
def skew(v: torch.Tensor) -> torch.Tensor:
    """
    v: [3]
    return: [3, 3]
    """
    return torch.tensor(
        [
            [0.0, -v[2].item(), v[1].item()],
            [v[2].item(), 0.0, -v[0].item()],
            [-v[1].item(), v[0].item(), 0.0],
        ],
        dtype=v.dtype,
        device=v.device,
    )


def quat_to_rotmat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """
    q: [..., 4], order = (x, y, z, w)
    return: [..., 3, 3]
    """
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    x, y, z, w = q.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return R


def relative_pose_world2cam(Ri: torch.Tensor, ti: torch.Tensor,
                            Rj: torch.Tensor, tj: torch.Tensor):
    """
    Given two world-to-camera extrinsics:
        x_i = Ri X + ti
        x_j = Rj X + tj

    Return relative pose from camera i to camera j:
        x_j = R_ji x_i + t_ji
    """
    R_ji = Rj @ Ri.transpose(-1, -2)
    t_ji = tj - R_ji @ ti
    return R_ji, t_ji


def essential_from_rt(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return skew(t) @ R


def fundamental_from_EK(E: torch.Tensor, Ki: torch.Tensor, Kj: torch.Tensor) -> torch.Tensor:
    F = torch.linalg.inv(Kj).transpose(-1, -2) @ E @ torch.linalg.inv(Ki)
    n = torch.norm(F)
    if n > 1e-12:
        F = F / n
    return F


# ============================================================
# Builder class
# ============================================================
@dataclass
class EpipolarSparseBuilder:
    # processed image size that backbone actually sees
    proc_h: int
    proc_w: int

    # patch/token layout
    patch_size: int
    patch_start_idx: int   # special tokens per view = camera + registers

    # intrinsics strategy
    k_mode: str = "fixed_prior"
    # fixed_prior:
    #   fx = fy = fixed_focal_ratio * proc_w   (unless fixed_focal_px is not None)
    fixed_focal_ratio: float = 0.9
    fixed_focal_px: Optional[float] = None

    # alternative interpretations for last2
    # "fov": last2 are (fovx, fovy) in radians
    # "focal_scale_wh": last2 are normalized focal scales wrt (W, H)

    # numerical
    line_eps: float = 1e-8

    def __post_init__(self):
        if self.proc_h % self.patch_size != 0 or self.proc_w % self.patch_size != 0:
            raise ValueError(
                f"proc_h/proc_w must be divisible by patch_size, "
                f"got ({self.proc_h}, {self.proc_w}) vs patch_size={self.patch_size}"
            )

        self.grid_h = self.proc_h // self.patch_size
        self.grid_w = self.proc_w // self.patch_size
        self.p_patch = self.grid_h * self.grid_w
        self.p_total = self.patch_start_idx + self.p_patch

        # precompute patch centers in homogeneous coordinates: [P_patch, 3]
        self.patch_centers_homo = self._build_patch_centers_homo()

    # ========================================================
    # patch center grid
    # ========================================================
    def _build_patch_centers_homo(self) -> torch.Tensor:
        """
        Patch center coordinates in processed image plane.
        shape: [P_patch, 3]
        """
        ys = torch.arange(self.grid_h, dtype=torch.float32) * self.patch_size + (self.patch_size / 2.0)
        xs = torch.arange(self.grid_w, dtype=torch.float32) * self.patch_size + (self.patch_size / 2.0)

        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.ones_like(xx).reshape(-1)], dim=-1)
        return centers  # [P_patch, 3]

    # ========================================================
    # decode pose_enc
    # ========================================================
    def decode_pose_enc(self, pose_enc: torch.Tensor):
        """
        pose_enc: [1, S, 9] (current sparse implementation assumes B=1)
        return:
            R:     [S, 3, 3]
            t:     [S, 3]
            last2: [S, 2]
        """
        if pose_enc.ndim != 3 or pose_enc.shape[-1] != 9:
            raise ValueError(f"pose_enc must be [B,S,9], got {tuple(pose_enc.shape)}")

        if pose_enc.shape[0] != 1:
            raise ValueError(
                "Current sparse builder assumes batch size B=1, "
                f"but got pose_enc.shape[0]={pose_enc.shape[0]}"
            )

        pose = pose_enc[0].float()  # [S, 9]
        t = pose[:, 0:3]
        q = pose[:, 3:7]            # xyzw
        last2 = pose[:, 7:9]

        R = quat_to_rotmat_xyzw(q)  # [S, 3, 3]
        return R, t, last2

    # ========================================================
    # intrinsics
    # ========================================================
    def _K_fixed_prior(self, device, dtype):
        if self.fixed_focal_px is not None:
            fx = fy = float(self.fixed_focal_px)
        else:
            fx = fy = float(self.fixed_focal_ratio) * float(self.proc_w)

        cx = self.proc_w / 2.0
        cy = self.proc_h / 2.0

        K = torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )
        return K

    def _K_from_fov(self, last2: torch.Tensor) -> torch.Tensor:
        """
        last2: [S, 2], interpreted as (fovx, fovy) in radians
        """
        S = last2.shape[0]
        device = last2.device
        dtype = last2.dtype

        Ks = []
        for s in range(S):
            fovx = float(last2[s, 0].item())
            fovy = float(last2[s, 1].item())

            fovx = max(1e-6, fovx)
            fovy = max(1e-6, fovy)

            fx = (self.proc_w / 2.0) / math.tan(fovx / 2.0)
            fy = (self.proc_h / 2.0) / math.tan(fovy / 2.0)
            cx = self.proc_w / 2.0
            cy = self.proc_h / 2.0

            K = torch.tensor(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=dtype,
                device=device,
            )
            Ks.append(K)

        return torch.stack(Ks, dim=0)

    def _K_from_focal_scale_wh(self, last2: torch.Tensor) -> torch.Tensor:
        """
        last2: [S, 2], interpreted as normalized focal wrt (W, H)
        fx = last2[:,0] * W
        fy = last2[:,1] * H
        """
        S = last2.shape[0]
        device = last2.device
        dtype = last2.dtype

        Ks = []
        for s in range(S):
            fx = float(last2[s, 0].item()) * float(self.proc_w)
            fy = float(last2[s, 1].item()) * float(self.proc_h)
            cx = self.proc_w / 2.0
            cy = self.proc_h / 2.0

            K = torch.tensor(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=dtype,
                device=device,
            )
            Ks.append(K)

        return torch.stack(Ks, dim=0)

    def build_intrinsics(self, last2: torch.Tensor) -> torch.Tensor:
        """
        last2: [S, 2]
        return: Ks [S, 3, 3]
        """
        S = last2.shape[0]
        device = last2.device
        dtype = last2.dtype

        if self.k_mode == "fixed_prior":
            K = self._K_fixed_prior(device=device, dtype=dtype)
            Ks = K.unsqueeze(0).repeat(S, 1, 1)
        elif self.k_mode == "fov":
            Ks = self._K_from_fov(last2)
        elif self.k_mode == "focal_scale_wh":
            Ks = self._K_from_focal_scale_wh(last2)
        else:
            raise ValueError(f"Unsupported k_mode: {self.k_mode}")

        return Ks

    # ========================================================
    # token index helpers
    # ========================================================
    def build_special_token_indices(self, num_views: int, device=None) -> torch.Tensor:
        """
        Return all special token indices in global token order.
        Note:
            global token order is by view block:
                [view0 special, view0 patch, view1 special, view1 patch, ...]
        """
        if device is None:
            device = self.patch_centers_homo.device

        out = []
        for v in range(num_views):
            v0 = v * self.p_total
            out.append(torch.arange(v0, v0 + self.patch_start_idx, dtype=torch.long, device=device))
        return torch.cat(out, dim=0)

    def build_same_view_patch_indices(self, num_views: int, device=None) -> List[torch.Tensor]:
        """
        Return list of length N_total.
        Each entry q_idx stores the same-view patch token global indices.
        """
        if device is None:
            device = self.patch_centers_homo.device

        N_total = num_views * self.p_total
        same_view_patch_indices: List[torch.Tensor] = []

        per_view_patch_idx = []
        for v in range(num_views):
            start = v * self.p_total + self.patch_start_idx
            end = (v + 1) * self.p_total
            per_view_patch_idx.append(torch.arange(start, end, dtype=torch.long, device=device))

        for q_idx in range(N_total):
            v = q_idx // self.p_total
            same_view_patch_indices.append(per_view_patch_idx[v])

        return same_view_patch_indices

    # ========================================================
    # main geometry -> sparse indices
    # ========================================================
    def build_sparse_meta_from_pose_enc(
        self,
        pose_enc: torch.Tensor,
        bandwidth_patches: int,
    ) -> Dict[str, Any]:
        """
        Build sparse meta from one anchor pose_enc.

        Args:
            pose_enc: [1, S, 9]
            bandwidth_patches: band half-width in patch units

        Returns:
            {
                "allowed_crossview_indices": list[LongTensor], length=N_total
                "same_view_patch_indices":   list[LongTensor], length=N_total
                "special_token_indices":     LongTensor
            }
        """
        if bandwidth_patches <= 0:
            raise ValueError(f"bandwidth_patches must be > 0, got {bandwidth_patches}")

        R, t, last2 = self.decode_pose_enc(pose_enc)  # [S,3,3], [S,3], [S,2]
        S = R.shape[0]
        device = pose_enc.device
        dtype = pose_enc.dtype

        Ks = self.build_intrinsics(last2)  # [S,3,3]

        patch_centers = self.patch_centers_homo.to(device=device, dtype=dtype)  # [P_patch, 3]
        P_patch = self.p_patch
        P_total = self.p_total
        N_total = S * P_total

        # bandwidth in pixels
        bandwidth_px = float(bandwidth_patches * self.patch_size)

        allowed_crossview_indices: List[List[torch.Tensor]] = [[] for _ in range(N_total)]

        # ----------------------------------------------------
        # For each pair (src -> dst):
        #   1) compute F_{ji}
        #   2) for each source patch center xi
        #   3) compute its epiline in dst
        #   4) keep dst patch centers within bandwidth
        # ----------------------------------------------------
        for src in range(S):
            Ri = R[src]
            ti = t[src]
            Ki = Ks[src]

            for dst in range(S):
                if dst == src:
                    continue

                Rj = R[dst]
                tj = t[dst]
                Kj = Ks[dst]

                R_ji, t_ji = relative_pose_world2cam(Ri, ti, Rj, tj)
                E_ji = essential_from_rt(R_ji, t_ji)
                F_ji = fundamental_from_EK(E_ji, Ki, Kj)  # [3,3]

                # lines for all source patch centers: [P_patch, 3]
                lines = (F_ji @ patch_centers.t()).t()

                # distances from all dst patch centers to all lines
                # numerators: [P_patch, P_patch]
                numerators = torch.abs((patch_centers @ lines.t()).t())

                # denominator per source line: [P_patch, 1]
                denom = torch.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2).clamp_min(self.line_eps).unsqueeze(-1)

                dists = numerators / denom  # [src_patch, dst_patch]

                # keep dst patches within band
                keep_mask = dists <= bandwidth_px

                for q_patch in range(P_patch):
                    global_q = src * P_total + self.patch_start_idx + q_patch

                    dst_patch_ids = torch.nonzero(keep_mask[q_patch], as_tuple=False).squeeze(-1)
                    if dst_patch_ids.numel() == 0:
                        continue

                    global_dst = dst * P_total + self.patch_start_idx + dst_patch_ids.to(torch.long)
                    allowed_crossview_indices[global_q].append(global_dst)

        # merge per-query lists
        allowed_crossview_indices_final: List[torch.Tensor] = []
        for q_idx in range(N_total):
            if len(allowed_crossview_indices[q_idx]) == 0:
                allowed_crossview_indices_final.append(
                    torch.empty(0, dtype=torch.long, device=device)
                )
            else:
                merged = torch.cat(allowed_crossview_indices[q_idx], dim=0)
                merged = torch.unique(merged, sorted=True)
                allowed_crossview_indices_final.append(merged)

        same_view_patch_indices = self.build_same_view_patch_indices(num_views=S, device=device)
        special_token_indices = self.build_special_token_indices(num_views=S, device=device)

        sparse_meta = {
            "allowed_crossview_indices": allowed_crossview_indices_final,
            "same_view_patch_indices": same_view_patch_indices,
            "special_token_indices": special_token_indices,
            # debug info
            "debug_bandwidth_patches": int(bandwidth_patches),
            "debug_bandwidth_px": float(bandwidth_px),
            "debug_num_views": int(S),
            "debug_p_total": int(P_total),
            "debug_p_patch": int(P_patch),
        }
        return sparse_meta

    # ========================================================
    # Build runtime dict for future layers
    # ========================================================
    def build_runtime_sparse_dict_for_anchor(
        self,
        anchor_global_idx: int,
        pose_enc: torch.Tensor,
        apply_global_layers: List[int],
        bandwidth_patches: int,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Convenience wrapper:
            pose_enc(anchor) -> sparse meta -> runtime_sparse_dict for later layers
        """
        sparse_meta = self.build_sparse_meta_from_pose_enc(
            pose_enc=pose_enc,
            bandwidth_patches=bandwidth_patches,
        )

        runtime_sparse_dict = {}
        for g in apply_global_layers:
            runtime_sparse_dict[g] = {
                "allowed_crossview_indices": sparse_meta["allowed_crossview_indices"],
                "same_view_patch_indices": sparse_meta["same_view_patch_indices"],
                "special_token_indices": sparse_meta["special_token_indices"],
            }
        return runtime_sparse_dict