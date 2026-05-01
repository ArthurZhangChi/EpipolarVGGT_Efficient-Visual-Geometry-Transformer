import os, glob, random
import torch
import numpy as np
from PIL import Image

# =========================
# 你只改这里
# =========================
BAND_INDEX_PT = r"outputs/token_attention/scene1_DTU/add_epipolar_band/scene1_band_bw3.pt"
DATA_DIR = r"datasets/scene1_DTU"

VIEW_ID_TO_POS_ID = {0: 1, 1: 3, 2: 5, 3: 7, 4: 9, 5: 11, 6: 13}
REF_VIEW_ID = 0

NUM_SAMPLE_Q = 200
NUM_SAMPLE_PAIRS = 10

# =========================
def find_rect_image(pos_id: int) -> str:
    c1 = sorted(glob.glob(os.path.join(DATA_DIR, f"rect_{pos_id:03d}_max.*")))
    if c1: return c1[0]
    c2 = sorted(glob.glob(os.path.join(DATA_DIR, f"rect_{pos_id:03d}.*")))
    if c2: return c2[0]
    raise FileNotFoundError(f"rect for pos_{pos_id:03d} not found in {DATA_DIR}")

def compute_patch_px_avg(W, H, Wp, Hp):
    patch_px_x = W / float(Wp)
    patch_px_y = H / float(Hp)
    return patch_px_x, patch_px_y, 0.5 * (patch_px_x + patch_px_y)

def main():
    obj = torch.load(BAND_INDEX_PT, map_location="cpu")
    meta = obj["meta"]

    Hp, Wp = meta["HpWp"]
    P_patch = int(meta["P_patch"])
    band_unit = meta.get("band_unit", "pixel")  # 旧版可能没有
    print("[meta] Hp,Wp =", (Hp, Wp), "P_patch=", P_patch, "band_unit=", band_unit)
    print("[meta keys]", sorted(list(meta.keys())))

    # 读取参考 rect 尺寸（用于把像素阈值换算成 patch）
    ref_pos = VIEW_ID_TO_POS_ID[REF_VIEW_ID]
    rp = find_rect_image(ref_pos)
    W, H = Image.open(rp).size
    patch_px_x, patch_px_y, patch_px = compute_patch_px_avg(W, H, Wp, Hp)

    print(f"[rect] ref={os.path.basename(rp)} size(W,H)=({W},{H})")
    print(f"[patch_px] x={patch_px_x:.3f}, y={patch_px_y:.3f}, avg={patch_px:.3f}")

    # =========================
    # A) 验证 band 半宽（patch 单位）
    # =========================
    if str(band_unit).lower() == "patch":
        bw_patch = float(meta["band_width_in_patch"])
        bw_px_ref = float(meta.get("ref_band_width_px_used", bw_patch * patch_px))
        print(f"[band] unit=patch | band_half_width = {bw_patch:.3f} patches (你配置的就是它)")
        print(f"[band] ref_px_used ≈ {bw_px_ref:.2f} px  (仅用于参考)")
    else:
        # 旧版：pixel
        band_px = float(meta["band_width_px"])
        half_width_patch = band_px / patch_px
        print(f"[band] unit=pixel | band_width_px = {band_px:.2f} px")
        print(f"[equiv] band_half_width ≈ {half_width_patch:.3f} patches  (例如目标=3.0 ?)")

    # =========================
    # B) CSR 抽样：每个 query 平均保留多少 patch
    # =========================
    pair = obj["pair_csr_patchid"]
    all_pairs = list(pair.keys())
    random.seed(0)
    sample_pairs = random.sample(all_pairs, k=min(NUM_SAMPLE_PAIRS, len(all_pairs)))

    print("\n[csr-check] sample pairs avg keep per query:")
    rng = np.random.default_rng(0)
    for (src, dst) in sample_pairs:
        csr = pair[(src, dst)]
        off = csr["offsets"].numpy()  # (P_patch+1,)
        keep_per_q = off[1:] - off[:-1]  # (P_patch,)

        idx = rng.choice(P_patch, size=min(NUM_SAMPLE_Q, P_patch), replace=False)
        m = float(np.mean(keep_per_q[idx]))
        p50 = float(np.median(keep_per_q[idx]))
        p90 = float(np.percentile(keep_per_q[idx], 90))
        print(f"  src{src:02d}->dst{dst:02d} | mean={m:.1f}  p50={p50:.1f}  p90={p90:.1f}")

    # =========================
    # C) 额外 sanity check：bw 增大时 mean keep 应该单调上升（你手动跑多个 pt 看）
    # =========================
    print("\n[tip] 你可以分别把 BAND_INDEX_PT 改成 bw0.5/bw1/bw1.5/...，观察上面 mean 是否随 bw 递增。")

if __name__ == "__main__":
    main()
