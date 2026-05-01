import torch

OLD_HOOK = "server/try/old_hook/predictions.pt"
OLD_NOHOOK = "server/try/old_nohook/predictions.pt"
NEW = "server/try/predictions.pt"

a = torch.load(OLD_HOOK, map_location="cpu")["pose_enc"]   # 你旧 pipeline（带hook）的输出
b = torch.load(OLD_NOHOOK, map_location="cpu")["pose_enc"]# 旧 pipeline（不hook）的输出
c = torch.load(NEW, map_location="cpu")["pose_enc"]       # 服务器 pipeline baseline（不band）的输出

def stat(x, y, name):
    d = (x - y).abs()
    print(name, "max", d.max().item(), "mean", d.mean().item())

stat(a, b, "old_hook vs old_nohook")
stat(b, c, "old_nohook vs new_baseline")
stat(a, c, "old_hook vs new_baseline")
