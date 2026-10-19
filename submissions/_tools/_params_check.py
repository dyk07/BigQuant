import torch
from Transformer_modelsave_train import build_model
import config as CFG

mc = dict(arch="multi", freqs=["bar30m", "bar15m", "bar5m"], **CFG.MODEL_CFG,
          use_xsec=bool(getattr(CFG, "USE_XSEC", False)),
          xsec_layers=int(getattr(CFG, "XSEC_LAYERS", 1)),
          xsec_heads=int(getattr(CFG, "XSEC_HEADS", 4)))
m = build_model(mc)
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
print("params =", n, "= %.2fM" % (n / 1e6))
for name in ["branch", "fuse", "xsec", "head"]:
    sub = getattr(m, name, None)
    if sub is None:
        continue
    if isinstance(sub, dict):
        nn_ = sum(p.numel() for mm in sub.values() for p in mm.parameters())
    else:
        nn_ = sum(p.numel() for p in sub.parameters())
    print(name, nn_)
for f, lb in [("bar30m", 4), ("bar15m", 8), ("bar5m", 24), ("bar1m", 120)]:
    print(f, "T=", lb)
