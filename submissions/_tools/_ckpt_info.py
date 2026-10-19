import torch

for n in ["cnn_3f", "cnn_best", "cnn_mf"]:
    ck = torch.load("miniquant/checkpoints/" + n + ".pt",
                    map_location="cpu", weights_only=False)
    print(n, "keys=", list(ck.keys()))
    for k in ["freq", "freqs", "val_ic", "lookback_minutes", "target", "loss",
              "ic_mean", "icir", "epoch"]:
        if k in ck:
            print("   ", k, "=", ck[k])
    # state dict 大小
    sd = ck.get("model", ck.get("state_dict", {}))
    print("   n_params=", sum(v.numel() for v in sd.values()))
