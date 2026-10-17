import sys, types, os, json
import numpy as np, pandas as pd, torch

ROOT = '/mnt/c/Users/Kelvin/Documents/GitHub/BigQuant'
PKG = os.path.join(ROOT, 'submissions', 'cnn_3f')
os.chdir(PKG); sys.path.insert(0, PKG)

import Transformer_modelsave_train as T

# load checkpoint
ck = T.load_model(os.path.join(PKG, 'transformer_model.json'), map_location='cpu')
model = T.build_model(ck['model_cfg']).eval()
model.load_state_dict(ck['state_dict'])
freqs = ck['freqs']
seq_lens = ck['seq_lens']
stats = {f: (np.asarray(ck['mean'][f], np.float32), np.asarray(ck['std'][f], np.float32)) for f in freqs}
minutes = ck['aligned_minutes']

DATA = os.path.join(ROOT, 'e2e_data', 'cpt_jyc_2026_e2e_bar1m')
MONTHS = ['202407', '202408', '202409', '202410', '202411', '202412']

def load(m):
    return pd.read_feather(os.path.join(DATA, f'{m}.0.feather'))

def spearman(a, b):
    if len(a) < 3: return np.nan
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra**2).sum()*(rb**2).sum())
    return float((ra*rb).sum()/den) if den > 0 else np.nan

all_p, all_y, all_day, all_mod = [], [], [], []
for m in MONTHS:
    df = load(m)
    Xs, y, idx_df, keys = T.build_dataset(
        df, m+'01', m+'28', 'eval', minutes, stats, freqs=freqs, seq_lens=seq_lens)
    if len(idx_df) == 0:
        continue
    Xt = [torch.from_numpy(np.ascontiguousarray(x)) for x in Xs]
    preds = []
    with torch.no_grad():
        for i in range(0, len(idx_df), 4096):
            xb = [t[i:i+4096] for t in Xt]
            preds.append(model(xb).numpy())
    preds = np.concatenate(preds)
    all_p.append(preds); all_y.append(y); all_day.append(keys[:,1]); all_mod.append(keys[:,2])
    print(f'  {m}: n={len(preds):,} pred_mean={preds.mean():.4f}')

p = np.concatenate(all_p); y = np.concatenate(all_y)
day = np.concatenate(all_day); mod = np.concatenate(all_mod)
m = np.isfinite(y)
df = pd.DataFrame({'p': p[m], 'y': y[m], 'day': day[m], 'mod': mod[m]})
ics = []
for _, g in df.groupby(['day','mod'], sort=False):
    if len(g) < 10: continue
    ics.append(spearman(g['p'].to_numpy(), g['y'].to_numpy()))
ics = np.array([x for x in ics if np.isfinite(x)])
print(f'\n=== factor vs label IC (local eval) ===')
print(f'  IC_mean = {ics.mean():+.4f}')
print(f'  IC_std  = {ics.std():.4f}')
print(f'  n_sections = {len(ics)}')
print(f'  => 若为正: 推理方向正确, 平台翻转来自数据口径; 若为负: 提交侧方向反了')
