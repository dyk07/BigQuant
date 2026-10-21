# -*- coding: utf-8 -*-
"""MG3_Transformer_T1 · Seed43 · 公榜模型与共享预处理。

1/5/15/30min → 自适应融合 → 因果CNN → 单层因果Transformer；
69维读出、竞价分支与完整截面 exclude_self_mean 保留。
本模块导入不读权重、不查询行情、不训练。公榜由 predict.main 惰性加载。
独立 train_and_save 只有显式调用才从初始化训练。
"""
from __future__ import annotations
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,gc,hashlib,json,math,platform,random,re,shutil,sys,time,uuid
from collections import deque,OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Mapping
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.checkpoint import checkpoint

HERE=Path(__file__).resolve().parent
MODEL_PATH=HERE/'mg3_t1_model.json'
FORMAT='mg3_multigran_crossmean_bigquant_json_v1'
PREPROCESSING_VERSION='mg3_original60grid_multigran_asof_v1'
GRAINS=(1,5,15,30)
MINUTE=60_000_000_000
DAY=1440*MINUTE
PREDICTION_CLOCKS=('09:30','10:00','10:30','11:00','11:30','13:30','14:00','14:30')
OUTPUT_COLUMNS=['date','instrument','factor']

def block_key(ids,starts):
    return (np.asarray(ids,dtype=np.int64)<<32)+np.asarray(starts,dtype=np.int64)//MINUTE

PROTOCOL={'history_bars': 60, 'input_bar_minutes': 15, 'label_horizon_minutes': 30, 'learning_rate': 0.0003, 'weight_decay': 0.0001, 'cpu_threads': 4, 'max_epochs': 8, 'validation_fractions': [0.25, 0.5, 0.75, 1.0], 'patience_validations': 8, 'min_delta': 0, 'optimizer': 'AdamW', 'loss': 'mean SmoothL1(beta=1) over labelled valid nodes in one full cross section', 'gradient_clipping': 'none', 'mixed_precision': 'none; float32', 'lr_schedule': 'constant', 'graphs_per_update': 1, 'gradient_accumulation': 1, 'encoder_chunk_nodes': 512, 'activation_checkpointing': False, 'graph_width': 32, 'top_k': 8, 'message_layers': 1, 'G1_width': 80, 'sampling': 'independent np.default_rng(seed) permutation of all training cross sections each epoch', 'label': 'end_close / future_30m_vwap - 1', 'input_cutoff': 'all history and auction timestamps strictly before decision', 'node_membership': 'current-day observed ID before decision and valid prior close; never label eligibility', 'official_submission_coverage_verified': False, 'history': 'exact real bounds of original 60 grid positions', 'grains': [1, 5, 15, 30], 'block_minutes': 30, 'grain_hidden': 8, 'grain_bottleneck': 4, 'grain_encoder': 'pointwise 76->4 GELU; left causal conv3 4->8 GELU; masked mean; linear 8->24', 'invalid_blocks': 'CNN masked after every residual block; LSTM skips invalid blocks (state held)', 'grain_scaler': 'all grains use unchanged 2020-2022 15min training flow statistics; per-node price/reference transform unchanged', 'routers': 8, 'router_heads': 2, 'router_layers': 1, 'residual_zero_initialization': False, 'data_prefetch': 'one CPU worker, deterministic explicit section order; no sampling RNG in worker', 'train_months': ['202001', '202002', '202003', '202004', '202005', '202006', '202007', '202008', '202009', '202010', '202011', '202012', '202101', '202102', '202103', '202104', '202105', '202106', '202107', '202108', '202109', '202110', '202111', '202112', '202201', '202202', '202203', '202204', '202205', '202206', '202207', '202208', '202209', '202210', '202211', '202212'], 'validation_months': ['202301', '202302', '202303', '202304', '202305', '202306', '202307', '202308', '202309', '202310', '202311', '202312'], 'diagnostic_months_after_freeze': ['202401', '202402', '202403', '202404', '202405', '202406', '202407', '202408', '202409', '202410', '202411', '202412'], 'readout': 'CNN last valid 24 + LSTM last valid 32 + auction 12 + availability 1; head 69->24->1', 'initialization': 'MG0 unchanged original G2; new shared tensors from same-seed MG6, mean-correction tensors from same-seed MG3', 'output_columns': ['date', 'instrument', 'score']}
DEFAULT_MODEL_CFG={'grains': [1, 5, 15, 30], 'feature_count': 25, 'encoded_channels': 76, 'cnn_hidden': 24, 'cnn_kernel_size': 3, 'cnn_dilations': [1, 2, 4, 8, 16, 32], 'lstm_hidden': 32, 'lstm_bidirectional': False, 'head_input': 69, 'head_hidden': 24, 'head_dropout': 0.0, 'auction_hidden': 12, 'use_auction': True, 'node_width': 32, 'readout_mode': 'last_valid'}

FEATURES = (
    "open", "high", "low", "close", "volume", "amount", "deal_number",
    "ask_price1", "ask_price2", "ask_price3", "bid_price1", "bid_price2", "bid_price3",
    "ask_volume1", "ask_volume2", "ask_volume3", "bid_volume1", "bid_volume2", "bid_volume3",
    "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
    "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
)

PRICE_IDX = np.array([i for i, name in enumerate(FEATURES) if name in
                      {"open", "high", "low", "close"} or "_price" in name], dtype=np.int64)

FLOW_IDX = np.array([i for i in range(len(FEATURES)) if i not in PRICE_IDX], dtype=np.int64)

class CausalBlock(nn.Module):
    """Matches the baseline block: left-only convolution, per-token LayerNorm."""

    def __init__(self, hidden: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(hidden, hidden, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv(F.pad(x, (self.left_padding, 0)))
        z = self.norm(z.transpose(1, 2)).transpose(1, 2)
        return F.gelu(x + z)

class MaskedEncoder(nn.Module):
    def __init__(self, channels: int, hidden: int, dilations, kernel_size: int):
        super().__init__()
        self.project = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.ModuleList(
            [CausalBlock(hidden, kernel_size, d) for d in dilations]
        )

    def forward(self, x: Tensor, padding: Tensor) -> Tensor:
        # masked_fill also removes NaN padding; NaN * 0 would remain NaN.
        clean = x.masked_fill(~padding[:, :, None], 0)
        mask = padding[:, None, :].to(x.dtype)
        z = self.project(clean.transpose(1, 2)) * mask
        for block in self.blocks:
            z = block(z) * mask
        return z.transpose(1, 2)

class GrainEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Conv1d(76,4,1)
        self.conv=nn.Conv1d(4,8,3)
        self.output=nn.Linear(8,24)
    def forward(self,raw,obs,present,refs):
        n,L,T,_=raw.shape
        obs=obs&present[:,:,:,None]
        present=present&obs.any(3)
        x=raw.clone()
        prices=((raw[:,:,:,PRICE_IDX]/refs[:,None,None,None]-1)*100).clamp(-50,50)
        x[:,:,:,PRICE_IDX]=prices
        x=x.masked_fill(~obs,0)
        # Same field masks and reserved 25 zero channels as the baseline.
        x=torch.cat((x,obs.to(x.dtype),torch.zeros_like(x),present[:,:,:,None].to(x.dtype)),3)
        x=x.reshape(n*L,T,76).transpose(1,2)
        mask=present.reshape(n*L,T)
        z=F.gelu(self.project(x)).masked_fill(~mask[:,None],0)
        z=F.gelu(self.conv(F.pad(z,(2,0)))).masked_fill(~mask[:,None],0)
        pooled=z.sum(2)/mask.sum(1).clamp_min(1)[:,None]
        available=mask.any(1).reshape(n,L)
        return self.output(pooled).reshape(n,L,24).masked_fill(~available[:,:,None],0),available

class BlockTemporal(nn.Module):
    def __init__(self,grains):
        super().__init__()
        # Build shared temporal/readout modules first so their RNG use is grain independent.
        self.continuous=MaskedEncoder(24,24,(1,2,4,8,16,32),3)
        self.lstm=nn.LSTM(24,32,batch_first=True,bidirectional=False)
        self.auction=MaskedEncoder(76,12,(1,2,4),3)
        self.head=nn.Sequential(nn.Linear(69,24),nn.GELU(),nn.Dropout(0),nn.Linear(24,1))
        self.grains=tuple(grains)
        # Per-grain independent initialization, invariant to the requested subset.
        self.grain=nn.ModuleDict()
        for g in grains:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+g*1009)
                self.grain[str(g)]=GrainEncoder()
        self.gate=nn.Sequential(nn.Linear(len(grains)*25,24),nn.GELU(),nn.Linear(24,len(grains))) if len(grains)>1 else None
    def representation(self,batch):
        emb=[];available=[]
        for g in self.grains:
            e,m=self.grain[str(g)](batch[f'raw_{g}'],batch[f'obs_{g}'],batch[f'present_{g}'],batch['refs'])
            emb.append(e);available.append(m)
        stack=torch.stack(emb,2);avail=torch.stack(available,2)&batch['block_padding'][:,:,None]
        gate_input=torch.cat((stack.flatten(2),avail.to(stack.dtype)),2)
        logits=(self.gate(gate_input) if self.gate is not None else torch.zeros_like(avail,dtype=stack.dtype)).masked_fill(~avail,-1e30)
        weights=logits.softmax(2)*avail
        weights=weights/weights.sum(2,keepdim=True).clamp_min(1)
        valid=avail.any(2)
        fused=(stack*weights[:,:,:,None]).sum(2)
        z=self.continuous(fused,valid)
        n,L,_=z.shape
        lengths=valid.sum(1)
        active=torch.where(lengths>0)[0]
        cnn=z.new_zeros((n,24));hidden=z.new_zeros((n,32))
        if len(active):
            pos=torch.arange(L,device=z.device)[None,:].expand(n,-1)
            last=pos.masked_fill(~valid,-1).max(1).values.clamp_min(0)
            cnn=z[torch.arange(n,device=z.device),last].masked_fill(~valid.any(1)[:,None],0)
            # Stable compaction skips internal invalid blocks: recurrent state is held there.
            order=torch.argsort((~valid[active]).to(torch.int32),dim=1,stable=True)
            compact=z[active].gather(1,order[:,:,None].expand(-1,-1,24))
            packed=pack_padded_sequence(compact,lengths[active].cpu(),batch_first=True,enforce_sorted=False)
            _,(hn,_)=self.lstm(packed)
            hidden=hidden.index_copy(0,active,hn[-1])
        a=self.auction(batch['auction'],batch['auction_padding'])[:,-1]
        a=a.masked_fill(~batch['auction_available'][:,None],0)
        u=torch.cat((cnn,hidden,a,batch['auction_available'][:,None].to(z.dtype)),1)
        # Weight diagnostics are descriptive, not gradients used as cached features.
        den=valid.sum(1).clamp_min(1)[:,None]
        alpha=(weights*valid[:,:,None]).sum(1)/den
        return u,alpha,weights

class LocalGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.q=nn.Linear(32,32,bias=False);self.k=nn.Linear(32,32,bias=False);self.v=nn.Linear(32,32,bias=False)
        self.self_linear=nn.Linear(32,32);self.neighbor_linear=nn.Linear(32,32,bias=False)
    def forward(self,h,ids,valid):
        order=torch.argsort(ids,stable=True);inverse=torch.argsort(order)
        hs=h[order];vs=valid[order];n=len(h)
        scores=self.q(hs)@self.k(hs).T/math.sqrt(32)
        allowed=vs[:,None]&vs[None,:]&~torch.eye(n,dtype=torch.bool,device=h.device)
        scores=scores.masked_fill(~allowed,-torch.inf)
        take=min(8,max(n-1,0))
        if take:
            neighbors=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:take]
            selected=scores.gather(1,neighbors);edges=torch.isfinite(selected)
            weights=selected.masked_fill(~edges,-1e30).softmax(1)*edges
            m=(weights[:,:,None]*self.v(hs)[neighbors]).sum(1)[inverse]
            degree=edges.sum(1).to(h.dtype)[inverse]
            maximum=weights.max(1).values[inverse]
            entropy=-(weights*weights.clamp_min(1e-30).log()).sum(1)[inverse]
        else:
            m=torch.zeros_like(h);degree=maximum=entropy=h.new_zeros(n)
        context=F.gelu(self.self_linear(h)+self.neighbor_linear(m)).masked_fill(~valid[:,None],0)
        return context,degree,maximum,entropy

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.routers=nn.Parameter(torch.randn(8,32)/math.sqrt(32))
        self.to_slots=nn.MultiheadAttention(32,2,batch_first=True,dropout=0)
        self.to_nodes=nn.MultiheadAttention(32,2,batch_first=True,dropout=0)
    def forward(self,h,valid):
        result=torch.zeros_like(h);active=torch.where(valid)[0]
        if len(active):
            nodes=h[active][None]
            slots,_=self.to_slots(self.routers[None],nodes,nodes,need_weights=False)
            context,_=self.to_nodes(nodes,slots,slots,need_weights=False)
            result=result.index_copy(0,active,context[0])
        return result

class MultiGranModel(nn.Module):
    def __init__(self,variant,chunk_nodes=None,activation_checkpointing=None):
        super().__init__();self.variant=variant;self.chunk_nodes=chunk_nodes or PROTOCOL['encoder_chunk_nodes']
        self.activation_checkpointing=PROTOCOL['activation_checkpointing'] if activation_checkpointing is None else activation_checkpointing
        self.temporal=BlockTemporal((15,) if variant=='MG1' else GRAINS)
        if variant!='MG2':self.project=nn.Linear(69,32)
        if variant in ('MG1','MG3'):
            self.self_linear=nn.Linear(32,32);self.neighbor_linear=nn.Linear(32,32,bias=False)
            self.correction=nn.Linear(32,1)
        if variant in ('MG4','MG6'):
            self.router=Router();self.global_gate=nn.Linear(64,32)
            self.global_delta=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
        if variant in ('MG5','MG6'):
            self.local=LocalGraph();self.local_gate=nn.Linear(64,32)
            self.local_delta=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,batch,node_ids,node_valid=None,diagnostics=False):
        if node_ids.ndim!=1 or len(torch.unique(node_ids))!=len(node_ids):raise ValueError('Unique aligned stock IDs required')
        n=len(node_ids)
        if len(batch['refs'])!=n:raise ValueError('One complete timestamp required')
        if 'decision_ns' in batch and len(torch.unique(batch['decision_ns']))!=1:
            raise ValueError('Cross-time aggregation is forbidden')
        valid=torch.ones(n,dtype=torch.bool,device=node_ids.device) if node_valid is None else node_valid
        if valid.dtype!=torch.bool or valid.shape!=(n,):raise ValueError('node_valid must be bool [N]')
        encoded=[];alphas=[];block_alphas=[]
        for lo in range(0,n,self.chunk_nodes):
            part={k:v[lo:lo+self.chunk_nodes] for k,v in batch.items()}
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                keys=tuple(part)
                def fn(*v,keys=keys):return self.temporal.representation(dict(zip(keys,v)))
                u,a,w=checkpoint(fn,*part.values(),use_reentrant=False)
            else:u,a,w=self.temporal.representation(part)
            encoded.append(u);alphas.append(a);block_alphas.append(w)
        u=torch.cat(encoded);alpha=torch.cat(alphas)
        own=self.temporal.head(u).squeeze(-1)
        zero=torch.zeros_like(own);dg=zero;dl=zero;dm=zero;degree=zero;maximum=zero;entropy=zero
        if self.variant!='MG2':h=self.project(u)
        if self.variant in ('MG1','MG3'):
            count=valid.sum()-valid.to(torch.int64)
            context=((h*valid[:,None]).sum(0,keepdim=True)-h*valid[:,None])/count.clamp_min(1)[:,None]
            context=context.masked_fill((count==0)[:,None],0)
            dm=self.correction(F.gelu(self.self_linear(h)+self.neighbor_linear(context))).squeeze(-1)
        if self.variant in ('MG4','MG6'):
            c=self.router(h,valid);gate=torch.sigmoid(self.global_gate(torch.cat((h,c),1)))
            dg=self.global_delta(torch.cat((h,gate*c),1)).squeeze(-1)
        if self.variant in ('MG5','MG6'):
            c,degree,maximum,entropy=self.local(h,node_ids,valid)
            gate=torch.sigmoid(self.local_gate(torch.cat((h,c),1)))
            dl=self.local_delta(torch.cat((h,gate*c),1)).squeeze(-1)
        correction=(dm+dg+dl).masked_fill(~valid,0)
        pred=own+correction
        details={'own':own,'correction':correction,'global_delta':dg,'local_delta':dl,'mean_delta':dm,
            'degree':degree,'max_weight':maximum,'entropy':entropy,'alpha':alpha,
            'alpha_blocks':torch.cat(block_alphas)}
        return (pred,details) if diagnostics else pred

CLOSE_IDX=FEATURES.index("close")

# -------- 与原MG3共享的预处理和可见性窗口 --------
BAR_CLOCKS = ("09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30",
              "13:15", "13:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00")

DECISIONS = PREDICTION_CLOCKS

FUTURE_CLOCKS = {
    "09:30": ("09:45", "10:00"), "10:00": ("10:15", "10:30"),
    "10:30": ("10:45", "11:00"), "11:00": ("11:15", "11:30"),
    "11:30": ("13:15", "13:30"), "13:30": ("13:45", "14:00"),
    "14:00": ("14:15", "14:30"), "14:30": ("14:45", "15:00"),
}
PROTOCOL['label_time_basis']='30 trading minutes; 11:30 uses 13:15/13:30 and ends 13:30 (user confirmed)'
PROTOCOL.update(training_decision_times=list(DECISIONS),prediction_times=list(PREDICTION_CLOCKS),
                output_columns=["date","instrument","factor"])


def normalize_fields(raw: np.ndarray, obs: np.ndarray, refs: np.ndarray,
               scaler: dict, auction: bool):
    result = np.zeros(raw.shape, dtype=np.float32)
    reference = refs[:, None, None]
    price = np.clip((raw[:, :, PRICE_IDX] / reference - 1.0) * 100.0, -50.0, 50.0)
    for j, k in enumerate(PRICE_IDX):
        result[:, :, k] = np.where(obs[:, :, k], price[:, :, j], 0.0)
    mean_key = "auction_flow_mean" if auction else "flow_mean"
    scale_key = "auction_flow_scale" if auction else "flow_scale"
    flows = np.log1p(np.maximum(raw[:, :, FLOW_IDX], 0))
    flows = ((flows - np.asarray(scaler[mean_key], dtype=np.float32)) /
             np.asarray(scaler[scale_key], dtype=np.float32))
    for j, k in enumerate(FLOW_IDX):
        result[:, :, k] = np.where(obs[:, :, k], flows[:, :, j], 0.0)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite encoded input")
    return result

def _sort_book(frame: pd.DataFrame):
    repairs = {"ask": 0, "bid": 0}
    for side in ("ask", "bid"):
        cols = [[f"{side}_price{i}", f"{side}_volume{i}", f"{side}_num_orders{i}"]
                for i in (1, 2, 3)]
        cube = np.stack([frame[c].to_numpy(dtype=np.float64) for c in cols], axis=1)
        p = cube[:, :, 0]
        eligible = np.isfinite(p).all(axis=1) & (p > 0).all(axis=1)
        order = np.argsort(p, axis=1, kind="stable")
        if side == "bid":
            order = order[:, ::-1]
        changed = eligible & (order != np.arange(3)).any(axis=1)
        repairs[side] = int(changed.sum())
        if changed.any():
            fixed = np.take_along_axis(cube[changed], order[changed, :, None], axis=1)
            for level, triple in enumerate(cols):
                for field, col in enumerate(triple):
                    frame.loc[changed, col] = fixed[:, level, field].astype(frame[col].dtype)
    return repairs

def restore_frame(frame):
    """Same row-wise repairs as the local 15min builder; no fitting/interpolation."""
    frame = frame.copy()
    required = {'date', 'instrument', *FEATURES}
    if not required.issubset(frame):
        raise ValueError(f'Missing platform fields: {sorted(required-set(frame))}')
    frame['date'] = pd.to_datetime(frame.date, errors='raise')
    if frame[['date', 'instrument']].isna().any().any():
        raise ValueError('Null source date/instrument')
    if frame.date.dt.tz is not None:
        raise ValueError('Expected naive Asia/Shanghai timestamps')
    frame['instrument'] = frame.instrument.astype(str)
    duplicated = frame.duplicated(['date', 'instrument'], keep=False)
    if duplicated.any() and frame.loc[duplicated].drop_duplicates().duplicated(['date', 'instrument']).any():
        raise ValueError('Conflicting duplicate source keys')
    frame = frame.drop_duplicates(['date', 'instrument']).sort_values(['instrument', 'date'], kind='stable').reset_index(drop=True)
    book = _sort_book(frame)
    v, c, a = (frame[name].to_numpy(dtype=np.float64) for name in ('volume', 'close', 'amount'))
    closing = frame.date.dt.strftime('%H:%M').eq('15:00').to_numpy()
    candidate = closing & np.isfinite(v) & np.isfinite(c) & (v > 0) & (c > 0)
    ratio = np.divide(a, v*c, out=np.full(len(frame), np.nan), where=candidate)
    repair = candidate & (~np.isfinite(ratio) | (ratio < .8) | (ratio > 1.2))
    if repair.any():
        frame.loc[repair, 'amount'] = (v[repair]*c[repair]).astype('int64')
    raw = frame[list(FEATURES)].to_numpy(dtype=np.float32)
    observed = np.isfinite(raw)
    observed[:, PRICE_IDX] &= raw[:, PRICE_IDX] > 0
    observed[:, FLOW_IDX] &= raw[:, FLOW_IDX] >= 0
    raw[~observed] = 0
    return frame[['date', 'instrument']], raw, observed, {
        'rows': len(frame), 'book_triples_reordered': book,
        'amount_1500_v_times_close_repaired': int(repair.sum())}

def training_requests(frame):
    """Generate the eight prediction times requested on 2026-10-20."""
    keys = frame[['date', 'instrument']].copy()
    keys['date'] = pd.to_datetime(keys.date).dt.normalize()
    keys = keys.drop_duplicates()
    parts = []
    for clock in DECISIONS:
        part = keys.copy()
        part['date'] += pd.Timedelta(clock+':00')
        parts.append(part)
    return pd.concat(parts, ignore_index=True)

def build_month(frame, carry, requests, with_labels=False):
    """Build shared token/index inputs for exactly the requested (time, stock).

    Inference does not read future rows to determine label validity. Requests
    come from the official pool and may include 09:30 / 13:00, unlike the old
    four-time research cache. State remains private to each stock.
    """
    keys, raw, observed, audit = restore_frame(frame)
    if keys.empty:
        return None, {}, {**audit, 'samples': 0, 'requested': len(requests)}
    if len(keys.date.dt.to_period('M').unique()) != 1:
        raise ValueError('build_month needs one calendar month at a time')
    values, masks, dates = [np.zeros((1, len(FEATURES)), np.float32), raw], [np.zeros((1, len(FEATURES)), bool), observed], [np.zeros(1, np.int64), keys.date.to_numpy(dtype='datetime64[ns]').astype('int64')]
    initial = {}
    offset = 1+len(raw)
    for instrument, (v, m, d) in carry.items():
        if len(v):
            values.append(v); masks.append(m); dates.append(d)
            initial[instrument] = list(range(offset, offset+len(v)))
            offset += len(v)
    token_values = np.concatenate(values)
    token_observed = np.concatenate(masks)
    token_dates = np.concatenate(dates)
    del values, masks, dates, raw, observed
    requested = {(str(r.instrument), int(pd.Timestamp(r.date).value)) for r in requests.itertuples(index=False)}
    request_clocks = set(pd.to_datetime(requests.date).dt.strftime('%H:%M')) if len(requests) else set()
    events = sorted(set(BAR_CLOCKS) | request_clocks | {'13:00'})
    ids = keys.instrument.to_numpy()
    days = keys.date.dt.normalize().to_numpy(dtype='datetime64[ns]')
    clocks = keys.date.dt.strftime('%H:%M').to_numpy()
    boundaries = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    histories, auctions, references, labels, sample_ids, sample_dates, label_ends = [], [], [], [], [], [], []
    new_carry = {}
    skipped_no_reference = skipped_bad_label = 0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        instrument = str(ids[lo])
        hist = deque(initial.get(instrument, ()), maxlen=60)
        ends = lo + np.flatnonzero(np.r_[True, days[lo+1:hi] != days[lo:hi-1], True])
        for day_lo, day_hi in zip(ends[:-1], ends[1:]):
            day = pd.Timestamp(days[day_lo])
            if hist:
                hist.append(0)
            by_clock = {clocks[pos]: pos for pos in range(day_lo, day_hi)}
            auction_pos = by_clock.get('09:25')
            auction = auction_pos+1 if auction_pos is not None else 0
            for clock in events:
                if clock == '13:00' and hist:
                    hist.append(0)
                decision = day+pd.Timedelta(clock+':00')
                if (instrument, decision.value) in requested:
                    if not (keys.date.iloc[day_lo:day_hi] < decision).any():
                        continue  # Future-only appearance cannot admit a graph node.
                    reference = next((float(token_values[i, CLOSE_IDX]) for i in reversed(hist)
                                      if i > 0 and token_observed[i, CLOSE_IDX]), None)
                    if reference is None:
                        skipped_no_reference += 1
                    else:
                        label, endpoint = np.nan, day+pd.Timedelta(FUTURE_CLOCKS[clock][-1]+':00')
                        if with_labels:
                            future_clocks = FUTURE_CLOCKS.get(clock)
                            if future_clocks is None:
                                raise ValueError('Training requires the configured eight label times')
                            positions = [by_clock.get(c) for c in future_clocks]
                            if all(p is not None for p in positions):
                                future = token_values[np.asarray(positions, np.int32)+1]
                                valid = token_observed[np.asarray(positions, np.int32)+1]
                                vc, ac = FEATURES.index('volume'), FEATURES.index('amount')
                                volume, amount = float(future[:, vc].sum()), float(future[:, ac].sum())
                                if valid[:, vc].all() and valid[:, ac].all() and valid[-1, CLOSE_IDX] and volume > 0 and amount > 0:
                                    label = float(future[-1, CLOSE_IDX]*volume/amount-1)
                        if with_labels and not np.isfinite(label):
                            skipped_bad_label += 1
                        index = np.full(60, -1, np.int32)
                        tail = list(hist)
                        if tail:
                            index[-len(tail):] = tail
                        if (token_dates[np.maximum(index, 0)] >= decision.value).any():
                            raise AssertionError('Future token in current input')
                        if auction and token_dates[auction] >= decision.value:
                            raise AssertionError('Auction data not yet available')
                        histories.append(index); auctions.append(auction); references.append(reference)
                        sample_ids.append(instrument); sample_dates.append(decision.value)
                        if with_labels:
                            labels.append(label); label_ends.append(endpoint.value)
                if clock in BAR_CLOCKS:
                    pos = by_clock.get(clock)
                    hist.append(pos+1 if pos is not None else 0)
        tail = np.asarray(hist, np.int32)
        new_carry[instrument] = (token_values[tail].copy(), token_observed[tail].copy(), token_dates[tail].copy())
    data = {'token_values': token_values, 'token_observed': token_observed, 'token_date_ns': token_dates,
            'history_idx': np.stack(histories) if histories else np.empty((0,60), np.int32),
            'auction_idx': np.asarray(auctions, np.int32), 'ref_close': np.asarray(references, np.float32),
            'instrument_id': np.asarray(sample_ids, dtype=str), 'date_ns': np.asarray(sample_dates, np.int64)}
    if with_labels:
        data['labels'] = np.asarray(labels, np.float32)
        data['label_valid'] = np.isfinite(data['labels'])
        data['label_end_ns'] = np.asarray(label_ends, np.int64)
    audit.update(samples=len(sample_ids), requested=len(requested), skipped_no_reference=skipped_no_reference,
                 skipped_bad_label=skipped_bad_label, label_free=not with_labels,
                 observed_auction_rows=int(keys.date.dt.strftime('%H:%M').eq('09:25').sum()))
    return data, new_carry, audit

def load_month(folder: Path):
    fields = ("token_values", "token_observed", "token_date_ns", "history_idx", "auction_idx", "ref_close",
              "labels", "instrument_id", "date_ns", "label_end_ns")
    return {field: np.load(folder / f"{field}.npy", mmap_mode="r", allow_pickle=False)
            for field in fields}

def fit_scaler(folders):
    """Training-only original log-flow/label formula, accumulated in small chunks."""
    count = np.zeros(len(FLOW_IDX), np.float64)
    sums, sums2 = np.zeros_like(count), np.zeros_like(count)
    y_count, y_sum, y_sum2 = 0, 0., 0.
    for folder in folders:
        data = load_month(Path(folder))
        raw_rows = json.loads((Path(folder)/'audit.json').read_text(encoding='utf-8'))['rows']
        for start in range(1, raw_rows+1, 32768):
            stop = min(start+32768, raw_rows+1)
            values = np.asarray(data['token_values'][start:stop][:, FLOW_IDX], np.float64)
            observed = np.asarray(data['token_observed'][start:stop][:, FLOW_IDX], bool)
            z = np.log1p(np.where(observed, values, 0.))
            count += observed.sum(axis=0); sums += (z*observed).sum(axis=0); sums2 += (z*z*observed).sum(axis=0)
        y = np.asarray(data['labels'], np.float64)
        y = y[np.isfinite(y)]
        y_count += len(y); y_sum += float(y.sum()); y_sum2 += float(np.dot(y,y))
    if y_count == 0 or (count == 0).any():
        raise ValueError('Insufficient training observations for scaling')
    mean = sums/count
    variance = np.maximum(sums2/count-mean**2, 0.)
    return {'flow_mean':mean.tolist(), 'flow_scale':np.maximum(np.sqrt(variance),.05).tolist(),
            'auction_flow_mean':np.zeros(len(count)).tolist(), 'auction_flow_scale':np.ones(len(count)).tolist(),
            'target_mean':y_sum/y_count, 'target_scale':max(np.sqrt(max(y_sum2/y_count-(y_sum/y_count)**2,0.)),1e-5),
            'fitted_samples':y_count, 'fit_scope':'configured training months only; unique current-month rows, no carry/validation fit'}

def validate_scaler(scaler):
    """校验训练期统计；推理阶段不重新拟合 mean/std。"""
    for name in ('flow_mean', 'flow_scale', 'auction_flow_mean', 'auction_flow_scale'):
        array = np.asarray(scaler[name], dtype=np.float32)
        if array.shape != (len(FLOW_IDX),) or not np.isfinite(array).all():
            raise ValueError(f'Invalid saved normalization statistic: {name}')
        if name.endswith('scale') and not (array > 0).all():
            raise ValueError(f'Normalization scales must be positive: {name}')
    for name in ('target_mean', 'target_scale'):
        if not np.isfinite(float(scaler[name])):
            raise ValueError(f'Non-finite saved target statistic: {name}')
    if float(scaler['target_scale']) <= 0:
        raise ValueError('target_scale must be positive')

def compute_metrics(frame: pd.DataFrame):
    if frame.empty:
        raise ValueError("No evaluation rows")
    grouped = frame.groupby("date", sort=True)
    rows = []
    for date, part in grouped:
        if len(part) < 3 or part.label.nunique() < 2 or part.score.nunique() < 2:
            ic = np.nan
            pearson = np.nan
        else:
            ic = part.label.rank().corr(part.score.rank())
            pearson = part.label.corr(part.score)
        rows.append({"date": date, "samples": len(part), "rank_ic": ic, "pearson_ic": pearson})
    daily = pd.DataFrame(rows)
    eligible = daily.rank_ic.dropna()
    eligible_pearson = daily.pearson_ic.dropna()
    return {"samples": int(len(frame)), "unique_ids": int(frame.instrument_id.nunique()),
            "cross_sections": int(len(daily)), "eligible_cross_sections": int(len(eligible)),
            "rank_ic_mean": float(eligible.mean()) if len(eligible) else None,
            "pearson_ic_mean": float(eligible_pearson.mean()) if len(eligible_pearson) else None,
            "rank_ic_ir": float(eligible.mean() / eligible.std(ddof=1)) if len(eligible) > 1 and eligible.std(ddof=1) > 0 else None,
            "mse": float(np.mean((frame.score.to_numpy() - frame.label.to_numpy()) ** 2)),
            "mae": float(np.mean(np.abs(frame.score.to_numpy() - frame.label.to_numpy())))}, daily

def clean_frame(frame):
    """Same book sorting and 15:00 V*C repair as the preserved data builder."""
    frame=frame.copy()
    frame['date']=pd.to_datetime(frame.date)
    keys=['instrument_id','date']
    dup=frame.duplicated(keys,keep=False)
    if dup.any():
        unique=frame.loc[dup].drop_duplicates()
        if unique.duplicated(keys).any(): raise ValueError('Conflicting stock/time rows')
        frame=frame.drop_duplicates(keys)
    frame=frame.sort_values(keys,kind='stable').reset_index(drop=True)
    _sort_book(frame)
    end=frame.date.dt.hour.eq(15)&frame.date.dt.minute.eq(0)
    v=frame.volume.to_numpy(np.float64);c=frame.close.to_numpy(np.float64)
    a=frame.amount.to_numpy(np.float64);den=v*c
    ratio=np.divide(a,den,out=np.full(len(frame),np.nan),where=den>0)
    repair=end.to_numpy()&(v>0)&(c>0)&(~np.isfinite(ratio)|(ratio<.8)|(ratio>1.2))
    frame.loc[repair,'amount']=den[repair].astype('int64')
    values=frame[list(FEATURES)].to_numpy(dtype=np.float32)
    obs=np.isfinite(values)
    obs[:,PRICE_IDX]&=values[:,PRICE_IDX]>0
    obs[:,FLOW_IDX]&=values[:,FLOW_IDX]>=0
    values[~obs]=0
    return frame[['date','instrument_id']],values,obs,int(repair.sum())

def session_blocks(dates):
    dates=np.asarray(dates,dtype=np.int64)
    days=dates//DAY*DAY;minutes=(dates-days)//MINUTE
    valid=((minutes>570)&(minutes<=690))|((minutes>780)&(minutes<=900))
    start=np.where(minutes<=690,570,780)
    block=(minutes-start-1)//30
    starts=days+(start+block*30)*MINUTE
    return valid, starts, (minutes-start-block*30).astype('int32')

def block_arrays(frame,g,sc=None):
    """Label-free in-memory equivalent of the monthly disk builder, also used online."""
    keys,v,o,repairs=clean_frame(frame)
    ns=keys.date.to_numpy(dtype='datetime64[ns]').astype('int64')
    valid,starts,offset=session_blocks(ns);valid&=offset%g==0
    k=block_key(keys.instrument_id.to_numpy(np.int64)[valid],starts[valid])
    unique,inverse=np.unique(k,return_inverse=True);count=30//g
    values=np.zeros((len(unique)+1,count,25),np.float32)
    observed=np.zeros(values.shape,bool);dates=np.zeros(values.shape[:2],np.int64)
    positions=offset[valid]//g-1
    values[inverse+1,positions]=v[valid];observed[inverse+1,positions]=o[valid]
    dates[inverse+1,positions]=ns[valid]
    if sc is None:raise ValueError("Frozen training scaler is required")
    flows=(np.log1p(np.maximum(values[:,:,FLOW_IDX],0))-np.asarray(sc['flow_mean'],np.float32))/np.asarray(sc['flow_scale'],np.float32)
    values[:,:,FLOW_IDX]=np.where(observed[:,:,FLOW_IDX],flows,0)
    return {'keys':unique,'values':values,'observed':observed,'dates':dates}

def window_records(keys, carry, decision_set=None):
    """Replay the original grid with real timestamps, including boundary markers.

    Window start is the first grid bar's start, or the session boundary for an
    overnight/lunch marker. Original missing positions are never called bars.
    Returns observable windows; neither labels nor subsequent rows admit nodes.
    """
    frame=keys.sort_values(['instrument_id','date'],kind='stable')
    output=[];new={}
    for stock,part in frame.groupby('instrument_id',sort=True):
        hist=deque(carry.get(int(stock),[]),maxlen=60)
        for day,dp in part.groupby(part.date.dt.normalize(),sort=True):
            day_ns=day.value
            if hist:hist.append((day_ns+570*MINUTE,True))
            lookup={d.strftime('%H:%M'):d.value for d in dp.date}
            seen=dp.date.min().value
            clocks=decision_set or DECISIONS
            events=sorted(set(BAR_CLOCKS)|set(clocks)|{'13:00'})
            for clock in events:
                if clock=='13:00' and hist:hist.append((day_ns+780*MINUTE,True))
                at=day_ns+pd.Timedelta(clock+':00').value
                if clock in clocks and hist and seen<at:
                    first,boundary=hist[0]
                    output.append((int(stock),at,first if boundary else first-15*MINUTE,
                        len(hist),np.asarray([x[0] for x in hist],dtype=np.int64)))
                if clock in BAR_CLOCKS:hist.append((at,False))
        new[int(stock)]=list(hist)
    return output,new

class BlockStore:
    """One complete online graph; no project disk-cache dependency."""

    def get(self, index, device='cpu', grains=GRAINS):
        (month, j, date) = self.graphs[int(index)]
        d = self.legacy.data[month]
        w = self.windows[month]
        (lo, hi) = map(int, d['graph_offsets'][j:j + 2])
        n = hi - lo
        offsets = np.array(w['offsets'][lo:hi + 1])
        lengths = np.diff(offsets)
        L = int(lengths.max())
        grid = offsets[:-1, None] + np.arange(L)[None, :]
        starts = np.where(np.arange(L)[None, :] < lengths[:, None], w['starts'][grid.clip(max=len(w['starts']) - 1)], 0)
        ids = np.array(d['instrument_id'][lo:hi])
        bounds = np.array(w['bounds'][lo:hi])
        keys = block_key(ids[:, None], starts)
        bn = starts > 0
        refs = np.array(d['ref_close'][lo:hi])
        tokens = d
        ai = np.array(d['auction_idx'][lo:hi])
        ao = np.array(tokens['token_observed'][ai])[:, None]
        av = np.array(tokens['token_values'][ai])[:, None]
        normalized = normalize_fields(av, ao, refs, self.scaler, True)
        one = np.concatenate((normalized, ao.astype(np.float32), np.zeros_like(normalized), (ai > 0)[:, None, None].astype(np.float32)), 2)
        auction = np.zeros((n, 10, 76), np.float32)
        auction[:, -1] = one[:, 0]
        padding = np.zeros((n, 10), bool)
        padding[:, -1] = ai > 0
        out = {'auction': torch.from_numpy(auction), 'auction_padding': torch.from_numpy(padding), 'auction_available': torch.from_numpy((ai > 0) & ao[:, 0].any(1))}
        y = torch.from_numpy(np.array(d['labels'][lo:hi])) if 'labels' in d else torch.full((n,), float('nan'))
        label_mask = torch.from_numpy(np.array(d['label_valid'][lo:hi])) if 'label_valid' in d else torch.zeros(n, dtype=torch.bool)
        out['block_padding'] = torch.from_numpy(bn)
        out['block_starts'] = torch.from_numpy(starts)
        out['decision_ns'] = torch.full((n,), date, dtype=torch.int64)
        month_positions = []
        for bm in sorted(set(pd.to_datetime(np.unique(starts[bn])).strftime('%Y%m'))):
            begin = pd.Timestamp(bm + '01')
            end = begin + pd.offsets.MonthBegin()
            pos = bn & (starts >= begin.value) & (starts < end.value)
            month_positions.append((bm, pos))
        for g in grains:
            count = 30 // g
            values = obs = td = None
            if len(month_positions) > 1:
                values = np.zeros((n, L, count, 25), np.float32)
                obs = np.zeros(values.shape, bool)
                td = np.zeros((n, L, count), np.int64)
            for (bm, pos) in month_positions:
                cache = self.blocks(bm, g)
                wanted = keys[pos]
                ix = np.searchsorted(cache['keys'], wanted)
                ok = ix < len(cache['keys'])
                ok[ok] &= cache['keys'][ix[ok]] == wanted[ok]
                row = np.where(ok, ix + 1, 0)
                if len(month_positions) == 1:
                    rows = np.zeros((n, L), np.int64)
                    rows[pos] = row
                    values = np.asarray(cache['values'][rows])
                    obs = np.asarray(cache['observed'][rows])
                    td = np.asarray(cache['dates'][rows])
                else:
                    values[pos] = cache['values'][row]
                    obs[pos] = cache['observed'][row]
                    td[pos] = cache['dates'][row]
            visible = (td < date) & (td - g * MINUTE >= bounds[:, None, None]) & (td > 0)
            out[f'raw_{g}'] = torch.from_numpy(values)
            out[f'obs_{g}'] = torch.from_numpy(obs)
            out[f'present_{g}'] = torch.from_numpy(visible)
        out['refs'] = torch.from_numpy(refs)
        return ({k: v.to(device) for (k, v) in out.items()}, torch.from_numpy(ids).to(device), y.to(device), label_mask.to(device), date, month)

class MemoryLegacy:

    def __init__(self, data):
        self.data = data
        self.graphs = []
        for (m, d) in data.items():
            for (j, t) in enumerate(d['graph_dates']):
                self.graphs.append((m, j, int(t)))

class LiveStore(BlockStore):
    """One calendar month's requested graphs plus a bounded warm-history block cache."""
    def __init__(self,sc):
        self.scaler=sc
        self.carry={};self.grid_carry={};self.mapping={};self.memory_blocks={};self.days=[]
    def blocks(self,month,g):
        if (month,g) not in self.memory_blocks:
            raise RuntimeError('History block was not loaded; extend configured warm history: '+month)
        return self.memory_blocks[month,g]
    def add_month(self,frames,requests,month,grains,with_labels=False):
        self.graphs=[]
        if 15 not in frames:raise ValueError('bar15m is required for common node/history eligibility')
        for code in sorted(set(str(x) for f in frames.values() for x in f.instrument.unique())):
            if code not in self.mapping:self.mapping[code]=len(self.mapping)+1
        inv={v:k for k,v in self.mapping.items()};base=frames[15]
        node,self.carry,audit=build_month(base,self.carry,requests,with_labels=with_labels)
        numeric=base[['date','instrument']].rename(columns={'instrument':'instrument_id'}).copy()
        numeric['instrument_id']=numeric.instrument_id.astype(str).map(self.mapping)
        self.days.extend(pd.to_datetime(base.date).dt.normalize().unique().astype('datetime64[ns]').astype('int64').tolist())
        clocks=set(pd.to_datetime(requests.date).dt.strftime('%H:%M')) if len(requests) else set()
        records,self.grid_carry=window_records(numeric,self.grid_carry,clocks or None)
        for g,f in frames.items():
            f=f.rename(columns={'instrument':'instrument_id'}).copy()
            f['instrument_id']=f.instrument_id.astype(str).map(self.mapping)
            self.memory_blocks[month,g]=block_arrays(f,g,self.scaler)
        if node is None:return audit
        ids=np.asarray([self.mapping[str(s)] for s in node['instrument_id']],np.int32)
        order=np.lexsort((ids,node['date_ns']))
        for key in [k for k in ('history_idx','auction_idx','ref_close','date_ns','instrument_id','labels','label_valid','label_end_ns') if k in node]:node[key]=node[key][order]
        node['instrument_id']=ids[order]
        gd,counts=np.unique(node['date_ns'],return_counts=True)
        node['graph_dates']=gd;node['graph_offsets']=np.r_[0,np.cumsum(counts)]
        lookup={(stock,t):start for stock,t,start,_,_ in records}
        bounds=np.asarray([lookup[(int(s),int(t))] for s,t in zip(node['instrument_id'],node['date_ns'])],np.int64)
        days=np.unique(self.days);starts=np.sort(np.concatenate([days+x*MINUTE for x in (570,600,630,660,780,810,840,870)]))
        offsets=[0];ragged=[]
        for lo,t in zip(bounds,node['date_ns']):
            blocks=starts[(starts+30*MINUTE>lo)&(starts<t)]
            ragged.extend(blocks.tolist());offsets.append(len(ragged))
        self.legacy=MemoryLegacy({month:node});self.graphs=self.legacy.graphs
        self.windows={month:{'bounds':bounds,'offsets':np.asarray(offsets,np.int64),'starts':np.asarray(ragged,np.int64)}}
        self.inverse_ids=inv
        # Retain exactly the months required by the next complete 60-slot history.
        earliest=min([pd.Timestamp(bounds.min()).strftime('%Y%m')] if len(bounds) else [str(pd.Period(month,freq='M')-1).replace('-','')])
        for key in list(self.memory_blocks):
            if key[0]<earliest:del self.memory_blocks[key]
        return audit

# -------- 平台接口：参考G2_test，股票池日度与日内口径分别处理 --------
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def normalize_range(start_date, end_date):
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if len(str(end_date)) <= 10:
        end += pd.Timedelta(days=1)-pd.Timedelta(seconds=1)
    if start.tz is not None or end.tz is not None or start > end:
        raise ValueError('Expected ordered, naive Asia/Shanghai start/end times')
    return start, end

def safe_table(value):
    if not isinstance(value, str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', value) is None:
        raise ValueError('datasources must map logical names to SQL table identifiers')
    return value

def query_frame(sql, start, end):
    # DAI 由 BigQuant 环境提供，函数调用时才导入；单纯导入模型不访问平台。
    import dai  # Supplied by BigQuant; deliberately no local credentials/network fallback.
    result = dai.query(sql, filters={'date':[str(start),str(end)]}, compression=True)
    frame = result.df()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('dai.query(...).df() did not return a DataFrame')
    return frame

def pool(datasources, start, end):
    """将正式股票池转换为预测请求，兼容日度名单和日内精确名单。

    日度名单的 date 为00:00，只展开当日成员到既定八个预测点；不跨日填充，
    不根据未来行情/标签决定成员。已有日内名单直接保留，不能先压成日度名单。
    同一日期混合日度和日内定义、或出现不明时刻时明确报错并给出样例。
    """
    start, end = normalize_range(start, end)
    table = safe_table(datasources.get('instruments','cpt_jyc_2026_instruments'))
    # 请求可能从10:30开始，但日度名单存于00:00，查询必须覆盖整天后再筛预测时点。
    day_start = start.normalize()
    day_end = end.normalize()+pd.Timedelta(days=1)-pd.Timedelta(seconds=1)
    frame = query_frame(f'SELECT date, instrument FROM {table}', day_start, day_end).copy()
    if not {'date','instrument'}.issubset(frame):
        raise ValueError('Official pool requires date and instrument')
    frame['date'] = pd.to_datetime(frame.date, errors='raise')
    if frame[['date','instrument']].isna().any().any():
        raise ValueError('Null official pool key')
    if frame.date.dt.tz is not None:
        raise ValueError('Pool timestamps must be naive Asia/Shanghai local times')
    frame['instrument'] = frame.instrument.astype(str)
    frame = frame.loc[frame.date.between(day_start,day_end),['date','instrument']].drop_duplicates()
    if frame.empty:
        raise ValueError(f'Official stock pool {table} is empty in the requested calendar dates')
    daily = frame.date.eq(frame.date.dt.normalize())
    daily_days = set(frame.loc[daily,'date'])
    intraday_days = set(frame.loc[~daily,'date'].dt.normalize())
    overlap = daily_days & intraday_days
    if overlap:
        examples = [str(day.date()) for day in sorted(overlap)[:5]]
        raise ValueError(f'Pool mixes daily and intraday membership on the same date: {examples}')
    intraday = frame.loc[~daily].copy()
    legal = (intraday.date.eq(intraday.date.dt.floor('min'))
             & intraday.date.dt.strftime('%H:%M').isin(PREDICTION_CLOCKS))
    if not legal.all():
        examples = intraday.loc[~legal].head(8).astype({'date':str}).to_dict(orient='records')
        clocks = intraday.date.dt.strftime('%H:%M:%S').value_counts().head(12).to_dict()
        raise ValueError(f'Unrecognized pool times; expected daily 00:00 rows or prediction clocks '
                         f'{list(PREDICTION_CLOCKS)}. Observed clocks={clocks}; examples={examples}')
    parts = [intraday]
    for clock in PREDICTION_CLOCKS:
        expanded = frame.loc[daily].copy()
        expanded['date'] += pd.Timedelta(clock+':00')
        parts.append(expanded)
    result = pd.concat(parts,ignore_index=True)
    result = result.loc[result.date.between(start,end)].drop_duplicates(['date','instrument'])
    if result.empty:
        raise ValueError('No prediction clocks fall within the requested start/end range')
    # A missing whole cross-section cannot be hidden by only measuring existing rows.
    source_days = sorted(set(frame.date.dt.normalize()))
    expected_times = {day+pd.Timedelta(clock+':00') for day in source_days for clock in PREDICTION_CLOCKS
                      if start <= day+pd.Timedelta(clock+':00') <= end}
    missing_times = sorted(expected_times-set(result.date))
    if missing_times:
        raise ValueError(f'Official pool has missing prediction cross-sections: '
                         f'{[str(value) for value in missing_times[:12]]}')
    result = result.sort_values(['date','instrument']).reset_index(drop=True)
    result.attrs['pool_audit'] = {
        'table':table,'source_rows':len(frame),'source_days':len(source_days),
        'schema':'daily' if daily.all() else ('intraday' if not daily.any() else 'daily_and_intraday_on_different_dates'),
        'daily_membership_rows':int(daily.sum()),'intraday_membership_rows':int((~daily).sum()),
        'prediction_clocks':list(PREDICTION_CLOCKS),'requested_rows':len(result),
        'query_start':str(day_start),'query_end':str(day_end),
    }
    print(json.dumps({'event':'pool_ready',**result.attrs['pool_audit']},ensure_ascii=False),flush=True)
    return result

def month_ranges(start, end):
    for period in pd.period_range(start.to_period('M'),end.to_period('M'),freq='M'):
        lo, hi = max(start,period.start_time), min(end,(period+1).start_time-pd.Timedelta(seconds=1))
        yield str(period).replace('-',''), lo, hi

def select_device(config):
    name = os.environ.get('BQ_DEVICE', config.get('device','auto'))
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name not in ('cuda','cpu'):
        raise ValueError('device must be auto, cuda or cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use BQ_DEVICE=cpu explicitly')
    return torch.device(name)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(PROTOCOL["cpu_threads"])
    return {"seed":seed,"python":platform.python_version(),"numpy":np.__version__,
        "pandas":pd.__version__,"torch":torch.__version__,"cuda":torch.version.cuda,
        "cudnn":torch.backends.cudnn.version(),"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,"float32_matmul_precision":torch.get_float32_matmul_precision(),
        "python_executable":sys.executable,"platform":platform.platform(),
        "cublas_workspace_config":os.getenv("CUBLAS_WORKSPACE_CONFIG"),
        "dataloader":"none; independent NumPy Generator for the sample order"}

def graph_ids(instruments, device):
    values = np.asarray(instruments)
    if len(np.unique(values)) != len(values):
        raise ValueError('Duplicate instrument within a cross section')
    try:
        ids = values.astype(np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Numeric aliases')
    except (ValueError,TypeError):
        ids = pd.factorize(values.astype(str),sort=True)[0]
    return torch.as_tensor(ids,device=device,dtype=torch.int64)

def resolve_training_config(config=None, model_path=None):
    """Explicit config wins; otherwise read recipe metadata or use code defaults.

    Training dates are separate from main's evaluation dates. The organizer has
    not specified how private retraining dates will be injected; no inference
    date range is automatically repurposed as a training range.
    """
    if config is not None:
        resolved=default_training_config()
        resolved.update(dict(config))
    elif model_path is not None:
        resolved=load_config(model_path)
    elif MODEL_PATH.is_file():
        resolved=load_config(MODEL_PATH)
    else:
        resolved=default_training_config()
    validate_config(resolved)
    return resolved

# -------- 配置、JSON读写及真正独立的多粒度训练入口 --------
def _baseline_training_config():
    """独立训练配置；即使提交JSON缺失，也能从初始化开始训练。"""
    return {'variant':'MG3','seed':43,'model_file':'mg3_model.json',
        'input_grains_minutes':[1,5,15,30],'history_grid_bar_minutes':15,'history_bars':60,
        'features':list(FEATURES),'model_cfg':copy.deepcopy(DEFAULT_MODEL_CFG),
        'graph_width':32,'message_layers':1,'aggregation':'exclude_self_mean',
        'device':'auto','cpu_threads':4,'encoder_chunk_nodes':512,
        'history_calendar_days':32,'max_missing_fraction':.4,
        'train_start':'2020-01-01 00:00:00','train_end':'2022-12-31 23:59:59',
        'val_start':'2023-01-01 00:00:00','val_end':'2023-12-31 23:59:59',
        'test_start':'2024-01-01 00:00:00','test_end':'2024-12-31 23:59:59',
        'test_usage':'not used for training, scaler fitting or checkpoint selection',
        'train_decision_times':list(DECISIONS),'prediction_times':list(PREDICTION_CLOCKS),
        'target_horizon_minutes':30,'platform_entry_output':list(OUTPUT_COLUMNS),
        'online_platform_validated':False,'platform_field_units_verified':False,
        'judge_contract_revision':'G2_test reference, 2026-10-19',
        'public_evaluation':'main loads supplied weights and forwards injected tables/dates',
        'private_training':'explicit train_and_save from initialization; dates from config only',
        'private_training_date_injection_verified':False}

def validate_datasources(datasources):
    if not isinstance(datasources,Mapping):
        raise TypeError('Expected platform-injected datasources mapping')
    for grain in GRAINS:
        key=f'bar{grain}m'
        if not datasources.get(key):
            raise KeyError(f'MG3 requires datasources[{key!r}]; no development-table fallback')
        safe_table(datasources[key])

def _validate_baseline_config(config):
    expected={'variant':'MG3','input_grains_minutes':list(GRAINS),'history_grid_bar_minutes':15,
        'history_bars':60,'target_horizon_minutes':30,'features':list(FEATURES),
        'model_cfg':DEFAULT_MODEL_CFG,'graph_width':32,'message_layers':1,
        'aggregation':'exclude_self_mean','platform_entry_output':OUTPUT_COLUMNS,
        'train_decision_times':list(DECISIONS),'prediction_times':list(PREDICTION_CLOCKS)}
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError('MG3 configuration mismatch: '+key)
    for key in ('cpu_threads','encoder_chunk_nodes','history_calendar_days'):
        value=config.get(key)
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(key+' must be a positive integer')
    if not 0<=config['max_missing_fraction']<=.4:raise ValueError('Missing fraction must be within [0,.4]')
    if isinstance(config['seed'],bool) or not isinstance(config['seed'],int):raise ValueError('seed must be integer')
    if not 0<=config['seed']<2**32:raise ValueError('seed must be within [0, 2**32)')
    a,b=normalize_range(config['train_start'],config['train_end'])
    c,d=normalize_range(config['val_start'],config['val_end'])
    if any(pd.isna(x) for x in (a,b,c,d)) or b>=c or b.to_period('M')>=c.to_period('M'):
        raise ValueError('Train/validation must be ordered and split at a calendar-month boundary')
    if (config.get('test_start') is None)!=(config.get('test_end') is None):
        raise ValueError('Provide both test dates or set both to None')
    if config.get('test_start') is not None:
        e,_=normalize_range(config['test_start'],config['test_end'])
        if d>=e:raise ValueError('Test period must follow validation')





def bars_multi(datasources,start,end):
    validate_datasources(datasources);values={}
    for grain in GRAINS:
        table=safe_table(datasources[f'bar{grain}m'])
        fields=', '.join(['date','instrument',*FEATURES])
        frame=query_frame(f'SELECT {fields} FROM {table} ORDER BY instrument, date',start,end).copy()
        if not {'date','instrument',*FEATURES}.issubset(frame):raise ValueError(f'bar{grain}m missing fields')
        frame['date']=pd.to_datetime(frame.date,errors='raise')
        values[grain]=frame.loc[frame.date.between(start,end)].reset_index(drop=True)
    return values

def write_json(path,value):
    """独立临时文件及短重试；避免Windows读取期间的替换锁冲突。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
    for attempt in range(30):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.05)

def cache_identity(datasources,config):
    value={'datasources':dict(datasources),'dates':{k:config[k] for k in ('train_start','train_end','val_start','val_end')},
        'features':list(FEATURES),'grains':list(GRAINS),'training_times':list(DECISIONS),
        'builder_sha256':file_sha256(Path(__file__)),'preprocessing_version':PREPROCESSING_VERSION}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),value

def prepare_cache(datasources,config,cache_root):
    """逐月查询四粒度，保留无标签节点；仅训练期拟合15min流字段与目标统计。"""
    validate_config(config);validate_datasources(datasources);cache_root=Path(cache_root)
    identity,description=cache_identity(datasources,config);marker=cache_root/'prepared.json'
    if marker.is_file():
        prepared=read_json(marker)
        if prepared['identity']!=identity:raise ValueError('Cache identity mismatch; use a different cache_dir')
        for item in prepared['manifest']:
            if file_sha256(cache_root/item['path'])!=item['sha256']:raise ValueError('Cache content changed: '+item['path'])
        if file_sha256(cache_root/'scaler.json')!=prepared['scaler_sha256']:raise ValueError('Cached scaler changed')
        return prepared
    if cache_root.exists() and any(cache_root.iterdir()):raise FileExistsError('Incomplete cache exists; use another cache_dir')
    cache_root.mkdir(parents=True,exist_ok=True)
    train_start,train_end=normalize_range(config['train_start'],config['train_end'])
    val_start,val_end=normalize_range(config['val_start'],config['val_end'])
    unit={'flow_mean':[0.]*len(FLOW_IDX),'flow_scale':[1.]*len(FLOW_IDX)}
    store=LiveStore(unit);train=[];val=[];audits=[];all_months=[]
    for month,lo,hi in month_ranges(train_start,val_end):
        frames=bars_multi(datasources,lo,hi)
        requests=training_requests(frames[15])
        requests=requests.loc[requests.date.between(train_start,train_end)|requests.date.between(val_start,val_end)]
        audit=store.add_month(frames,requests,month,GRAINS,with_labels=True)
        folder=cache_root/month;folder.mkdir()
        node=store.legacy.data[month]
        node['instrument_code']=np.asarray([store.inverse_ids[int(x)] for x in node['instrument_id']],dtype=str)
        for name,array in node.items():np.save(folder/(name+'.npy'),array,allow_pickle=False)
        write_json(folder/'audit.json',audit)
        window=folder/'windows';window.mkdir()
        for name,array in store.windows[month].items():np.save(window/(name+'.npy'),array,allow_pickle=False)
        for grain in GRAINS:
            blocks=store.memory_blocks[month,grain];target=cache_root/'blocks'/month/str(grain);target.mkdir(parents=True)
            for name,array in blocks.items():np.save(target/(name+'.npy'),array,allow_pickle=False)
        has_train=bool(requests.date.between(train_start,train_end).any())
        has_val=bool(requests.date.between(val_start,val_end).any())
        if has_train and has_val:raise ValueError('Monthly train/validation overlap')
        if has_train:train.append(month)
        if has_val:val.append(month)
        audits.append({'month':month,**audit});all_months.append(month)
        print(json.dumps({'event':'prepared_month','month':month,'nodes':len(node['date_ns']),
            'labelled_nodes':int(node['label_valid'].sum()),'grains':list(GRAINS)}),flush=True)
        del frames,node;gc.collect()
    if not train or not val:raise ValueError('No training or validation months')
    scaler=fit_scaler([cache_root/m for m in train]);validate_scaler(scaler)
    write_json(cache_root/'scaler.json',scaler)
    # First-pass blocks contain log1p(raw flow); finalize with train-only stats.
    mean=np.asarray(scaler['flow_mean'],np.float32);scale=np.asarray(scaler['flow_scale'],np.float32)
    for month in all_months:
        for grain in GRAINS:
            base=cache_root/'blocks'/month/str(grain)
            values=np.load(base/'values.npy',mmap_mode='r+');observed=np.load(base/'observed.npy',mmap_mode='r')
            for lo in range(0,len(values),4096):
                hi=min(lo+4096,len(values))
                part=values[lo:hi]
                part[:,:,FLOW_IDX]=np.where(observed[lo:hi,:,FLOW_IDX],(part[:,:,FLOW_IDX]-mean)/scale,0)
            values.flush();del values,observed
    manifest=[{'path':p.relative_to(cache_root).as_posix(),'sha256':file_sha256(p)}
        for p in sorted(cache_root.rglob('*')) if p.is_file()]
    prepared={'identity':identity,'description':description,'train_months':train,'val_months':val,
        'all_months':all_months,'audits':audits,'manifest':manifest,'scaler_sha256':file_sha256(cache_root/'scaler.json')}
    write_json(marker,prepared);return prepared

class CachedBlockStore(BlockStore):
    """训练缓存只保留少量mmap句柄；一张图仍是一个完整股票截面。"""
    def __init__(self,folders):
        self.folders={Path(f).name:Path(f) for f in folders}
        roots={f.parent for f in self.folders.values()}
        if len(roots)!=1:raise ValueError('Folders must share one cache root')
        self.root=roots.pop();self.graphs=[];self.windows={};self.legacy=SimpleNamespace(data={})
        self.metadata=OrderedDict();self.block_cache=OrderedDict()
        for month,folder in self.folders.items():
            for j,date in enumerate(np.load(folder/'graph_dates.npy',mmap_mode='r')):
                self.graphs.append((month,j,int(date)))
    def load_metadata(self,month):
        if month not in self.metadata:
            if len(self.metadata)>=2:
                prior,_=self.metadata.popitem(last=False);self.legacy.data.pop(prior);self.windows.pop(prior)
            folder=self.folders[month]
            data={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in folder.glob('*.npy')}
            self.legacy.data[month]=data
            self.windows[month]={p.stem:np.load(p,mmap_mode='r') for p in (folder/'windows').glob('*.npy')}
            self.metadata[month]=True
        self.metadata.move_to_end(month)
    def blocks(self,month,grain):
        key=(month,grain)
        if key not in self.block_cache:
            if len(self.block_cache)>=12:self.block_cache.popitem(last=False)
            base=self.root/'blocks'/month/str(grain)
            self.block_cache[key]={n:np.load(base/(n+'.npy'),mmap_mode='r') for n in ('keys','values','observed','dates')}
        self.block_cache.move_to_end(key);return self.block_cache[key]

class FullSectionStore:
    def __init__(self,folders):
        self.folders=[Path(f) for f in folders];self.reader=CachedBlockStore(self.folders)
        self.graphs=[(self.reader.folders[m],date) for m,j,date in self.reader.graphs]
    def get(self,index,scaler,device):
        month,j,date=self.reader.graphs[int(index)];self.reader.load_metadata(month);self.reader.scaler=scaler
        batch,_,labels,valid,_,_=self.reader.get(index,device,GRAINS)
        data=self.reader.legacy.data[month];lo,hi=map(int,data['graph_offsets'][j:j+2])
        instruments=np.asarray(data['instrument_code'][lo:hi])
        return batch,graph_ids(instruments,device),labels,valid,date,instruments

def train_and_save(datasources,output_path=None,cache_dir=None,*,config=None,model_path=None,work_dir=None):
    """显式重训入口；从初始化训练，main的评测日期不会被用作训练日期。"""
    validate_datasources(datasources);config=resolve_training_config(config,model_path)
    def absolute_path(value):
        value=Path(value);return value if value.is_absolute() else HERE/value
    work=absolute_path(work_dir or 'work');key,_=cache_identity(datasources,config)
    cache=absolute_path(cache_dir) if cache_dir is not None else work/('training_cache_'+key[:16])
    prepared=prepare_cache(datasources,config,cache);scaler=read_json(cache/'scaler.json')
    if file_sha256(cache/'scaler.json')!=prepared['scaler_sha256']:raise ValueError('Cached scaler changed')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f');run_dir=work/f'train_{stamp}'
    checkpoint,summary=run_training([cache/m for m in prepared['train_months']],
        [cache/m for m in prepared['val_months']],scaler,config,run_dir)
    write_json(run_dir/'data_identity.json',{'cache_identity':prepared['identity'],'description':prepared['description'],
        'scaler_sha256':prepared['scaler_sha256'],'train_months':prepared['train_months'],'val_months':prepared['val_months']})
    target=absolute_path(output_path or config['model_file']);target.parent.mkdir(parents=True,exist_ok=True)
    staged=target.with_name(target.name+'.'+stamp+'.tmp');shutil.copy2(checkpoint,staged);load_model(staged,'cpu')
    if target.exists():shutil.copy2(target,target.with_name(target.stem+'.previous_'+stamp+target.suffix))
    staged.replace(target)
    return {'model_path':str(target),'run_dir':str(run_dir),'summary':summary}

def evaluate(model, folders, scaler, device, keep=False, limit=None):
    store = folders if isinstance(folders,FullSectionStore) else FullSectionStore(folders)
    previous = model.training
    model.eval(); frames = []; started = time.perf_counter()
    try:
        with torch.inference_mode():
            for i in range(min(len(store.graphs),limit or len(store.graphs))):
                batch,ids,y,valid,date,instruments = store.get(i,scaler,device)
                value = model(batch,ids).cpu().numpy()*scaler['target_scale']+scaler['target_mean']
                frames.append(pd.DataFrame({'date':pd.Timestamp(date),'instrument_id':instruments,
                    'label':y.cpu().numpy(),'label_valid':valid.cpu().numpy(),'score':value}))
        frame = pd.concat(frames,ignore_index=True)
        metrics, points = compute_metrics(frame.loc[frame.label_valid])
        metrics['inference_seconds'] = time.perf_counter()-started
        return metrics,points,frame if keep else None
    finally:
        model.train(previous)

def run_training(train_folders, val_folders, scaler, config, run_dir, smoke_steps=None):
    """Explicit retraining from initialization, never called by prediction main."""
    validate_config(config); validate_scaler(scaler)
    run_dir = Path(run_dir); run_dir.mkdir(parents=True,exist_ok=False)
    device = select_device(config)
    environment = configure_runtime(config['seed'])
    model = new_model(config).to(device)
    train, val = FullSectionStore(train_folders), FullSectionStore(val_folders)
    if not train.graphs or not val.graphs:
        raise ValueError('Training and validation need full cross sections')
    if max(d for _,d in train.graphs) >= min(d for _,d in val.graphs):
        raise ValueError('Train and validation graph dates overlap')
    for folder in train.folders:
        ends = np.load(folder/'label_end_ns.npy',mmap_mode='r')
        if int(ends.max()) >= min(d for _,d in val.graphs):
            raise ValueError('Training label horizon crosses validation boundary')
    optimizer = torch.optim.AdamW(model.parameters(),lr=PROTOCOL['learning_rate'],weight_decay=PROTOCOL['weight_decay'])
    rng = np.random.default_rng(config['seed'])
    count = len(train.graphs) if smoke_steps is None else min(int(smoke_steps),len(train.graphs))
    if count<1: raise ValueError('smoke_steps must be positive')
    schedule = sorted({int(math.ceil(count*f)) for f in PROTOCOL['validation_fractions']})
    best,stale,step,history = -math.inf,0,0,[]
    started = time.perf_counter(); stop_reason = 'max_epochs'
    provenance = {'seed':config['seed'],'training_decision_times':list(DECISIONS),
        'train_start':config['train_start'],'train_end':config['train_end'],
        'val_start':config['val_start'],'val_end':config['val_end'],
        'source':'explicit platform retraining; full cross section per update',
        'smoke_only':smoke_steps is not None,'online_platform_validated':False,
        'official_score':False,'model_source_sha256':file_sha256(Path(__file__))}
    write_json(run_dir/'config.json',{'config':config,'environment':environment,'protocol':PROTOCOL})
    checkpoint = run_dir/'best_model.json'
    for epoch in range(1,(1 if smoke_steps is not None else PROTOCOL['max_epochs'])+1):
        model.train()
        order = rng.permutation(len(train.graphs))[:count]
        write_json(run_dir/f'order_epoch{epoch}.json',[str(pd.Timestamp(train.graphs[int(i)][1])) for i in order])
        for position,index in enumerate(order,1):
            batch,ids,y,mask,date,instruments = train.get(index,scaler,device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch,ids)
            if bool(mask.any()):
                target = (y[mask]-scaler['target_mean'])/scaler['target_scale']
                loss = F.smooth_l1_loss(prediction[mask],target,beta=1.0)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
                loss.backward(); optimizer.step(); step+=1
            if position in schedule:
                metrics,_,_ = evaluate(model,val,scaler,device,limit=2 if smoke_steps is not None else None)
                value = metrics['rank_ic_mean']
                improved = value is not None and np.isfinite(value) and value>best
                row = {'epoch_fraction':epoch-1+position/count,'global_step':step,
                       'improved':bool(improved),'val_rank_ic_mean':value}
                history.append(row)
                if improved:
                    best=value; stale=0
                    provenance.update(best_epoch_fraction=row['epoch_fraction'],best_global_step=step,
                                      research_val_rank_ic=float(value))
                    save_model(checkpoint,model,scaler,provenance,config)
                else: stale+=1
                pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False)
                print(json.dumps(row),flush=True)
                if stale>=PROTOCOL['patience_validations']:
                    stop_reason='early_stop_8_validations'; break
        if stale>=PROTOCOL['patience_validations']: break
    if not checkpoint.is_file(): raise RuntimeError('No valid validation checkpoint')
    restored,_,saved=load_model(checkpoint,device)
    metrics,points,predictions=evaluate(restored,val,scaler,device,keep=True,limit=2 if smoke_steps is not None else None)
    if abs(metrics['rank_ic_mean']-best)>1e-7: raise AssertionError('Checkpoint reload metric mismatch')
    points.to_csv(run_dir/'validation_ic.csv',index=False)
    predictions.to_pickle(run_dir/'validation_predictions.pkl')
    summary={'variant':'MG3','model_type':EXPECTED_MODEL_TYPE,'seed':config['seed'],'smoke_only':smoke_steps is not None,
        'best_epoch_fraction':saved['provenance']['best_epoch_fraction'],'actual_updates':step,
        'training_seconds':time.perf_counter()-started,'stop_reason':stop_reason,**metrics}
    write_json(run_dir/'summary.json',summary)
    return checkpoint,summary



# -------- 原始位置编码、因果时间注意力和 T1 模型 --------
"""Portable architecture metadata; importing this module performs no I/O."""
MODEL_TYPES=('MG3_LSTM','MG3_Transformer_T1')
T1=dict(d_model=32,nhead=4,num_layers=1,dim_feedforward=64,dropout=0.,activation='gelu',batch_first=True,
    norm_first=True,final_layer_norm=32,enable_nested_tensor=False,causal_mask=True,
    positional_encoding='fixed sinusoidal at ORIGINAL block positions before stable valid-token compaction',
    empty_history='zero 32d summary, same as original LSTM')

"""MG3 with its original LSTM or one explicitly causal temporal Transformer."""
import math
import torch
from torch import nn


class CausalTemporalTransformer(nn.Module):
    """Operate along each stock's history, never across stocks.

    Add original-position sinusoids BEFORE compaction. Only nonempty rows enter
    attention, avoiding all-masked Softmax. True masks blocked attention/keys.
    """
    def __init__(self):
        super().__init__()
        self.input_projection=nn.Linear(24,32)
        layer=nn.TransformerEncoderLayer(d_model=32,nhead=4,dim_feedforward=64,dropout=0.,
            activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=1,norm=nn.LayerNorm(32),enable_nested_tensor=False)

    @staticmethod
    def positions(length,device,dtype):
        positions=torch.arange(length,device=device,dtype=torch.float32)[:,None]
        frequency=torch.exp(torch.arange(0,32,2,device=device,dtype=torch.float32)*(-math.log(10000.)/32))
        encoding=torch.zeros((length,32),device=device,dtype=torch.float32)
        encoding[:,0::2]=torch.sin(positions*frequency);encoding[:,1::2]=torch.cos(positions*frequency)
        return encoding.to(dtype)

    def encode(self,z,valid,return_sequence=False):
        if z.ndim!=3 or z.shape[-1]!=24 or valid.shape!=z.shape[:2] or valid.dtype!=torch.bool:
            raise ValueError('Expected z [N,L,24], valid bool [N,L]')
        n,length,_=z.shape
        hidden=z.new_zeros((n,32))
        sequence=z.new_zeros((n,length,32))
        lengths=valid.sum(1);active=torch.where(lengths>0)[0]
        if len(active):
            clean=z.masked_fill(~valid[:,:,None],0)
            tokens=self.input_projection(clean)+self.positions(length,z.device,z.dtype)[None]
            # Stable sort preserves actual token order and its original PE indices.
            order=torch.argsort((~valid[active]).to(torch.int32),dim=1,stable=True)
            max_length=int(lengths[active].max())
            selected=order[:,:max_length]
            compact=tokens[active].gather(1,selected[:,:,None].expand(-1,-1,32))
            padding=torch.arange(max_length,device=z.device)[None]>=lengths[active,None]
            compact=compact.masked_fill(padding[:,:,None],0)
            causal=torch.ones((max_length,max_length),dtype=torch.bool,device=z.device).triu(1)
            output=self.encoder(compact,mask=causal,src_key_padding_mask=padding)
            last=output[torch.arange(len(active),device=z.device),lengths[active]-1]
            hidden=hidden.index_copy(0,active,last)
            if return_sequence:
                restored=torch.zeros_like(tokens[active]).scatter(1,selected[:,:,None].expand(-1,-1,32),output.masked_fill(padding[:,:,None],0))
                sequence=sequence.index_copy(0,active,restored)
        return (hidden,sequence) if return_sequence else hidden

    def forward(self,z,valid):return self.encode(z,valid)


class TransformerTemporal(nn.Module):
    def __init__(self,original,seed):
        super().__init__()
        # Reuse only fresh same-seed modules, never pretrained weights.
        for name in ('continuous','auction','head','grain','gate'):
            setattr(self,name,getattr(original,name))
        self.grains=original.grains
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+9137)
            self.transformer=CausalTemporalTransformer()

    def representation(self,batch):
        emb=[];available=[]
        for g in self.grains:
            e,m=self.grain[str(g)](batch[f'raw_{g}'],batch[f'obs_{g}'],batch[f'present_{g}'],batch['refs'])
            emb.append(e);available.append(m)
        stack=torch.stack(emb,2);avail=torch.stack(available,2)&batch['block_padding'][:,:,None]
        gate_input=torch.cat((stack.flatten(2),avail.to(stack.dtype)),2)
        logits=(self.gate(gate_input) if self.gate is not None else torch.zeros_like(avail,dtype=stack.dtype)).masked_fill(~avail,-1e30)
        weights=logits.softmax(2)*avail
        weights=weights/weights.sum(2,keepdim=True).clamp_min(1)
        valid=avail.any(2)
        fused=(stack*weights[:,:,:,None]).sum(2)
        z=self.continuous(fused,valid)
        n,length,_=z.shape
        pos=torch.arange(length,device=z.device)[None].expand(n,-1)
        last=pos.masked_fill(~valid,-1).max(1).values.clamp_min(0)
        cnn=z[torch.arange(n,device=z.device),last].masked_fill(~valid.any(1)[:,None],0)
        hidden=self.transformer(z,valid)
        a=self.auction(batch['auction'],batch['auction_padding'])[:,-1]
        a=a.masked_fill(~batch['auction_available'][:,None],0)
        u=torch.cat((cnn,hidden,a,batch['auction_available'][:,None].to(z.dtype)),1)
        if u.shape!=(n,69):raise AssertionError('Fusion must remain [N,69]')
        den=valid.sum(1).clamp_min(1)[:,None]
        alpha=(weights*valid[:,:,None]).sum(1)/den
        return u,alpha,weights


class MG3Model(MultiGranModel):
    def __init__(self,model_type,seed=43,chunk_nodes=512):
        if model_type not in MODEL_TYPES:raise ValueError(model_type)
        # Fully construct the original fresh MG3 first, aligning ALL common tensors.
        super().__init__('MG3',chunk_nodes=chunk_nodes,activation_checkpointing=False)
        self.model_type=model_type;self.seed=seed
        if model_type=='MG3_Transformer_T1':self.temporal=TransformerTemporal(self.temporal,seed)


def configure_runtime(seed):
    # This CUDA build uses numerically different fused encoder kernels in
    # no-grad evaluation. Keep one attention path for training and inference.
    # This setting has no effect on the original MG3 LSTM/CNN operations.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    torch.set_float32_matmul_precision('highest')
    environment=seed_everything(seed)
    environment.update(transformer_nested_tensor=False,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        mha_fastpath_enabled=torch.backends.mha.get_fastpath_enabled())
    return environment


def make_model(model_type,seed=43,device='cpu',chunk_nodes=512):
    configure_runtime(seed)
    return MG3Model(model_type,seed,chunk_nodes).to(device)


def architecture(model_type):
    cfg=dict(DEFAULT_MODEL_CFG,model_type=model_type,sequence_encoder='LSTM' if model_type=='MG3_LSTM' else 'TransformerEncoder')
    if model_type=='MG3_Transformer_T1':
        cfg.pop('lstm_hidden');cfg.pop('lstm_bidirectional');cfg['transformer']=T1
    return cfg


# -------- Transformer 专用配置和 JSON 权重；拒绝误载 LSTM --------
FORMAT='mg3_temporal_comparison_json_v1'
EXPECTED_MODEL_TYPE='MG3_Transformer_T1'
DEFAULT_MODEL_PATH=MODEL_PATH
PROTOCOL=dict(PROTOCOL,model_type=EXPECTED_MODEL_TYPE,
    readout='CNN last-valid 24 + causal Transformer last-valid 32 + auction 12 + flag 1 = 69',
    sequence_encoder='TransformerEncoder',transformer=T1,output_columns=OUTPUT_COLUMNS)

def default_training_config():
    """显式独立重训的配置；main 不使用评测日期来重训。"""
    config=_baseline_training_config()
    config.update(model_type=EXPECTED_MODEL_TYPE,model_cfg=architecture(EXPECTED_MODEL_TYPE),
        model_file=MODEL_PATH.name,private_training='explicit train_and_save from initialization')
    return config

def validate_config(config):
    if config.get('model_type')!=EXPECTED_MODEL_TYPE:
        raise ValueError('This package accepts MG3_Transformer_T1 only')
    if config.get('model_cfg')!=architecture(EXPECTED_MODEL_TYPE):
        raise ValueError('Transformer architecture differs from the saved configuration')
    _validate_baseline_config(dict(config,model_cfg=DEFAULT_MODEL_CFG))

def load_config(model_path=None,expected_model_type=EXPECTED_MODEL_TYPE):
    artifact=read_json(model_path or MODEL_PATH)
    if expected_model_type!=EXPECTED_MODEL_TYPE or artifact.get('model_type')!=EXPECTED_MODEL_TYPE:
        raise ValueError('Model type mismatch: LSTM weights are not Transformer weights')
    if artifact.get('format')!=FORMAT or artifact.get('architecture')!=architecture(EXPECTED_MODEL_TYPE):
        raise ValueError('Unsupported artifact or architecture mismatch')
    config=dict(artifact['submission_config']);validate_config(config)
    if artifact['provenance']['seed']!=config['seed']:raise ValueError('Seed mismatch')
    validate_scaler(artifact['scaler'])
    return config

def new_model(config):
    """从初始化构建 T1，绝不读取训练过的 LSTM 权重。"""
    validate_config(config)
    return make_model(EXPECTED_MODEL_TYPE,config['seed'],'cpu',config['encoder_chunk_nodes'])

def save_model(path,model,scaler,provenance,submission_config=None):
    config=dict(submission_config or default_training_config())
    config.update(model_file=Path(path).name)
    validate_config(config);validate_scaler(scaler)
    if model.model_type!=EXPECTED_MODEL_TYPE:raise ValueError('Expected T1 model')
    if provenance['seed']!=config['seed']:raise ValueError('Seed mismatch')
    tensors={}
    for name,tensor in model.state_dict().items():
        value=tensor.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(value).all():raise ValueError('Nonfinite weight: '+name)
        tensors[name]=dict(shape=list(value.shape),dtype='float32',data=value.reshape(-1).tolist(),
            sha256=hashlib.sha256(value.tobytes()).hexdigest())
    artifact=dict(format=FORMAT,model_type=EXPECTED_MODEL_TYPE,architecture=architecture(EXPECTED_MODEL_TYPE),
        submission_config=config,scaler=scaler,provenance=dict(provenance,model_type=EXPECTED_MODEL_TYPE),
        state_dict=tensors,features=list(FEATURES),preprocessing_version=PREPROCESSING_VERSION,
        target=PROTOCOL['label'])
    write_json(path,artifact)
    return artifact

def load_model(path=MODEL_PATH,map_location='cpu',expected_model_type=EXPECTED_MODEL_TYPE):
    """严格核对模型类型、结构、scaler 和每个张量的 SHA-256。"""
    config=load_config(path,expected_model_type);artifact=read_json(path)
    if (artifact.get('features')!=list(FEATURES)
        or artifact.get('preprocessing_version')!=PREPROCESSING_VERSION
        or artifact.get('target')!=PROTOCOL['label']):raise ValueError('Preprocessing/target identity mismatch')
    model=new_model(config);state={}
    for name,record in artifact['state_dict'].items():
        if record['dtype']!='float32':raise ValueError('Unexpected tensor dtype')
        value=np.asarray(record['data'],np.float32).reshape(record['shape'])
        if not np.isfinite(value).all() or hashlib.sha256(value.tobytes()).hexdigest()!=record['sha256']:
            raise ValueError('Corrupt weight: '+name)
        state[name]=torch.from_numpy(value.copy())
    model.load_state_dict(state,strict=True);model.to(map_location).eval()
    return model,artifact['scaler'],artifact

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Explicit T1 training; never called by prediction main')
    parser.add_argument('--datasources-json',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--config-json',type=Path)
    args=parser.parse_args()
    print(json.dumps(train_and_save(read_json(args.datasources_json),args.output,args.cache_dir,
        config=read_json(args.config_json) if args.config_json else None,work_dir=args.work_dir),ensure_ascii=False,indent=2))
