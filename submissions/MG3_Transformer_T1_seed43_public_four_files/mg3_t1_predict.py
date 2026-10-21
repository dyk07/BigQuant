# -*- coding: utf-8 -*-
"""MG3 G2_test风格推理：main返回且仅返回date、instrument、factor。

四文件同目录上传；只读取已训练JSON，不拟合标准化、不更新权重、不自动重训。
使用平台原样注入的四粒度表映射与评测日期。默认device=auto，可用BQ_DEVICE覆盖。
"""
from __future__ import annotations
import mg3_t1_model as shared  # 先配置MKL，再导入数值库。
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

LAST_DIAGNOSTICS=None


def predict_range(datasources,start_date,end_date,model_path=None):
    global LAST_DIAGNOSTICS
    LAST_DIAGNOSTICS=None;shared.validate_datasources(datasources);started=time.perf_counter()
    path=Path(model_path) if model_path is not None else shared.DEFAULT_MODEL_PATH
    if not path.is_absolute():path=shared.HERE/path
    config=shared.load_config(path);torch.set_num_threads(config['cpu_threads'])
    device=shared.select_device(config);model,scaler,artifact=shared.load_model(path,device)
    if artifact['provenance'].get('smoke_only',False):raise ValueError('Smoke checkpoints cannot be submitted')
    start,end=shared.normalize_range(start_date,end_date);requested=shared.pool(datasources,start,end)
    pool_audit=dict(requested.attrs.get('pool_audit',{}))
    warm=start.normalize()-pd.Timedelta(days=config['history_calendar_days'])
    store=shared.LiveStore(scaler);outputs=[];audits=[];coverage=[]
    with torch.inference_mode():
        for month,lo,hi in shared.month_ranges(warm,end):
            frames=shared.bars_multi(datasources,lo,hi)
            requests=requested.loc[requested.date.dt.strftime('%Y%m').eq(month)]
            audit=store.add_month(frames,requests,month,shared.GRAINS);audits.append(dict(audit,month=month))
            graphs={pd.Timestamp(date):i for i,(_,_,date) in enumerate(store.graphs)}
            for date,req in requests.groupby('date',sort=True):
                if date not in graphs:raise RuntimeError('Missing whole prediction cross section: '+str(date))
                batch,ids,_,_,_,_=store.get(graphs[date],device,shared.GRAINS)
                instruments=np.asarray([store.inverse_ids[int(i)] for i in ids.cpu().numpy()],dtype=str)
                excluded=sorted(set(req.instrument.astype(str))-set(instruments));missing=len(excluded)/len(req)
                coverage.append({'date':str(date),'requested':len(req),'eligible_nodes':len(instruments),
                    'missing_fraction':missing,'excluded_without_asof_input':excluded})
                if missing>config['max_missing_fraction'] or not len(instruments):raise RuntimeError(f'Insufficient as-of input at {date}: {missing:.2%} missing')
                if set(instruments)-set(req.instrument.astype(str)):raise AssertionError('Foreign stock in graph')
                score=model(batch,shared.graph_ids(instruments,device)).cpu().numpy()
                factor=score*scaler['target_scale']+scaler['target_mean']
                if not np.isfinite(factor).all():raise RuntimeError('Non-finite prediction')
                outputs.append(pd.DataFrame({'date':date,'instrument':instruments,'factor':factor.astype(np.float64)}))
    if len(outputs)!=requested.date.nunique():raise RuntimeError('Missing whole prediction cross section')
    result=pd.concat(outputs,ignore_index=True).sort_values(['date','instrument']).reset_index(drop=True)
    if result.duplicated(['date','instrument']).any() or not result.date.between(start,end).all():raise AssertionError('Invalid output keys')
    if len(result)!=sum(c['eligible_nodes'] for c in coverage):raise AssertionError('Graph output count mismatch')
    result=result[shared.OUTPUT_COLUMNS]
    LAST_DIAGNOSTICS={'model_type':artifact['model_type'],'seed':artifact['provenance']['seed'],'rows':len(result),
        'prediction_times':len(outputs),'coverage':coverage,'input_audits':audits,'pool_audit':pool_audit,
        'elapsed_seconds':time.perf_counter()-started,'device':str(device),'input_grains_minutes':list(shared.GRAINS),
        'full_cross_section_aggregation':True,'label_free_inference':True,
        'training_or_scaler_fit_during_inference':False,'online_platform_validated':False,
        'training_validated_clocks':list(shared.DECISIONS),'output_columns':list(result.columns)}
    return result


def main(datasources,start_date,end_date):
    """平台三参数入口，按G2_test约定返回factor。"""
    return predict_range(datasources,start_date,end_date)

def evaluate_platform(factor_data, show=True, *, start_date=None, end_date=None):
    """Forward the same factor-only frame and optional dates as the official judge."""
    if not isinstance(factor_data,pd.DataFrame) or list(factor_data.columns)!=shared.OUTPUT_COLUMNS:
        raise ValueError('Expected exactly date, instrument, factor')
    if (start_date is None)!=(end_date is None):
        raise ValueError('Provide both evaluation start_date and end_date')
    kwargs={'factor_data':factor_data.copy(deep=True),'show':show}
    if start_date is not None:
        shared.normalize_range(start_date,end_date)
        kwargs.update(start_date=start_date,end_date=end_date)
    from bigmodule import M
    return M.jyc_eval._latest(**kwargs)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasources-json',type=Path,required=True)
    parser.add_argument('--start-date',required=True)
    parser.add_argument('--end-date',required=True)
    parser.add_argument('--model-path',type=Path,default=shared.DEFAULT_MODEL_PATH)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    result=predict_range(shared.read_json(args.datasources_json),args.start_date,args.end_date,args.model_path)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    if args.evaluate: evaluate_platform(result,start_date=args.start_date,end_date=args.end_date)
