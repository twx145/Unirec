import argparse, json, os, subprocess, numpy as np
from utils.config import load_config

def fit(xs, ys):
    X=np.stack([np.ones_like(xs), np.log10(xs)], axis=1)
    beta=np.linalg.lstsq(X, ys, rcond=None)[0]
    return float(beta[0]), float(beta[1])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--configs', nargs='+', required=True); ap.add_argument('--run_train', action='store_true'); args=ap.parse_args()
    res=[]
    for c in args.configs:
        cfg=load_config(c); name=cfg.get('name', os.path.splitext(os.path.basename(c))[0]); hist=os.path.join(cfg['train']['output_dir'],name,'history.json')
        if args.run_train and not os.path.exists(hist): subprocess.run(['python','train.py','--config',c], check=True)
        with open(hist,'r',encoding='utf-8') as f: h=json.load(f)
        best=max((x for x in h if x.get('auc')==x.get('auc')), key=lambda x:x['auc'])
        proxy=float(cfg['model']['d_model'])*float(cfg['model']['n_layers'])*float(cfg['model']['num_queries'])
        res.append({'config':c,'params_proxy':proxy,'auc':best['auc']})
    xs=np.asarray([x['params_proxy'] for x in res],dtype=float); ys=np.asarray([x['auc'] for x in res],dtype=float)
    a,b=fit(xs,ys); print(json.dumps({'results':res,'fit':{'intercept':a,'slope_log10_params':b}}, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
