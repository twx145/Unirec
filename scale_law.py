"""
Scaling Law 实验 + 多模型对比
======================

用法 1 — 同一模型不同规模:
python scale_law.py --configs configs/small.yaml configs/base.yaml configs/large.yaml

用法 2 — 不同模型相同规模 (横向对比):
python scale_law.py --configs configs/base.yaml configs/interformer.yaml configs/onetrans.yaml configs/hyformer.yaml

加 --run_train 可自动训练尚未有结果的配置。
"""

import argparse
import json
import os
import subprocess

import numpy as np
from utils.config import load_config

def fit_log_linear(xs, ys):
    """拟合 y = a + b * log10(x)"""
    X = np.stack([np.ones_like(xs), np.log10(xs)], axis=1)
    beta = np.linalg.lstsq(X, ys, rcond=None)[0]
    return float(beta[0]), float(beta[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', required=True)
    ap.add_argument('--run_train', action='store_true')
    args = ap.parse_args()

    results = []

    for c in args.configs:
        cfg = load_config(c)
        name = cfg.get('name', os.path.splitext(os.path.basename(c))[0])
        model_class = cfg['model'].get('model_class', 'UniScaleFormer')
        hist_path = os.path.join(cfg['train']['output_dir'], name, 'history.json')

        # 自动训练
        if args.run_train and not os.path.exists(hist_path):
            print(f'Training {name} ({model_class})...')
            subprocess.run(['python', 'train.py', '--config', c], check=True)

        if not os.path.exists(hist_path):
            print(f'Skipping {name}: no history.json found')
            continue

        with open(hist_path, 'r', encoding='utf-8') as f:
            h = json.load(f)

        valid = [x for x in h if x.get('auc') == x.get('auc')]  # 排除 nan
        if not valid:
            print(f'Skipping {name}: no valid AUC')
            continue

        best = max(valid, key=lambda x: x['auc'])
        proxy = (
            float(cfg['model'].get('d_model', 192))
            * float(cfg['model'].get('n_layers', 4))
            * float(cfg['model'].get('num_queries', cfg['model'].get('n_layers', 4)))
        )

        results.append({
            'config': c,
            'name': name,
            'model_class': model_class,
            'params_proxy': proxy,
            'best_auc': best['auc'],
            'best_epoch': best.get('epoch', '?'),
        })

    # 打印对比表
    print('\n' + '=' * 80)
    print(f'{"Name":<20} {"Model":<18} {"Proxy":>10} {"AUC":>10} {"Epoch":>6}')
    print('-' * 80)
    for r in sorted(results, key=lambda x: -x['best_auc']):
        print(f'{r["name"]:<20} {r["model_class"]:<18} {r["params_proxy"]:>10.0f} {r["best_auc"]:>10.6f} {r["best_epoch"]:>6}')
    print('=' * 80)

    # 若有 3+ 结果且参数规模不同，拟合 scaling law
    proxies = [r['params_proxy'] for r in results]
    if len(results) >= 3 and len(set(proxies)) >= 3:
        xs = np.asarray(proxies, dtype=float)
        ys = np.asarray([r['best_auc'] for r in results], dtype=float)
        a, b = fit_log_linear(xs, ys)
        print(f'\nScaling Law Fit: AUC ≈ {a:.6f} + {b:.6f} × log10(params_proxy)')
        if b > 0:
            print('→ slope > 0: 模型越大，AUC 越高 ✓')
        else:
            print('→ slope ≤ 0: Scaling Law 不成立，可能数据不足或架构瓶颈')

    # 保存 JSON
    out = {'results': results}
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__=='__main__': main()
