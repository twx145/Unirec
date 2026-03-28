import argparse, json, os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import UniScaleFormer
from utils.collate import TAACCollator
from utils.config import load_config, ensure_dir
from utils.metrics import binary_metrics
from utils.parquet_dataset import TAACDataset
from utils.training import set_seed, move_to_device, cosine_lr

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); preds=[]; labels=[]; losses=[]
    for batch in tqdm(loader,desc='eval',leave=False):
        batch=move_to_device(batch,device); out=model(batch); loss=model.compute_loss(batch,out)['loss']
        preds.extend(out['probs'].cpu().tolist()); labels.extend(batch['targets'].cpu().tolist()); losses.append(float(loss.item()))
    m=binary_metrics(labels,preds); m['loss']=sum(losses)/max(1,len(losses)); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args()
    cfg=load_config(args.config); set_seed(int(cfg['train']['seed'])); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name=cfg.get('name', os.path.splitext(os.path.basename(args.config))[0]); out_dir=os.path.join(cfg['train']['output_dir'], name); ensure_dir(out_dir)
    collate=TAACCollator(cfg)
    train_ds=TAACDataset(cfg['data']['data_dir'], cfg['data'].get('train_split','train'), cfg['data']['positive_action_types'], cfg['data'].get('train_limit'))
    valid_ds=TAACDataset(cfg['data']['data_dir'], cfg['data'].get('valid_split','valid'), cfg['data']['positive_action_types'], cfg['data'].get('valid_limit'))
    train_loader=DataLoader(train_ds,batch_size=int(cfg['train']['batch_size']),shuffle=True,num_workers=int(cfg['train']['num_workers']),collate_fn=collate)
    valid_loader=DataLoader(valid_ds,batch_size=int(cfg['train']['eval_batch_size']),shuffle=False,num_workers=int(cfg['train']['num_workers']),collate_fn=collate)
    model=UniScaleFormer(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=float(cfg['train']['lr']), weight_decay=float(cfg['train']['weight_decay']))
    total=max(1,int(cfg['train']['epochs'])*len(train_loader)); warmup=int(cfg['train'].get('warmup_steps',0)); use_amp=bool(cfg['train'].get('use_amp',True)) and device.type=='cuda'; scaler=torch.amp.GradScaler('cuda', enabled=use_amp)
    best=float('-inf'); history=[]; step=0
    for epoch in range(1,int(cfg['train']['epochs'])+1):
        model.train(); pbar=tqdm(train_loader,desc=f'train {epoch}')
        for batch in pbar:
            batch=move_to_device(batch,device)
            lr=cosine_lr(step,total,warmup,float(cfg['train']['lr']),float(cfg['train'].get('min_lr',1e-5)))
            for g in opt.param_groups: g['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out=model(batch); loss=model.compute_loss(batch,out)['loss']
            scaler.scale(loss).backward()
            if cfg['train'].get('grad_clip',0.0):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['train']['grad_clip']))
            scaler.step(opt); scaler.update(); step+=1; pbar.set_postfix(loss=f'{float(loss.item()):.5f}', lr=f'{lr:.2e}')
        metrics=evaluate(model,valid_loader,device); history.append({'epoch':epoch, **metrics}); print(metrics)
        ckpt={'model':model.state_dict(),'config':cfg,'metrics':metrics,'epoch':epoch}; torch.save(ckpt, os.path.join(out_dir,'last.pt'))
        if metrics.get('auc',float('nan'))==metrics.get('auc',float('nan')) and metrics['auc']>best: best=metrics['auc']; torch.save(ckpt, os.path.join(out_dir,'best.pt'))
        with open(os.path.join(out_dir,'history.json'),'w',encoding='utf-8') as f: json.dump(history,f,ensure_ascii=False,indent=2)
if __name__=='__main__': main()
