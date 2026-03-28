import argparse, csv, os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import UniScaleFormer
from utils.collate import TAACCollator
from utils.config import load_config, ensure_dir
from utils.parquet_dataset import TAACDataset
from utils.training import move_to_device

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--split',default='test'); ap.add_argument('--output',required=True); args=ap.parse_args()
    cfg=load_config(args.config); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=TAACDataset(cfg['data']['data_dir'], args.split, cfg['data']['positive_action_types'], cfg['data'].get('test_limit'))
    loader=DataLoader(ds,batch_size=int(cfg['train']['eval_batch_size']),shuffle=False,num_workers=int(cfg['train']['num_workers']),collate_fn=TAACCollator(cfg))
    ckpt=torch.load(args.checkpoint,map_location='cpu'); model=UniScaleFormer(cfg); model.load_state_dict(ckpt['model']); model.to(device).eval()
    rows=[]; rid=0
    with torch.no_grad():
        for batch in tqdm(loader,desc='infer'):
            batch=move_to_device(batch,device); out=model(batch); probs=out['probs'].cpu().tolist(); item_ids=batch['item_ids'].cpu().tolist()
            for item_id,prob in zip(item_ids,probs): rows.append({'row_id':rid,'item_id':item_id,'score':float(prob)}); rid+=1
    ensure_dir(os.path.dirname(args.output) or '.')
    with open(args.output,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['row_id','item_id','score']); w.writeheader(); w.writerows(rows)
    print(f'saved {len(rows)} rows to {args.output}')
if __name__=='__main__': main()
