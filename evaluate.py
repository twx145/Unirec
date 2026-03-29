import argparse, json, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import build_model
from utils.collate import TAACCollator
from utils.config import load_config
from utils.metrics import binary_metrics
from utils.parquet_dataset import TAACDataset
from utils.training import move_to_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default='valid')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = TAACDataset(cfg['data']['data_dir'], args.split,
                    cfg['data']['positive_action_types'],
                    cfg['data'].get('valid_limit'))
    loader = DataLoader(ds, batch_size=int(cfg['train']['eval_batch_size']),
                        shuffle=False, num_workers=int(cfg['train']['num_workers']),
                        collate_fn=TAACCollator(cfg))

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model = build_model(cfg)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    preds, labels, losses = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='evaluate'):
            batch = move_to_device(batch, device)
            out = model(batch)
            losses.append(float(model.compute_loss(batch, out)['loss'].item()))
            preds.extend(out['probs'].cpu().tolist())
            labels.extend(batch['targets'].cpu().tolist())

    m = binary_metrics(labels, preds)
    m['loss'] = sum(losses) / max(1, len(losses))
    print(json.dumps(m, ensure_ascii=False, indent=2))
    
if __name__=='__main__': main()
