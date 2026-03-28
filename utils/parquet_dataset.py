from __future__ import annotations
import glob, os, hashlib
import pyarrow.parquet as pq
from functools import lru_cache
from torch.utils.data import Dataset

class IndexedParquetDataset(Dataset):
    def __init__(self, paths, limit=None):
        self.files=[pq.ParquetFile(p) for p in paths]
        self.index=[]
        for fi,pf in enumerate(self.files):
            for rg in range(pf.num_row_groups):
                n=pf.metadata.row_group(rg).num_rows
                for ri in range(n):
                    self.index.append((fi,rg,ri))
                    if limit and len(self.index)>=limit: return
    def __len__(self): return len(self.index)
    @lru_cache(maxsize=8)
    def _read_rg(self, fi, rg): return self.files[fi].read_row_group(rg).to_pylist()
    def __getitem__(self, idx):
        fi,rg,ri=self.index[idx]
        return self._read_rg(fi,rg)[ri]

class TAACDataset(Dataset):
    def __init__(self, data_dir, split, positive_action_types, limit=None, file_pattern='*.parquet'):
        split_dir=os.path.join(data_dir, split)
        paths=sorted(glob.glob(os.path.join(split_dir,file_pattern))) if os.path.isdir(split_dir) else sorted(glob.glob(os.path.join(data_dir, f'{split}*.parquet')))
        if not paths: raise FileNotFoundError(f'No parquet files found for {split} under {data_dir}')
        self.ds=IndexedParquetDataset(paths, limit)
        self.pos=set(int(x) for x in positive_action_types)
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        row=self.ds[idx]
        label=row.get('label') or []
        y=0.0
        for x in label:
            if x and x.get('action_type') is not None and int(x['action_type']) in self.pos:
                y=1.0; break
        return {
            'item_id': row.get('item_id',0),
            'item_feature': row.get('item_feature') or [],
            'label': label,
            'seq_feature': row.get('seq_feature') or {},
            'timestamp': row.get('timestamp',0),
            'user_feature': row.get('user_feature') or [],
            'user_id': row.get('user_id',''),
            'target': y,
        }
