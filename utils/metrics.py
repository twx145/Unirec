import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

def binary_metrics(y_true, y_pred):
    y_true=np.asarray(y_true,dtype=float)
    y_pred=np.clip(np.asarray(y_pred,dtype=float),1e-6,1-1e-6)
    out={'logloss':float(log_loss(y_true,y_pred))}
    out['auc']=float(roc_auc_score(y_true,y_pred)) if len(np.unique(y_true))>1 else float('nan')
    return out
