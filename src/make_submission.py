import os
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from features import build, D

OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs')
WEIGHTS = {'lgb': 3, 'lr': 2}

_, y, _, ids, _ = build()
res = pd.read_pickle(os.path.join(OUT, 'res_lgb_lr.pkl'))
rank = lambda v: rankdata(v) / len(v)

oof = sum(w * rank(res[k][0]) for k, w in WEIGHTS.items()) / sum(WEIGHTS.values())
test = sum(w * rank(res[k][1]) for k, w in WEIGHTS.items()) / sum(WEIGHTS.values())
print(f'blend CV AUROC: {roc_auc_score(y, oof):.5f}')

sample = pd.read_csv(D + 'sample_submission.csv')
sub = pd.DataFrame({'ID': ids, 'Predicted': test}).set_index('ID').loc[sample.ID].reset_index()
assert len(sub) == len(sample) and sub.Predicted.between(0, 1).all()
path = os.path.join(OUT, 'submission.csv')
sub.to_csv(path, index=False)
print('wrote', path, sub.shape)
