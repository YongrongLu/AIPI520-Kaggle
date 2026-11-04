import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from features import build

X, y, Xt, ids, cats = build()
X = X.drop(columns=['id_prefix', 'billing_city'])
groups = {
    'geo': ['lat', 'lon', 'state_ca', 'wage_per_return', 'zip_pop', 'd_sf', 'd_berk', 'd_palo', 'd_cc', 'd_min', 'zip3', 'zip'],
    'idbatch': ['id_gap_prev', 'id_gap_next', 'id_batch_size', 'id_batch_sub13', 'id_batch_tk', 'id_rank'],
    'id': ['id_num', 'id_rank', 'id_gap_prev', 'id_gap_next', 'id_batch_size', 'id_batch_sub13', 'id_batch_tk'],
    'tickets': [c for c in X.columns if c.startswith('tk_')] + ['in_tk'],
    'donation': ['don_2013', 'don_life', 'don_n', 'don_avg', 'don_first_yr', 'don_years', 'fd_days', 'fd_after_apr14'],
    'subyears': [f'sub_{y}' for y in range(2005, 2013)],
    'subcats': ['last_package', 'last_location', 'last_section', 'relationship'],
}

def cv(cols, seeds=(0, 1, 2, 3)):
    a = []
    for sd in seeds:
        oof = np.zeros(len(y))
        for tri, vai in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
            m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.6, reg_lambda=2, random_state=sd, verbose=-1)
            m.fit(X.iloc[tri][cols], y[tri]); oof[vai] = m.predict_proba(X.iloc[vai][cols])[:, 1]
        a.append(roc_auc_score(y, oof))
    return np.mean(a)

allc = list(X.columns)
b = cv(allc); print(f'all            {b:.5f}')
for g, cs in groups.items():
    v = cv([c for c in allc if c not in cs]); print(f'- {g:12s} {v:.5f}  ({v - b:+.5f})', flush=True)
