import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import build
import os
OUT = os.path.join(os.path.dirname(__file__), '..', 'outputs'); os.makedirs(OUT, exist_ok=True)

X, y, Xt, ids, cats = build()
DROP = ['id_prefix', 'billing_city',  # constant / high-card
        'lat', 'lon', 'state_ca', 'wage_per_return', 'zip_pop', 'd_sf', 'd_berk', 'd_palo', 'd_cc', 'zip3', 'zip']  # geo noise (ablation)
X = X.drop(columns=DROP); Xt = Xt.drop(columns=DROP)
cats = [c for c in cats if c not in DROP]
SEEDS = [0, 1, 2, 3, 4]

def run(name, fit_pred):
    oof = np.zeros((len(SEEDS), len(y))); tp = np.zeros(len(Xt))
    for si, sd in enumerate(SEEDS):
        for tri, vai in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
            p_va, p_te = fit_pred(X.iloc[tri], y[tri], X.iloc[vai], Xt, sd)
            oof[si, vai] = p_va; tp += p_te / (5 * len(SEEDS))
    aucs = [roc_auc_score(y, o) for o in oof]
    print(f'{name:10s} AUC {np.mean(aucs):.5f} ± {np.std(aucs):.5f}', flush=True)
    return oof.mean(0), tp

def lgbm(Xa, ya, Xv, Xt_, sd):
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=7, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=2,
                           random_state=sd, verbose=-1)
    m.fit(Xa, ya)
    return m.predict_proba(Xv)[:, 1], m.predict_proba(Xt_)[:, 1]

def num(df):
    d = df.copy()
    for c in cats: d[c] = d[c].cat.codes
    return d

def xgbm(Xa, ya, Xv, Xt_, sd):
    m = xgb.XGBClassifier(n_estimators=700, learning_rate=0.02, max_depth=4, subsample=0.8,
                          colsample_bytree=0.6, min_child_weight=2, reg_lambda=2, random_state=sd,
                          enable_categorical=True, tree_method='hist')
    m.fit(Xa, ya)
    return m.predict_proba(Xv)[:, 1], m.predict_proba(Xt_)[:, 1]

def cbm(Xa, ya, Xv, Xt_, sd):
    def prep(d):
        d = d.copy()
        for c in cats: d[c] = d[c].astype(object).fillna("NA").astype(str)
        return d
    m = CatBoostClassifier(iterations=1200, learning_rate=0.03, depth=5, l2_leaf_reg=5,
                           random_seed=sd, verbose=0, cat_features=cats)
    m.fit(prep(Xa), ya)
    return m.predict_proba(prep(Xv))[:, 1], m.predict_proba(prep(Xt_))[:, 1]

def lr(Xa, ya, Xv, Xt_, sd):
    def prep(d):
        d = num(d).copy()
        for c in d.columns:
            if c not in cats and d[c].max() > 50: d[c] = np.log1p(d[c].clip(lower=0))
        return d.fillna(-1)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    m.fit(prep(Xa), ya)
    return m.predict_proba(prep(Xv))[:, 1], m.predict_proba(prep(Xt_))[:, 1]

if __name__ == '__main__':
    models = {'lgb': lgbm, 'xgb': xgbm, 'cat': cbm, 'lr': lr}
    which = sys.argv[1:] or list(models)
    res = {k: run(k, models[k]) for k in which}
    pd.to_pickle(res, os.path.join(OUT, 'res_' + '_'.join(which) + '.pkl'))
    if len(res) > 1:
        r = lambda v: rankdata(v) / len(v)
        bl = np.mean([r(v[0]) for v in res.values()], 0)
        print('rank blend', roc_auc_score(y, bl))
        trees = [k for k in res if k != 'lr']
        print('trees blend', roc_auc_score(y, np.mean([r(res[k][0]) for k in trees], 0)))
