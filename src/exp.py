"""Experiments on top of the final feature set. Usage: python exp.py <variant>"""
import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import cv as C
from features import R, id_num, sidx

X, y, Xt = C.X.copy(), C.y, C.Xt.copy()
tr_ids = np.array(X.index); te_ids = np.array(Xt.index)

# global ordering over every account
a = R('account').rename(columns={'account.id': 'id'}).set_index('id')
order = pd.Series([id_num(i) for i in a.index], index=a.index).sort_values()
pos = pd.Series(np.arange(len(order)), index=order.index)
zipc = a['billing.zip.code'].astype(str).str[:5].reindex(order.index)
s = R('subscriptions').rename(columns={'account.id': 'id'}); s['yr'] = sidx(s.season)
s13 = set(s[s.yr == 2013].id)

def household_feats(w=3):
    """label-free: same billing ZIP among the w accounts on each side"""
    idx = order.index; z = zipc.values; h13 = np.array([i in s13 for i in idx]); n = len(idx)
    same = np.zeros(n); same13 = np.zeros(n)
    for k in range(1, w + 1):
        for sh in (k, -k):
            j = np.arange(n) + sh; ok = (j >= 0) & (j < n); jj = np.clip(j, 0, n - 1)
            m = ok & (z == z[jj]) & (z != 'nan')
            same += m; same13 += m & h13[jj]
    return pd.DataFrame({'hh_same': same, 'hh_same13': same13}, index=idx)

def nb_label(train_idx_ids, train_labels, query_ids, w):
    """mean label of *labelled* accounts within +-w global positions (self excluded)"""
    lab = np.full(len(order), np.nan)
    lab[pos.loc[train_idx_ids].values] = train_labels
    v = np.nan_to_num(lab); c = (~np.isnan(lab)).astype(float)
    cs_v = np.concatenate([[0], np.cumsum(v)]); cs_c = np.concatenate([[0], np.cumsum(c)])
    p = pos.loc[query_ids].values
    lo = np.clip(p - w, 0, len(order)); hi = np.clip(p + w + 1, 0, len(order))
    sv = cs_v[hi] - cs_v[lo]; sc = cs_c[hi] - cs_c[lo]
    selfl = lab[p]; has = ~np.isnan(selfl)
    sv = sv - np.where(has, selfl, 0); sc = sc - has
    return np.where(sc > 0, sv / np.maximum(sc, 1), np.nan), sc

def params(sd):
    return dict(n_estimators=600, learning_rate=0.02, num_leaves=7, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.3, reg_lambda=2, random_state=sd, verbose=-1)

LAST_AUC = {}  # variant -> (mean, sd) of per-seed AUROC

def run(variant, seeds=range(5), W=(5, 20)):
    Xv, Xtv = X.copy(), Xt.copy()
    if 'hh' in variant:
        h = household_feats(); Xv = Xv.join(h); Xtv = Xtv.join(h)
    aucs = []; oofs = []; tp = np.zeros(len(Xt))
    for sd in seeds:
        oof = np.zeros(len(y))
        for tri, vai in StratifiedKFold(5, shuffle=True, random_state=sd).split(Xv, y):
            A, V, T = Xv.iloc[tri].copy(), Xv.iloc[vai].copy(), Xtv.copy()
            if 'nbl' in variant:
                # inner OOF for the training rows so their feature matches what val/test see
                for w in W:
                    A[f'nbl{w}'] = np.nan
                    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=99).split(A, y[tri]):
                        A.iloc[iva, A.columns.get_loc(f'nbl{w}')] = nb_label(tr_ids[tri][itr], y[tri][itr], tr_ids[tri][iva], w)[0]
                    V[f'nbl{w}'] = nb_label(tr_ids[tri], y[tri], tr_ids[vai], w)[0]
                    T[f'nbl{w}'] = nb_label(tr_ids[tri], y[tri], te_ids, w)[0]
            m = lgb.LGBMClassifier(**params(sd)).fit(A, y[tri])
            oof[vai] = m.predict_proba(V)[:, 1]; tp += m.predict_proba(T)[:, 1] / (5 * len(seeds))
        aucs.append(roc_auc_score(y, oof)); oofs.append(oof)
    LAST_AUC[variant] = (round(np.mean(aucs), 5), round(np.std(aucs), 5))
    print(f'{variant:10s} {np.mean(aucs):.5f} ± {np.std(aucs):.5f}', flush=True)
    return np.mean(oofs, 0), tp

if __name__ == '__main__':
    out = {v: run(v) for v in sys.argv[1:]}
    pd.to_pickle(out, os.path.join(C.OUT, 'exp_' + '_'.join(sys.argv[1:]) + '.pkl'))
