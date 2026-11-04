import os, itertools, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

import features as F
import cv as C

import sys
STEPS = set(sys.argv[1:]) or {'models', 'tuning', 'ablation', 'buildup', 'experiments', 'leakage', 'dictionary'}
OUTD = os.path.join(os.path.dirname(__file__), '..', 'outputs'); os.makedirs(OUTD, exist_ok=True)
RES = os.path.join(os.path.dirname(__file__), '..', 'results'); os.makedirs(RES, exist_ok=True)
X, y, Xt, cats = C.X, C.y, C.Xt, C.cats
rank = lambda v: rankdata(v) / len(v)
GEO = ['lat', 'lon', 'state_ca', 'wage_per_return', 'zip_pop', 'd_sf', 'd_berk', 'd_palo', 'd_cc', 'd_min', 'zip3', 'zip']
ID = ['id_num', 'id_rank', 'id_gap_prev', 'id_gap_next', 'id_batch_size', 'id_batch_sub13', 'id_batch_tk']

def save(df, name):
    df.to_csv(os.path.join(RES, name), index=False); print(f'\n== {name}\n{df.to_string(index=False)}', flush=True)

FINAL_LGB = dict(n_estimators=600, learning_rate=0.02, num_leaves=7, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.3, reg_lambda=2, verbose=-1)
EARLY_LGB = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.6, reg_lambda=2, verbose=-1)

def cv_lgb(D, params, seeds):
    """mean and sd of out-of-fold AUROC over seeds"""
    a = []
    for sd in seeds:
        oof = np.zeros(len(y))
        for tri, vai in StratifiedKFold(5, shuffle=True, random_state=sd).split(D, y):
            m = lgb.LGBMClassifier(**params, random_state=sd).fit(D.iloc[tri], y[tri])
            oof[vai] = m.predict_proba(D.iloc[vai])[:, 1]
        a.append(roc_auc_score(y, oof))
    return round(np.mean(a), 5), round(np.std(a), 5)

def cv_oof(fit_pred, D=X, seeds=range(5)):
    """seed-averaged OOF predictions + per-seed AUROC for any fit_pred(Xa, ya, Xv, sd)"""
    oofs = []
    for sd in seeds:
        oof = np.zeros(len(y))
        for tri, vai in StratifiedKFold(5, shuffle=True, random_state=sd).split(D, y):
            oof[vai] = fit_pred(D.iloc[tri], y[tri], D.iloc[vai], sd)
        oofs.append(oof)
    a = [roc_auc_score(y, o) for o in oofs]
    return np.mean(oofs, 0), round(np.mean(a), 5), round(np.std(a), 5)
sub = [c for c in X.columns if c.startswith('sub_') or c in ['tier_mean', 'seats_mean', 'price_mean', 'multi_subs', 'last_tier',
       'last_seats', 'last_price', 'last_package', 'last_location', 'last_section', 'tier_trend', 'tier_slope5', 'n_locations', 'in_subs']]

# ---------------------------------------------------------------- 1. models and blends
if 'models' in STEPS:
    wrap = lambda f: (lambda Xa, ya, Xv, sd: f(Xa, ya, Xv, Xv.iloc[:1], sd)[0])
    oof, rows = {}, []
    for name, f in [('LightGBM', C.lgbm), ('XGBoost', C.xgbm), ('CatBoost', C.cbm), ('Logistic regression', C.lr)]:
        o, m, s = cv_oof(wrap(f)); oof[name] = rank(o); rows.append([name, 'model', m, s])
    def blend(w): return round(roc_auc_score(y, sum(v * oof[k] for k, v in w.items())), 5)
    for name, w in [('LightGBM + LR (3:2, final)', {'LightGBM': 3, 'Logistic regression': 2}),
                    ('LightGBM + XGBoost + LR', {'LightGBM': 1, 'XGBoost': 1, 'Logistic regression': 1}),
                    ('All four models, equal weights', {k: 1 for k in oof}),
                    ('Three tree models', {'LightGBM': 1, 'XGBoost': 1, 'CatBoost': 1})]:
        rows.append([name, 'rank blend', blend(w), None])
    save(pd.DataFrame(rows, columns=['model', 'type', 'cv_auroc_mean', 'cv_auroc_sd']), 'cv_summary.csv')
    pd.to_pickle(oof, os.path.join(OUTD, 'oof_ranks.pkl'))

    grid = [(blend(dict(zip(oof, w))), *w) for w in itertools.product(range(5), repeat=4) if sum(w)]
    g = pd.DataFrame(grid, columns=['cv_auroc', 'w_lightgbm', 'w_xgboost', 'w_catboost', 'w_logreg'])
    save(g.sort_values('cv_auroc', ascending=False).head(10), 'blend_weights.csv')

# ---------------------------------------------------------------- 2. LightGBM settings (final features, 3 seeds)
if 'tuning' in STEPS:
    tune = [{}, dict(num_leaves=7), dict(num_leaves=4, n_estimators=1000), dict(num_leaves=7, min_child_samples=40),
            dict(num_leaves=7, colsample_bytree=0.3), dict(num_leaves=7, n_estimators=1000, learning_rate=0.01),
            dict(num_leaves=7, reg_lambda=10, min_child_samples=30), dict(num_leaves=15, colsample_bytree=0.3, min_child_samples=40),
            dict(max_depth=3, num_leaves=8, n_estimators=900)]
    save(pd.DataFrame([[str(t or 'default: 15 leaves, 60% columns'), *cv_lgb(X, {**EARLY_LGB, **t}, range(3))] for t in tune],
                      columns=['change_from_default', 'cv_auroc_mean', 'cv_auroc_sd']), 'lgbm_tuning.csv')

# ---------------------------------------------------------------- 3. ablation (full 78 features, early settings, 4 seeds)
if 'ablation' in STEPS:
    Xa = F.build()[0].drop(columns=['id_prefix', 'billing_city'])
    groups = {
        'Geography': GEO, 'Account-ID batch stats only': ID[1:], 'Account-ID order (all)': ID,
        'Tickets': [c for c in Xa.columns if c.startswith('tk_')] + ['in_tk'],
        'Donations': ['don_2013', 'don_life', 'don_n', 'don_avg', 'don_first_yr', 'don_years', 'fd_days', 'fd_after_apr14'],
        'Per-season subscription seats 2005-12': [f'sub_{k}' for k in range(2005, 2013)],
        'Last package, location, section': ['last_package', 'last_location', 'last_section', 'relationship'],
    }
    base = cv_lgb(Xa, EARLY_LGB, range(4))
    rows = [['None (baseline, 78 features)', len(Xa.columns), *base, 0.0]]
    for gname, cs in groups.items():
        r = cv_lgb(Xa.drop(columns=cs), EARLY_LGB, range(4)); rows.append([f'Remove {gname}', len(Xa.columns) - len(cs), *r, round(r[0] - base[0], 5)])
    nogeo = [c for c in Xa.columns if c not in GEO]
    for gname, keep in [('Geography -> keep d_min only', ['d_min']), ('Geography -> keep d_min + ZIP wages', ['d_min', 'wage_per_return'])]:
        r = cv_lgb(Xa[nogeo + keep], EARLY_LGB, range(4)); rows.append([gname, len(nogeo) + len(keep), *r, round(r[0] - base[0], 5)])
    save(pd.DataFrame(rows, columns=['variant', 'n_features', 'cv_auroc_mean', 'cv_auroc_sd', 'change_vs_baseline']), 'ablation.csv')

# ---------------------------------------------------------------- 4. build-up chart (final features, final settings, 5 seeds)
if 'buildup' in STEPS:
    steps = [('Subscription history', sub), ('+ Ticket purchases', [c for c in X.columns if c.startswith('tk_')] + ['in_tk']),
             ('+ Donation totals', ['don_2013', 'don_life', 'don_n', 'don_avg', 'don_first_yr', 'don_years', 'has_ship', 'relationship']),
             ('+ Distance to nearest venue', ['d_min']), ('+ Exact first-donation date', ['fd_days', 'fd_after_apr14']),
             ('+ Account-ID order', ID)]
    cols, rows = [], []
    for s, cs in steps:
        cols += cs; rows.append([s, len(cols), *cv_lgb(X[cols], FINAL_LGB, range(5))])
    bu = pd.DataFrame(rows, columns=['step', 'n_features', 'cv_auroc_mean', 'cv_auroc_sd'])
    bu['gain'] = bu.cv_auroc_mean.diff().round(5)
    save(bu, 'feature_build_up.csv')

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.6))
    g_ = bu.gain.fillna(0).values[::-1]; lab = bu.step.values[::-1]
    colors = ['#2c7be5' if 'Donation' in l or 'Account-ID' in l else '#c4c4b8' for l in lab]
    ax.barh(lab, g_, color=colors)
    for i, (v, c) in enumerate(zip(g_, bu.cv_auroc_mean.values[::-1])):
        ax.text(v + 0.0002, i, f'+{v:.4f}' if v else 'start', va='center', fontsize=9)
        ax.text(0.0158, i, f'{c:.4f}', va='center', ha='right', fontsize=9)
    ax.set_xlim(0, 0.016); ax.set_xlabel('Gain in CV AUROC'); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Donations and account-ID order add the most AUROC on top of subscription history', fontsize=10, loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'feature_gain.png'), dpi=160)

# ---------------------------------------------------------------- 5. experiments that did not help
if 'experiments' in STEPS:
    if 'models' not in STEPS:  # reuse OOF ranks from an earlier models run
        oof = pd.read_pickle(os.path.join(OUTD, 'oof_ranks.pkl'))
        blend = lambda w: round(roc_auc_score(y, sum(v * oof[k] for k, v in w.items())), 5)
    rows = []
    # 5a label-free ID-neighbour activity (early settings, 3 seeds, as when it was tested)
    F.NEIGHBOURS = True
    Xn = F.build()[0].drop(columns=C.DROP); F.NEIGHBOURS = False
    rows.append(['ID-neighbour activity (label-free, +-5/+-25 IDs)', 'LightGBM, early settings',
                 cv_lgb(X, EARLY_LGB, range(3))[0], cv_lgb(Xn, EARLY_LGB, range(3))[0]])
    # 5b household / neighbour labels, via exp.py
    import exp as E
    for v, label in [('hh', 'Household: same ZIP within +-3 IDs'), ('nbl', 'Neighbour labels, out of fold')]:
        E.run(v); rows.append([label, 'LightGBM, final settings', cv_lgb(X, FINAL_LGB, range(5))[0], E.LAST_AUC[v][0]])
    # 5c extra tree ensembles and spline LR in the blend
    def numeric(d):
        d = d.copy()
        for c in cats: d[c] = d[c].cat.codes
        return d.fillna(-999)
    ref = blend({'LightGBM': 3, 'Logistic regression': 2})
    for label, mk in [('ExtraTrees added to the blend', lambda sd: ExtraTreesClassifier(1000, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=sd)),
                      ('Random forest added to the blend', lambda sd: RandomForestClassifier(1000, min_samples_leaf=3, max_features=0.3, n_jobs=-1, random_state=sd))]:
        o, _, _ = cv_oof(lambda A, ya, V, sd: mk(sd).fit(numeric(A), ya).predict_proba(numeric(V))[:, 1])
        rows.append([label, 'LightGBM 3 + LR 2 + this 1', ref, round(roc_auc_score(y, 3 * oof['LightGBM'] + 2 * oof['Logistic regression'] + rank(o)), 5)])
    nums = [c for c in X.columns if c not in cats]
    key = ['id_num', 'sub_last', 'sub_streak', 'sub_since_last', 'fd_days', 'd_min', 'sub_n', 'don_life', 'tk_rows', 'id_batch_size']
    def spline_lr(A, ya, V, sd):
        def prep(d):
            d = d.copy()
            for c in nums:
                if d[c].max() > 50: d[c] = np.log1p(d[c].clip(lower=0))
            for c in cats: d[c] = d[c].astype(object).fillna('NA').astype(str)
            miss = d[nums].isna().astype(float).add_suffix('_na')
            miss = miss.loc[:, miss.std() > 0] if len(d) > 3000 else miss
            d[nums] = d[nums].fillna(d[nums].median())
            return pd.concat([d, miss], axis=1)
        PA = prep(A); PV = prep(V).reindex(columns=PA.columns, fill_value=0)
        ct = ColumnTransformer([('spl', make_pipeline(StandardScaler(), SplineTransformer(n_knots=6, degree=3)), key),
                                ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=20), cats)], remainder=StandardScaler())
        m = make_pipeline(ct, LogisticRegression(C=0.03, max_iter=5000)).fit(PA, ya)
        return m.predict_proba(PV)[:, 1]
    o, _, _ = cv_oof(spline_lr)
    rows.append(['Spline logistic regression in place of plain LR', 'LightGBM 3 + this 2', ref, round(roc_auc_score(y, 3 * oof['LightGBM'] + 2 * rank(o)), 5)])
    ex = pd.DataFrame(rows, columns=['experiment', 'compared_on', 'cv_auroc_reference', 'cv_auroc_with_change'])
    ex['change'] = (ex.cv_auroc_with_change - ex.cv_auroc_reference).round(5)
    save(ex, 'experiments.csv')

# ---------------------------------------------------------------- 6. leakage check
if 'leakage' in STEPS:
    a = F.R('account').set_index('account.id')
    late = (pd.to_datetime(a['first.donated'], errors='coerce').reindex(X.index) >= '2014-01-01').values
    B = X.drop(columns=['fd_days', 'fd_after_apr14'])
    Cc = B.copy()
    Cc.loc[late, ['don_first_yr', 'don_years', 'don_avg']] = np.nan
    Cc.loc[late, ['don_life', 'don_n']] = 0
    D = Cc.drop(columns=ID)
    rows = [[n, *cv_lgb(d, FINAL_LGB, range(5))] for n, d in
            [('Final LightGBM', X), ('Without the exact first-donation date', B),
             ('Also masking all donation information from 2014', Cc), ('Also removing account-ID order', D)]]
    lk = pd.DataFrame(rows, columns=['version', 'cv_auroc_mean', 'cv_auroc_sd'])
    lk['note'] = ['', '', f'{late.sum()} training accounts first donated in 2014 ({y[late].mean():.1%} renewed)', '']
    save(lk, 'leakage_check.csv')

# ---------------------------------------------------------------- 7. feature dictionary
if 'dictionary' in STEPS:
    desc = {
        'sub_n': 'Number of distinct seasons subscribed', 'sub_rows': 'Number of subscription records',
        'sub_first': 'First season subscribed (start year)', 'sub_last': 'Last season subscribed (start year)',
        'sub_span': 'Seasons from first to last subscription', 'sub_density': 'Share of seasons in that span with a subscription',
        'sub_since_last': 'Seasons between the last subscription and 2014-15', 'sub_streak': 'Consecutive seasons subscribed ending in 2013-14',
        'sub_longest': 'Longest run of consecutive seasons', 'sub_n_last5': 'Seasons subscribed in 2009-10 to 2013-14',
        'sub_n_last3': 'Seasons subscribed in 2011-12 to 2013-14', 'tier_mean': 'Mean subscription tier',
        'seats_mean': 'Mean seats per subscription', 'price_mean': 'Mean price level', 'multi_subs': 'Records flagged as multiple subscriptions',
        'last_tier': 'Tier of the latest subscription', 'last_seats': 'Seats in the latest subscription', 'last_price': 'Price level of the latest subscription',
        'last_package': 'Package of the latest subscription', 'last_location': 'Venue of the latest subscription', 'last_section': 'Section of the latest subscription',
        'tier_trend': 'Latest tier minus mean tier', 'tier_slope5': 'Slope of seats x tier over the last five seasons',
        'sub_full_frac': 'Share of subscriptions that were the Full package', 'n_locations': 'Distinct venues subscribed to', 'in_subs': 'Has any subscription record',
        'tk_rows': 'Single-ticket purchases', 'tk_seats': 'Seats bought as single tickets', 'tk_seasons': 'Seasons with a ticket purchase',
        'tk_last': 'Last season with a ticket purchase', 'tk_first': 'First season with a ticket purchase', 'tk_sets': 'Distinct concert sets attended',
        'tk_price': 'Mean ticket price level', 'tk_mkt': 'Purchases with a recorded marketing source', 'tk_family': 'Family-concert purchases', 'in_tk': 'Has any ticket record',
        'don_2013': 'Amount donated in 2013', 'don_life': 'Lifetime amount donated', 'don_n': 'Lifetime number of donations', 'don_avg': 'Average donation',
        'don_first_yr': 'Year of first donation', 'don_years': 'Years since first donation', 'has_ship': 'Has a shipping address', 'relationship': 'Relationship type (e.g. vendor, education participant)',
        'd_min': 'Miles from billing ZIP to the nearest of four venues',
        'fd_days': 'Days from 1 Jan 2014 to the first donation (may postdate the season, see leakage_check.csv)',
        'fd_after_apr14': 'First donation on or after 1 Apr 2014',
        'id_num': 'Account ID decoded from base 62 (creation order)', 'id_rank': 'Percentile of id_num among all 19,833 accounts',
        'id_gap_prev': 'Gap to the previous account ID', 'id_gap_next': 'Gap to the next account ID', 'id_batch_size': 'Accounts in the same creation batch (IDs within 1,000)',
        'id_batch_sub13': 'Share of the batch with a 2013-14 subscription (no labels used)', 'id_batch_tk': 'Share of the batch with a ticket purchase',
    }
    for yv in range(2005, 2014): desc[f'sub_{yv}'] = f'Seats x tier subscribed in {yv}-{str(yv + 1)[2:]}'
    for yv in range(2010, 2014): desc[f'tk_{yv}'] = f'Ticket purchases in {yv}-{str(yv + 1)[2:]}'
    group = lambda c: ('Subscription history' if c in sub else 'Tickets' if c.startswith('tk_') or c == 'in_tk' else
                       'Account-ID order' if c in ID else 'First-donation date' if c.startswith('fd_') else 'Distance' if c == 'd_min' else 'Donations')
    fd = pd.DataFrame({'feature': X.columns, 'group': [group(c) for c in X.columns], 'description': [desc[c] for c in X.columns]})
    save(fd.sort_values(['group', 'feature']), 'feature_dictionary.csv')
    print('\ndone:', sorted(os.listdir(RES)))
