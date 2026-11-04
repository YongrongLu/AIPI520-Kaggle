import numpy as np, pandas as pd

import os
D = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(__file__), '..', 'data')) + '/'
NEIGHBOURS = False  # label-free ID-neighbour activity features (tested, not used)
R = lambda f: pd.read_csv(D + f + '.csv', encoding='latin-1', low_memory=False)

def sidx(s):  # '2013-2014' -> 2013
    return s.str[:4].astype(int)

B62 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
def id_num(x):
    n = 0
    for ch in x[-6:]:
        n = n * 62 + B62.index(ch)
    return n

def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 7918 * np.arcsin(np.sqrt(a))

def build():
    tr = R('train').rename(columns={'account.id': 'id'})
    te = R('test').rename(columns={'ID': 'id'})
    base = pd.concat([tr[['id']], te[['id']]], ignore_index=True)

    # ---------- subscriptions ----------
    s = R('subscriptions').rename(columns={'account.id': 'id'})
    s['yr'] = sidx(s.season)
    s['tier_seats'] = s.subscription_tier * s['no.seats']
    g = s.groupby('id')
    F = pd.DataFrame(index=base.id.unique())
    F['sub_n'] = g.yr.nunique()
    F['sub_rows'] = g.size()
    F['sub_first'] = g.yr.min()
    F['sub_last'] = g.yr.max()
    F['sub_span'] = F.sub_last - F.sub_first + 1
    F['sub_density'] = F.sub_n / F.sub_span
    F['sub_since_last'] = 2014 - F.sub_last
    for y in range(2005, 2014):
        F[f'sub_{y}'] = s[s.yr == y].groupby('id').tier_seats.sum()
    # streak ending in 2013 and longest streak
    yrs = g.yr.apply(lambda v: sorted(set(v)))
    def streak_end(v):
        c = 0; y = 2013
        while y in v: c += 1; y -= 1
        return c
    def longest(v):
        best = cur = 0; prev = None
        for y in v:
            cur = cur + 1 if prev is not None and y == prev + 1 else 1
            best = max(best, cur); prev = y
        return best
    F['sub_streak'] = yrs.apply(lambda v: streak_end(set(v)))
    F['sub_longest'] = yrs.apply(longest)
    F['sub_n_last5'] = s[s.yr >= 2009].groupby('id').yr.nunique()
    F['sub_n_last3'] = s[s.yr >= 2011].groupby('id').yr.nunique()
    F['tier_mean'] = g.subscription_tier.mean()
    F['seats_mean'] = g['no.seats'].mean()
    F['price_mean'] = g['price.level'].mean()
    F['multi_subs'] = (s['multiple.subs'] == 'yes').groupby(s.id).sum()
    last = s.sort_values('yr').groupby('id').tail(1).set_index('id')
    F['last_tier'] = last.subscription_tier
    F['last_seats'] = last['no.seats']
    F['last_price'] = last['price.level']
    F['last_package'] = last.package
    F['last_location'] = last.location
    F['last_section'] = last.section
    F['tier_trend'] = F.last_tier - F.tier_mean
    # tier slope over last 5 seasons
    rec = s[s.yr >= 2009].groupby(['id', 'yr']).tier_seats.sum().reset_index()
    def slope(d):
        return np.polyfit(d.yr, d.tier_seats, 1)[0] if len(d) > 1 else 0.0
    F['tier_slope5'] = rec.groupby('id').apply(slope, include_groups=False)
    F['sub_full_frac'] = (s.package == 'Full').groupby(s.id).mean()
    F['n_locations'] = g.location.nunique()

    # ---------- tickets ----------
    t = R('tickets_all').rename(columns={'account.id': 'id'})
    t['yr'] = sidx(t.season)
    t['pl'] = pd.to_numeric(t['price.level'], errors='coerce')
    gt = t.groupby('id')
    F['tk_rows'] = gt.size()
    F['tk_seats'] = gt['no.seats'].sum()
    F['tk_seasons'] = gt.yr.nunique()
    F['tk_last'] = gt.yr.max()
    F['tk_first'] = gt.yr.min()
    F['tk_sets'] = gt.set.nunique()
    F['tk_price'] = gt.pl.mean()
    for y in range(2010, 2014):
        F[f'tk_{y}'] = t[t.yr == y].groupby('id').size()
    F['tk_mkt'] = gt['marketing.source'].count()
    F['tk_family'] = (t.location == 'Family concert').groupby(t.id).sum()

    # ---------- account ----------
    a = R('account').rename(columns={'account.id': 'id'}).set_index('id')
    F['don_2013'] = a['amount.donated.2013']
    F['don_life'] = a['amount.donated.lifetime']
    F['don_n'] = a['no.donations.lifetime']
    F['don_avg'] = F.don_life / F.don_n.replace(0, np.nan)
    fd = pd.to_datetime(a['first.donated'], errors='coerce')
    F['don_first_yr'] = fd.dt.year
    F['don_years'] = 2014 - F.don_first_yr
    F['has_ship'] = a['shipping.zip.code'].notna().astype(int)
    F['relationship'] = a.relationship
    F['billing_city'] = a['billing.city']
    zipc = pd.to_numeric(a['billing.zip.code'].astype(str).str[:5], errors='coerce')
    F['zip'] = zipc

    # ---------- zip geo ----------
    z = R('zipcodes').drop_duplicates('Zipcode').set_index('Zipcode')
    F['lat'] = F.zip.map(z.Lat)
    F['lon'] = F.zip.map(z.Long)
    F['state_ca'] = (F.zip.map(z.State) == 'CA').astype(float)
    F.loc[F.zip.isna(), 'state_ca'] = np.nan
    F['wage_per_return'] = F.zip.map(z.TotalWages / z.TaxReturnsFiled)
    F['zip_pop'] = F.zip.map(z.EstimatedPopulation)
    venues = {'sf': (37.7793, -122.4193), 'berk': (37.8716, -122.2727), 'palo': (37.4419, -122.1430), 'cc': (37.9101, -122.0652)}
    for k, (la, lo) in venues.items():
        F[f'd_{k}'] = haversine(F.lat, F.lon, la, lo)
    F['d_min'] = F[[f'd_{k}' for k in venues]].min(axis=1)
    F['zip3'] = F.zip // 100

    # ---------- id ordering ----------
    F['id_num'] = [id_num(i) for i in F.index]
    F['id_prefix'] = [i[:9] for i in F.index]

    # ---------- v2: precise donation date + id batch structure ----------
    F['fd_days'] = (fd - pd.Timestamp('2014-01-01')).dt.days
    F['fd_after_apr14'] = (fd >= '2014-04-01').astype(float)
    allid = pd.Series([id_num(i) for i in a.index], index=a.index).sort_values()
    gp = allid.diff(); gn = -allid.diff(-1)
    F['id_gap_prev'] = gp; F['id_gap_next'] = gn
    batch = (gp.fillna(1e9) > 1000).cumsum()
    bsize = batch.map(batch.value_counts())
    F['id_batch_size'] = bsize
    F['id_rank'] = allid.rank(pct=True)
    # share of each id batch that held a 2013-14 subscription (no label used)
    s13 = set(s[s.yr == 2013].id)
    bsub = pd.Series(allid.index.isin(s13).astype(float), index=allid.index).groupby(batch).transform('mean')
    F['id_batch_sub13'] = bsub
    tkset = set(t.id)
    F['id_batch_tk'] = pd.Series(allid.index.isin(tkset).astype(float), index=allid.index).groupby(batch).transform('mean')

    # ---------- v3: id-neighbourhood activity (label-free) ----------
    order = allid.index
    has13 = pd.Series(order.isin(s13).astype(float), index=order)
    hasany = pd.Series(order.isin(set(s.id)).astype(float), index=order)
    hasrecent = pd.Series(order.isin(set(s[s.yr >= 2011].id)).astype(float), index=order)
    hastk = pd.Series(order.isin(tkset).astype(float), index=order)
    hasdon = pd.Series((a.loc[order, 'amount.donated.lifetime'] > 0).astype(float).values, index=order)
    for w in ((5, 25) if NEIGHBOURS else ()):  # tested: hurt CV by ~0.0006, off by default
        for nm, ser in [('s13', has13), ('sany', hasany), ('srec', hasrecent), ('tk', hastk), ('don', hasdon)]:
            roll = ser.rolling(2 * w + 1, center=True, min_periods=1)
            F[f'nb{w}_{nm}'] = (roll.sum() - ser) / (roll.count() - 1)

    F['in_subs'] = F.sub_n.notna().astype(int)
    F['in_tk'] = F.tk_rows.notna().astype(int)

    cats = ['last_package', 'last_location', 'last_section', 'relationship', 'billing_city', 'id_prefix']
    for c in cats:
        F[c] = F[c].astype('category')
    F.index.name = 'id'
    X_tr = F.loc[tr.id]; X_te = F.loc[te.id]
    return X_tr, tr.label.values, X_te, te.id.values, cats

if __name__ == '__main__':
    X, y, Xt, ids, cats = build()
    print(X.shape, Xt.shape)
    print(X.assign(y=y).groupby('in_subs').y.agg(['mean', 'size']))
    print(X.assign(y=y).groupby('sub_streak').y.agg(['mean', 'size']))
    print(X.id_prefix.value_counts().head())
