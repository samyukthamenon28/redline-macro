"""
features.py — Redline Macro
Build the feature matrix that feeds the crisis prediction model.
~200 features per country-month. Output: data/processed/features.parquet

Design philosophy: features a macro analyst would actually use, not
whatever sklearn.preprocessing spits out. Every regime signal has
an economic reason to exist.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# The core macro indicators we track per country.
# Not every country has all of these — we handle NaN gracefully.
MACRO_INDICATORS = [
    "gdp_growth",          # real GDP YoY %
    "inflation",           # CPI YoY %
    "unemployment",        # unemployment rate %
    "current_account_pct", # CA balance as % of GDP
    "debt_gdp",            # general government debt / GDP
    "broad_money_growth",  # M2 or M3 YoY %
    "credit_private_gdp",  # private sector credit / GDP
    "fx_reserves_months",  # import coverage in months
    "policy_rate",         # central bank policy rate
    "real_gdp_pc",         # real GDP per capita (level)
]

# Global/market indicators — same value for all countries at a given date,
# but the *interaction* with country fundamentals is what matters
GLOBAL_INDICATORS = [
    "T10Y2Y",     # US 10y-2y yield spread — the recession oracle
    "VIX",        # fear gauge
    "BAA_spread", # Moody's BAA corporate OAS — credit stress
    "DXY",        # dollar index — EM destroyer
    "GSCI",       # commodities — terms of trade shock proxy
    "EEM",        # EM equities — capital flow signal
    "HYG",        # high yield bonds — risk appetite
    "GS10",       # 10yr treasury yield
    "CPIAUCSL",   # US CPI — global inflation anchor
]

LAG_MONTHS = [3, 6, 9, 12]
ROLLING_WINDOWS = [3, 6, 12, 24]

# ── helpers ───────────────────────────────────────────────────────────────────

def _zscore_rolling(s: pd.Series, window: int) -> pd.Series:
    # z-score vs rolling mean — tells you how extreme current reading is
    # relative to recent history, not absolute level
    mu = s.rolling(window, min_periods=window // 2).mean()
    sigma = s.rolling(window, min_periods=window // 2).std()
    return (s - mu) / sigma.replace(0, np.nan)


def _percentile_rank(s: pd.Series, window: int) -> pd.Series:
    # where is today's reading in the country's own history?
    # 95th percentile debt/GDP is a different beast than 50th
    def _pct(x):
        if len(x) < 2:
            return np.nan
        return stats.percentileofscore(x[:-1], x[-1], kind="rank")
    return s.rolling(window, min_periods=window // 2).apply(_pct, raw=True)


def _pct_change_window(s: pd.Series, window: int) -> pd.Series:
    return s.pct_change(periods=window) * 100


def _safe_diff(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.diff(periods)

# ── macro momentum block ──────────────────────────────────────────────────────

def build_momentum_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    All the momentum machinery for one indicator column.
    Returns a wide dataframe with prefix = col.
    Called per indicator, then concat'd — verbose but debuggable.
    """
    s = df[col]
    feats = {}

    for w in ROLLING_WINDOWS:
        # level context — smoothed signal cuts through monthly noise
        feats[f"{col}_rm{w}"] = s.rolling(w, min_periods=w // 2).mean()
        feats[f"{col}_rstd{w}"] = s.rolling(w, min_periods=w // 2).std()

        # rate of change over window — trend direction matters as much as level
        feats[f"{col}_pct{w}"] = _pct_change_window(s, w)

        # where are we vs the last 10 years? spot regime shifts
        feats[f"{col}_z10yr_{w}"] = _zscore_rolling(s, min(120, len(s)))

        # percentile rank in country history — cross-country comparable
        feats[f"{col}_pctrank{w}"] = _percentile_rank(s, w)

    # momentum reversal — the single most useful turning-point signal.
    # a value that was falling for 12m and now flattens is a recovery candidate.
    # a value that was rising for 12m and now reverses is a cliff edge.
    feats[f"{col}_reversal12"] = s - s.shift(12)
    feats[f"{col}_reversal6"] = s - s.shift(6)

    return pd.DataFrame(feats, index=df.index)

# ── regime signals ────────────────────────────────────────────────────────────

def build_regime_signals(df: pd.DataFrame) -> pd.DataFrame:
    feats = {}

    # ── yield curve ──────────────────────────────────────────────────────────

    if "T10Y2Y" in df.columns:
        t = df["T10Y2Y"]

        # the recession oracle. every US recession since 1955 was preceded
        # by inversion. not perfect but nothing is.
        feats["yield_inverted"] = (t < 0).astype(int)

        # duration matters — a 1-month inversion is noise,
        # 12 months is a policy crisis in slow motion
        inv = (t < 0).astype(int)
        duration = []
        count = 0
        for v in inv:
            count = count + 1 if v else 0
            duration.append(count)
        feats["yield_inversion_duration"] = duration

    # ── VIX regime ────────────────────────────────────────────────────────────

    if "VIX" in df.columns:
        v = df["VIX"]
        # discretize volatility — regime matters more than exact level.
        # <15: goldilocks, 15-25: normal fear, 25-35: stress, >35: crisis/crash
        feats["vix_regime"] = pd.cut(
            v,
            bins=[-np.inf, 15, 25, 35, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(float)
        feats["vix_zscore"] = _zscore_rolling(v, 60)  # 5yr window

    # ── credit stress ─────────────────────────────────────────────────────────

    if "BAA_spread" in df.columns:
        baa = df["BAA_spread"]
        baa_z = _zscore_rolling(baa, 60)  # 5yr rolling z
        feats["credit_spread_zscore"] = baa_z

        # >2 std above 5yr mean = credit market pricing in real risk.
        # historically coincides with or leads equity drawdowns by 2-6m
        feats["credit_stress"] = (baa_z > 2.0).astype(int)

    # ── GDP deceleration ──────────────────────────────────────────────────────

    if "gdp_growth" in df.columns:
        gdp = df["gdp_growth"]
        # 3 consecutive quarters of falling growth = textbook late-cycle.
        # doesn't require contraction — deceleration alone is the signal.
        delta = gdp.diff()
        feats["gdp_decel"] = (
            (delta < 0) & (delta.shift(1) < 0) & (delta.shift(2) < 0)
        ).astype(int)
        feats["gdp_delta"] = delta  # raw momentum

    # ── debt dynamics ─────────────────────────────────────────────────────────

    if "debt_gdp" in df.columns:
        d = df["debt_gdp"]
        # 5pp jump in 12m = structural fiscal deterioration.
        # Japan 1990s, US 2008, EM 2020 — all show this before the break.
        feats["debt_acceleration"] = (d.diff(12) > 5).astype(int)
        feats["debt_change_12m"] = d.diff(12)

    # ── real rates ────────────────────────────────────────────────────────────

    if "policy_rate" in df.columns and "inflation" in df.columns:
        r = df["policy_rate"]
        pi = df["inflation"]
        real = r - pi  # ex-post real rate
        feats["real_rate"] = real

        # negative real rates = financial repression.
        # savers punished, capital misallocation incentivized,
        # asset bubbles subsidized. great for equities until it isn't.
        feats["negative_real_rate"] = (real < 0).astype(int)

    # ── dollar surge ──────────────────────────────────────────────────────────

    if "DXY" in df.columns:
        dxy = df["DXY"]
        dxy_3m_ret = _pct_change_window(dxy, 3)
        dxy_std = dxy_3m_ret.rolling(60, min_periods=30).std()

        # dollar move > 1.5 std in 3m = EM funding crisis risk.
        # dollar-denominated debt becomes unpayable, capital flees.
        # 2014-15, 2018, 2022 — all had this signature.
        feats["dollar_surge"] = (dxy_3m_ret > 1.5 * dxy_std).astype(int)
        feats["dxy_3m_return"] = dxy_3m_ret
        feats["dxy_zscore"] = _zscore_rolling(dxy, 60)

    # ── commodity crash ───────────────────────────────────────────────────────

    if "GSCI" in df.columns:
        gsci = df["GSCI"]
        gsci_3m = gsci.pct_change(3) * 100

        # -20% in 3m = terms of trade shock for commodity exporters.
        # also signals demand collapse globally (2008, 2014, 2020).
        feats["commodity_crash"] = (gsci_3m < -20).astype(int)
        feats["gsci_3m_return"] = gsci_3m

    # ── EM flight ─────────────────────────────────────────────────────────────

    if "EEM" in df.columns and "DXY" in df.columns:
        eem_3m = df["EEM"].pct_change(3) * 100
        dxy_rising = df["DXY"].diff(3) > 0

        # EM equities down hard AND dollar rising = classic risk-off capital flight.
        # the combination is the signal — either alone could be idiosyncratic.
        feats["em_flight"] = ((eem_3m < -15) & dxy_rising).astype(int)
        feats["eem_3m_return"] = eem_3m

    # ── high yield stress ─────────────────────────────────────────────────────

    if "HYG" in df.columns:
        hyg_3m = df["HYG"].pct_change(3) * 100

        # HYG down >10% in 3m = leveraged credit market refusing to fund.
        # high yield seizes before equities every time.
        feats["high_yield_stress"] = (hyg_3m < -10).astype(int)
        feats["hyg_3m_return"] = hyg_3m

    return pd.DataFrame(feats, index=df.index)

# ── interaction features ──────────────────────────────────────────────────────

def build_interaction_features(df: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    feats = {}

    # joint stress — both spiking together is far worse than either alone.
    # the 2008 signature: credit AND vol exploding simultaneously.
    # each variable alone might be 2 std; the product captures the tail.
    if "vix_zscore" in regime.columns and "credit_spread_zscore" in regime.columns:
        feats["vix_x_credit"] = regime["vix_zscore"] * regime["credit_spread_zscore"]

    # debt sustainability — high debt + rising rates = fiscal stress.
    # italy 2011, argentina 2018, uk 2022 mini-budget all fit this.
    # the threshold that matters isn't debt alone, it's debt × rate.
    if "debt_gdp" in df.columns and "real_rate" in regime.columns:
        feats["debt_x_rate"] = df["debt_gdp"] * regime["real_rate"]

    # EM vulnerability — deficit countries get crushed by dollar strength.
    # a country with -5% CA deficit and a surging dollar needs to either
    # hike aggressively or devalue. neither is fun.
    if "current_account_pct" in df.columns and "dxy_3m_return" in regime.columns:
        feats["ca_x_dollar"] = df["current_account_pct"] * regime["dxy_3m_return"]

    # misery index variant — stagflation detector.
    # high inflation AND high unemployment = political instability risk.
    # Turkey 2021, Argentina perennially, UK 2022-23.
    if "inflation" in df.columns and "unemployment" in df.columns:
        feats["inflation_x_unemployment"] = df["inflation"] * df["unemployment"]
        feats["misery_index"] = df["inflation"] + df["unemployment"]  # classic version too

    # monetary excess signal — money supply running ahead of inflation
    # is unsustainable. the excess eventually shows up in prices or asset bubbles.
    # Friedman was right about lags — this leads inflation by 12-18m.
    if "broad_money_growth" in df.columns and "inflation" in df.columns:
        feats["money_x_inflation"] = df["broad_money_growth"] * df["inflation"]
        feats["money_less_inflation"] = df["broad_money_growth"] - df["inflation"]  # real money growth

    return pd.DataFrame(feats, index=df.index)

# ── lag structure ─────────────────────────────────────────────────────────────

def build_lag_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Lag every feature at t-3, t-6, t-9, t-12.
    The model learns which horizon matters for each signal.
    Yield curve inversion leads recession by 12-18m; credit spreads by 6-9m.
    Giving the model all lags lets it figure this out.
    """
    lagged = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        s = df[col]
        for lag in LAG_MONTHS:
            lagged[f"{col}_lag{lag}"] = s.shift(lag)
    return pd.DataFrame(lagged, index=df.index)

# ── target construction ───────────────────────────────────────────────────────

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Four target definitions — the model predicts all four,
    ensemble at inference time. Different definitions capture
    different crisis types.
    """
    tgts = {}

    if "gdp_growth" in df.columns:
        gdp = df["gdp_growth"]

        # hard recession: 2+ quarters below -1% — the headline definition
        tgts["crisis_recession_6m"] = (
            gdp.shift(-6).rolling(2).min() < -1.0
        ).astype(int)

        # growth shock: sharp deceleration even if not technically negative.
        # -3pp drop in 6m is a crisis for an EM country at 2% baseline.
        tgts["crisis_growth_shock_6m"] = (
            (gdp.shift(-6) - gdp) < -3.0
        ).astype(int)

    # financial stress composite — any two of: VIX crisis + credit stress + HYG stress
    stress_signals = []
    if "vix_regime" in df.columns:
        stress_signals.append((df["vix_regime"] >= 3).astype(int))
    if "credit_stress" in df.columns:
        stress_signals.append(df["credit_stress"])
    if "high_yield_stress" in df.columns:
        stress_signals.append(df["high_yield_stress"])

    if len(stress_signals) >= 2:
        stress_sum = sum(s.shift(-6) for s in stress_signals)
        tgts["crisis_financial_6m"] = (stress_sum >= 2).astype(int)

    # broad crisis: recession OR financial stress — catches everything
    if "crisis_recession_6m" in tgts and "crisis_financial_6m" in tgts:
        tgts["crisis_any_6m"] = (
            (tgts["crisis_recession_6m"] == 1) | (tgts["crisis_financial_6m"] == 1)
        ).astype(int)

    return pd.DataFrame(tgts, index=df.index)

# ── per-country pipeline ──────────────────────────────────────────────────────

def process_country(country: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline for one country.
    df is indexed by date, contains whatever indicators are available.
    Returns wide feature matrix for this country.
    """
    df = df.sort_index()

    # momentum features for every available macro indicator
    momentum_parts = []
    for col in MACRO_INDICATORS + GLOBAL_INDICATORS:
        if col in df.columns and df[col].notna().sum() > 24:  # need at least 2yr of data
            momentum_parts.append(build_momentum_features(df, col))

    momentum = pd.concat(momentum_parts, axis=1) if momentum_parts else pd.DataFrame(index=df.index)

    # regime signals (needs raw df)
    regime_df = pd.concat([df, momentum], axis=1)
    regime = build_regime_signals(regime_df)

    # interactions (needs both raw and regime signals)
    interactions = build_interaction_features(df, regime)

    # concatenate everything before lagging
    all_features = pd.concat([df, momentum, regime, interactions], axis=1)
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]

    # lag every feature — model sees history, not the future
    feature_cols = [c for c in all_features.columns
                    if c not in ["date", "country", "region"]
                    and not c.startswith("crisis_")]

    lagged = build_lag_features(all_features, feature_cols)

    # targets — built from regime signals baked into all_features
    targets = build_targets(all_features)

    # assemble final frame for this country
    out = pd.concat([all_features, lagged, targets], axis=1)
    out["country"] = country

    # drop first 24 rows — not enough history for long windows
    out = out.iloc[24:]

    return out


# ── main ──────────────────────────────────────────────────────────────────────

def load_raw_data() -> dict:
    """
    Load per-country CSVs from data/raw/.
    Each file: date index + whatever indicators exist.
    Returns dict: country_iso -> DataFrame.

    Hack: we also merge global indicators (same for all countries)
    from data/raw/global.csv if it exists.
    """
    data = {}

    global_df = None
    global_path = RAW / "global.csv"
    if global_path.exists():
        global_df = pd.read_csv(global_path, index_col="date", parse_dates=True)
        print(f"  loaded global indicators: {list(global_df.columns)}")

    for path in sorted(RAW.glob("*.csv")):
        if path.stem == "global":
            continue
        try:
            df = pd.read_csv(path, index_col="date", parse_dates=True)
            if global_df is not None:
                df = df.join(global_df, how="left", rsuffix="_g")
                df = df.loc[:, ~df.columns.str.endswith("_g")]
            country = path.stem.upper()
            data[country] = df
        except Exception as e:
            print(f"  warning: failed to load {path.name}: {e}")

    return data


def build_feature_matrix(country_data: dict) -> pd.DataFrame:
    """
    Run the pipeline for all countries, stack into one big parquet.
    ~200 features x N countries x M months.
    """
    frames = []
    n = len(country_data)

    for i, (country, df) in enumerate(country_data.items(), 1):
        print(f"  [{i:3d}/{n}] {country} — {len(df)} months raw", end="")
        try:
            feat = process_country(country, df)
            frames.append(feat)
            print(f" → {len(feat)} rows, {feat.shape[1]} features")
        except Exception as e:
            print(f" ✗ {e}")

    if not frames:
        raise RuntimeError("no country data processed — check data/raw/")

    combined = pd.concat(frames, axis=0, ignore_index=False)
    combined.index.name = "date"
    combined = combined.reset_index().sort_values(["country", "date"]).set_index("date")

    # reorder: date, country, region, features, targets
    meta_cols = ["country", "region"]
    target_cols = [c for c in combined.columns if c.startswith("crisis_")]
    feature_cols = [c for c in combined.columns
                    if c not in meta_cols + target_cols]

    col_order = meta_cols + feature_cols + target_cols
    col_order = [c for c in col_order if c in combined.columns]
    combined = combined[col_order]

    nan_rate = combined.isna().mean().mean()
    print(f"\n  feature matrix: {combined.shape[0]:,} rows x {combined.shape[1]} columns")
    print(f"  target columns: {target_cols}")
    print(f"  NaN rate: {nan_rate:.1%} (expected ~30% from rolling windows)")

    return combined


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quick sanity check — call this after building features."""
    numeric = df.select_dtypes(include=[np.number])
    summary = pd.DataFrame({
        "mean": numeric.mean(),
        "std": numeric.std(),
        "nan_pct": numeric.isna().mean() * 100,
        "min": numeric.min(),
        "max": numeric.max(),
        "n_unique": numeric.nunique(),
    }).round(3)
    return summary.sort_values("nan_pct", ascending=False)


def _synthetic_data() -> dict:
    """
    Fake but structurally realistic data for 5 countries.
    Good enough to test the pipeline end-to-end.
    Not a substitute for real FRED/World Bank data.
    """
    np.random.seed(42)
    dates = pd.date_range("2000-01", "2024-12", freq="MS")
    countries = {"USA": "Americas", "DEU": "Europe", "BRA": "Americas",
                 "TUR": "EMEA", "JPN": "Asia"}

    data = {}
    for iso, region in countries.items():
        n = len(dates)
        def ar1(n, phi=0.92, mu=0, sigma=1):
            x = np.zeros(n)
            x[0] = np.random.randn() * sigma
            for t in range(1, n):
                x[t] = phi * x[t-1] + (1-phi)*mu + sigma*np.sqrt(1-phi**2)*np.random.randn()
            return x

        df = pd.DataFrame({
            "gdp_growth":          ar1(n, 0.85, mu=2.5, sigma=1.5),
            "inflation":           np.abs(ar1(n, 0.92, mu=2.0, sigma=1.2)),
            "unemployment":        np.clip(ar1(n, 0.95, mu=6.0, sigma=1.0), 2, 25),
            "current_account_pct": ar1(n, 0.88, mu=-1.5, sigma=2.0),
            "debt_gdp":            np.cumsum(ar1(n, 0.6, mu=0.1, sigma=1.0)) + 60,
            "broad_money_growth":  ar1(n, 0.80, mu=5.0, sigma=2.5),
            "policy_rate":         np.clip(ar1(n, 0.94, mu=3.0, sigma=1.2), 0, 25),
            "T10Y2Y":              ar1(n, 0.90, mu=1.2, sigma=0.8),
            "VIX":                 np.abs(ar1(n, 0.82, mu=18, sigma=5)) + 10,
            "BAA_spread":          np.abs(ar1(n, 0.88, mu=2.0, sigma=0.6)) + 1.0,
            "DXY":                 np.cumsum(ar1(n, 0.5, mu=0, sigma=0.3)) + 95,
            "GSCI":                np.cumsum(ar1(n, 0.5, mu=0, sigma=1.5)) + 400,
            "EEM":                 np.cumsum(ar1(n, 0.6, mu=0.3, sigma=2.0)) + 35,
            "HYG":                 np.cumsum(ar1(n, 0.7, mu=0.1, sigma=0.8)) + 85,
        }, index=dates)

        df["region"] = region
        data[iso] = df

    return data


def main():
    print("Redline Macro — feature engineering pipeline")
    print("=" * 60)

    print("\n[1/3] loading raw data...")
    country_data = load_raw_data()
    if not country_data:
        print("  no raw data found — generating synthetic data for 5 countries")
        country_data = _synthetic_data()

    print(f"\n[2/3] building features for {len(country_data)} countries...")
    features = build_feature_matrix(country_data)

    out_path = PROCESSED / "features.parquet"
    print(f"\n[3/3] writing {out_path}...")
    features.to_parquet(out_path, index=True, engine="pyarrow", compression="snappy")
    print(f"  done. {out_path.stat().st_size / 1e6:.1f} MB")

    summary = feature_summary(features)
    print("\n  top 10 features by NaN rate:")
    print(summary.head(10).to_string())

    return features


if __name__ == "__main__":
    features = main()