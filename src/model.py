"""
src/model.py — Redline Macro
Walk-forward crisis prediction: LogReg, XGBoost, LSTM, Ensemble.
Regimes break slowly, then all at once.
"""

import json
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
import optuna
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("redline")

REPORTS_DIR = Path("reports")
MODELS_DIR  = Path("models")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

TRAIN_YEARS   = 10
TEST_YEARS    = 1
LOOKBACK      = 12
OPTUNA_TRIALS = 50
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
XGB_WEIGHT    = 0.5
THRESHOLDS    = [0.5, 0.6, 0.7]

CRISIS_EVENTS: dict = {
    "GFC_2008":       {"country": "USA",   "start": "2008-09", "label": "Global Financial Crisis"},
    "EURO_DEBT_2011": {"country": "GRC",   "start": "2011-07", "label": "European Debt Crisis"},
    "RUSSIA_2014":    {"country": "RUS",   "start": "2014-12", "label": "Russia Crisis 2014"},
    "BRAZIL_2015":    {"country": "BRA",   "start": "2015-08", "label": "Brazil Recession 2015"},
    "TURKEY_2018":    {"country": "TUR",   "start": "2018-08", "label": "Turkey Currency Crisis"},
    "ARGENTINA_2019": {"country": "ARG",   "start": "2019-04", "label": "Argentina Crisis 2019"},
    "COVID_2020":     {"country": "WORLD", "start": "2020-03", "label": "COVID-19 Economic Shock"},
    "SRI_LANKA_2022": {"country": "LKA",   "start": "2022-04", "label": "Sri Lanka Default 2022"},
}

EXCLUDE_COLS = {
    'region', 'crisis_6m', 'crisis_severity', 'months_to_crisis',
    'crisis_recession_6m', 'crisis_growth_shock_6m',
    'crisis_financial_6m', 'crisis_any_6m',
}

NUMERIC_DTYPES = {'float64', 'float32', 'int64', 'int32', 'int8', 'uint8'}


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class WalkForwardSplit:
    fold:         int
    train_start:  pd.Timestamp
    train_end:    pd.Timestamp
    test_start:   pd.Timestamp
    test_end:     pd.Timestamp
    X_train:      np.ndarray
    y_train:      np.ndarray
    X_test:       np.ndarray
    y_test:       np.ndarray
    dates_test:   pd.DatetimeIndex
    country_test: np.ndarray


@dataclass
class FoldResult:
    fold:          int
    dates:         pd.DatetimeIndex
    countries:     np.ndarray
    y_true:        np.ndarray
    prob_lr:       np.ndarray
    prob_xgb:      np.ndarray
    prob_lstm:     np.ndarray
    prob_ensemble: np.ndarray
    metrics:       dict = field(default_factory=dict)


# ─────────────────────────────────────────────
#  Walk-forward splitter
# ─────────────────────────────────────────────

def make_splits(df: pd.DataFrame,
                date_col: str = "date",
                target_col: str = "crisis",
                country_col: str = "country",
                feature_cols: Optional[list] = None) -> list:

    df = df.sort_values(date_col).reset_index(drop=True)

    if feature_cols is None:
        exclude = EXCLUDE_COLS | {date_col, target_col, country_col}
        feature_cols = [
            c for c in df.columns
            if c not in exclude
            and str(df[c].dtype) in NUMERIC_DTYPES
        ]
        log.info(f"Using {len(feature_cols)} feature columns")

    # clean inf and NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)

    # drop rows where target is missing
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    dates = df[date_col].values.astype("datetime64[M]")
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df[target_col].values.astype(np.float32)
    c_all = df[country_col].values

    min_date = pd.Timestamp(dates[0])
    max_date = pd.Timestamp(dates[-1])

    splits = []
    fold   = 0
    train_start = min_date

    while True:
        train_end  = train_start + pd.DateOffset(years=TRAIN_YEARS) - pd.DateOffset(months=1)
        test_start = train_end   + pd.DateOffset(months=1)
        test_end   = test_start  + pd.DateOffset(years=TEST_YEARS)  - pd.DateOffset(months=1)

        if test_end > max_date:
            break

        tr_mask = (pd.DatetimeIndex(dates) >= train_start) & (pd.DatetimeIndex(dates) <= train_end)
        te_mask = (pd.DatetimeIndex(dates) >= test_start)  & (pd.DatetimeIndex(dates) <= test_end)

        if tr_mask.sum() < 24 or te_mask.sum() < 1:
            break

        splits.append(WalkForwardSplit(
            fold         = fold,
            train_start  = train_start,
            train_end    = train_end,
            test_start   = test_start,
            test_end     = test_end,
            X_train      = X_all[tr_mask],
            y_train      = y_all[tr_mask],
            X_test       = X_all[te_mask],
            y_test       = y_all[te_mask],
            dates_test   = pd.DatetimeIndex(dates[te_mask]),
            country_test = c_all[te_mask],
        ))

        fold += 1
        train_start += pd.DateOffset(years=1)

    if not splits:
        raise ValueError("No valid walk-forward splits created — check date range and data size")

    log.info(f"Created {len(splits)} walk-forward folds "
             f"({splits[0].train_start.date()} -> {splits[-1].test_end.date()})")
    return splits


# ─────────────────────────────────────────────
#  Model 1 — Logistic Regression (ElasticNet)
# ─────────────────────────────────────────────

def fit_logreg(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    best_auc, best_pipe = -1, None

    for C in [0.001, 0.01, 0.1, 1.0]:
        for l1 in [0.0, 0.15, 0.5, 0.85, 1.0]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=l1,
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                )),
            ])
            split_at = int(len(X_train) * 0.8)
            try:
                pipe.fit(X_train[:split_at], y_train[:split_at])
                prob = pipe.predict_proba(X_train[split_at:])[:, 1]
                if y_train[split_at:].sum() < 2:
                    continue
                auc = average_precision_score(y_train[split_at:], prob)
                if auc > best_auc:
                    best_auc, best_pipe = auc, pipe
            except Exception:
                continue

    if best_pipe is None:
        best_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="elasticnet", solver="saga",
                l1_ratio=0.5, C=0.01,
                class_weight="balanced", max_iter=2000, random_state=42,
            )),
        ])

    best_pipe.fit(X_train, y_train)
    return best_pipe


# ─────────────────────────────────────────────
#  Model 2 — XGBoost + Optuna
# ─────────────────────────────────────────────

def _xgb_default(pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        max_depth=4, learning_rate=0.05, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        verbosity=0, random_state=42, tree_method="hist",
    )


def fit_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    split_at  = int(len(X_train) * 0.8)
    Xtr, Xval = X_train[:split_at], X_train[split_at:]
    ytr, yval = y_train[:split_at], y_train[split_at:]

    pos_weight = float((ytr == 0).sum()) / float(max((ytr == 1).sum(), 1))

    # skip tuning if val has no positives
    if yval.sum() < 2:
        log.warning("XGBoost: val set has no positives — using default params")
        model = _xgb_default(pos_weight)
        model.fit(X_train, y_train, verbose=False)
        return model

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            n_estimators     = trial.suggest_int("n_estimators", 200, 800),
            scale_pos_weight = pos_weight,
            eval_metric      = "aucpr",
            verbosity        = 0,
            random_state     = 42,
            tree_method      = "hist",
        )
        clf = xgb.XGBClassifier(**params)
        clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        prob = clf.predict_proba(Xval)[:, 1]
        return average_precision_score(yval, prob)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        log.warning("XGBoost: all Optuna trials pruned — using default params")
        model = _xgb_default(pos_weight)
        model.fit(X_train, y_train, verbose=False)
        return model

    best = study.best_params
    best.update(dict(
        scale_pos_weight = pos_weight,
        eval_metric      = "aucpr",
        verbosity        = 0,
        random_state     = 42,
        tree_method      = "hist",
    ))
    model = xgb.XGBClassifier(**best)
    model.fit(X_train, y_train, verbose=False)
    return model


# ─────────────────────────────────────────────
#  Model 3 — LSTM (PyTorch)
# ─────────────────────────────────────────────

class AttentionPool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.w(h).squeeze(-1)
        alpha  = torch.softmax(scores, dim=-1)
        return (h * alpha.unsqueeze(-1)).sum(dim=1)


class CrisisLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.attn = AttentionPool(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        h, _ = self.lstm(x)
        c = self.attn(h)
        return self.head(c).squeeze(-1)


def _make_sequences(X: np.ndarray, y: np.ndarray, lookback: int = LOOKBACK):
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i - lookback: i])
        ys.append(y[i])
    if not xs:
        return np.zeros((0, lookback, X.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def fit_lstm(X_train: np.ndarray, y_train: np.ndarray, n_features: int):
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_train)
    Xseq, yseq = _make_sequences(X_s, y_train)

    if len(Xseq) < 20:
        log.warning("LSTM: not enough sequences — returning untrained model")
        return CrisisLSTM(n_features=n_features).to(DEVICE), scaler

    split_at  = int(len(Xseq) * 0.85)
    Xtr, Xval = Xseq[:split_at], Xseq[split_at:]
    ytr, yval = yseq[:split_at], yseq[split_at:]

    pos_w      = float((ytr == 0).sum()) / float(max((ytr == 1).sum(), 1))
    pos_weight = torch.tensor([pos_w], dtype=torch.float32).to(DEVICE)

    ds_tr  = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    ds_val = TensorDataset(torch.tensor(Xval), torch.tensor(yval))
    dl_tr  = DataLoader(ds_tr, batch_size=64, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=256)

    model = CrisisLSTM(n_features=n_features).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    crit  = nn.BCELoss(reduction="none")

    best_val_auc   = -1.0
    patience_count = 0
    best_state     = None
    PATIENCE       = 10

    for epoch in range(200):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            w    = torch.where(yb == 1, pos_weight.expand_as(yb), torch.ones_like(yb))
            loss = (crit(pred, yb) * w).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        preds_val = []
        with torch.no_grad():
            for xb, _ in dl_val:
                preds_val.append(model(xb.to(DEVICE)).cpu().numpy())
        preds_val = np.concatenate(preds_val)

        if yval.sum() >= 2:
            val_auc = average_precision_score(yval, preds_val)
            if val_auc > best_val_auc:
                best_val_auc   = val_auc
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, scaler


def predict_lstm(model: CrisisLSTM, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    X_s     = scaler.transform(X)
    Xseq, _ = _make_sequences(X_s, np.zeros(len(X_s)))
    padded  = np.full(len(X), np.nan, dtype=np.float32)

    if len(Xseq) == 0:
        return padded

    ds = TensorDataset(torch.tensor(Xseq))
    dl = DataLoader(ds, batch_size=256)
    model.eval()
    probs = []
    with torch.no_grad():
        for (xb,) in dl:
            probs.append(model(xb.to(DEVICE)).cpu().numpy())
    probs = np.concatenate(probs)
    padded[LOOKBACK:] = probs
    return padded


# ─────────────────────────────────────────────
#  Calibration
# ─────────────────────────────────────────────

def calibrate(prob_raw: np.ndarray, y_true: np.ndarray,
              prob_test: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(prob_raw)
    if valid.sum() < 20 or y_true[valid].sum() < 2:
        return prob_test
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_raw[valid], y_true[valid])
    out = np.full_like(prob_test, np.nan)
    nv  = ~np.isnan(prob_test)
    out[nv] = iso.predict(prob_test[nv])
    return out


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────

def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins    = np.linspace(0, 1, n_bins + 1)
    total   = len(y_true)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc     = y_true[mask].mean()
        conf    = probs[mask].mean()
        ece_val += mask.sum() / total * abs(acc - conf)
    return float(ece_val)


def optimal_threshold_f1(y_true: np.ndarray, probs: np.ndarray):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, label: str) -> dict:
    valid = ~np.isnan(probs)
    yt    = y_true[valid]
    yp    = probs[valid]
    if len(yt) == 0 or yt.sum() < 2:
        return {"model": label, "note": "insufficient positives"}
    t, f1 = optimal_threshold_f1(yt, yp)
    return {
        "model":     label,
        "auc_roc":   round(float(roc_auc_score(yt, yp)), 4),
        "auc_pr":    round(float(average_precision_score(yt, yp)), 4),
        "f1":        round(float(f1), 4),
        "f1_thresh": round(float(t), 3),
        "brier":     round(float(brier_score_loss(yt, yp)), 4),
        "ece":       round(float(ece(yt, yp)), 4),
    }


# ─────────────────────────────────────────────
#  Single fold runner
# ─────────────────────────────────────────────

def run_fold(split: WalkForwardSplit) -> FoldResult:
    fold   = split.fold
    n_feat = split.X_train.shape[1]

    log.info(f"Fold {fold:02d} | train {split.train_start.date()}->{split.train_end.date()} "
             f"| test {split.test_start.date()}->{split.test_end.date()} "
             f"| n_train={len(split.X_train)} pos={int(split.y_train.sum())}")

    lr_pipe   = fit_logreg(split.X_train, split.y_train)
    prob_lr   = lr_pipe.predict_proba(split.X_test)[:, 1]

    xgb_model = fit_xgboost(split.X_train, split.y_train)
    prob_xgb  = xgb_model.predict_proba(split.X_test)[:, 1]

    X_for_lstm     = np.vstack([split.X_train[-LOOKBACK:], split.X_test])
    lstm_model, scaler_lstm = fit_lstm(split.X_train, split.y_train, n_feat)
    prob_lstm_full = predict_lstm(lstm_model, scaler_lstm, X_for_lstm)
    prob_lstm      = prob_lstm_full[LOOKBACK:]

    prob_xgb_train = xgb_model.predict_proba(split.X_train)[:, 1]
    prob_xgb_cal   = calibrate(prob_xgb_train, split.y_train, prob_xgb)

    lstm_train_full = predict_lstm(lstm_model, scaler_lstm, split.X_train)
    valid_tr        = ~np.isnan(lstm_train_full)
    prob_lstm_cal   = calibrate(
        lstm_train_full[valid_tr],
        split.y_train[valid_tr],
        prob_lstm,
    )

    prob_ens   = np.full(len(split.y_test), np.nan, dtype=np.float32)
    both_valid = (~np.isnan(prob_xgb_cal)) & (~np.isnan(prob_lstm_cal))
    prob_ens[both_valid] = (
        XGB_WEIGHT       * prob_xgb_cal[both_valid] +
        (1 - XGB_WEIGHT) * prob_lstm_cal[both_valid]
    )
    only_xgb  = ~np.isnan(prob_xgb_cal) & np.isnan(prob_lstm_cal)
    only_lstm = np.isnan(prob_xgb_cal) & ~np.isnan(prob_lstm_cal)
    prob_ens[only_xgb]  = prob_xgb_cal[only_xgb]
    prob_ens[only_lstm] = prob_lstm_cal[only_lstm]

    metrics = {
        "logreg":   compute_metrics(split.y_test, prob_lr,       "logreg"),
        "xgboost":  compute_metrics(split.y_test, prob_xgb_cal,  "xgboost"),
        "lstm":     compute_metrics(split.y_test, prob_lstm_cal,  "lstm"),
        "ensemble": compute_metrics(split.y_test, prob_ens,       "ensemble"),
    }

    return FoldResult(
        fold          = fold,
        dates         = split.dates_test,
        countries     = split.country_test,
        y_true        = split.y_test,
        prob_lr       = prob_lr,
        prob_xgb      = prob_xgb_cal,
        prob_lstm     = prob_lstm_cal,
        prob_ensemble = prob_ens,
        metrics       = metrics,
    )


# ─────────────────────────────────────────────
#  Aggregate metrics
# ─────────────────────────────────────────────

def aggregate_metrics(results: list) -> dict:
    y_all   = np.concatenate([r.y_true        for r in results])
    pr_lr   = np.concatenate([r.prob_lr       for r in results])
    pr_xgb  = np.concatenate([r.prob_xgb      for r in results])
    pr_lstm = np.concatenate([r.prob_lstm     for r in results])
    pr_ens  = np.concatenate([r.prob_ensemble for r in results])

    return {
        "logreg":   compute_metrics(y_all, pr_lr,   "logreg"),
        "xgboost":  compute_metrics(y_all, pr_xgb,  "xgboost"),
        "lstm":     compute_metrics(y_all, pr_lstm,  "lstm"),
        "ensemble": compute_metrics(y_all, pr_ens,   "ensemble"),
    }


# ─────────────────────────────────────────────
#  Backtest
# ─────────────────────────────────────────────

def _months_before(crisis_start: pd.Timestamp,
                   prob_series: pd.Series,
                   threshold: float) -> Optional[int]:
    crossed = prob_series[prob_series >= threshold]
    if crossed.empty:
        return None
    first_cross = crossed.index.min()
    delta = (crisis_start.to_period("M") - first_cross.to_period("M")).n
    return int(delta)


def backtest_crisis(results: list):
    rows = []
    for r in results:
        for i, d in enumerate(r.dates):
            val = r.prob_ensemble[i]
            rows.append({
                "date":     d,
                "country":  r.countries[i],
                "y_true":   float(r.y_true[i]),
                "ensemble": float(val) if not np.isnan(val) else np.nan,
            })
    df_ts = pd.DataFrame(rows).sort_values("date")

    backtest_out = {}
    summary_rows = []

    for event_id, meta in CRISIS_EVENTS.items():
        country      = meta["country"]
        crisis_start = pd.Timestamp(meta["start"])
        label        = meta["label"]

        window_start = crisis_start - pd.DateOffset(months=36)
        window_end   = crisis_start + pd.DateOffset(months=12)

        sub = df_ts[
            (df_ts["country"] == country) &
            (df_ts["date"] >= window_start) &
            (df_ts["date"] <= window_end)
        ].copy()

        if sub.empty:
            backtest_out[event_id] = {"note": "no data for this country/period"}
            summary_rows.append({
                "event": event_id, "label": label, "verdict": "NO DATA",
                "lead_0.5": "-", "lead_0.6": "-", "lead_0.7": "-",
                "peak_prob": "-", "false_alarm_rate": "-",
            })
            continue

        prob_series = sub.set_index("date")["ensemble"].dropna()
        lead_times  = {t: _months_before(crisis_start, prob_series, t) for t in THRESHOLDS}
        peak_prob   = float(prob_series.max()) if not prob_series.empty else 0.0

        pre = df_ts[
            (df_ts["country"] == country) &
            (df_ts["date"] >= crisis_start - pd.DateOffset(months=24)) &
            (df_ts["date"] < crisis_start)
        ]
        if not pre.empty and pre["ensemble"].notna().sum() > 0:
            fa_denom = pre["ensemble"].notna().sum()
            fa_num   = ((pre["ensemble"] >= 0.5) & (pre["y_true"] == 0)).sum()
            false_alarm_rate = round(float(fa_num) / float(fa_denom), 3)
        else:
            false_alarm_rate = None

        lead_05 = lead_times.get(0.5)
        if lead_05 is None:
            verdict = "MISSED"
        elif lead_05 >= 6:
            verdict = "CAUGHT EARLY"
        else:
            verdict = "CAUGHT LATE"

        if peak_prob < 0.4 and false_alarm_rate is not None and false_alarm_rate > 0.3:
            verdict = "FALSE ALARM"

        backtest_out[event_id] = {
            "label":            label,
            "country":          country,
            "crisis_start":     str(crisis_start.date()),
            "lead_months":      {str(t): lead_times[t] for t in THRESHOLDS},
            "peak_prob":        round(peak_prob, 4),
            "false_alarm_rate": false_alarm_rate,
            "verdict":          verdict,
        }

        summary_rows.append({
            "event":            event_id,
            "label":            label[:28].ljust(28),
            "verdict":          verdict,
            "lead_0.5":         f"{lead_times[0.5]}m" if lead_times[0.5] is not None else "-",
            "lead_0.6":         f"{lead_times[0.6]}m" if lead_times[0.6] is not None else "-",
            "lead_0.7":         f"{lead_times[0.7]}m" if lead_times[0.7] is not None else "-",
            "peak_prob":        f"{peak_prob:.2f}",
            "false_alarm_rate": f"{false_alarm_rate:.2f}" if false_alarm_rate is not None else "-",
        })

    return backtest_out, summary_rows


# ─────────────────────────────────────────────
#  ASCII tables
# ─────────────────────────────────────────────

VERDICT_EMOJI = {
    "CAUGHT EARLY": "V",
    "CAUGHT LATE":  "~",
    "MISSED":       "X",
    "FALSE ALARM":  "!",
    "NO DATA":      "?",
}


def print_metrics_table(global_metrics: dict):
    header = f"{'MODEL':<12} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>7} {'BRIER':>8} {'ECE':>7}"
    sep    = "-" * len(header)
    print("\n" + "=" * len(header))
    print("  REDLINE MACRO -- Model Performance (walk-forward, pooled)")
    print("=" * len(header))
    print(header)
    print(sep)
    for key, m in global_metrics.items():
        if "note" in m:
            print(f"  {key:<10}  (insufficient positives)")
            continue
        print(f"  {m['model']:<10} "
              f"{m.get('auc_roc', '-'):>8} "
              f"{m.get('auc_pr', '-'):>8} "
              f"{m.get('f1', '-'):>7} "
              f"{m.get('brier', '-'):>8} "
              f"{m.get('ece', '-'):>7}")
    print("=" * len(header) + "\n")


def print_backtest_table(summary_rows: list):
    hdr = (f"  {'EVENT':<22} {'LABEL':<30} {'VRD':>4}  "
           f"{'@0.5':>5} {'@0.6':>5} {'@0.7':>5}  "
           f"{'PEAK':>5}  {'FA%':>5}")
    sep = "-" * (len(hdr) - 2)
    print("=" * (len(hdr) - 2))
    print("  REDLINE MACRO -- Crisis Backtest Summary")
    print("=" * (len(hdr) - 2))
    print(hdr)
    print(sep)
    for r in summary_rows:
        sym = VERDICT_EMOJI.get(r["verdict"], "?")
        print(f"  {r['event']:<22} {r['label']:<30} {sym:>4}  "
              f"{r['lead_0.5']:>5} {r['lead_0.6']:>5} {r['lead_0.7']:>5}  "
              f"{r['peak_prob']:>5}  {r['false_alarm_rate']:>5}")
    print("=" * (len(hdr) - 2))
    print()
    print("  Verdicts:  V CAUGHT EARLY (>=6m advance)  ~ CAUGHT LATE  X MISSED  ! FALSE ALARM")
    print("  Lead times = months of advance warning at each probability threshold")
    print()


# ─────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame,
                 date_col: str    = "date",
                 target_col: str  = "crisis",
                 country_col: str = "country",
                 feature_cols: Optional[list] = None) -> dict:

    log.info("=" * 60)
    log.info("  REDLINE MACRO  |  Walk-Forward Crisis Prediction")
    log.info("=" * 60)

    splits  = make_splits(df, date_col, target_col, country_col, feature_cols)
    results = []

    for split in splits:
        r = run_fold(split)
        results.append(r)
        m = r.metrics.get("ensemble", {})
        if "auc_pr" in m:
            log.info(f"  Fold {r.fold:02d}  AUC-PR={m['auc_pr']:.4f}  AUC-ROC={m['auc_roc']:.4f}")

    global_metrics = aggregate_metrics(results)
    print_metrics_table(global_metrics)

    backtest_data, summary_rows = backtest_crisis(results)
    print_backtest_table(summary_rows)

    report = {
        "meta": {
            "n_folds":       len(splits),
            "train_years":   TRAIN_YEARS,
            "test_years":    TEST_YEARS,
            "lookback":      LOOKBACK,
            "xgb_weight":    XGB_WEIGHT,
            "optuna_trials": OPTUNA_TRIALS,
            "device":        DEVICE,
        },
        "global_metrics":   global_metrics,
        "per_fold_metrics": [{"fold": r.fold, **r.metrics} for r in results],
        "backtest":         backtest_data,
    }

    out_path = REPORTS_DIR / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Report saved -> {out_path}")

    return report


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os

    features_path = "data/processed/features.parquet"

    if os.path.exists(features_path):
        log.info("Loading real data from features.parquet ...")
        df = pd.read_parquet(features_path).reset_index()
        log.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        run_pipeline(
            df,
            date_col    = "date",
            target_col  = "crisis_any_6m",
            country_col = "country",
        )
    else:
        log.warning("features.parquet not found — running smoke test with synthetic data")
        rng = np.random.default_rng(42)

        n         = 15 * 12 * 5
        dates     = pd.date_range("2005-01", periods=15 * 12, freq="MS").tolist() * 5
        ctries    = ["USA", "DEU", "BRA", "TUR", "GRC"]
        countries = [c for c in ctries for _ in range(15 * 12)]

        df = pd.DataFrame({"date": sorted(dates), "country": countries})
        df["crisis"] = (rng.random(n) < 0.05).astype(int)
        for i in range(8):
            df[f"feat_{i}"] = rng.standard_normal(n)

        run_pipeline(df, date_col="date", target_col="crisis", country_col="country")
        