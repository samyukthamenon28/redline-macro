"""
explain.py — SHAP explainability layer for Redline Macro

TreeExplainer on XGBoost. Fast. No kernel tricks.
Answers the question every risk manager actually wants:
"What was the model seeing before [crisis] blew up?"
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ── brand ─────────────────────────────────────────────────────────────────────
RED      = "#DC2626"
RED_DARK = "#991B1B"
RED_MID  = "#EF4444"
BG       = "#0A0A0A"
SURFACE  = "#111111"
BORDER   = "#1F1F1F"
TEXT_HI  = "#F5F5F5"
TEXT_MID = "#A3A3A3"
TEXT_LO  = "#525252"

# custom SHAP colormap — blue (safe) → dark → red (risk)
_SHAP_CMAP = LinearSegmentedColormap.from_list(
    "redline_shap",
    ["#3B82F6", "#1E3A5F", BG, RED_DARK, RED],
    N=256,
)

# ── named crisis windows ──────────────────────────────────────────────────────
# onset = first month of acute phase; we look 1–6 months prior
CRISIS_EVENTS: dict[str, dict] = {
    "Global Financial Crisis": {
        "onset":       "2008-09-01",
        "label":       "GFC",
        "description": "Lehman collapse → systemic credit freeze",
    },
    "Eurozone Debt Crisis": {
        "onset":       "2011-08-01",
        "label":       "EZ",
        "description": "Sovereign spread blowout, ECB backstop",
    },
    "China Taper Shock": {
        "onset":       "2015-08-01",
        "label":       "CNY",
        "description": "Renminbi devaluation + EM capital flight",
    },
    "COVID Macro Shock": {
        "onset":       "2020-03-01",
        "label":       "COVID",
        "description": "Pandemic demand collapse + supply freeze",
    },
    "Fed Tightening Cycle": {
        "onset":       "2022-03-01",
        "label":       "FEDTIGHT",
        "description": "Fastest hiking cycle in 40 years → EM stress",
    },
}


# ── matplotlib dark theme ─────────────────────────────────────────────────────
def _apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT_MID,
        "axes.titlecolor":   TEXT_HI,
        "xtick.color":       TEXT_LO,
        "ytick.color":       TEXT_LO,
        "text.color":        TEXT_HI,
        "grid.color":        BORDER,
        "grid.alpha":        0.5,
        "font.family":       "monospace",
        "font.size":         9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class RedlineExplainer:

    def __init__(self, model: xgb.XGBClassifier, X: pd.DataFrame):
        self.model = model
        self.X     = X  # full feature matrix, DatetimeIndex

        print("[explain] fitting TreeExplainer...")
        self.explainer   = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X)

        # XGBoost binary → shap_values may be list[array]; take positive class
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]

        self.shap_df = pd.DataFrame(
            self.shap_values,
            index=X.index,
            columns=X.columns,
        )

        # global importance — used everywhere
        self.mean_abs = (
            self.shap_df.abs()
            .mean()
            .sort_values(ascending=False)
        )

        print(f"[explain] ready. {X.shape[1]} features, {X.shape[0]} samples.")

    # ── global plots ──────────────────────────────────────────────────────────

    def plot_beeswarm(self, top_n: int = 20, out: Optional[Path] = None):
        """
        Beeswarm of top_n features. SHAP native plot, Redline colors.
        Saves to reports/shap_beeswarm.png.
        """
        out = out or (REPORTS / "shap_beeswarm.png")
        _apply_dark_theme()

        top_features = self.mean_abs.head(top_n).index.tolist()
        X_top        = self.X[top_features]
        sv_top       = self.shap_df[top_features].values

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.sca(ax)

        shap.summary_plot(
            sv_top,
            X_top,
            feature_names=top_features,
            show=False,
            color=_SHAP_CMAP,
            plot_type="dot",
            max_display=top_n,
            alpha=0.55,
        )

        fig = plt.gcf()
        ax  = plt.gca()
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("SHAP value  (← lowers crisis prob  |  raises crisis prob →)",
                      color=TEXT_MID, fontsize=8)
        ax.tick_params(colors=TEXT_MID)
        ax.axvline(0, color=RED_DARK, linewidth=0.8, alpha=0.5, zorder=0)

        # brand stamp
        fig.text(0.01, 0.99,
                 "REDLINE MACRO  //  GLOBAL FEATURE ATTRIBUTION",
                 color=RED, fontsize=8, fontweight="bold",
                 ha="left", va="top", transform=fig.transFigure,
                 fontfamily="monospace")
        fig.text(0.01, 0.965,
                 f"Top {top_n} features  ·  mean |SHAP| ranking  ·  {len(self.X):,} obs",
                 color=TEXT_LO, fontsize=7,
                 ha="left", va="top", transform=fig.transFigure,
                 fontfamily="monospace")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"[explain] beeswarm → {out}")

    def plot_importance_bar(self, top_n: int = 20, out: Optional[Path] = None):
        """
        Horizontal bar chart: mean |SHAP|. Clean terminal aesthetic.
        Saves to reports/shap_importance.png.
        """
        out = out or (REPORTS / "shap_importance.png")
        _apply_dark_theme()

        top = self.mean_abs.head(top_n).sort_values()  # ascending → top at apex
        norm = top.values / top.max()

        # interpolate BORDER → RED
        hex_colors = [
            "#{:02X}{:02X}{:02X}".format(
                int(0x1F + (0xDC - 0x1F) * n),
                int(0x1F + (0x26 - 0x1F) * n),
                int(0x1F + (0x26 - 0x1F) * n),
            )
            for n in norm
        ]

        fig, ax = plt.subplots(figsize=(10, 7))

        bars = ax.barh(
            range(len(top)), top.values,
            color=hex_colors, height=0.65, linewidth=0,
        )

        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=8, color=TEXT_MID, fontfamily="monospace")
        ax.set_xlabel("mean |SHAP value|", color=TEXT_MID, fontsize=8)

        for i, (bar, val) in enumerate(zip(bars, top.values)):
            ax.text(val + top.max() * 0.01, i, f"{val:.4f}",
                    va="center", fontsize=7, color=TEXT_LO, fontfamily="monospace")

        for i in range(len(top)):
            rank = len(top) - i
            ax.text(-top.max() * 0.02, i, f"#{rank:02d}",
                    va="center", ha="right", fontsize=6,
                    color=RED_MID if rank <= 5 else TEXT_LO,
                    fontfamily="monospace")

        ax.set_xlim(-top.max() * 0.08, top.max() * 1.12)
        ax.set_facecolor(SURFACE)
        fig.patch.set_facecolor(BG)
        ax.grid(axis="x", color=BORDER, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)

        fig.text(0.02, 0.98,
                 "REDLINE MACRO  //  FEATURE IMPORTANCE",
                 color=RED, fontsize=8, fontweight="bold",
                 ha="left", va="top", transform=fig.transFigure,
                 fontfamily="monospace")
        fig.text(0.02, 0.955,
                 f"mean |SHAP| across all {len(self.X):,} observations  ·  XGBoost TreeExplainer",
                 color=TEXT_LO, fontsize=7,
                 ha="left", va="top", transform=fig.transFigure,
                 fontfamily="monospace")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"[explain] importance bar → {out}")

    # ── per-crisis driver analysis ────────────────────────────────────────────

    def analyze_crisis_drivers(
        self,
        lookback_months: int = 6,
        top_k:           int = 5,
        out: Optional[Path]  = None,
    ) -> dict:
        """
        For each named crisis, average SHAP values in the [onset-6m, onset-1m] window.

        Tells you what the model was reacting to while the crisis was building —
        not what happened after the break, but the 6-month signal accumulation.

        Returns the dict and saves to reports/crisis_drivers.json.
        """
        out = out or (REPORTS / "crisis_drivers.json")
        results = {}

        for name, meta in CRISIS_EVENTS.items():
            onset        = pd.Timestamp(meta["onset"])
            window_end   = onset - pd.DateOffset(months=1)
            window_start = onset - pd.DateOffset(months=lookback_months)

            mask        = (self.shap_df.index >= window_start) & (self.shap_df.index <= window_end)
            window_shap = self.shap_df.loc[mask]

            if window_shap.empty:
                print(f"[explain] {meta['label']}: no data in window, skipping")
                continue

            # mean SHAP per feature — this is the core signal
            mean_shap = window_shap.mean().sort_values(key=abs, ascending=False)
            top_abs   = mean_shap.abs().sort_values(ascending=False).head(top_k)
            top_drivers = top_abs.index.tolist()

            # regime characterisation
            n_pos = (mean_shap.head(20) > 0).sum()
            n_neg = (mean_shap.head(20) < 0).sum()
            regime_bias = (
                "risk-OFF amplification" if n_pos >= n_neg
                else "risk-ON suppression"
            )

            dominant = mean_shap.abs().idxmax()
            dom_dir  = "elevated" if mean_shap[dominant] > 0 else "suppressed"
            dom_val  = float(mean_shap[dominant])

            # what share of total global signal did these drivers represent?
            signal_share = (
                top_abs.sum() / max(self.mean_abs.sum(), 1e-9) * 100
            )

            interpretation = (
                f"{dominant} was {dom_dir} (SHAP={dom_val:+.4f}), "
                f"dominating the {lookback_months}-month pre-onset window for {meta['label']}. "
                f"Top {top_k} drivers carry {signal_share:.1f}% of total model signal in this period. "
                f"Regime: {regime_bias} ({n_pos} amplifiers vs {n_neg} dampeners in top 20). "
                f"Context: {meta['description']}."
            )

            results[name] = {
                "onset":   meta["onset"],
                "label":   meta["label"],
                "top_drivers": top_drivers,
                "drivers_detail": {
                    feat: {
                        "mean_shap": round(float(mean_shap[feat]), 6),
                        "direction": "amplifier" if mean_shap[feat] > 0 else "dampener",
                        "abs_rank":  int(list(top_abs.index).index(feat)) + 1,
                    }
                    for feat in top_drivers
                },
                "window": {
                    "start": str(window_start.date()),
                    "end":   str(window_end.date()),
                    "obs":   int(mask.sum()),
                },
                "regime":         regime_bias,
                "interpretation": interpretation,
            }

            print(
                f"[explain] {name:30s} | top: {top_drivers[0]:30s} | "
                f"SHAP={dom_val:+.4f} | obs={mask.sum()}"
            )

        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[explain] crisis drivers → {out}")
        return results

    # ── country snapshot ──────────────────────────────────────────────────────

    def get_country_snapshot(
        self,
        country:         str,
        date:            str | pd.Timestamp,
        country_col:     str = "country",
        lookback_months: int = 6,
    ) -> dict:
        """
        Dashboard drill-down for a single country at a point in time.

        Returns:
          - current risk score + tier
          - 6-month trend with direction
          - top 5 SHAP drivers with values and directions
          - comparison to historical baseline (z-score + percentile)

        Used by dashboard/app.py → country drill-down page.
        """
        date = pd.Timestamp(date)

        # filter to country if country_col present; else assume X is pre-filtered
        if country_col in self.X.columns:
            c_mask = self.X[country_col] == country
            X_c    = self.X.loc[c_mask]
            shap_c = self.shap_df.loc[c_mask]
        else:
            X_c    = self.X
            shap_c = self.shap_df

        if X_c.empty:
            return {"error": f"no data for country={country}"}

        # closest available date at or before requested date
        available = X_c.index.unique().sort_values()
        prior     = available[available <= date]
        if prior.empty:
            return {"error": f"no data on or before {date.date()} for {country}"}
        current_date = prior[-1]

        # feature cols = everything except country label
        feat_cols = [c for c in X_c.columns if c != country_col]

        # ── current risk score ────────────────────────────────────────────────
        row  = X_c.loc[[current_date], feat_cols].iloc[[-1]]
        prob = float(self.model.predict_proba(row)[0, 1])

        # ── 6-month trend ─────────────────────────────────────────────────────
        t_start  = date - pd.DateOffset(months=lookback_months)
        t_mask   = (X_c.index >= t_start) & (X_c.index <= date)
        X_trend  = X_c.loc[t_mask, feat_cols]

        if len(X_trend) < 2:
            trend_scores    = [prob]
            trend_dates_str = [str(current_date.date())]
            delta           = 0.0
            direction       = "insufficient_data"
        else:
            trend_scores = [
                float(self.model.predict_proba(X_trend.iloc[[i]])[0, 1])
                for i in range(len(X_trend))
            ]
            trend_dates_str = X_trend.index.strftime("%Y-%m-%d").tolist()
            delta = trend_scores[-1] - trend_scores[0]
            if delta > 0.05:
                direction = "deteriorating"
            elif delta < -0.05:
                direction = "improving"
            else:
                direction = "stable"

        # ── top 5 SHAP drivers at current date ───────────────────────────────
        shap_row = shap_c.loc[current_date]
        if isinstance(shap_row, pd.DataFrame):
            shap_row = shap_row.iloc[-1]
        shap_row = shap_row.drop(labels=[country_col], errors="ignore")

        top5_feats = shap_row.abs().sort_values(ascending=False).head(5).index
        top5_drivers = [
            {
                "feature":     feat,
                "shap_value":  round(float(shap_row[feat]), 6),
                "direction":   "↑ risk" if shap_row[feat] > 0 else "↓ risk",
                "feature_val": round(float(row[feat].iloc[0]), 4) if feat in feat_cols else None,
            }
            for feat in top5_feats
        ]

        # ── historical baseline comparison ────────────────────────────────────
        shap_hist = shap_c.drop(columns=[country_col], errors="ignore")

        if len(shap_hist) >= 6:
            top_feat = top5_feats[0]
            feat_history = shap_hist[top_feat]
            h_mean = float(feat_history.mean())
            h_std  = float(feat_history.std()) if feat_history.std() > 1e-9 else 1.0
            z      = (float(shap_row[top_feat]) - h_mean) / h_std

            # crisis prob percentile rank over last 5yrs (max 60 months)
            hist_dates = available[-min(60, len(available)):]
            hist_probs = np.array([
                float(self.model.predict_proba(X_c.loc[[d], feat_cols].iloc[[-1]])[0, 1])
                for d in hist_dates
            ])
            pct_rank = float(np.mean(hist_probs <= prob)) * 100

            if z > 2.0:
                regime_label = f"CRITICAL — top driver {z:.1f}σ above historical mean"
            elif z > 1.5:
                regime_label = f"ELEVATED — top driver {z:.1f}σ above historical mean"
            elif z < -1.5:
                regime_label = f"SUPPRESSED — top driver {z:.1f}σ below historical mean"
            else:
                regime_label = "NORMAL — within historical range"

            baseline = {
                "crisis_prob_pct_rank":       round(pct_rank, 1),
                "top_driver_z_score":         round(z, 3),
                "top_driver_historical_mean": round(h_mean, 6),
                "regime":                     regime_label,
            }
        else:
            baseline = {"note": "insufficient history for baseline (<6 obs)"}

        return {
            "country":     country,
            "as_of":       str(current_date.date()),
            "crisis_prob": round(prob, 4),
            "risk_tier":   _risk_tier(prob),
            "trend": {
                "direction":  direction,
                "delta_6m":   round(delta, 4),
                "scores":     [round(s, 4) for s in trend_scores],
                "dates":      trend_dates_str,
            },
            "top_drivers": top5_drivers,
            "baseline":    baseline,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _risk_tier(prob: float) -> str:
    # thresholds calibrated to historical crisis base rates (~8% annual)
    if prob >= 0.75:   return "CRITICAL"
    elif prob >= 0.55: return "HIGH"
    elif prob >= 0.35: return "ELEVATED"
    elif prob >= 0.15: return "MODERATE"
    else:              return "LOW"


def run_full_explain_pipeline(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> "RedlineExplainer":
    """
    One-call entry from train.py / notebooks:
        explainer = run_full_explain_pipeline(model, X_test)

    Saves all reports/ artifacts, returns the explainer object for dashboard use.
    """
    exp = RedlineExplainer(model, X)
    exp.plot_beeswarm()
    exp.plot_importance_bar()
    exp.analyze_crisis_drivers()
    print("[explain] pipeline complete. reports/ is current.")
    return exp


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[explain] smoke test — synthetic macro data...")

    rng   = np.random.default_rng(42)
    n     = 500
    dates = pd.date_range("2004-01-01", periods=n, freq="MS")

    feature_names = [
        "credit_spread_zscore", "yield_curve_slope", "vix_x_credit",
        "fx_vol_zscore",        "reserves_months_import", "ca_gdp_ratio",
        "debt_gdp_ratio",       "real_rate_diff",     "bank_credit_growth",
        "equity_vol",           "pmi_composite",       "cpi_momentum",
        "gdp_gap",              "m2_growth_zscore",    "term_premium",
        "sovereign_cds_5y",     "loan_deposit_ratio",  "reer_deviation",
        "capital_flow_zscore",  "commodity_terms_trade",
    ]

    X = pd.DataFrame(
        rng.standard_normal((n, len(feature_names))),
        columns=feature_names, index=dates,
    )
    y = (
        X["credit_spread_zscore"] * 0.4
        + X["vix_x_credit"]       * 0.3
        + X["yield_curve_slope"]  * (-0.2)
        + rng.standard_normal(n)  * 0.3
    > 0.5).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X, y)

    exp = run_full_explain_pipeline(model, X)

    snap = exp.get_country_snapshot("Turkey", "2022-01-01")
    print("\n[snapshot] Turkey @ 2022-01-01:")
    print(json.dumps(snap, indent=2))

    print("\n[explain] smoke test passed ✓")