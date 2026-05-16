# Redline Macro: Model Methodology

**Internal Research Note — v1.2**
*Quant Research, Redline Macro*

---

## 1. Data Sources and Coverage

Five primary data feeds. All pulled programmatically; no manual downloads in the pipeline.

**IMF World Economic Outlook (WEO)** — Annual. 190 countries from 1980. Core macro aggregates: GDP growth, current account balance (% GDP), gross government debt (% GDP), inflation (CPI), output gap. Used as the backbone of the feature matrix. Lag: ~6 months post-year-end on April release; October release preferred for most recent vintage.

**BIS Total Credit Statistics** — Quarterly. ~45 advanced and emerging economies. Credit-to-GDP gap is the single most predictive feature in the model; this dataset is its source. The credit gap (deviation from HP-filtered trend) is constructed using a one-sided filter on the expanding window — no future data leaks into the trend estimate.

**World Bank Development Indicators (WDI)** — Annual. 217 countries. FX reserves (months of import cover), short-term external debt (% of reserves), trade openness. Fills coverage gaps for smaller economies where BIS data is absent.

**FRED (Federal Reserve Economic Data)** — Monthly/quarterly. US and global. DXY index, federal funds rate, VIX, US 10Y yield. Used as global push-factor controls — EM crises cluster around dollar tightening cycles and these features capture that.

**Laeven & Valencia Systemic Banking, Currency, and Debt Crisis Database** — Binary crisis labels, 151 countries, 1970–2017. Extended manually to 2023 using IMF Article IV consultations, Reuters event data, and academic working papers. Label construction is the weakest link in the pipeline.

**Coverage note**: 180 countries scored. Reliable estimates (>20 years of training data, no major gaps) for ~110. The remaining 70 should be treated as illustrative — low data density countries have wide implicit uncertainty that the point scores don't communicate.

---

## 2. Model Architecture

**Task**: Binary classification. Target $y_{i,t} = 1$ if a currency, sovereign debt, or banking crisis begins in country $i$ within the 12-month window $[t+6, t+18]$. The 6-month offset is intentional — sub-6-month warnings are operationally useless for most users, and the signal-to-noise ratio degrades sharply at horizons under 3 months.

**Algorithm**: XGBoost gradient-boosted classifier. `n_estimators=800`, `max_depth=4`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.7`. `scale_pos_weight` set to inverse class frequency (~23x) to handle imbalance. These hyperparameters were tuned via Optuna on the 1980–2005 training fold; not re-tuned per fold in walk-forward validation (that would overfit to the validation set selection process).

**Feature engineering**: 47 features total across 6 groups.

| Feature Group | Count | Key Variables |
|---|---|---|
| Credit cycle | 8 | Credit-to-GDP gap, credit growth YoY, corporate vs household decomp |
| External balance | 9 | CA deficit, FX reserves/imports, ST debt/reserves, REER deviation |
| Fiscal | 7 | Primary balance, debt/GDP, debt/revenue, maturity profile |
| Monetary | 6 | Real policy rate, inflation momentum, FX vs PPP |
| Growth momentum | 5 | GDP growth, output gap, investment/GDP delta |
| Global push factors | 5 | DXY, US real rates, VIX, commodity terms-of-trade |
| Structural | 7 | Trade openness, dollarization proxies, IMF program dummy |

All features expressed as z-scores (rolling 10-year window, country-specific mean/std) or percentage deviations from trend. Raw levels not used — the goal is regime deviation, not absolute position.

**Calibration**: Isotonic regression on a held-out calibration set (most recent 20% of training data, not used in tree fitting). Reliability diagram confirms calibrated scores track empirical frequencies reasonably well between 0.3–0.8. Tails are noisy.

**Limitations of this architecture**: Tree models can't extrapolate. A feature combination outside the training distribution produces a score near the base rate, not an extreme one — the model fails silently rather than loudly. Panel structure is partially ignored: the model treats country-year observations as exchangeable after z-scoring, which is a reasonable approximation but misses country-specific crisis propensity (Argentina is not Indonesia, even at the same z-score).

---

## 3. Walk-Forward Validation

Time-series data cannot be validated with standard k-fold cross-validation. Randomly shuffling country-year observations would let future data inform past predictions — the crisis of 2008 would help predict the crisis of 1997. That's not a model, it's a lookup.

Walk-forward validation trains on all data up to year $T$, evaluates on year $T+1$, then expands the training window and repeats. This mimics live deployment: the model always uses only what would have been available at the time of prediction.

```
Fold 1:  Train [1980–1994]  →  Eval [1995]
Fold 2:  Train [1980–1995]  →  Eval [1996]
...
Fold N:  Train [1980–2021]  →  Eval [2022]
```

Reported metrics (AUROC, precision at threshold) are the mean and standard deviation across all evaluation years. A single aggregate AUROC hides year-to-year variance; 2008 and 2020 are much harder years than 2013.

**Why this matters**: A model that claims 0.85 AUROC from random cross-validation on macro panel data is almost certainly overfitting. Walk-forward AUROC of 0.79–0.84 is a more honest number — and it still benefits from lookahead on hyperparameter selection (one structural limitation of the current setup).

---

## 4. Feature Importance (SHAP)

SHAP beeswarm plot is generated at `reports/figures/shap_beeswarm.png` by running `python src/explain.py`. It shows the distribution of SHAP values for each feature across all country-year observations in the most recent evaluation fold.

Top 5 features by mean |SHAP| (most recent fold, 2022 eval):

1. **Credit-to-GDP gap** — by far the dominant predictor. Positive deviations above +8pp are highly associated with crisis onset. Consistent with BIS Basel III countercyclical capital buffer research.
2. **FX reserves / imports** — fewer than 3 months of cover reliably elevates scores. Simple ratio but hard to game.
3. **REER deviation from 5Y trend** — overvalued currencies precede current account adjustments. The signal is stronger for peg or managed-float regimes.
4. **Short-term external debt / reserves** — Greenspan-Guidotti ratio proxy. Measures rollover risk.
5. **Current account deficit (% GDP)** — directional but noisy. High false positive rate for commodity exporters with persistent deficits that never crisis.

The credit gap's dominance is both reassuring (consistent with theory) and a concern — if that single feature is mismeasured or structurally shifted post-COVID, the model's top signal degrades.

Feature interactions matter more than individual features. A country with an elevated credit gap *and* low FX reserves *and* REER overvaluation scores materially higher than the sum of parts. XGBoost captures this; linear models miss it.

---

## 5. Known Failure Modes

**Contagion episodes**: The model scores countries independently. The 1997 Asian crisis spread from Thailand to Korea through trade and capital flow linkages that aren't in the feature matrix. The model caught Thailand's domestic vulnerabilities but underscored the contagion targets until they showed domestic deterioration. A network-based contagion layer is the clearest known gap.

**Commodity super-cycles**: Oil exporters accumulate reserves and run surpluses during boom periods, suppressing crisis scores, then deteriorate rapidly when commodity prices collapse. The model is slow to update on price regime shifts. Venezuela 2014–2016 was underscored until the collapse was already visible in reserves data.

**Political shocks**: Sudden political transitions — coups, election-driven policy reversals, sanctions — are not in the feature matrix. Turkey 2018 was partially predicted (REER overvaluation, CAD) but the idiosyncratic Erdoğan monetary policy factor accelerated the timeline beyond what macro features alone suggested.

**Post-COVID structural breaks**: Many trend-deviation features were constructed relative to 2010–2019 baselines. Post-2020 inflation regimes, fiscal expansions, and supply chain shocks pushed multiple features into historically extreme territory simultaneously — crowding out genuine crisis signals with false positives in 2021–2022. Recalibrating baselines is a known to-do.

**Small states and data-poor countries**: Countries with <15 years of non-missing data in the training set get pooled estimates. Their scores are effectively the global model applied to partial features — uninformative at best, misleading at worst. These are flagged in the dashboard with a `LOW DATA` badge but the risk of misuse is real.

---

## 6. References

[1] Laeven, L. and Valencia, F. (2020). *Systemic Banking Crises Database II*. IMF Economic Review, 68(2), 307–361. https://doi.org/10.1057/s41308-020-00107-3

[2] Drehmann, M. and Tsatsaronis, K. (2014). *The credit-to-GDP gap and countercyclical capital buffers: questions and answers*. BIS Quarterly Review, March 2014.

[3] Reinhart, C.M. and Rogoff, K.S. (2009). *This Time Is Different: Eight Centuries of Financial Folly*. Princeton University Press.

[4] Chinn, M. and Ito, H. (2006). *What matters for financial development? Capital controls, institutions, and interactions*. Journal of Development Economics, 81(1), 163–192.

[5] Frankel, J. and Saravelos, G. (2012). *Can leading indicators assess country vulnerability? Evidence from the 2008–09 global financial crisis*. Journal of International Economics, 87(2), 216–231.

[6] Lundberg, S. and Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017. https://arxiv.org/abs/1705.07874

[7] Greenspan, A. (1999). *Currency reserves and debt*. Remarks before the World Bank Conference on Recent Trends in Reserves Management, Washington D.C.

[8] BIS (2021). *Early warning indicators of banking crises: expanding the family*. BIS Working Papers No. 904.

---

*This note describes the model as of v1.2. Walk-forward results are subject to revision as crisis label data is updated. Do not treat scores as investment advice.*