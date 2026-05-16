import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
CACHE_MAX_AGE_DAYS = 7

WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG":    "gdp_growth",
    "FP.CPI.TOTL.ZG":       "cpi_inflation",
    "SL.UEM.TOTL.ZS":       "unemployment",
    "BN.CAB.XOKA.GD.ZS":   "current_account_gdp",
    "GC.DOD.TOTL.GD.ZS":   "govt_debt_gdp",
    "NE.TRD.GNFS.ZS":       "trade_openness",
    "FM.LBL.BMNY.GD.ZS":   "broad_money_gdp",
    "NY.GNS.ICTR.ZS":       "gross_savings",
    "GC.REV.XGRT.GD.ZS":   "tax_revenue_gdp",
    "BX.KLT.DINV.WD.GD.ZS":"fdi_inflows_gdp",
}

FRED_SERIES = {
    "T10Y2Y":       "yield_curve",
    "VIXCLS":       "vix",
    "BAA10Y":       "baa_spread",
    "FEDFUNDS":     "fed_funds",
    "M2SL":         "m2_money_supply",
    "DCOILWTICO":   "oil_wti",
    "DTWEXBGS":     "usd_broad_index",
    "MORTGAGE30US": "mortgage_30y",
}

YFINANCE_TICKERS = {
    "^GSPC":    "sp500",
    "DX-Y.NYB": "dxy",
    "GC=F":     "gold",
    "^GSCI":    "gsci_commodities",
    "EEM":      "em_equities",
    "^TNX":     "us10y_yield",
    "HYG":      "hy_bonds",
    "TLT":      "long_treasury",
}

CRISIS_EVENTS = {
    "Global Financial Crisis": {
        "start": "2008-09", "end": "2009-06",
        "regions": ["global"], "severity": "critical",
    },
    "Asian Financial Crisis": {
        "start": "1997-07", "end": "1998-12",
        "regions": ["asia", "emerging"], "severity": "critical",
    },
    "Russian Default": {
        "start": "1998-08", "end": "1999-03",
        "regions": ["russia", "emerging"], "severity": "severe",
    },
    "Dot-com Recession": {
        "start": "2001-03", "end": "2001-11",
        "regions": ["us", "developed"], "severity": "moderate",
    },
    "Eurozone Sovereign Crisis": {
        "start": "2011-07", "end": "2012-09",
        "regions": ["europe"], "severity": "severe",
    },
    "COVID Shock": {
        "start": "2020-02", "end": "2020-06",
        "regions": ["global"], "severity": "critical",
    },
    "LatAm Crisis": {
        "start": "2001-12", "end": "2002-12",
        "regions": ["latam", "emerging"], "severity": "severe",
    },
    "UK LDI Crisis": {
        "start": "2022-09", "end": "2022-11",
        "regions": ["europe"], "severity": "moderate",
    },
}

SEVERITY_MAP = {"none": 0, "moderate": 1, "severe": 2, "critical": 3}

WB_START_YEAR = 1990
WB_END_YEAR   = 2023