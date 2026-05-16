"""
utils.py — shared plumbing for the whole pipeline.
If it's used more than once, it lives here.
"""

import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
import time

import pandas as pd
import numpy as np


# ── Logging ──────────────────────────────────────────────────────────────────

class _RedlineFormatter(logging.Formatter):
    # ANSI codes — red for warnings/errors, dim for debug, white for info
    LEVELS = {
        logging.DEBUG:    "\033[2m",        # dim
        logging.INFO:     "\033[0m",        # normal
        logging.WARNING:  "\033[33m",       # yellow
        logging.ERROR:    "\033[31m",       # red
        logging.CRITICAL: "\033[1;31m",     # bold red
    }
    RESET = "\033[0m"

    def format(self, record):
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        color = self.LEVELS.get(record.levelno, "")
        level_tag = record.levelname[0]  # D I W E C — single char, clean
        return f"{color}[{ts}] {level_tag} {record.getMessage()}{self.RESET}"


def get_logger(name: str = "redline") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log  # already configured, don't double-add handlers

    log.setLevel(logging.DEBUG)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_RedlineFormatter())
    log.addHandler(h)
    log.propagate = False
    return log


log = get_logger()


# ── File caching ──────────────────────────────────────────────────────────────

CACHE_TTL_DAYS = 7


def cache_valid(path: Union[str, Path], max_age_days: int = CACHE_TTL_DAYS) -> bool:
    """True if file exists and is fresh enough to skip re-fetching."""
    p = Path(path)
    if not p.exists():
        return False
    age = time.time() - p.stat().st_mtime
    return age < max_age_days * 86400


def cache_path(name: str, base_dir: Union[str, Path] = "data/raw") -> Path:
    """Canonical cache path so every caller agrees on the filename."""
    p = Path(base_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_or_fetch(path: Union[str, Path], fetch_fn, *args, max_age_days=CACHE_TTL_DAYS, **kwargs):
    """
    If cache is warm, load from disk. Otherwise call fetch_fn and save.
    fetch_fn must return a DataFrame.
    """
    p = Path(path)
    if cache_valid(p, max_age_days):
        log.debug(f"cache hit → {p}")
        return pd.read_parquet(p)

    log.debug(f"cache miss, fetching → {p}")
    df = fetch_fn(*args, **kwargs)
    df.to_parquet(p, index=True)
    return df


# ── Date helpers ──────────────────────────────────────────────────────────────

def date_range_str(years_back: int = 30) -> tuple[str, str]:
    """(start, end) as 'YYYY-MM-DD' strings. WB API loves this format."""
    end = datetime.today()
    start = end.replace(year=end.year - years_back)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def quarter_end_dates(start: str, end: str) -> pd.DatetimeIndex:
    """Quarter-end dates in range — useful for aligning mixed-frequency data."""
    return pd.date_range(start, end, freq="QE")


def fiscal_year(dt: pd.Timestamp, fy_start_month: int = 1) -> int:
    """Fiscal year for a timestamp. Default is calendar year."""
    if dt.month >= fy_start_month:
        return dt.year
    return dt.year - 1


def months_between(a: str, b: str) -> int:
    """Quick month count — used when we need to assert data recency."""
    da, db = pd.Timestamp(a), pd.Timestamp(b)
    return (db.year - da.year) * 12 + (db.month - da.month)


# ── Safe math ─────────────────────────────────────────────────────────────────

def safe_div(num: pd.Series, denom: pd.Series, fill: float = np.nan) -> pd.Series:
    """
    Elementwise division without blowing up on zeros.
    fill=0 makes sense for ratio features; fill=nan surfaces the problem.
    """
    return num.div(denom.replace(0, np.nan)).fillna(fill)


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip to quantile bounds — stops one bad WB observation from wrecking a feature."""
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def pct_change_safe(s: pd.Series, periods: int = 1) -> pd.Series:
    """pct_change but zero-base values produce nan not inf."""
    shifted = s.shift(periods).replace(0, np.nan)
    return (s - s.shift(periods)) / shifted


def zscore(s: pd.Series) -> pd.Series:
    """Standard z-score. Used for composite index construction."""
    std = s.std()
    if std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


# ── DataFrame validation ──────────────────────────────────────────────────────

def check_cols(df: pd.DataFrame, required: list[str], context: str = "") -> None:
    """Crash loudly if expected columns are missing. Better than silent NaN hell."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        tag = f"[{context}] " if context else ""
        raise ValueError(f"{tag}Missing columns: {missing}")


def check_nulls(
    df: pd.DataFrame,
    threshold: float = 0.3,
    cols: Optional[list[str]] = None,
    context: str = "",
) -> pd.DataFrame:
    """
    Warn (don't crash) if any column has >threshold null fraction.
    Returns a summary frame for logging. Caller decides what to do.
    """
    target = df[cols] if cols else df
    null_frac = target.isnull().mean()
    bad = null_frac[null_frac > threshold]
    if not bad.empty:
        tag = f"[{context}] " if context else ""
        log.warning(f"{tag}high null rate:\n{bad.to_string()}")
    return bad.to_frame("null_frac")


def assert_date_index(df: pd.DataFrame, context: str = "") -> None:
    """We assume DatetimeIndex everywhere. Break early if not."""
    if not isinstance(df.index, pd.DatetimeIndex):
        tag = f"[{context}] " if context else ""
        raise TypeError(f"{tag}Expected DatetimeIndex, got {type(df.index)}")


def drop_all_null_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely null — happens often with sparse WB data."""
    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    dropped = before - df.shape[1]
    if dropped:
        log.debug(f"dropped {dropped} fully-null columns")
    return df


def align_countries(dfs: list[pd.DataFrame], on: str = "country") -> pd.DataFrame:
    """
    Inner-join a list of DataFrames on the country column.
    Keeps only countries that appear in all frames — avoids leaky merges.
    """
    base = dfs[0]
    for other in dfs[1:]:
        base = base.merge(other, on=on, how="inner")
    return base


# ── Progress / status printing ────────────────────────────────────────────────

class status:
    """
    Context manager for timed status blocks.

        with status("fetching World Bank GDP"):
            df = wb.download(...)

    Prints start + elapsed on exit. Clean. No tqdm dependency.
    """
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        log.info(f"→ {self.label}...")
        self._t = time.time()
        return self

    def __exit__(self, exc_type, *_):
        elapsed = time.time() - self._t
        if exc_type:
            log.error(f"✗ {self.label} failed ({elapsed:.1f}s)")
        else:
            log.info(f"✓ {self.label} ({elapsed:.1f}s)")


def progress(msg: str) -> None:
    """One-liner status print for when a context manager is overkill."""
    log.info(f"→ {msg}")


def done(msg: str) -> None:
    log.info(f"✓ {msg}")


def warn(msg: str) -> None:
    log.warning(f"⚠ {msg}")


# ── Misc ──────────────────────────────────────────────────────────────────────

def flatten_multiindex(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """
    After a groupby().agg() you get MultiIndex columns. Flatten them.
    e.g. ('gdp', 'mean') → 'gdp_mean'
    """
    df.columns = [sep.join(c).strip(sep) if isinstance(c, tuple) else c for c in df.columns]
    return df


def country_iso2_to_iso3() -> dict:
    """
    Hardcoded map for the handful of codes WB and IMF disagree on.
    Extend as we hit them — not worth pulling a whole library for this.
    """
    return {
        "US": "USA", "GB": "GBR", "DE": "DEU", "FR": "FRA",
        "JP": "JPN", "CN": "CHN", "IN": "IND", "BR": "BRA",
        "ZA": "ZAF", "NG": "NGA", "MX": "MEX", "KR": "KOR",
        "AU": "AUS", "CA": "CAN", "IT": "ITA", "ES": "ESP",
        "AR": "ARG", "TR": "TUR", "ID": "IDN", "SA": "SAU",
        # add more as the data sources surprise us
    }


def ensure_dirs(*paths) -> None:
    """Create directory trees. Call once at pipeline start."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    """Timestamp string for filenames — avoids spaces and colons."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def memo(fn):
    """
    Dead-simple memoization for pure functions called many times per run.
    Not thread-safe. Don't care — pipeline is single-threaded.
    """
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]
    return wrapper