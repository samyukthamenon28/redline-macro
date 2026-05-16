"""
fetch.py — pull everything we need for Redline Macro.

Sources:
  - World Bank (wbgapi): annual panel, 180 countries, 1990-2023
  - FRED: key US macro/financial daily series
  - yfinance: equity/bond/commodity prices daily
  - Crisis labels: built from CRISIS_EVENTS registry in config

Cache logic: skip fetch if parquet exists and is < CACHE_MAX_AGE_DAYS old.
All failures are loud but non-fatal — we log and continue.
"""

import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# make src importable when run directly
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CACHE_MAX_AGE_DAYS, CRISIS_EVENTS, DATA_RAW,
    FRED_API_KEY, FRED_SERIES, SEVERITY_MAP,
    WB_END_YEAR, WB_INDICATORS, WB_START_YEAR, YFINANCE_TICKERS,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich import box
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None


# ── terminal helpers ────────────────────────────────────────────────────────

def _print(msg, style=""):
    if RICH:
        console.print(msg, style=style)
    else:
        print(msg)

def _ok(label, detail=""):
    tag = "[bold green]✓[/bold green]" if RICH else "✓"
    _print(f"  {tag}  {label}  [dim]{detail}[/dim]" if detail else f"  {tag}  {label}")

def _warn(label, detail=""):
    tag = "[bold yellow]⚠[/bold yellow]" if RICH else "⚠"
    _print(f"  {tag}  {label}  [dim]{detail}[/dim]" if detail else f"  {tag}  {label}", style="yellow")

def _err(label, detail=""):
    tag = "[bold red]✗[/bold red]" if RICH else "✗"
    _print(f"  {tag}  {label}  [dim]{detail}[/dim]" if detail else f"  {tag}  {label}", style="red")

def _section(title):
    if RICH:
        console.rule(f"[bold red]{title}[/bold red]")
    else:
        print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ── cache helpers ────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return DATA_RAW / f"{name}.parquet"

def _is_fresh(name: str) -> bool:
    p = _cache_path(name)
    if not p.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    return age < timedelta(days=CACHE_MAX_AGE_DAYS)

def _save(df: pd.DataFrame, name: str):
    p = _cache_path(name)
    df.to_parquet(p, index=True)
    kb = p.stat().st_size / 1024
    _ok(name, f"{len(df):,} rows · {df.shape[1]} cols · {kb:.0f} KB")

def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(_cache_path(name))


# ── World Bank ───────────────────────────────────────────────────────────────

def fetch_world_bank(force=False) -> pd.DataFrame:
    _section("World Bank — Annual Panel")

    if not force and _is_fresh("world_bank"):
        _ok("world_bank", "cache hit — skipping fetch")
        return _load("world_bank")

    try:
        import wbgapi as wb
    except ImportError:
        _err("wbgapi not installed", "pip install wbgapi")
        return pd.DataFrame()

    frames = []
    indicators = list(WB_INDICATORS.keys())
    friendly   = list(WB_INDICATORS.values())

    progress_kw = dict(
        SpinnerColumn(spinner_name="dots", style="red"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, style="red", complete_style="bright_red"),
        TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
        TimeElapsedColumn(),
    ) if RICH else ()

    if RICH:
        progress = Progress(*progress_kw, console=console)
    
    failed = []

    for i, (code, name) in enumerate(zip(indicators, friendly)):
        desc = f"[cyan]{name:<25}[/cyan]" if RICH else name
        try:
            raw = wb.data.DataFrame(
                code,
                time=range(WB_START_YEAR, WB_END_YEAR + 1),
                labels=False,
                numericTimeKeys=True,
            )
            # wbgapi returns (economy × time) — melt to long
            df = (
                raw
                .reset_index()
                .melt(id_vars="economy", var_name="year", value_name=name)
                .rename(columns={"economy": "country"})
                .assign(year=lambda x: x["year"].astype(int))
                .dropna(subset=[name])
            )
            frames.append(df)
            _ok(f"{name:<25}", f"{len(df):,} obs")
        except Exception as e:
            _err(f"{name:<25}", str(e)[:80])
            failed.append(name)
        time.sleep(0.15)  # be kind to the API

    if not frames:
        _err("No World Bank data retrieved")
        return pd.DataFrame()

    # merge all indicators on (country, year)
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=["country", "year"], how="outer")

    if failed:
        _warn(f"Missing indicators: {', '.join(failed)}")

    _save(merged, "world_bank")
    return merged


# ── FRED ─────────────────────────────────────────────────────────────────────

def fetch_fred(force=False) -> pd.DataFrame:
    _section("FRED — US Financial Series")

    if not force and _is_fresh("fred"):
        _ok("fred", "cache hit — skipping fetch")
        return _load("fred")

    if not FRED_API_KEY:
        _warn("FRED_API_KEY not set in env", "set it in .env or export FRED_API_KEY=...")
        _warn("Attempting unauthenticated fetch via requests fallback")

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY if FRED_API_KEY else "anonymous")
    except ImportError:
        _err("fredapi not installed", "pip install fredapi")
        return pd.DataFrame()

    frames = {}
    failed = []

    for series_id, name in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start="1990-01-01")
            s.name = name
            frames[name] = s
            _ok(f"{name:<20}", f"{len(s):,} obs  {s.index[0].date()} → {s.index[-1].date()}")
        except Exception as e:
            _err(f"{name:<20}", str(e)[:80])
            failed.append(series_id)
        time.sleep(0.1)

    if not frames:
        _err("No FRED data retrieved")
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = "date"

    # forward-fill gaps ≤ 7 days (weekends, holidays) — don't fill structural gaps
    df = df.resample("D").last().ffill(limit=7)

    if failed:
        _warn(f"Failed series: {', '.join(failed)}")

    _save(df, "fred")
    return df


# ── yfinance ─────────────────────────────────────────────────────────────────

def fetch_yfinance(force=False) -> pd.DataFrame:
    _section("yfinance — Market Prices")

    if not force and _is_fresh("market"):
        _ok("market", "cache hit — skipping fetch")
        return _load("market")

    try:
        import yfinance as yf
    except ImportError:
        _err("yfinance not installed", "pip install yfinance")
        return pd.DataFrame()

    tickers = list(YFINANCE_TICKERS.keys())
    names   = list(YFINANCE_TICKERS.values())

    try:
        raw = yf.download(
            tickers,
            start="1990-01-01",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        _err("yfinance download failed", str(e)[:120])
        return pd.DataFrame()

    # pull Close prices, rename columns
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()

    close.columns = [YFINANCE_TICKERS.get(c, c) for c in close.columns]
    close.index.name = "date"
    close = close.resample("D").last().ffill(limit=7)

    # add log returns — useful downstream
    for col in close.columns:
        close[f"{col}_ret"] = np.log(close[col] / close[col].shift(1))

    for name in names:
        if name in close.columns:
            n_valid = close[name].notna().sum()
            first   = close[name].first_valid_index()
            last    = close[name].last_valid_index()
            _ok(f"{name:<20}", f"{n_valid:,} obs  {first.date() if first else '?'} → {last.date() if last else '?'}")
        else:
            _warn(f"{name:<20}", "not in download result")

    _save(close, "market")
    return close


# ── Crisis labels ─────────────────────────────────────────────────────────────

def build_crisis_labels(force=False) -> pd.DataFrame:
    """
    Build a daily crisis label series from CRISIS_EVENTS.

    Columns:
      in_crisis       — 1 if date falls within any crisis window
      crisis_severity — 0/1/2/3 max severity active on that date
      crisis_name     — name of most severe active crisis (or None)
      crisis_12m      — 1 if any crisis STARTS within next 12 months
      crisis_6m       — 1 if any crisis STARTS within next 6 months
      months_to_crisis — months until next crisis start (NaN if none)
    """
    _section("Crisis Labels")

    if not force and _is_fresh("crisis_labels"):
        _ok("crisis_labels", "cache hit — skipping build")
        return _load("crisis_labels")

    idx = pd.date_range("1990-01-01", "2024-12-31", freq="D")
    df  = pd.DataFrame(index=idx)
    df.index.name = "date"

    df["in_crisis"]        = 0
    df["crisis_severity"]  = 0
    df["crisis_name"]      = None
    df["crisis_start"]     = 0  # 1 on the exact start date

    parsed = {}  # name → (start_dt, end_dt, severity_int)

    for name, meta in CRISIS_EVENTS.items():
        start = pd.to_datetime(meta["start"])
        end   = pd.to_datetime(meta["end"]) + pd.offsets.MonthEnd(0)  # inclusive month end
        sev   = SEVERITY_MAP[meta["severity"]]
        parsed[name] = (start, end, sev)

        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, "in_crisis"] = 1
        # keep highest severity if windows overlap
        df.loc[mask & (df["crisis_severity"] < sev), "crisis_severity"] = sev
        df.loc[mask & (df["crisis_severity"] == sev), "crisis_name"] = name
        df.loc[start, "crisis_start"] = 1

    # forward-looking labels — loop once, vectorise per crisis start
    starts = sorted([(v[0], v[2], k) for k, v in parsed.items()])

    months_to = pd.Series(np.nan, index=idx)
    c12 = pd.Series(0, index=idx)
    c6  = pd.Series(0, index=idx)
    max_sev_fwd = pd.Series(0, index=idx)

    for start_dt, sev, name in starts:
        if start_dt not in idx:
            continue
        # every date up to 12m before this crisis
        window_12 = pd.date_range(
            max(idx[0], start_dt - pd.DateOffset(months=12)),
            start_dt - pd.Timedelta(days=1),
            freq="D",
        )
        window_6 = pd.date_range(
            max(idx[0], start_dt - pd.DateOffset(months=6)),
            start_dt - pd.Timedelta(days=1),
            freq="D",
        )
        c12.loc[window_12] = 1
        c6.loc[window_6]   = 1

        # months_to: for each date in window_12, compute months until start_dt
        for d in window_12:
            m = (start_dt.year - d.year) * 12 + (start_dt.month - d.month)
            if np.isnan(months_to.loc[d]) or m < months_to.loc[d]:
                months_to.loc[d] = m

        max_sev_fwd.loc[window_12] = np.maximum(
            max_sev_fwd.loc[window_12].values, sev
        )

    df["crisis_12m"]       = c12.values
    df["crisis_6m"]        = c6.values
    df["months_to_crisis"] = months_to.values
    df["crisis_severity_fwd"] = max_sev_fwd.values  # severity of the upcoming crisis

    # summary
    _print(f"\n  [dim]Crisis windows:[/dim]" if RICH else "\n  Crisis windows:")
    if RICH:
        tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold red")
        tbl.add_column("Crisis", style="white")
        tbl.add_column("Start",  style="dim")
        tbl.add_column("End",    style="dim")
        tbl.add_column("Severity", style="yellow")
        for name, (s, e, sev) in parsed.items():
            tbl.add_row(name, str(s.date()), str(e.date()), str(sev))
        console.print(tbl)

    _print(f"  crisis_12m prevalence : {df['crisis_12m'].mean():.1%}")
    _print(f"  crisis_6m  prevalence : {df['crisis_6m'].mean():.1%}")
    _print(f"  in_crisis  prevalence : {df['in_crisis'].mean():.1%}")

    _save(df, "crisis_labels")
    return df


# ── orchestrator ──────────────────────────────────────────────────────────────

def fetch_all(force=False) -> dict[str, pd.DataFrame]:
    if RICH:
        console.print(
            Panel.fit(
                "[bold red]REDLINE MACRO[/bold red]  [dim]data ingestion pipeline[/dim]\n"
                "[dim italic]Macro regimes break slowly, then all at once.[/dim italic]",
                border_style="red",
            )
        )
    else:
        print("\n" + "="*60)
        print("  REDLINE MACRO — data ingestion pipeline")
        print("  Macro regimes break slowly, then all at once.")
        print("="*60 + "\n")

    t0 = time.time()
    results = {}

    results["world_bank"]    = fetch_world_bank(force=force)
    results["fred"]          = fetch_fred(force=force)
    results["market"]        = fetch_yfinance(force=force)
    results["crisis_labels"] = build_crisis_labels(force=force)

    elapsed = time.time() - t0

    _section("Done")
    if RICH:
        tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        tbl.add_column("Dataset",  style="cyan")
        tbl.add_column("Rows",     justify="right")
        tbl.add_column("Cols",     justify="right")
        tbl.add_column("Status",   style="green")
        for name, df in results.items():
            if df is not None and len(df):
                tbl.add_row(name, f"{len(df):,}", str(df.shape[1]), "✓ ok")
            else:
                tbl.add_row(name, "—", "—", "[red]✗ empty[/red]")
        console.print(tbl)
        console.print(f"\n  [dim]Total time: {elapsed:.1f}s[/dim]")
    else:
        for name, df in results.items():
            status = f"{len(df):,} rows × {df.shape[1]} cols" if df is not None and len(df) else "EMPTY"
            print(f"  {name:<20} {status}")
        print(f"\n  Total time: {elapsed:.1f}s")

    return results


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Redline Macro — fetch all data sources")
    parser.add_argument("--force", action="store_true", help="Ignore cache, re-fetch everything")
    parser.add_argument(
        "--source",
        choices=["wb", "fred", "market", "crisis", "all"],
        default="all",
        help="Which source to fetch",
    )
    args = parser.parse_args()

    if args.source == "all":
        fetch_all(force=args.force)
    elif args.source == "wb":
        fetch_world_bank(force=args.force)
    elif args.source == "fred":
        fetch_fred(force=args.force)
    elif args.source == "market":
        fetch_yfinance(force=args.force)
    elif args.source == "crisis":
        build_crisis_labels(force=args.force)