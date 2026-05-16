# theme.py — design tokens, plotly template, and CSS injection
# single source of truth. change here, propagates everywhere.

import plotly.graph_objects as go
import plotly.io as pio

# ── color tokens ──────────────────────────────────────────────────────────────

BG_DARK   = '#0A0F1E'
BG_CARD   = '#0D1424'
BG_BORDER = '#1E2A45'

RED       = '#DC2626'
RED_LIGHT = '#FCA5A5'
RED_DIM   = 'rgba(220,38,38,0.12)'
AMBER     = '#F59E0B'
AMBER_DIM = 'rgba(245,158,11,0.12)'
GREEN     = '#22C55E'
GREEN_DIM = 'rgba(34,197,94,0.12)'
BLUE      = '#38BDF8'
BLUE_DIM  = 'rgba(56,189,248,0.10)'

TEXT_PRIMARY   = '#F1F5F9'
TEXT_SECONDARY = '#94A3B8'
TEXT_MUTED     = '#475569'

# ── typography ────────────────────────────────────────────────────────────────
# Bebas Neue for the terminal wordmark + big numbers
# IBM Plex Mono everywhere else — data deserves a mono font

FONT_DISPLAY = "'Bebas Neue', 'Impact', sans-serif"
FONT_MONO    = "'IBM Plex Mono', 'JetBrains Mono', 'Courier New', monospace"
FONT_BODY    = "'IBM Plex Sans', 'Helvetica Neue', sans-serif"

# ── plotly template ───────────────────────────────────────────────────────────

_template = go.layout.Template()
_template.layout = go.Layout(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_CARD,
    font=dict(family=FONT_MONO, color=TEXT_SECONDARY, size=11),
    title=dict(
        font=dict(family=FONT_MONO, color=TEXT_PRIMARY, size=13),
        pad=dict(l=4),
    ),
    colorway=[RED, AMBER, GREEN, BLUE, '#A78BFA', '#FB923C', '#34D399'],
    xaxis=dict(
        gridcolor=BG_BORDER, gridwidth=1,
        linecolor=BG_BORDER,
        tickcolor=TEXT_MUTED,
        tickfont=dict(color=TEXT_MUTED, size=10, family=FONT_MONO),
        zerolinecolor=BG_BORDER, zerolinewidth=1,
    ),
    yaxis=dict(
        gridcolor=BG_BORDER, gridwidth=1,
        linecolor=BG_BORDER,
        tickcolor=TEXT_MUTED,
        tickfont=dict(color=TEXT_MUTED, size=10, family=FONT_MONO),
        zerolinecolor=BG_BORDER, zerolinewidth=1,
    ),
    legend=dict(
        bgcolor='rgba(13,20,36,0.9)',
        bordercolor=BG_BORDER, borderwidth=1,
        font=dict(color=TEXT_SECONDARY, size=10, family=FONT_MONO),
    ),
    margin=dict(l=52, r=20, t=36, b=44),
    hoverlabel=dict(
        bgcolor=BG_CARD, bordercolor=BG_BORDER,
        font=dict(family=FONT_MONO, color=TEXT_PRIMARY, size=11),
    ),
)

pio.templates['redline'] = _template
pio.templates.default   = 'redline'

# ── css injection ─────────────────────────────────────────────────────────────

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* kill streamlit chrome */
#MainMenu {{ visibility: hidden; }}
header    {{ visibility: hidden; height: 0 !important; }}
footer    {{ visibility: hidden; height: 0 !important; }}
.stDeployButton {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}

html, body {{
    background: {BG_DARK} !important;
    color: {TEXT_PRIMARY} !important;
}}

.stApp, [data-testid="stAppViewContainer"] {{
    background: {BG_DARK} !important;
    font-family: {FONT_MONO} !important;
}}

.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
    background: {BG_CARD} !important;
    border-right: 1px solid {BG_BORDER} !important;
    box-shadow: 6px 0 32px rgba(0,0,0,0.5) !important;
}}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
    color: {TEXT_SECONDARY} !important;
    font-family: {FONT_MONO} !important;
    font-size: 11px !important;
    letter-spacing: 0.04em;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: {TEXT_MUTED} !important;
    font-size: 10px !important;
}}
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
    background: {BG_DARK} !important;
    border: 1px solid {BG_BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    font-family: {FONT_MONO} !important;
    font-size: 11px !important;
}}

/* ── tabs ── */
[data-testid="stTabs"] [role="tablist"] {{
    border-bottom: 1px solid {BG_BORDER} !important;
    background: transparent !important;
    gap: 0 !important;
}}
[data-testid="stTabs"] button[role="tab"] {{
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: {TEXT_MUTED} !important;
    font-family: {FONT_MONO} !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
    transition: color 0.2s, border-color 0.2s;
}}
[data-testid="stTabs"] button[role="tab"]:hover {{
    color: {TEXT_SECONDARY} !important;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    color: {RED} !important;
    border-bottom: 2px solid {RED} !important;
    background: transparent !important;
}}

/* ── metric card ── */
.rl-card {{
    background: {BG_CARD};
    border: 1px solid {BG_BORDER};
    border-top: 2px solid {RED};
    padding: 16px 18px 14px;
    border-radius: 2px;
    position: relative;
    overflow: hidden;
    height: 100%;
}}
.rl-card::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(220,38,38,0.04) 0%, transparent 55%);
    pointer-events: none;
}}
.rl-card-label {{
    font-family: {FONT_MONO};
    font-size: 9px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    margin-bottom: 8px;
}}
.rl-card-value {{
    font-family: {FONT_DISPLAY};
    font-size: 32px;
    letter-spacing: 0.04em;
    color: {TEXT_PRIMARY};
    line-height: 1;
    margin-bottom: 4px;
}}
.rl-card-sub {{
    font-family: {FONT_MONO};
    font-size: 10px;
    color: {TEXT_MUTED};
    margin-top: 4px;
    letter-spacing: 0.02em;
}}

/* ── section headers ── */
.rl-sh {{
    font-family: {FONT_MONO};
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 18px 0 10px 0;
}}
.rl-sh::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {BG_BORDER};
}}

/* ── pills ── */
.pill-high   {{ background: rgba(220,38,38,0.15);  color: {RED};   border: 1px solid rgba(220,38,38,0.3);  padding: 2px 9px; border-radius: 2px; font-size:10px; font-family:{FONT_MONO}; letter-spacing:0.06em; }}
.pill-medium {{ background: rgba(245,158,11,0.15); color: {AMBER}; border: 1px solid rgba(245,158,11,0.3); padding: 2px 9px; border-radius: 2px; font-size:10px; font-family:{FONT_MONO}; letter-spacing:0.06em; }}
.pill-low    {{ background: rgba(34,197,94,0.15);  color: {GREEN}; border: 1px solid rgba(34,197,94,0.3);  padding: 2px 9px; border-radius: 2px; font-size:10px; font-family:{FONT_MONO}; letter-spacing:0.06em; }}

/* ── status badges ── */
.badge {{ display:inline-block; padding:3px 10px; border-radius:2px; font-family:{FONT_MONO}; font-size:9px; letter-spacing:0.12em; text-transform:uppercase; }}
.badge-caught  {{ background:rgba(34,197,94,0.15);  color:{GREEN}; border:1px solid rgba(34,197,94,0.35);  }}
.badge-partial {{ background:rgba(245,158,11,0.15); color:{AMBER}; border:1px solid rgba(245,158,11,0.35); }}
.badge-missed  {{ background:rgba(220,38,38,0.15);  color:{RED};   border:1px solid rgba(220,38,38,0.35);  }}

/* ── expanders ── */
[data-testid="stExpander"] {{
    background: {BG_CARD} !important;
    border: 1px solid {BG_BORDER} !important;
    border-radius: 2px !important;
    margin-bottom: 8px;
}}
[data-testid="stExpander"] summary {{
    font-family: {FONT_MONO} !important;
    font-size: 11px !important;
    color: {TEXT_SECONDARY} !important;
    letter-spacing: 0.04em;
}}

/* ── dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BG_BORDER} !important;
    border-radius: 2px;
}}

/* ── markdown ── */
.stMarkdown p, .stMarkdown li {{
    font-family: {FONT_MONO} !important;
    color: {TEXT_SECONDARY} !important;
    font-size: 12px !important;
    line-height: 1.7;
}}

/* ── live dot ── */
.live-dot {{
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {GREEN};
    margin-right: 5px;
    vertical-align: middle;
    animation: blink 2s ease-in-out infinite;
}}
@keyframes blink {{
    0%,100% {{ opacity:1; }} 50% {{ opacity:0.25; }}
}}

/* ── methodology blocks ── */
.meth-section {{ margin-bottom: 28px; }}
.meth-title {{
    font-family: {FONT_MONO};
    font-size: 9px; letter-spacing: 0.22em;
    text-transform: uppercase; color: {RED};
    padding-bottom: 7px;
    border-bottom: 1px solid {BG_BORDER};
    margin-bottom: 12px;
}}
.meth-body {{
    font-family: {FONT_MONO};
    font-size: 11.5px; color: {TEXT_SECONDARY}; line-height: 1.82;
}}
.meth-body strong {{ color: {TEXT_PRIMARY}; }}
.meth-body code {{
    background: {BG_DARK}; border: 1px solid {BG_BORDER};
    padding: 1px 6px; border-radius: 2px;
    font-size: 10px; color: {AMBER};
}}

/* ── failure block ── */
.fail-box {{
    background: rgba(220,38,38,0.05);
    border-left: 3px solid {RED};
    padding: 14px 18px; margin: 8px 0;
    border-radius: 0 2px 2px 0;
}}
.fail-title {{
    font-family: {FONT_MONO}; font-size: 10px; letter-spacing: 0.1em;
    text-transform: uppercase; color: {RED_LIGHT}; margin-bottom: 8px;
}}
.fail-body {{
    font-family: {FONT_MONO}; font-size: 11px;
    color: {TEXT_SECONDARY}; line-height: 1.78;
    white-space: pre-wrap;
}}

/* ── note block ── */
.rl-note {{
    background: {BG_CARD};
    border-left: 2px solid {RED};
    padding: 14px 18px; margin: 10px 0;
    font-family: {FONT_MONO}; font-size: 11.5px;
    color: {TEXT_SECONDARY}; line-height: 1.78;
    border-radius: 0 2px 2px 0;
}}

/* ── header ── */
.rl-header {{
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 14px 0 14px;
    border-bottom: 1px solid {BG_BORDER};
    margin-bottom: 18px;
}}
.rl-wordmark {{
    font-family: {FONT_DISPLAY};
    font-size: 26px; letter-spacing: 0.14em;
    color: {TEXT_PRIMARY}; line-height: 1;
}}
.rl-wordmark .r {{ color: {RED}; }}
.rl-tagline {{
    font-family: {FONT_MONO};
    font-size: 9px; letter-spacing: 0.14em;
    color: {TEXT_MUTED}; margin-top: 3px;
    text-transform: uppercase;
}}
.rl-ts {{
    font-family: {FONT_MONO};
    font-size: 10px; color: {TEXT_MUTED};
    letter-spacing: 0.06em; text-align: right;
}}
</style>
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def risk_color(r: float) -> str:
    if r >= 0.65: return RED
    if r >= 0.40: return AMBER
    return GREEN

def risk_label(r: float) -> str:
    if r >= 0.65: return "HIGH"
    if r >= 0.40: return "MEDIUM"
    return "LOW"

def risk_pill_html(r: float) -> str:
    cls = f"pill-{risk_label(r).lower()}"
    return f'<span class="{cls}">{risk_label(r)}</span>'

def trend_sym(d: float) -> str:
    if d >  0.02: return "▲"
    if d < -0.02: return "▼"
    return "━"

def trend_color(d: float) -> str:
    if d >  0.02: return RED
    if d < -0.02: return GREEN
    return TEXT_MUTED