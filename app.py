import hashlib
import hmac
import html
import importlib.util
import io
import json
import math
import secrets
import sqlite3
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

st.set_page_config(
    page_title="Banana Leaf Health and Ripeness Tracking",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src_Banana_ripeness"))
RIPENESS_CHECKPOINT = PROJECT_DIR / "checkpoints" / "best_vit.pth"
LEAF_CHECKPOINT = PROJECT_DIR / "checkpoints" / "banana_leaf_disease_vit.pth"
LANDING_PAGE_PATH = PROJECT_DIR / "landing_premium.html"
LANDING_HEIGHT = 4200  # adjust if the landing page looks cut off or has a large gap

DB_PATH = PROJECT_DIR / "banana_app.db"
SESSION_TIMEOUT_MIN = 30
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png")
ROLE_CUSTOMER = "customer"
ROLE_ADMIN = "admin"
SHOW_DEMO_HINTS = True  # set to False before sharing the app
ENABLE_SUBJECT_CHECK = True
PAGE_SIZE = 12
FMT = "%Y-%m-%d %H:%M:%S"
DARK_THRESHOLD = 60
BLUR_THRESHOLD = 60
DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
SUBJECT_PROMPTS = [
    "a photo of a banana leaf",
    "a photo of a banana fruit",
    "a photo of something that is not a banana",
]
TRACKS = {
    "Banana leaf": {"title": "Leaf Health", "desc": "Detect Cordana or Sigatoka leaf spot, or confirm a healthy leaf."},
    "Banana fruit": {"title": "Ripeness Tracking", "desc": "Check whether the fruit is best to eat right now."},
}
ACTION_TEXT = {
    "login": "signed in",
    "logout": "signed out",
    "timeout": "was signed out after inactivity",
    "upload": "uploaded a photo",
    "login_failed": "failed a sign-in attempt",
    "user_updated": "updated a user",
    "user_deleted": "deleted a user",
    "results_deleted": "deleted results",
}

# ---------- Design system ----------

THEME_VARS = {
    "customer_dark": """
      --bg:#0F172A; --bg2:#111C32; --text:#F8FAFC; --muted:rgba(248,250,252,.68);
      --border:rgba(148,163,184,.16); --surface:rgba(30,41,59,.92);
      --accent:#FACC15; --accent2:#22C55E; --accent3:#22C55E; --danger:#FB7185;
      --shadow:0 22px 56px rgba(2,6,23,.46); --dot:rgba(148,163,184,.24);
    """,
    "customer_light": """
      --bg:#F4FBF4; --bg2:#EAF6EA; --text:#1B5E20; --muted:rgba(27,94,32,.68);
      --border:rgba(46,125,50,.14); --surface:rgba(255,255,255,.88);
      --accent:#FBC02D; --accent2:#81C784; --accent3:#2E7D32; --danger:#A64B3C;
      --shadow:0 18px 44px rgba(46,125,50,.10); --dot:rgba(46,125,50,.20);
    """,
    "admin": """
      --bg:#F4FBF4; --bg2:#EAF6EA; --text:#1B5E20; --muted:rgba(27,94,32,.68);
      --border:rgba(46,125,50,.16); --surface:rgba(255,255,255,.86);
      --accent:#FBC02D; --accent2:#81C784; --accent3:#2E7D32; --danger:#A64B3C;
      --shadow:0 0 0 1px rgba(46,125,50,.06), 0 22px 52px rgba(46,125,50,.12); --dot:rgba(46,125,50,.18);
    """,
    "admin_dark": """
      --bg:#0F172A; --bg2:#111C32; --text:#F8FAFC; --muted:rgba(248,250,252,.68);
      --border:rgba(148,163,184,.16); --surface:rgba(30,41,59,.92);
      --accent:#FACC15; --accent2:#22C55E; --accent3:#22C55E; --danger:#FB7185;
      --shadow:0 0 0 1px rgba(34,197,94,.10), 0 22px 56px rgba(2,6,23,.46); --dot:rgba(148,163,184,.24);
    """,
}

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');
:root { --ease: cubic-bezier(0.22, 0.8, 0.2, 1); --r: 24px; }
html, body { background: var(--bg); }
.stApp { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; overflow-x: hidden; }
.stApp::before {
  content: ""; position: fixed; inset: -20%; z-index: 0; pointer-events: none;
  background:
    radial-gradient(45% 40% at 15% 20%, color-mix(in srgb, var(--accent2) 26%, transparent), transparent 70%),
    radial-gradient(40% 35% at 85% 15%, color-mix(in srgb, var(--accent) 18%, transparent), transparent 70%),
    radial-gradient(50% 45% at 60% 95%, color-mix(in srgb, var(--accent3) 14%, transparent), transparent 70%);
  animation: mesh 22s ease-in-out infinite alternate;
}
@keyframes mesh { 0% { transform: translate3d(-2%, -1%, 0) scale(1); } 100% { transform: translate3d(3%, 2%, 0) scale(1.08); } }
.stApp::after {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none; opacity: 0.35;
  background-image: radial-gradient(var(--dot) 1px, transparent 1px); background-size: 26px 26px;
  mask-image: radial-gradient(ellipse at center, #000 30%, transparent 75%);
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1280px !important; padding: 1.6rem clamp(1rem, 3vw, 3rem) 4rem !important; position: relative; z-index: 1; }
[data-testid="stMainBlockContainer"] { animation: enter .5s var(--ease) both; }
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] { color: var(--text); }
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }
[data-testid="stMetric"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 1rem 1.2rem;
  backdrop-filter: blur(14px); transition: transform .45s var(--ease), box-shadow .45s var(--ease), border-color .45s;
}
[data-testid="stMetric"]:hover { transform: translateY(-4px); box-shadow: var(--shadow); border-color: color-mix(in srgb, var(--accent) 45%, transparent); }
[data-testid="stMetricValue"] { font-family: 'Inter Tight', sans-serif; font-weight: 800; color: var(--accent); }

.field { position: relative; overflow: hidden; }
.particles { position: absolute; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; }
.particle { position: absolute; bottom: -12px; width: var(--s); height: var(--s); border-radius: 50%; background: var(--accent); opacity: 0; animation: drift var(--d) linear infinite; animation-delay: var(--dl); }
@keyframes drift { 0% { transform: translateY(0); opacity: 0; } 10% { opacity: .7; } 90% { opacity: .4; } 100% { transform: translateY(-120%) translateX(30px); opacity: 0; } }
.shape { position: absolute; border-radius: 30% 70% 60% 40% / 40% 40% 60% 60%; z-index: 0; opacity: .6; filter: blur(2px);
  background: linear-gradient(135deg, color-mix(in srgb, var(--accent2) 55%, transparent), color-mix(in srgb, var(--accent) 30%, transparent));
  animation: float 9s ease-in-out infinite; }
@keyframes float { 0%, 100% { transform: translateY(0) rotate(0deg); } 50% { transform: translateY(-22px) rotate(8deg); } }

.glass { position: relative; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 1.5rem;
  backdrop-filter: blur(18px) saturate(150%); box-shadow: var(--shadow);
  transition: transform .45s var(--ease), box-shadow .45s var(--ease), border-color .45s; animation: enter .6s var(--ease) both; }
.glass:hover { transform: translateY(-5px); border-color: color-mix(in srgb, var(--accent) 50%, transparent); box-shadow: 0 30px 70px rgba(0,0,0,.3); }
.glass h3 { font-family: 'Inter Tight', sans-serif; font-weight: 700; font-size: 1.3rem; margin: 0 0 .3rem; }
.sub { color: var(--muted); margin: 0 0 1rem; line-height: 1.55; }
.eyebrow { color: var(--muted); font-size: .9rem; margin: 0; }
@keyframes enter { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: none; } }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }

.hero { position: relative; overflow: hidden; border-radius: 30px; padding: 3rem 2.4rem; border: 1px solid var(--border);
  background: linear-gradient(135deg, color-mix(in srgb, var(--accent2) 20%, transparent), color-mix(in srgb, var(--accent) 6%, transparent)), var(--surface);
  backdrop-filter: blur(20px); box-shadow: var(--shadow); animation: enter .8s var(--ease) both; }
.hero h1 { font-family: 'Inter Tight', sans-serif; font-weight: 800; font-size: clamp(2.1rem, 4.4vw, 3.6rem); letter-spacing: -.035em;
  line-height: 1.05; margin: 0 0 .9rem; max-width: 19ch;
  background: linear-gradient(120deg, var(--text) 35%, var(--accent) 70%, var(--accent2)); background-size: 200% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: shine 7s linear infinite; }
@keyframes shine { from { background-position: 0% 50%; } to { background-position: 200% 50%; } }
.hero p.lede { color: var(--muted); font-size: 1.08rem; line-height: 1.6; max-width: 58ch; margin: 0; }
.chip { display: inline-block; border: 1px solid var(--border); border-radius: 999px; padding: .38rem .85rem; font-size: .86rem;
  margin: .3rem .35rem 0 0; background: var(--surface); transition: transform .3s var(--ease), border-color .3s; }
.chip:hover { transform: translateY(-2px); border-color: var(--accent); }
.chip.accent { border-color: color-mix(in srgb, var(--accent) 60%, transparent); }
.feature { display: flex; gap: .75rem; align-items: center; margin: .6rem 0; font-size: 1.02rem; animation: rise .6s var(--ease) both; }
.feature i { width: 30px; height: 30px; border-radius: 10px; display: grid; place-items: center; font-style: normal;
  background: color-mix(in srgb, var(--accent) 16%, transparent); color: var(--accent); }

.stButton > button, .stFormSubmitButton > button, [data-testid="stDownloadButton"] > button {
  border-radius: 14px; border: 1px solid transparent; font-weight: 600; padding: .6rem 1.2rem;
  background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #1B5E20;
  box-shadow: 0 10px 26px color-mix(in srgb, var(--accent2) 30%, transparent); transition: transform .2s var(--ease), box-shadow .2s; }
.stButton > button:hover, .stFormSubmitButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
  transform: translateY(-2px); box-shadow: 0 16px 36px color-mix(in srgb, var(--accent2) 45%, transparent); }
.stButton > button[kind="secondary"] { background: var(--surface); color: var(--text); border-color: var(--border); box-shadow: none; }
[data-testid="stTextInput"] input { background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; transition: border-color .2s, box-shadow .2s; }
[data-testid="stTextInput"] input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 16%, transparent) !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div { background: var(--surface) !important; border-color: var(--border) !important; border-radius: 12px !important; color: var(--text) !important; }
[data-testid="stForm"] { border: 1px solid var(--border); border-radius: var(--r); padding: 1.5rem; background: var(--surface); box-shadow: var(--shadow); }
[data-testid="stRadio"] [role="radiogroup"] { gap: .6rem; }
[data-testid="stRadio"] [role="radiogroup"] > label { border: 1px solid var(--border); border-radius: 999px; padding: .5rem 1.1rem; background: var(--surface); transition: all .3s var(--ease); }
[data-testid="stRadio"] [role="radiogroup"] > label:hover { transform: translateY(-2px); border-color: var(--accent); }
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 14%, transparent); }
[data-testid="stFileUploader"] section { background: var(--surface) !important; border: 1.5px dashed color-mix(in srgb, var(--accent) 55%, transparent) !important;
  border-radius: 22px !important; min-height: 160px; transition: all .35s var(--ease); }
[data-testid="stFileUploader"] section:hover { border-color: var(--accent) !important; transform: translateY(-2px); box-shadow: 0 0 0 6px color-mix(in srgb, var(--accent) 10%, transparent); }
[data-testid="stImage"] img { border-radius: 18px; border: 1px solid var(--border); }
[data-baseweb="tab-list"] { gap: .4rem; background: var(--surface); padding: .35rem; border-radius: 14px; border: 1px solid var(--border); width: fit-content; }
[data-baseweb="tab"] { border-radius: 10px !important; color: var(--muted) !important; padding: .5rem .95rem !important; transition: all .25s var(--ease); }
[data-baseweb="tab"][aria-selected="true"] { background: linear-gradient(135deg, var(--accent), var(--accent2)) !important; color: #1B5E20 !important; font-weight: 600; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 16px; overflow: hidden; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--bg2), var(--bg)); border-right: 1px solid var(--border); }
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] { border-radius: 12px !important; transition: transform .3s var(--ease), background .3s; }
[data-testid="stSidebarNav"] a:hover, [data-testid="stSidebarNavLink"]:hover { transform: translateX(4px); background: var(--surface); }
[data-testid="stSidebarNav"] a[aria-current="page"], [data-testid="stSidebarNavLink"][aria-current="page"] {
  background: linear-gradient(90deg, color-mix(in srgb, var(--accent) 18%, transparent), transparent) !important;
  box-shadow: inset 3px 0 var(--accent); }
[data-testid="stDataFrame"] { background: var(--surface); }
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] { background: var(--surface) !important; color: var(--text) !important; }
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stDateInput"] input { background: var(--surface) !important; color: var(--text) !important; border-color: var(--border) !important; }

.skeleton { border-radius: 18px; min-height: 120px; background: linear-gradient(90deg, var(--surface) 25%, color-mix(in srgb, var(--text) 9%, transparent) 37%, var(--surface) 63%);
  background-size: 400% 100%; animation: shimmer 1.4s ease infinite; border: 1px solid var(--border); }
@keyframes shimmer { 0% { background-position: 100% 50%; } 100% { background-position: 0 50%; } }
.loader-ring { width: 56px; height: 56px; border-radius: 50%; margin: 0 auto;
  background: conic-gradient(from 0deg, var(--accent), var(--accent2), transparent 70%);
  -webkit-mask: radial-gradient(circle, transparent 55%, #000 56%); mask: radial-gradient(circle, transparent 55%, #000 56%);
  animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.success-badge { display: inline-flex; gap: .5rem; align-items: center; padding: .4rem .9rem; border-radius: 999px;
  background: color-mix(in srgb, var(--accent3) 16%, transparent); color: var(--accent3); font-weight: 600; animation: pop .5s var(--ease) both; }
@keyframes pop { 0% { transform: scale(.6); opacity: 0; } 100% { transform: none; opacity: 1; } }

.result-grid { display: grid; grid-template-columns: 240px 1fr; gap: 1.2rem; align-items: stretch; margin-top: 1rem; }
.gauge-wrap { position: relative; width: 170px; height: 170px; margin: 0 auto; }
.gauge-wrap svg { width: 100%; height: 100%; transform: rotate(-90deg); }
.gauge-track { stroke: color-mix(in srgb, var(--text) 9%, transparent); }
.gauge-val { stroke: var(--gc, var(--accent)); stroke-linecap: round; stroke-dasharray: 339.3; stroke-dashoffset: 339.3; animation: draw 1.4s var(--ease) .15s forwards; }
@keyframes draw { to { stroke-dashoffset: var(--off); } }
.gauge-center { position: absolute; inset: 0; display: grid; place-items: center; font-family: 'Inter Tight', sans-serif; font-weight: 800; font-size: 2rem; }
.gauge-center small { font-size: .95rem; opacity: .7; margin-left: 2px; }
.status { display: inline-flex; gap: .5rem; align-items: center; padding: .34rem .8rem; border-radius: 999px; font-weight: 600; font-size: .85rem; margin-top: .8rem; }
.status::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; animation: pulse 1.8s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 color-mix(in srgb, currentColor 60%, transparent); } 70% { box-shadow: 0 0 0 10px transparent; } 100% { box-shadow: 0 0 0 0 transparent; } }
.s-ok { color: var(--accent3); background: color-mix(in srgb, var(--accent3) 14%, transparent); }
.s-warn { color: var(--accent); background: color-mix(in srgb, var(--accent) 14%, transparent); }
.s-danger { color: var(--danger); background: color-mix(in srgb, var(--danger) 14%, transparent); }
.result-title { font-family: 'Inter Tight', sans-serif; font-weight: 800; font-size: clamp(1.9rem, 3.2vw, 2.8rem); letter-spacing: -.03em; line-height: 1.05; margin: .2rem 0 .5rem; }
.meter { height: 10px; border-radius: 99px; background: color-mix(in srgb, var(--text) 8%, transparent); overflow: hidden; margin: .8rem 0 1rem; }
.meter i { display: block; height: 100%; border-radius: 99px; width: var(--w); background: linear-gradient(90deg, var(--accent2), var(--accent)); transform-origin: left; animation: grow 1s var(--ease) both; }
.bars { display: grid; gap: .75rem; }
.bar-row { display: grid; grid-template-columns: 120px 1fr 58px; gap: .8rem; align-items: center; color: var(--muted); font-size: .95rem; }
.bar-row.top { color: var(--text); font-weight: 600; }
.bar-track { height: 10px; border-radius: 99px; background: color-mix(in srgb, var(--text) 8%, transparent); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, color-mix(in srgb, var(--accent) 40%, transparent), var(--accent)); transform-origin: left; animation: grow .9s var(--ease) both; }
.bar-row:not(.top) .bar-fill { background: color-mix(in srgb, var(--text) 30%, transparent); }
.bar-val { text-align: right; font-variant-numeric: tabular-nums; }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.sev { margin-top: 1rem; padding: 1rem 1.2rem; border-radius: 18px; border: 1px solid var(--border); background: var(--surface); }
.sev-head { display: flex; justify-content: space-between; align-items: baseline; color: var(--muted); margin-bottom: .5rem; }
.sev-head b { font-family: 'Inter Tight', sans-serif; font-size: 1.35rem; color: var(--danger); }
.insight { border-left: 3px solid var(--accent2); padding: .85rem 1rem; border-radius: 0 14px 14px 0; background: color-mix(in srgb, var(--accent2) 9%, transparent); margin: .5rem 0; line-height: 1.6; }
.section-title { font-family: 'Inter Tight', sans-serif; font-weight: 700; font-size: 1.6rem; letter-spacing: -.02em; margin: 2rem 0 .4rem; }
.empty-state { border: 1px dashed var(--border); border-radius: 22px; padding: 2.4rem 1.5rem; text-align: center; color: var(--muted); min-height: 180px; display: grid; place-items: center; }
.empty-state strong { display: block; font-family: 'Inter Tight', sans-serif; font-size: 1.3rem; color: var(--text); margin-bottom: .4rem; }
.hm-row { display: grid; grid-template-columns: 44px 1fr; gap: 8px; align-items: center; margin-bottom: 5px; font-size: .78rem; color: var(--muted); }
.hm-cells { display: grid; grid-template-columns: repeat(24, 1fr); gap: 3px; }
.hm-cells i { height: 16px; border-radius: 4px; background: var(--accent); transition: transform .2s var(--ease); }
.hm-cells i:hover { transform: scale(1.35); }
.tl { position: relative; padding-left: 1.3rem; }
.tl::before { content: ""; position: absolute; left: 5px; top: 6px; bottom: 6px; width: 1px; background: color-mix(in srgb, var(--accent) 25%, transparent); }
.tl-item { position: relative; padding: .4rem 0 .9rem; font-size: .92rem; animation: rise .5s var(--ease) both; }
.tl-item::before { content: ""; position: absolute; left: -1.3rem; top: .55rem; width: 10px; height: 10px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 12px var(--accent); }
.tl-item small { color: var(--muted); display: block; }
.live-dot { display: inline-flex; align-items: center; gap: .45rem; font-size: .82rem; color: var(--accent3); }
.live-dot::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--accent3); animation: livepulse 1.5s infinite; }
@keyframes livepulse { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }
.login-left { position: relative; overflow: hidden; border-radius: 30px; padding: 3rem; border: 1px solid var(--border); background: var(--surface); min-height: 480px; }
.login-left h1 { font-family: 'Inter Tight', sans-serif; font-weight: 800; font-size: clamp(2.2rem, 4.2vw, 3.8rem); line-height: 1.03; letter-spacing: -.03em;
  background: linear-gradient(120deg, var(--text) 30%, var(--accent) 60%, var(--accent2)); background-size: 200% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: shine 6s linear infinite; }
.login-card { padding: 2rem; border-radius: 26px; border: 1px solid var(--border); background: var(--surface); backdrop-filter: blur(22px); box-shadow: var(--shadow); animation: enter .6s .1s var(--ease) both; }
.login-card h2 { font-family: 'Inter Tight', sans-serif; font-weight: 800; font-size: 1.9rem; margin: .2rem 0 .3rem; letter-spacing: -.02em; }
.sec-grid { position: absolute; inset: 0; pointer-events: none; background-image: linear-gradient(color-mix(in srgb, var(--accent) 7%, transparent) 1px, transparent 1px), linear-gradient(90deg, color-mix(in srgb, var(--accent) 7%, transparent) 1px, transparent 1px); background-size: 44px 44px; }
.sec-scan { position: absolute; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), transparent); box-shadow: 0 0 22px var(--accent); animation: scan 5s linear infinite; }
@keyframes scan { from { top: 0; opacity: 0; } 10%, 90% { opacity: 1; } to { top: 100%; opacity: 0; } }
.radar { width: 230px; height: 230px; border-radius: 50%; position: relative; overflow: hidden; margin: 1.4rem auto;
  border: 1px solid color-mix(in srgb, var(--accent) 35%, transparent);
  background: repeating-radial-gradient(circle, transparent 0 30px, color-mix(in srgb, var(--accent) 12%, transparent) 30px 31px); }
.radar::before { content: ""; position: absolute; inset: 0; background: conic-gradient(from 0deg, color-mix(in srgb, var(--accent) 50%, transparent), transparent 70deg); animation: sweep 3.2s linear infinite; }
@keyframes sweep { to { transform: rotate(360deg); } }
.blip { position: absolute; width: 9px; height: 9px; border-radius: 50%; background: var(--accent3); box-shadow: 0 0 16px var(--accent3); animation: blip 3.2s infinite; }
@keyframes blip { 0%, 100% { opacity: 0; } 10% { opacity: 1; } 40% { opacity: 0; } }
.admin-badge { display: inline-block; font-size: .72rem; letter-spacing: .08em; padding: .25rem .65rem; border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 15%, transparent); color: var(--accent); font-weight: 600; }
.admin-brand { font-family: 'Inter Tight', sans-serif; font-weight: 800; font-size: 1.15rem; margin: .5rem 0 .2rem; }
.foot { margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--border); color: var(--muted); font-size: .88rem; }
.app-topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:1.2rem; }
.brand-lockup { display:flex; align-items:center; gap:.7rem; }
.brand-mark { width:42px; height:42px; display:grid; place-items:center; border-radius:14px; font-size:1.35rem;
  background:linear-gradient(135deg,var(--accent),var(--accent2)); box-shadow:0 10px 24px color-mix(in srgb,var(--accent2) 28%,transparent); }
.brand-name { font-family:'Inter Tight',sans-serif; font-size:1.08rem; font-weight:800; letter-spacing:-.02em; }
.brand-meta { display:block; color:var(--muted); font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; margin-top:.12rem; }
.live-pill { display:inline-flex; align-items:center; gap:.45rem; color:var(--accent3); border:1px solid color-mix(in srgb,var(--accent3) 26%,transparent);
  background:color-mix(in srgb,var(--accent3) 8%,transparent); border-radius:999px; padding:.42rem .72rem; font-size:.78rem; font-weight:600; }
.live-pill::before { content:""; width:7px; height:7px; border-radius:50%; background:currentColor; box-shadow:0 0 0 5px color-mix(in srgb,currentColor 14%,transparent); animation:livepulse 1.6s infinite; }
.workspace-hero { position:relative; overflow:hidden; padding:2rem 2.1rem; border:1px solid var(--border); border-radius:28px;
  background:linear-gradient(115deg,color-mix(in srgb,var(--accent2) 18%,transparent),color-mix(in srgb,var(--accent) 9%,transparent)),var(--surface);
  box-shadow:var(--shadow); }
.workspace-hero::after { content:""; position:absolute; width:280px; height:280px; right:-70px; top:-100px; border-radius:50%;
  border:1px solid color-mix(in srgb,var(--accent) 25%,transparent); box-shadow:0 0 0 28px color-mix(in srgb,var(--accent) 5%,transparent),0 0 0 56px color-mix(in srgb,var(--accent) 3%,transparent); }
.hero-kicker { color:var(--accent); font-size:.75rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin:0 0 .55rem; }
.workspace-hero h1 { font-family:'Inter Tight',sans-serif; font-size:clamp(2rem,4vw,3.35rem); line-height:1.04; letter-spacing:-.045em; max-width:15ch; margin:0 0 .7rem; }
.workspace-hero p { color:var(--muted); line-height:1.6; max-width:54ch; margin:0; }
.hero-side { position:relative; z-index:1; padding:1.15rem; border:1px solid var(--border); border-radius:20px; background:color-mix(in srgb,var(--surface) 72%,transparent); backdrop-filter:blur(16px); }
.hero-side-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:.8rem; font-size:.82rem; color:var(--muted); }
.ready { color:var(--accent3); font-weight:700; }
.ready::before { content:""; display:inline-block; width:7px; height:7px; border-radius:50%; background:currentColor; margin-right:.35rem; }
.model-row { display:flex; justify-content:space-between; gap:1rem; padding:.7rem 0; border-top:1px solid var(--border); font-size:.85rem; }
.model-row span:last-child { color:var(--accent3); font-weight:600; }
.metric-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem; margin:1rem 0 1.5rem; }
.mini-metric { padding:1rem 1.1rem; border-radius:18px; border:1px solid var(--border); background:var(--surface); }
.mini-metric b { display:block; font-family:'Inter Tight',sans-serif; font-size:1.5rem; color:var(--text); margin-top:.25rem; }
.mini-metric span { color:var(--muted); font-size:.78rem; }
.scan-heading { display:flex; align-items:end; justify-content:space-between; gap:1rem; margin:1.7rem 0 .75rem; }
.scan-heading h2 { font-family:'Inter Tight',sans-serif; font-size:1.45rem; letter-spacing:-.025em; margin:0; }
.scan-heading p { color:var(--muted); font-size:.86rem; margin:0; }
.track-panel { border:1px solid var(--border); border-radius:22px; background:var(--surface); padding:1.1rem 1.25rem; }
.track-panel h3 { font-family:'Inter Tight',sans-serif; font-size:1.15rem; margin:.1rem 0 .25rem; }
.track-panel p { color:var(--muted); font-size:.88rem; line-height:1.5; margin:0 0 .9rem; }
.upload-panel { border:1px solid var(--border); border-radius:22px; background:linear-gradient(145deg,var(--surface),color-mix(in srgb,var(--accent) 4%,transparent)); padding:1.1rem; min-height:100%; }
.upload-panel .hint { display:flex; gap:.6rem; align-items:flex-start; color:var(--muted); font-size:.82rem; line-height:1.45; margin-top:.75rem; }
.upload-panel .hint b { color:var(--accent); font-size:1.05rem; }
.recent-card { display:flex; align-items:center; justify-content:space-between; gap:.8rem; padding:.85rem 1rem; border:1px solid var(--border); border-radius:16px; background:var(--surface); margin-bottom:.55rem; }
.recent-card:hover { border-color:color-mix(in srgb,var(--accent) 45%,transparent); transform:translateX(3px); transition:transform .25s var(--ease),border-color .25s; }
.recent-main { min-width:0; }.recent-main b { display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.recent-main small { color:var(--muted); }.recent-score { text-align:right; white-space:nowrap; }
.recent-score b { display:block; color:var(--accent3); }.recent-score small { color:var(--muted); }
.landing-top { display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:1.2rem; }
.landing-brand { display:flex; align-items:center; gap:.7rem; }
.landing-brand .brand-mark { width:48px; height:48px; }
.landing-hero { position:relative; overflow:hidden; border-radius:32px; padding:clamp(2rem,6vw,4.7rem) clamp(1.4rem,5vw,4.5rem); margin-bottom:1.1rem;
  background:linear-gradient(120deg,color-mix(in srgb,var(--accent2) 22%,transparent),color-mix(in srgb,var(--accent) 12%,transparent)),var(--surface);
  border:1px solid var(--border); box-shadow:var(--shadow); }
.landing-hero::before { content:""; position:absolute; width:340px; height:340px; right:-80px; top:-145px; border-radius:50%;
  background:radial-gradient(circle,color-mix(in srgb,var(--accent) 28%,transparent),transparent 67%); animation:float 9s ease-in-out infinite; }
.landing-hero h1 { position:relative; z-index:1; font-family:'Inter Tight',sans-serif; font-size:clamp(2.5rem,6vw,5.2rem); line-height:.97; letter-spacing:-.06em; max-width:11ch; margin:0 0 1rem; }
.landing-hero p { position:relative; z-index:1; color:var(--muted); line-height:1.65; max-width:57ch; font-size:1.05rem; margin:0; }
.portal-icon { display:grid; place-items:center; width:48px; height:48px; border-radius:15px; font-size:1.45rem; background:color-mix(in srgb,var(--accent) 14%,transparent); margin-bottom:1.1rem; }
.portal-card h3 { font-family:'Inter Tight',sans-serif; font-size:1.35rem; margin:0 0 .4rem; }
.portal-card p { color:var(--muted); line-height:1.55; min-height:3.1rem; margin:0 0 1rem; }
.portal-meta { display:flex; flex-wrap:wrap; gap:.4rem; margin-bottom:1.1rem; }
.portal-meta span { color:var(--muted); border:1px solid var(--border); border-radius:999px; padding:.28rem .55rem; font-size:.72rem; }
.landing-note { text-align:center; color:var(--muted); font-size:.82rem; margin:1.1rem 0 0; }
.st-key-landing_customer_card_overlay button,
.st-key-landing_admin_card_overlay button {
  height:260px; min-height:260px; width:100%; position:relative; overflow:hidden;
  border-radius:22px; border:1px solid rgba(148,163,184,.13);
  padding:1.55rem 2.1rem 3.2rem; text-align:left; white-space:pre-line;
  align-items:flex-start; justify-content:flex-start;
  background:linear-gradient(145deg,rgba(55,74,118,.68),rgba(24,38,68,.92));
  color:#F8FAFC; box-shadow:0 18px 40px rgba(2,6,23,.18);
  backdrop-filter:blur(18px); font-family:'Inter',sans-serif;
  font-size:.96rem; line-height:1.55;
  transition:transform .32s var(--ease),border-color .32s var(--ease),box-shadow .32s var(--ease),background .32s var(--ease);
}
.st-key-landing_customer_card_overlay button::before,
.st-key-landing_admin_card_overlay button::before {
  content:""; position:absolute; width:140px; height:140px; right:-78px; bottom:-88px;
  border-radius:50%; border:1px solid rgba(250,204,21,.12);
  box-shadow:0 0 0 22px rgba(250,204,21,.025),0 0 0 44px rgba(250,204,21,.018);
}
.st-key-landing_customer_card_overlay button::after,
.st-key-landing_admin_card_overlay button::after {
  content:"→"; position:absolute; right:1.35rem; bottom:1.1rem;
  color:#FACC15; font-size:1.35rem; line-height:1;
}
.st-key-landing_customer_card_overlay button:hover,
.st-key-landing_admin_card_overlay button:hover {
  transform:translateY(-7px); border-color:rgba(250,204,21,.40);
  background:linear-gradient(145deg,rgba(63,84,132,.76),rgba(27,43,78,.96));
  box-shadow:0 24px 52px rgba(2,6,23,.30);
}
.portal-card-divider {
  width:58px; height:2px; border-radius:999px; margin:.55rem 0 1rem;
  background:linear-gradient(90deg,#FACC15,#22C55E);
}
@media (max-width: 700px) { .metric-strip { grid-template-columns:1fr; } .app-topbar { align-items:flex-start; } .live-pill { display:none; } .workspace-hero { padding:1.5rem; } }
@media (max-width: 900px) { .result-grid { grid-template-columns: 1fr; } .login-left { min-height: auto; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
"""

CUSTOMER_HIDE_ADMIN_CSS = """
[data-testid="collapsedControl"] { display: none !important; }
"""


def theme_css(role: str, light: bool) -> str:
    if role == ROLE_ADMIN:
        dark = bool(st.session_state.get("dark_mode"))
        return f":root {{ {THEME_VARS['admin_dark' if dark else 'admin']} }}"
    if bool(st.session_state.get("dark_mode")):
        return f":root {{ {THEME_VARS['customer_dark']} }}"
    return f":root {{ {THEME_VARS['customer_light']} }}"


def palette(role: str, light: bool) -> dict:
    if bool(st.session_state.get("dark_mode")):
        return {"text": "#F8FAFC", "muted": "#94A3B8", "grid": "rgba(148,163,184,.14)",
                "accent": "#FACC15", "accent2": "#22C55E", "accent3": "#22C55E", "danger": "#FB7185"}
    if role == ROLE_ADMIN:
        return {"text": "#1B5E20", "muted": "#4F684E", "grid": "rgba(46,125,50,.12)",
                "accent": "#FBC02D", "accent2": "#81C784", "accent3": "#2E7D32", "danger": "#A64B3C"}
    return {"text": "#1B5E20", "muted": "#4F684E", "grid": "rgba(46,125,50,.12)",
            "accent": "#FBC02D", "accent2": "#81C784", "accent3": "#2E7D32", "danger": "#A64B3C"}


def current_palette() -> dict:
    user = st.session_state.get("user")
    role = user["role"] if user else ROLE_CUSTOMER
    light = bool(st.session_state.get("light_mode")) and role == ROLE_CUSTOMER
    return palette(role, light)


# ---------- Utilities ----------

def compact(markup: str) -> str:
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def render_theme_toggle(key: str = "global_theme", label: str = "Dark mode") -> None:
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True
    selected = st.toggle(label, key=key, value=bool(st.session_state.dark_mode))
    if selected != bool(st.session_state.dark_mode):
        st.session_state.dark_mode = selected
        st.query_params["theme"] = "dark" if selected else "light"
        st.rerun()
    components.html(
        f"""<script>
        try {{
          const params = new URLSearchParams(window.location.search);
          const stored = localStorage.getItem("banalens_theme");
          if (!params.has("theme") && stored) {{
            params.set("theme", stored);
            window.location.search = params.toString();
          }} else {{
            localStorage.setItem("banalens_theme", "{'dark' if st.session_state.dark_mode else 'light'}");
          }}
        }} catch (e) {{}}
        </script>""",
        height=0,
    )


def now_str() -> str:
    return datetime.now().strftime(FMT)


def normalize_label(name) -> str:
    return str(name).lower().replace("_", "").replace(" ", "").replace("-", "")


def pretty(name) -> str:
    return str(name).replace("_", " ").title()


def greeting() -> str:
    hour = datetime.now().hour
    return "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"


def device_label() -> str:
    try:
        ua = st.context.headers.get("user-agent", "")
    except Exception:
        ua = ""
    return ua[:80] or "Unknown device"


def particles_html(count: int = 14) -> str:
    spans = []
    for i in range(count):
        left, size, dur, delay = (i * 37) % 100, 3 + (i * 7) % 6, 12 + (i * 5) % 10, -(i * 2.3)
        spans.append(f'<span class="particle" style="left:{left}%;--s:{size}px;--d:{dur}s;--dl:{delay:.1f}s"></span>')
    return f'<div class="particles" aria-hidden="true">{"".join(spans)}</div>'


def skeleton(height: int = 140) -> str:
    return f'<div class="skeleton" style="min-height:{height}px"></div>'


def gauge_html(percent: float, color: str) -> str:
    p = max(0.0, min(100.0, percent))
    offset = 339.3 * (1 - p / 100)
    return compact(f"""
        <div class="gauge-wrap" style="--off:{offset:.1f}; --gc:{color};">
          <svg viewBox="0 0 120 120" aria-hidden="true">
            <circle class="gauge-track" cx="60" cy="60" r="54" fill="none" stroke-width="10"/>
            <circle class="gauge-val" cx="60" cy="60" r="54" fill="none" stroke-width="10"/>
          </svg>
          <div class="gauge-center">{p:.0f}<small>%</small></div>
        </div>""")


def bars_html(probabilities: dict, top_label: str) -> str:
    rows = "".join(
        compact(f"""
            <div class="bar-row{' top' if name == top_label else ''}">
              <span>{html.escape(name)}</span>
              <div class="bar-track"><div class="bar-fill" style="width:{max(0.0, min(100.0, v)):.1f}%"></div></div>
              <span class="bar-val">{v:.1f}%</span>
            </div>""")
        for name, v in sorted(probabilities.items(), key=lambda item: -item[1])
    )
    return f'<div class="bars">{rows}</div>'


def parse_details(raw) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}


def paginate(df: pd.DataFrame, key: str, size: int = PAGE_SIZE) -> pd.DataFrame:
    """Server-side style pagination over an already-cached DataFrame."""
    total = len(df)
    pages = max(1, math.ceil(total / size))
    state_key = f"page_{key}"
    page = min(max(1, st.session_state.get(state_key, 1)), pages)
    left, middle, right = st.columns([1, 2, 1])
    if left.button("← Previous", key=f"{key}_prev", disabled=page <= 1, type="secondary"):
        st.session_state[state_key] = page - 1
        st.rerun()
    middle.markdown(f"<p class='sub' style='text-align:center;margin:.6rem 0'>Page {page} of {pages} · {total} records</p>",
                    unsafe_allow_html=True)
    if right.button("Next →", key=f"{key}_next", disabled=page >= pages, type="secondary"):
        st.session_state[state_key] = page + 1
        st.rerun()
    start = (page - 1) * size
    return df.iloc[start:start + size]


def require_role(role: str) -> None:
    user = st.session_state.get("user")
    if not user or user["role"] != role:
        st.error("You do not have access to this page.")
        st.stop()


# ---------- Charts (Chart.js in an embedded frame) ----------

CHART_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>html,body{margin:0;height:100%;background:transparent;font-family:Inter,system-ui,sans-serif}
canvas{width:100%!important;height:100%!important}</style></head>
<body><canvas id="c"></canvas><script>
const P = __PAL__;
const cfg = __CFG__;
const base = {
  responsive: true, maintainAspectRatio: false,
  animation: { duration: 900, easing: 'easeOutQuart' },
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { labels: { color: P.muted, usePointStyle: true, boxWidth: 8 } },
    tooltip: { backgroundColor: 'rgba(27,94,32,.96)', padding: 12, cornerRadius: 12, titleColor: '#FFFFFF', bodyColor: '#F4FBF4' }
  },
  scales: {
    x: { ticks: { color: P.muted, maxRotation: 0, autoSkipPadding: 12 }, grid: { display: false } },
    y: { ticks: { color: P.muted }, grid: { color: P.grid }, beginAtZero: true }
  }
};
cfg.options = Object.assign(base, cfg.options || {});
new Chart(document.getElementById('c'), cfg);
</script></body></html>"""


def chart_frame(config: dict, height: int = 260) -> None:
    payload = json.dumps(config).replace("</", "<\\/")
    doc = CHART_TEMPLATE.replace("__PAL__", json.dumps(current_palette())).replace("__CFG__", payload)
    components.html(doc, height=height)


def line_config(labels, series):
    datasets = [{
        "label": name, "data": values, "borderColor": color, "backgroundColor": color + "26",
        "fill": True, "tension": 0.35, "pointRadius": 2, "pointHoverRadius": 6, "borderWidth": 2.5,
    } for name, values, color in series]
    return {"type": "line", "data": {"labels": labels, "datasets": datasets}}


def bar_config(labels, values, color):
    return {"type": "bar", "data": {"labels": labels, "datasets": [{
        "label": "Count", "data": values, "backgroundColor": color + "CC", "borderRadius": 8, "maxBarThickness": 26,
    }]}}


def doughnut_config(labels, values, colors):
    return {"type": "doughnut", "data": {"labels": labels, "datasets": [{
        "data": values, "backgroundColor": colors, "borderWidth": 0, "hoverOffset": 8,
    }]}, "options": {"cutout": "68%"}}


KPI_TEMPLATE = """<!DOCTYPE html><html><head><style>
body{margin:0;font-family:Inter,system-ui,sans-serif;background:transparent}
.g{display:grid;grid-template-columns:repeat(__COLS__,1fr);gap:12px}
.k{position:relative;overflow:hidden;background:__SURF__;border:1px solid __BORDER__;border-radius:20px;padding:16px 18px;color:__TEXT__;
  backdrop-filter:blur(14px);transition:transform .45s cubic-bezier(.22,.8,.2,1),box-shadow .45s,border-color .45s;animation:rise .6s cubic-bezier(.22,.8,.2,1) both}
.k:hover{transform:translateY(-4px);box-shadow:0 18px 40px rgba(0,0,0,.28);border-color:__ACC__}
.k::after{content:"";position:absolute;right:-28px;top:-28px;width:90px;height:90px;border-radius:50%;background:radial-gradient(circle,__ACC__40,transparent 70%)}
.kl{color:__MUTE__;font-size:12.5px;font-weight:500}
.kv{font-family:"Inter Tight",Inter,sans-serif;font-size:30px;font-weight:800;margin-top:6px;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.kt{margin-top:6px;font-size:12px;font-weight:600}.up{color:__OK__}.down{color:__DANGER__}
@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
</style></head><body><div class="g">__CARDS__</div><script>
document.querySelectorAll('.kv').forEach((el,i)=>{
  const t=parseFloat(el.dataset.v), d=900, s=performance.now()+i*70;
  function f(n){ if(n<s){requestAnimationFrame(f);return}
    const p=Math.min((n-s)/d,1), e=1-Math.pow(1-p,3);
    el.textContent=Math.round(t*e).toLocaleString(); if(p<1) requestAnimationFrame(f); }
  requestAnimationFrame(f);
});
</script></body></html>"""


def kpi_counters(items, height: int = 260) -> None:
    pal = current_palette()
    admin = st.session_state.get("user", {}).get("role") == ROLE_ADMIN
    dark = bool(st.session_state.get("dark_mode"))
    if dark:
        surface = "rgba(30,41,59,.92)"
        border = "rgba(148,163,184,.16)"
    else:
        surface = "rgba(255,255,255,.86)" if admin else "rgba(255,255,255,.88)"
        border = "rgba(46,125,50,.16)" if admin else "rgba(46,125,50,.14)"
    cards = "".join(
        f'<div class="k"><div class="kl">{html.escape(label)}</div>'
        f'<div class="kv" data-v="{int(value)}">0</div>'
        f'<div class="kt {"up" if up else "down"}">{html.escape(trend)}</div></div>'
        for label, value, trend, up in items
    )
    doc = (KPI_TEMPLATE.replace("__COLS__", str(min(3, len(items))))
           .replace("__SURF__", surface).replace("__BORDER__", border)
           .replace("__TEXT__", pal["text"]).replace("__MUTE__", pal["muted"])
           .replace("__ACC__", pal["accent"]).replace("__OK__", pal["accent3"])
           .replace("__DANGER__", pal["danger"]).replace("__CARDS__", cards))
    components.html(doc, height=height)


# ---------- Database: cached reads, versioned invalidation, writes ----------

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, full_name TEXT NOT NULL,
  role TEXT NOT NULL, pw_hash TEXT NOT NULL, email TEXT, status TEXT NOT NULL DEFAULT 'active',
  created_at TEXT NOT NULL, last_active TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, username TEXT NOT NULL, role TEXT NOT NULL,
  login_at TEXT NOT NULL, logout_at TEXT, device TEXT, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, username TEXT NOT NULL, track TEXT NOT NULL,
  file_name TEXT, label TEXT, confidence REAL, risk TEXT, severity REAL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, action TEXT NOT NULL, detail TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, login_at);
CREATE INDEX IF NOT EXISTS idx_sessions_time ON sessions(login_at);
CREATE INDEX IF NOT EXISTS idx_scans_user ON scans(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_scans_time ON scans(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(username, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action, created_at);
"""


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    return hmac.compare_digest(hash_password(password, stored.split("$", 1)[0]), stored)


def _connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_resource(show_spinner=False)
def _version_box() -> dict:
    return {"v": 0}


def _version() -> int:
    return _version_box()["v"]


def _bump() -> None:
    _version_box()["v"] += 1


@st.cache_data(ttl=60, show_spinner=False, max_entries=512)
def _fetch_cached(sql: str, params: tuple, version: int) -> pd.DataFrame:
    """Cached read. The version argument invalidates every cached read after any write."""
    conn = _connect()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def fetch_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    return _fetch_cached(sql, tuple(params), _version())


def execute(sql: str, params: tuple = ()) -> int:
    conn = _connect()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        new_id = cur.lastrowid
    finally:
        conn.close()
    _bump()
    return new_id


@st.cache_resource(show_spinner=False)
def setup_database() -> bool:
    conn = _connect()
    try:
        conn.executescript(SCHEMA)
        user_cols = [row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "email" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL")
        cols = [row[1] for row in conn.execute("PRAGMA table_info(scans)").fetchall()]
        if "details" not in cols:
            conn.execute("ALTER TABLE scans ADD COLUMN details TEXT")
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            demo = [
                ("admin", "Platform Admin", ROLE_ADMIN, "Admin@123"),
                ("customer", "Demo Customer", ROLE_CUSTOMER, "Customer@123"),
            ]
            for username, full_name, role, password in demo:
                conn.execute(
                    "INSERT INTO users (username, full_name, role, pw_hash, created_at) VALUES (?,?,?,?,?)",
                    (username, full_name, role, hash_password(password), now_str()),
                )
            conn.commit()
    finally:
        conn.close()
    return True


def log_audit(username, action: str, detail: str = "") -> None:
    execute("INSERT INTO audit_log (username, action, detail, created_at) VALUES (?,?,?,?)",
            (username, action, detail, now_str()))


# ---------- Authentication and sessions ----------

def authenticate(role: str, username: str, password: str):
    df = fetch_df("SELECT * FROM users WHERE username = ?", (username.strip(),))
    ok = False
    if not df.empty:
        row = df.iloc[0]
        ok = (verify_password(password, row["pw_hash"]) and row["role"] == role and row["status"] == "active")
    if not ok:
        log_audit(username, "login_failed", role)
        return None
    return {k: row[k] for k in ("id", "username", "full_name", "role", "status")}


def register_customer(full_name: str, username: str, email: str, password: str) -> tuple[bool, str]:
    full_name = full_name.strip()
    username = username.strip().lower()
    email = email.strip().lower()
    if len(full_name) < 2:
        return False, "Please enter your full name."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if " " in username:
        return False, "Username cannot contain spaces."
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not any(char.isupper() for char in password) or not any(char.isdigit() for char in password):
        return False, "Password must include at least one uppercase letter and one number."

    existing = fetch_df(
        "SELECT username, email FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?",
        (username, email),
    )
    if not existing.empty:
        for row in existing.itertuples():
            if str(getattr(row, "username", "")).lower() == username:
                return False, "This username is already registered."
            if str(getattr(row, "email", "")).lower() == email:
                return False, "This email address is already registered."

    try:
        execute(
            "INSERT INTO users (username, full_name, role, pw_hash, email, status, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (username, full_name, ROLE_CUSTOMER, hash_password(password), email, "active", now_str()),
        )
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."
    log_audit(username, "user_registered", "Customer account created")
    return True, "Account created successfully. You can sign in now."


def start_session(user: dict) -> None:
    session_id = execute(
        "INSERT INTO sessions (user_id, username, role, login_at, device, status) VALUES (?,?,?,?,?,?)",
        (int(user["id"]), user["username"], user["role"], now_str(), device_label(), "Active"),
    )
    execute("UPDATE users SET last_active = ? WHERE id = ?", (now_str(), int(user["id"])))
    st.session_state.user = {k: (v.item() if hasattr(v, "item") else v) for k, v in user.items()}
    st.session_state.session_id = session_id
    st.session_state.last_active = datetime.now()
    log_audit(user["username"], "login", f"{user['role']} signed in")


def end_session(status: str) -> None:
    user = st.session_state.get("user")
    if not user:
        return
    execute("UPDATE sessions SET logout_at = ?, status = ? WHERE id = ?",
            (now_str(), status, st.session_state.session_id))
    execute("UPDATE users SET last_active = ? WHERE id = ?", (now_str(), int(user["id"])))
    log_audit(user["username"], "logout" if status == "Logged out" else "timeout", status)
    for key in ("user", "session_id", "last_active", "last_scan"):
        st.session_state.pop(key, None)


def enforce_timeout() -> None:
    user = st.session_state.get("user")
    if user and datetime.now() - st.session_state.last_active > timedelta(minutes=SESSION_TIMEOUT_MIN):
        end_session("Timed out")
        st.session_state.flash = "Your session timed out after 30 minutes. Please sign in again."


def touch_session() -> None:
    if st.session_state.get("user"):
        st.session_state.last_active = datetime.now()


# ---------- Model loading (lazy: only when a customer analyses a photo) ----------

@st.cache_resource(show_spinner=False)
def get_predictors():
    spec = importlib.util.spec_from_file_location(
        "banana_ripeness_predict", PROJECT_DIR / "src_Banana_ripeness" / "predict.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the banana ripeness prediction module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from banana_leaf_disease.predict import predict as predict_leaf
    return module.predict, predict_leaf


@st.cache_resource(show_spinner="Loading the subject check...")
def get_clip():
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    return CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name)


def normalize_image(raw: bytes):
    """Shrink and re-encode once. Faster model calls, smaller cache keys, smaller storage."""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    original_size = image.size
    image.thumbnail((1024, 1024))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    return buffer.getvalue(), image, original_size


@st.cache_data(show_spinner=False, max_entries=256)
def detect_subject(image_bytes: bytes) -> str:
    import torch
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model, processor = get_clip()
    inputs = processor(text=SUBJECT_PROMPTS, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
    best = int(probs.argmax())
    if float(probs[best]) < 0.5:
        return "unclear"
    return ["leaf", "fruit", "other"][best]


def photo_issues(image: Image.Image) -> list:
    gray = np.asarray(image.convert("L").resize((512, 512)), dtype=np.float32)
    issues = []
    if gray.mean() < DARK_THRESHOLD:
        issues.append("The photo is quite dark. Use brighter, even light for a more reliable result.")
    lap = (4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
    if lap.var() < BLUR_THRESHOLD:
        issues.append("The photo looks blurry. Hold the camera steady and focus on the leaf or fruit.")
    return issues


@st.cache_data(show_spinner=False, max_entries=256)
def run_model(image_bytes: bytes, mode: str) -> dict:
    predict_ripeness, predict_leaf = get_predictors()
    with tempfile.TemporaryDirectory() as folder:
        image_path = Path(folder) / "image.jpg"
        image_path.write_bytes(image_bytes)
        if mode == "Banana fruit":
            return predict_ripeness(str(image_path), checkpoint=str(RIPENESS_CHECKPOINT))
        return predict_leaf(str(image_path), checkpoint_path=str(LEAF_CHECKPOINT))


@st.cache_data(show_spinner=False, max_entries=256)
def leaf_severity(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((256, 256))
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    hue, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    healthy_leaf = (sat > 0.2) & (val > 0.15) & (hue >= 35) & (hue <= 170)
    lesion = (val < 0.35) & (hue >= 5) & (hue <= 50)
    leaf_total = int((healthy_leaf | lesion).sum())
    if leaf_total < 2000:
        return None
    return float(lesion.sum() / leaf_total * 100)


def risk_level(track: str, label: str, confidence_pct: float) -> str:
    key = normalize_label(label)
    if track == "Leaf Health":
        if key == "healthy":
            return "Low"
        return "High" if confidence_pct >= 60 else "Medium"
    return "Medium" if key == "besttoeat" else "Low"


def validate_upload(uploaded):
    if uploaded is None:
        return None
    if uploaded.name.rsplit(".", 1)[-1].lower() not in ALLOWED_EXTENSIONS:
        return "Only JPG and PNG images are supported."
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        return f"Images must be under {MAX_UPLOAD_MB} MB."
    return None


LEAF_GUIDANCE = {
    "Healthy": "No strong disease pattern was detected. Keep monitoring the plant and keep leaves well ventilated.",
    "Cordana": "Possible Cordana leaf spot. Remove severely affected leaves, avoid overhead watering, and consult a local expert about approved treatment.",
    "Sigatoka": "Possible Sigatoka leaf spot. Inspect nearby leaves, remove badly affected material, improve airflow, and consult a local expert about approved fungicide options.",
}
LEAF_MEANING = {
    "Healthy": "No strong disease pattern was detected on this leaf.",
    "Cordana": "Cordana leaf spot is a fungal disease. It shows as dark streaks or spots and spreads in humid, wet conditions.",
    "Sigatoka": "Sigatoka leaf spot is a fungal disease. It shows as yellow streaks that turn brown or black and reduces the leaf's ability to feed the plant.",
}


# ---------- Result rendering ----------

def headline_for(mode: str, label_raw: str) -> str:
    if mode == "Banana leaf":
        return pretty(label_raw)
    return "Best to eat" if normalize_label(label_raw) == "besttoeat" else "Not yet best to eat"


def render_result(label_raw: str, confidence: float, probabilities: dict, mode: str):
    headline = headline_for(mode, label_raw)
    if confidence >= 80:
        gauge_color, level_cls, level_text = "var(--accent3)", "s-ok", "High confidence"
    elif confidence >= 60:
        gauge_color, level_cls, level_text = "var(--accent)", "s-warn", "Moderate confidence"
    else:
        gauge_color, level_cls, level_text = "var(--danger)", "s-danger", "Uncertain"
    kicker = "Leaf screening result" if mode == "Banana leaf" else "Ripeness result"
    pretty_probs = {pretty(k): float(v) * 100 for k, v in probabilities.items()}
    st.markdown(compact(f"""
        <div class="result-grid">
          <div class="glass" style="text-align:center">
            {gauge_html(confidence, gauge_color)}
            <div><span class="status {level_cls}">{level_text}</span></div>
          </div>
          <div class="glass">
            <p class="eyebrow">{kicker}</p>
            <h2 class="result-title">{html.escape(headline)}</h2>
            <div class="meter"><i style="--w:{confidence:.1f}%"></i></div>
            {bars_html(pretty_probs, pretty(label_raw))}
          </div>
        </div>"""), unsafe_allow_html=True)
    return headline


def render_severity(percent) -> None:
    if percent is None:
        st.markdown('<div class="sev"><p class="sub" style="margin:0">The leaf area could not be measured. '
                    'Use a photo where the whole leaf is visible.</p></div>', unsafe_allow_html=True)
        return
    width = max(0.0, min(100.0, percent))
    st.markdown(compact(f"""
        <div class="sev">
          <div class="sev-head"><span>Estimated leaf area affected</span><b>{percent:.1f}%</b></div>
          <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:linear-gradient(90deg,var(--danger),var(--accent))"></div></div>
          <p class="sub" style="margin:.6rem 0 0;font-size:.88rem">Rough estimate from lesion colour. Treat it as a guide until a segmentation model is trained on labelled leaf masks.</p>
        </div>"""), unsafe_allow_html=True)


def render_explanation(track: str, headline: str, confidence: float, risk: str) -> None:
    if track == "Leaf Health":
        meaning = LEAF_MEANING.get(headline, "The model could not match this leaf to a known condition.")
        recommendation = LEAF_GUIDANCE.get(headline, "Inspect the plant and consult a local expert.")
        steps = ["Re-photograph the same leaf in daylight and compare the result.",
                 "Check nearby plants for the same pattern.",
                 "Revisit in a few days and compare with your history."]
    else:
        meaning = ("The model estimates this fruit has reached the stage most people prefer to eat."
                   if headline == "Best to eat" else "The model estimates this fruit is still ripening.")
        recommendation = ("Eat soon, or refrigerate to slow ripening." if headline == "Best to eat"
                          else "Leave it at room temperature and check again in a day or two.")
        steps = ["Photograph the fruit again tomorrow in the same light.",
                 "Compare the score with earlier scans to see the trend."]
    st.markdown("<h3 class='section-title' style='font-size:1.35rem'>Understanding this result</h3>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["What it means", "How to read it", "Recommendation", "Next steps"])
    with t1:
        st.markdown(f"<div class='insight'>{html.escape(meaning)}</div>", unsafe_allow_html=True)
    with t2:
        st.markdown(f"<p class='sub'>80% and above is high confidence. 60% to 79% is moderate, so check another sample. "
                    f"Below 60%, treat the result as a screening result only.</p>"
                    f"<p class='sub'>This result: <b>{confidence:.0f}%</b> confidence, <b>{risk}</b> risk.</p>",
                    unsafe_allow_html=True)
    with t3:
        st.markdown(f"<div class='insight'>{html.escape(recommendation)}</div>", unsafe_allow_html=True)
    with t4:
        for step in steps:
            st.markdown(f"<p class='sub'>• {html.escape(step)}</p>", unsafe_allow_html=True)


# ---------- Landing and login ----------

@st.cache_data(show_spinner=False)
def load_landing_page() -> str | None:
    """Read the landing page once per server process. Returns None if the file is missing."""
    if LANDING_PAGE_PATH.exists():
        return LANDING_PAGE_PATH.read_text(encoding="utf-8")
    return None


def render_landing_fallback() -> None:
    st.markdown(compact("""
        <div class="landing-top">
          <div class="landing-brand"><div class="brand-mark">🍌</div><div>
            <div class="brand-name">BanaLens</div><span class="brand-meta">Plant intelligence workspace</span>
          </div></div>
          <span class="live-pill">AI systems ready</span>
        </div>
        <section class="landing-hero">
          <div style="position:relative;z-index:2">
            <div class="hero-kicker">Banana health intelligence</div>
            <h1>Better decisions, from one photo.</h1>
            <p>Screen banana leaf health and fruit ripeness with explainable AI. Get a clear result, confidence score, and practical next steps for the field.</p>
          </div>
        </section>"""), unsafe_allow_html=True)
    st.markdown(compact("""
        <div class="metric-strip">
          <div class="mini-metric"><span>AI models</span><b>2</b></div>
          <div class="mini-metric"><span>Detection tracks</span><b>5</b></div>
          <div class="mini-metric"><span>Analysis mode</span><b style="font-size:1.12rem">Edge-ready</b></div>
        </div>"""), unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        if st.button(
            "◉  FIELD USER\n\nScan a plant or fruit\n\nUpload a photo and receive a focused health or ripeness assessment with a private scan history.\n\nLeaf disease · Ripeness · Guidance",
            key="landing_customer_card_overlay",
            use_container_width=True,
        ):
            st.session_state.portal = ROLE_CUSTOMER
            st.rerun()
    with col2:
        if st.button(
            "◆  OPERATIONS\n\nManage the platform\n\nReview system health, customer activity, model results, audit events, and operational trends.\n\nAnalytics · Users · Audit trail",
            key="landing_admin_card_overlay",
            use_container_width=True,
        ):
            st.session_state.portal = ROLE_ADMIN
            st.rerun()
    st.markdown("<p class='landing-note'>A screening aid for education and field triage · Not a replacement for professional agricultural diagnosis</p>",
                unsafe_allow_html=True)


def render_landing() -> None:
    render_theme_toggle("landing_theme")
    landing_html = load_landing_page()
    if landing_html:
        components.html(landing_html, height=LANDING_HEIGHT, scrolling=True)
    else:
        render_landing_fallback()


def render_registration() -> None:
    render_theme_toggle("registration_theme")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(compact("""
            <div class="login-left field">
              <div class="shape" style="width:260px;height:260px;top:-60px;right:-40px"></div>
              <div style="position:relative;z-index:2">
                <span class="chip accent">CREATE CUSTOMER ACCOUNT</span>
                <h1 style="margin-top:1.2rem">Start your BanaLens workspace.</h1>
                <p class="lede" style="font-size:1.04rem;max-width:44ch;margin-top:1rem">Create a secure account to run banana health scans and keep your analysis history private.</p>
                <div style="margin-top:1.4rem">
                  <div class="feature"><i>✓</i> Personal scan history</div>
                  <div class="feature"><i>✓</i> Leaf and ripeness analysis</div>
                  <div class="feature"><i>✓</i> Clear AI recommendations</div>
                </div>
              </div>
            </div>"""), unsafe_allow_html=True)
    with right:
        st.markdown(compact("""
            <div class="login-card">
              <span class="admin-badge">CUSTOMER REGISTRATION</span>
              <h2>Create an account</h2>
              <p class="sub">Complete the form below. Your account is created as a Customer automatically.</p>
            </div>"""), unsafe_allow_html=True)
        with st.form("customer_registration", clear_on_submit=False):
            full_name = st.text_input("Full name", placeholder="Example: Alex Kumar")
            username = st.text_input("Username", placeholder="Example: alexbanana")
            email = st.text_input("Email address", placeholder="Example: alex@example.com")
            password = st.text_input("Password", type="password", placeholder="At least 8 characters")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            if password != confirm_password:
                st.error("Password and confirm password do not match.")
            else:
                success, message = register_customer(full_name, username, email, password)
                if success:
                    st.success(message)
                    st.session_state.registration_success_username = username.strip().lower()
                    st.session_state.portal = ROLE_CUSTOMER
                    st.rerun()
                else:
                    st.error(message)
        if st.button("Already have an account? Sign in", key="registration_back", type="secondary", use_container_width=True):
            st.session_state.portal = ROLE_CUSTOMER
            st.rerun()


def render_login(role: str) -> None:
    render_theme_toggle("login_theme")
    if st.session_state.pop("flash", None):
        st.info(st.session_state.get("flash_text") or "Your session timed out. Please sign in again.")
    if role == ROLE_ADMIN:
        left_html = compact("""
            <div class="login-left">
              <div class="sec-grid"></div><div class="sec-scan"></div>
              <span class="admin-badge">SECURE ADMIN CONSOLE</span>
              <h1 style="margin-top:1rem">Monitor. Protect. Analyse.</h1>
              <p class="sub" style="font-size:1.04rem;max-width:40ch">Every sign-in is logged, sessions time out after inactivity, and customer activity stays visible to you.</p>
              <div class="radar">
                <div class="blip" style="top:30%;left:62%"></div>
                <div class="blip" style="top:60%;left:28%;animation-delay:-1.4s"></div>
                <div class="blip" style="top:44%;left:45%;animation-delay:-2.4s"></div>
              </div>
            </div>""")
        badge, title = "ADMIN", "Administrator sign in"
        sub = "Restricted to platform administrators. Every attempt is recorded."
        demo = "admin / Admin@123"
        default_username = "admin"
        default_password = "Admin@123"
    else:
        left_html = compact(f"""
            <div class="login-left field">
              {particles_html(18)}
              <div class="shape" style="width:260px;height:260px;top:-60px;right:-40px"></div>
              <div class="shape" style="width:120px;height:120px;bottom:12%;left:44%;animation-delay:-3s"></div>
              <div style="position:relative;z-index:2">
                <span class="chip accent">AI BANANA HEALTH PLATFORM</span>
                <h1 style="margin-top:1.2rem">Know your banana before it knows you.</h1>
                <p class="lede" style="font-size:1.04rem;max-width:44ch;margin-top:1rem">Upload a photo. Get a vision-transformer result, a risk level, and clear next steps.</p>
                <div style="margin-top:1.4rem">
                  <div class="feature" style="animation-delay:.2s"><i>✓</i> Leaf disease and ripeness in one place</div>
                  <div class="feature" style="animation-delay:.35s"><i>✓</i> Confidence scores you can read at a glance</div>
                  <div class="feature" style="animation-delay:.5s"><i>✓</i> Your scan history, kept private to you</div>
                </div>
              </div>
            </div>""")
        badge, title = "CUSTOMER", "Welcome back"
        sub = "Sign in to upload a photo and see your results."
        demo = "customer / Customer@123"
        default_username = "customer"
        default_password = "Customer@123"
        registered_username = st.session_state.pop("registration_success_username", "")
        if registered_username:
            default_username = registered_username
            default_password = ""
            st.success("Account created successfully. Sign in with your new username and password.")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(left_html, unsafe_allow_html=True)
    with right:
        st.markdown(compact(f"""
            <div class="login-card">
              <span class="admin-badge">{badge}</span>
              <h2>{title}</h2>
              <p class="sub">{sub}</p>
            </div>"""), unsafe_allow_html=True)
        with st.form(f"login_{role}"):
            username = st.text_input("Username", value=default_username)
            password = st.text_input("Password", value=default_password, type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            user = authenticate(role, username, password)
            if user is not None:
                start_session(user)
                st.session_state.portal = None
                st.rerun()
            st.error("Invalid username or password.")
        if SHOW_DEMO_HINTS:
            st.caption(f"Demo login: {demo}. Set SHOW_DEMO_HINTS to False before sharing the app.")
        if role == ROLE_CUSTOMER:
            if st.button("Create a new account", key=f"register_{role}", use_container_width=True):
                st.session_state.portal = "register"
                st.rerun()
        if st.button("Back", key=f"back_{role}", type="secondary"):
            st.session_state.portal = None
            st.rerun()


# ---------- Customer pages ----------

def page_analyse() -> None:
    user = st.session_state.user
    st.markdown(compact(f"""
        <div class="app-topbar">
          <div class="brand-lockup"><div class="brand-mark">🍌</div><div>
            <div class="brand-name">BanaLens</div><span class="brand-meta">Plant intelligence workspace</span>
          </div></div>
          <div style="display:flex;align-items:center;gap:.9rem"><span class="live-pill">Models online</span></div>
        </div>"""), unsafe_allow_html=True)
    top_l, top_r = st.columns([3, 1], vertical_alignment="center")
    with top_l:
        st.markdown(f"<p class='sub' style='margin:0'>{greeting()}, <b>{html.escape(user['full_name'])}</b> · Ready for a new scan</p>",
                    unsafe_allow_html=True)
    with top_r:
        render_theme_toggle("customer_theme", "Dark mode")

    scan_count = int(fetch_df("SELECT COUNT(*) AS n FROM scans WHERE user_id = ?", (int(user["id"]),)).iloc[0]["n"])
    avg_conf = fetch_df("SELECT AVG(confidence) AS n FROM scans WHERE user_id = ?", (int(user["id"]),)).iloc[0]["n"]
    last_scan = fetch_df("SELECT created_at FROM scans WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                         (int(user["id"]),))
    last_seen = str(last_scan.iloc[0]["created_at"])[5:16] if not last_scan.empty else "No scans yet"
    ripeness_ready = RIPENESS_CHECKPOINT.exists()
    leaf_ready = LEAF_CHECKPOINT.exists()
    st.markdown(compact(f"""
        <section class="workspace-hero">
          <div style="position:relative;z-index:1">
            <div class="hero-kicker">AI-powered field screening</div>
            <h1>See what your banana needs next.</h1>
            <p>Choose a scan, upload a focused photo, and get an explainable result for fruit ripeness or leaf health in seconds.</p>
            <div style="margin-top:1.1rem"><span class="chip accent">Vision Transformer</span><span class="chip">Private history</span><span class="chip">Field-ready</span></div>
          </div>
        </section>"""), unsafe_allow_html=True)
    st.markdown(compact(f"""
        <div class="metric-strip">
          <div class="mini-metric"><span>Total scans</span><b>{scan_count}</b></div>
          <div class="mini-metric"><span>Average confidence</span><b>{f"{float(avg_conf):.0f}%" if pd.notna(avg_conf) else "—"}</b></div>
          <div class="mini-metric"><span>Last activity</span><b style="font-size:1.12rem">{html.escape(last_seen)}</b></div>
        </div>"""), unsafe_allow_html=True)
    st.markdown("<div class='scan-heading'><div><h2>Start a new scan</h2><p>Select what you want to understand, then drop in one clear image.</p></div><span class='live-pill'>Instant analysis</span></div>",
                unsafe_allow_html=True)
    option_col, upload_col = st.columns([1, 1.35], gap="large")
    with option_col:
        mode = st.radio("Scan type", list(TRACKS.keys()),
                        format_func=lambda key: "🌿  Leaf health" if key == "Banana leaf" else "🍌  Fruit ripeness",
                        horizontal=True)
        ready = leaf_ready if mode == "Banana leaf" else ripeness_ready
        st.markdown(compact(f"""
            <div class="track-panel">
              <span class="chip accent">{'LEAF LAB' if mode == 'Banana leaf' else 'FRUIT LAB'}</span>
              <h3>{TRACKS[mode]["title"]}</h3>
              <p>{TRACKS[mode]["desc"]}</p>
              <div class="model-row"><span>Model status</span><span>{'Ready' if ready else 'Checkpoint missing'}</span></div>
              <div class="model-row"><span>Best input</span><span>One focused subject</span></div>
            </div>"""), unsafe_allow_html=True)
    with upload_col:
        version = st.session_state.get("uploader_version", 0)
        st.markdown("<div class='upload-panel'><div class='hero-kicker'>Upload your sample</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop a JPG or PNG here", type=list(ALLOWED_EXTENSIONS),
                                    label_visibility="visible", key=f"uploader_{version}")
        st.markdown(compact(f"""
            <div class="hint"><b>✦</b><span>Use bright, even light and keep the whole leaf or fruit in frame. Maximum file size: {MAX_UPLOAD_MB} MB.</span></div>
            </div>"""), unsafe_allow_html=True)

    error = validate_upload(uploaded)
    if uploaded is None:
        st.markdown(compact("""
            <div class="empty-state"><div><strong>Your result will appear here</strong>
            Upload a sample to reveal the class, confidence, risk level, and next best action.</div></div>"""),
            unsafe_allow_html=True)
    elif error:
        st.error(error)
    else:
        render_analysis(uploaded, mode, user, version)

    st.markdown("<div class='scan-heading'><div><h2>Recent scans</h2><p>A quick view of your latest field checks.</p></div></div>",
                unsafe_allow_html=True)
    recent = fetch_df("SELECT created_at, track, label, confidence, risk FROM scans WHERE user_id = ? "
                      "ORDER BY created_at DESC LIMIT 5", (int(user["id"]),))
    if recent.empty:
        st.markdown("<div class='empty-state' style='min-height:120px'><div><strong>Your scan timeline is empty</strong>Complete your first scan to start tracking changes.</div></div>",
                    unsafe_allow_html=True)
    else:
        cards = []
        for row in recent.itertuples():
            tone = "var(--accent3)" if str(row.risk) == "Low" else "var(--accent)" if str(row.risk) == "Medium" else "var(--danger)"
            cards.append(compact(f"""
                <div class="recent-card">
                  <div class="recent-main"><b>{'🌿' if row.track == 'Leaf Health' else '🍌'} {html.escape(str(row.label))}</b>
                    <small>{html.escape(str(row.track))} · {html.escape(str(row.created_at))}</small></div>
                  <div class="recent-score"><b style="color:{tone}">{float(row.confidence):.0f}%</b><small>{html.escape(str(row.risk))} risk</small></div>
                </div>"""))
        st.markdown("".join(cards), unsafe_allow_html=True)


def render_analysis(uploaded, mode: str, user: dict, version: int) -> None:
    placeholder = st.empty()
    placeholder.markdown(compact(f"""
        <div class="result-grid">
          {skeleton(200)}{skeleton(200)}
        </div>
        <div class="glass" style="text-align:center;margin-top:1rem">
          <div class="loader-ring"></div>
          <p class="sub" style="margin:.8rem 0 0">Checking photo quality, confirming the subject, running the model...</p>
        </div>"""), unsafe_allow_html=True)

    raw = uploaded.getvalue()
    image_bytes, image, original_size = normalize_image(raw)
    scan_key = f"{user['id']}-{uploaded.name}-{uploaded.size}-{mode}"
    issues = photo_issues(image)
    detected = detect_subject(image_bytes) if ENABLE_SUBJECT_CHECK else "unclear"

    mismatch = None
    if detected == "other":
        mismatch = "This photo does not look like a banana leaf or fruit. Upload a clear photo of one."
    elif detected == "unclear":
        mismatch = "The photo could not be identified clearly. Upload a sharper, brighter photo containing one banana leaf or fruit."
    elif mode == "Banana fruit" and detected == "leaf":
        mismatch = "Wrong scan selected: this image looks like a banana leaf. Choose Leaf Health for this photo."
    elif mode == "Banana leaf" and detected == "fruit":
        mismatch = "Wrong scan selected: this image looks like a banana fruit. Choose Fruit Ripeness for this photo."
    if mismatch:
        placeholder.empty()
        st.error(mismatch)
        st.caption("Tip: Select the correct radio option above, then upload the image again.")
        return

    result = run_model(image_bytes, mode)
    placeholder.empty()

    width_px, height_px = original_size
    st.markdown(compact(f"""
        <div style="display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0">
          <span class="chip">{html.escape(uploaded.name)}</span>
          <span class="chip">{uploaded.size / 1024:.0f} KB</span>
          <span class="chip">{width_px} × {height_px} px</span>
          <span class="chip">JPG preview</span>
        </div>
        <span class="success-badge">✓ Analysis complete</span>"""), unsafe_allow_html=True)
    if st.session_state.get("last_toast") != scan_key:
        st.toast("Analysis complete", icon="✅")
        st.session_state.last_toast = scan_key
    st.image(image, caption="Uploaded image", width=380)
    for issue in issues:
        st.warning(issue)

    label_raw = str(result["label"])
    confidence = float(result["confidence"]) * 100
    probabilities = result.get("probabilities") or {label_raw: float(result["confidence"])}
    probabilities = {str(k): float(v) for k, v in probabilities.items()}

    headline = render_result(label_raw, confidence, probabilities, mode)
    severity = leaf_severity(image_bytes) if mode == "Banana leaf" else None
    if mode == "Banana leaf":
        render_severity(severity)
    risk = risk_level(TRACKS[mode]["title"], label_raw, confidence)
    risk_cls = {"Low": "s-ok", "Medium": "s-warn", "High": "s-danger"}[risk]
    st.markdown(f"<p><span class='status {risk_cls}'>Risk level: {risk}</span></p>", unsafe_allow_html=True)

    if st.session_state.get("last_scan") != scan_key:
        details = json.dumps({"probabilities": probabilities, "headline": headline,
                              "track": TRACKS[mode]["title"]})
        execute(
            "INSERT INTO scans (user_id, username, track, file_name, label, confidence, risk, severity, created_at, details) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (int(user["id"]), user["username"], TRACKS[mode]["title"], uploaded.name, headline,
             round(confidence, 1), risk, severity, now_str(), details),
        )
        log_audit(user["username"], "upload", f"{TRACKS[mode]['title']} · {uploaded.name}")
        st.session_state.last_scan = scan_key

    render_explanation(TRACKS[mode]["title"], headline, confidence, risk)
    if st.button("Analyse another photo", key="another", type="secondary"):
        st.session_state.uploader_version = version + 1
        st.session_state.pop("last_scan", None)
        st.rerun()


def render_scan_table(df: pd.DataFrame, key: str) -> None:
    if df.empty:
        st.markdown(compact("""
            <div class="empty-state"><div><strong>No uploads yet</strong>Scans and results will appear here.</div></div>"""),
            unsafe_allow_html=True)
        return
    page = paginate(df, key)
    st.dataframe(
        page[["created_at", "track", "file_name", "label", "confidence", "risk"]].rename(columns={
            "created_at": "When", "track": "Track", "file_name": "File", "label": "Result",
            "confidence": "Confidence (%)", "risk": "Risk"}),
        use_container_width=True, hide_index=True,
    )


def render_scan_detail(df: pd.DataFrame, key: str) -> None:
    if df.empty:
        return
    options = {f"#{r.id} · {r.created_at} · {r.label}": r for r in df.itertuples()}
    choice = st.selectbox("Open a scan for detailed analysis", list(options.keys()), key=f"detail_{key}")
    row = options[choice]
    details = parse_details(row.details)
    probs = details.get("probabilities", {})
    pretty_probs = {pretty(k): float(v) for k, v in probs.items()} if probs else {}
    color = ("var(--accent3)" if (row.confidence or 0) >= 80 else
             "var(--accent)" if (row.confidence or 0) >= 60 else "var(--danger)")
    st.markdown(compact(f"""
        <div class="result-grid">
          <div class="glass" style="text-align:center">
            {gauge_html(float(row.confidence or 0), color)}
            <div><span class="status">{html.escape(str(row.risk))} risk</span></div>
          </div>
          <div class="glass">
            <p class="eyebrow">{html.escape(str(row.track))}</p>
            <h2 class="result-title">{html.escape(str(row.label))}</h2>
            <p class="sub">Scanned {html.escape(str(row.created_at))} · File {html.escape(str(row.file_name))}</p>
            {bars_html(pretty_probs, pretty(row.label)) if pretty_probs else "<p class='sub'>No per-class scores were stored for this scan.</p>"}
            {f"<p class='sub' style='margin-top:1rem'>Leaf area affected (estimate): <b>{row.severity:.1f}%</b></p>" if row.severity == row.severity and row.severity is not None else ""}
          </div>
        </div>"""), unsafe_allow_html=True)


def page_my_history() -> None:
    user = st.session_state.user
    st.markdown("<h2 class='section-title'>My history</h2>", unsafe_allow_html=True)
    scans = fetch_df("SELECT id, created_at, track, file_name, label, confidence, risk, severity, details "
                     "FROM scans WHERE user_id = ? ORDER BY created_at DESC", (int(user["id"]),))
    if scans.empty:
        render_scan_table(scans, "my")
        return
    tab_list, tab_trend, tab_detail = st.tabs(["All uploads", "Confidence trend", "Detailed analysis"])
    with tab_list:
        render_scan_table(scans, "my_list")
    with tab_trend:
        ordered = scans.iloc[::-1]
        pal = current_palette()
        chart_frame(line_config(ordered["created_at"].str[5:16].tolist(),
                                [("Confidence %", ordered["confidence"].round(1).tolist(), pal["accent"])]), height=260)
    with tab_detail:
        render_scan_detail(scans, "my")


# ---------- Admin pages ----------

def admin_counts() -> dict:
    now = datetime.now()
    week = (now - timedelta(days=7)).strftime(FMT)
    fortnight = (now - timedelta(days=14)).strftime(FMT)
    day_ago = (now - timedelta(hours=24)).strftime(FMT)
    today = date.today().isoformat()
    df = fetch_df("""
        SELECT
          (SELECT COUNT(*) FROM users) AS users,
          (SELECT COUNT(*) FROM users WHERE created_at >= ?) AS users_week,
          (SELECT COUNT(*) FROM users WHERE status='active' AND last_active >= ?) AS active_24h,
          (SELECT COUNT(*) FROM sessions WHERE login_at >= ?) AS logins_week,
          (SELECT COUNT(*) FROM sessions WHERE login_at >= ? AND login_at < ?) AS logins_prev,
          (SELECT COUNT(*) FROM audit_log WHERE action='upload' AND created_at >= ?) AS uploads_week,
          (SELECT COUNT(*) FROM audit_log WHERE action='upload' AND created_at >= ? AND created_at < ?) AS uploads_prev,
          (SELECT COUNT(*) FROM audit_log WHERE action='upload' AND created_at LIKE ?) AS uploads_today,
          (SELECT COUNT(*) FROM scans) AS results_total,
          (SELECT COUNT(*) FROM scans WHERE created_at >= ?) AS results_week,
          (SELECT COUNT(*) FROM scans WHERE created_at >= ? AND created_at < ?) AS results_prev
        """, (week, day_ago, week, fortnight, week, week, fortnight, week, f"{today}%",
              week, fortnight, week))
    return df.iloc[0].to_dict()


def trend(current: int, previous: int) -> tuple:
    if previous == 0:
        return ("▲ New this week" if current else "No change vs last week", True)
    change = (current - previous) / previous * 100
    return (f"{'▲' if change >= 0 else '▼'} {abs(change):.0f}% vs last week", change >= 0)


def system_health() -> int:
    checks = [DB_PATH.exists(), RIPENESS_CHECKPOINT.exists(), LEAF_CHECKPOINT.exists()]
    try:
        fetch_df("SELECT 1")
        checks.append(True)
    except Exception:
        checks.append(False)
    return round(100 * sum(checks) / len(checks))


@st.fragment(run_every="20s")
def live_kpis() -> None:
    c = admin_counts()
    up_t, up_ok = trend(int(c["uploads_week"]), int(c["uploads_prev"]))
    lg_t, lg_ok = trend(int(c["logins_week"]), int(c["logins_prev"]))
    rs_t, rs_ok = trend(int(c["results_week"]), int(c["results_prev"]))
    health = system_health()
    kpi_counters([
        ("Total users", c["users"], f"▲ {int(c['users_week'])} new this week", True),
        ("Active users (24 h)", c["active_24h"], "Signed in within a day", True),
        ("Daily logins (7 days)", c["logins_week"], lg_t, lg_ok),
        ("Upload trend (7 days)", c["uploads_week"], up_t, up_ok),
        ("Prediction count", c["results_total"], rs_t, rs_ok),
        ("System health (%)", health, "All checks passed" if health == 100 else "Some checks failing", health == 100),
    ], height=300)


def daily_series(table: str, column: str, days: int = 30, extra: str = ""):
    start = (date.today() - timedelta(days=days - 1)).isoformat()
    df = fetch_df(
        f"SELECT substr({column},1,10) AS day, COUNT(*) AS n FROM {table} "
        f"WHERE substr({column},1,10) >= ? {extra} GROUP BY day", (start,))
    counts = dict(zip(df["day"], df["n"])) if not df.empty else {}
    all_days = [(date.today() - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    return [d[5:] for d in all_days], [int(counts.get(d, 0)) for d in all_days]


def upload_matrix():
    df = fetch_df(
        "SELECT CAST(strftime('%w', created_at) AS INTEGER) AS d, CAST(strftime('%H', created_at) AS INTEGER) AS h, "
        "COUNT(*) AS n FROM audit_log WHERE action='upload' GROUP BY d, h")
    matrix = [[0] * 24 for _ in range(7)]
    for r in df.itertuples():
        matrix[int(r.d)][int(r.h)] = int(r.n)
    return matrix


def heatmap_html(matrix) -> str:
    peak = max(1, max(max(row) for row in matrix))
    rows = []
    for day, row in zip(DAY_NAMES, matrix):
        cells = "".join(
            f'<i title="{day} {hour:02d}:00 · {count} uploads" style="opacity:{0.1 + 0.9 * count / peak:.2f}"></i>'
            for hour, count in enumerate(row))
        rows.append(f'<div class="hm-row"><span>{day}</span><div class="hm-cells">{cells}</div></div>')
    return compact(f'<div>{"".join(rows)}</div>')


def timeline_html(rows: pd.DataFrame) -> str:
    items = []
    for i, row in enumerate(rows.itertuples()):
        text = ACTION_TEXT.get(row.action, row.action)
        items.append(f'<div class="tl-item" style="animation-delay:{i * 0.05:.2f}s">'
                     f'<b>{html.escape(str(row.username or "system"))}</b> {html.escape(text)}'
                     f'<small>{html.escape(str(row.created_at))}</small></div>')
    return compact(f'<div class="tl">{"".join(items)}</div>')


def page_overview() -> None:
    require_role(ROLE_ADMIN)
    st.markdown("<h2 class='section-title'>Overview</h2>", unsafe_allow_html=True)
    live_kpis()

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("<p class='sub'>Logins and uploads, last 30 days</p>", unsafe_allow_html=True)
        labels, logins = daily_series("sessions", "login_at")
        _, uploads = daily_series("audit_log", "created_at", extra="AND action = 'upload'")
        pal = current_palette()
        chart_frame(line_config(labels, [("Logins", logins, pal["accent3"]), ("Uploads", uploads, pal["accent"])]), height=260)
    with right:
        st.markdown("<p class='sub'>System status</p>", unsafe_allow_html=True)
        rows = [
            ("Database", "Reachable", True),
            ("Ripeness checkpoint", "Found" if RIPENESS_CHECKPOINT.exists() else "Missing", RIPENESS_CHECKPOINT.exists()),
            ("Leaf checkpoint", "Found" if LEAF_CHECKPOINT.exists() else "Missing", LEAF_CHECKPOINT.exists()),
        ]
        body = "".join(
            f"<div style='display:flex;justify-content:space-between;padding:.7rem .9rem;border-radius:14px;"
            f"background:var(--surface);border:1px solid var(--border);margin-bottom:.5rem'>"
            f"<span>{html.escape(n)}</span><b style='color:{'var(--accent3)' if ok else 'var(--danger)'}'>{html.escape(v)}</b></div>"
            for n, v, ok in rows)
        st.markdown(f"<div class='glass'>{body}</div>", unsafe_allow_html=True)

    h1, h2 = st.columns([1.3, 1], gap="large")
    with h1:
        st.markdown("<p class='sub'>Upload heat map (weekday by hour)</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='glass'>{heatmap_html(upload_matrix())}</div>", unsafe_allow_html=True)
    with h2:
        st.markdown("<p class='sub'>Live activity</p>", unsafe_allow_html=True)
        recent = fetch_df("SELECT username, action, created_at FROM audit_log ORDER BY id DESC LIMIT 8")
        st.markdown(f"<div class='glass'>{timeline_html(recent)}</div>" if not recent.empty
                    else "<div class='empty-state'>No activity yet.</div>", unsafe_allow_html=True)


def page_analytics() -> None:
    require_role(ROLE_ADMIN)
    st.markdown("<h2 class='section-title'>Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sub'>Interactive charts. Hover any point for details.</p>", unsafe_allow_html=True)
    pal = current_palette()
    labels, logins = daily_series("sessions", "login_at")
    _, uploads = daily_series("audit_log", "created_at", extra="AND action = 'upload'")
    _, results = daily_series("scans", "created_at")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<p class='sub'>Login trend, 30 days</p>", unsafe_allow_html=True)
        chart_frame(line_config(labels, [("Logins", logins, pal["accent3"])]), height=260)
    with b:
        st.markdown("<p class='sub'>Upload activity, 30 days</p>", unsafe_allow_html=True)
        chart_frame(bar_config(labels, uploads, pal["accent"]), height=260)
    c, d = st.columns(2, gap="large")
    with c:
        st.markdown("<p class='sub'>Results generated, 30 days</p>", unsafe_allow_html=True)
        chart_frame(line_config(labels, [("Results", results, pal["accent2"])]), height=260)
    with d:
        st.markdown("<p class='sub'>Risk distribution</p>", unsafe_allow_html=True)
        risk_df = fetch_df("SELECT risk, COUNT(*) AS n FROM scans GROUP BY risk")
        risk_map = dict(zip(risk_df["risk"], risk_df["n"])) if not risk_df.empty else {}
        names = ["Low", "Medium", "High"]
        chart_frame(doughnut_config(names, [int(risk_map.get(n, 0)) for n in names],
                                    [pal["accent3"], pal["accent"], pal["danger"]]), height=260)


@st.fragment(run_every="15s")
def login_monitor() -> None:
    latest = int(fetch_df("SELECT COALESCE(MAX(id), 0) AS m FROM audit_log").iloc[0]["m"])
    last = st.session_state.get("last_audit_id")
    if last is None:
        st.session_state.last_audit_id = latest
    elif latest > last:
        fresh = fetch_df(
            "SELECT username, action FROM audit_log WHERE id > ? AND action IN ('login','upload','login_failed') "
            "ORDER BY id DESC LIMIT 3", (int(last),))
        for row in fresh.itertuples():
            icon = {"login": "🔐", "upload": "🖼️", "login_failed": "⚠️"}.get(row.action, "•")
            st.toast(f"{row.username} {ACTION_TEXT.get(row.action, row.action)}", icon=icon)
        st.session_state.last_audit_id = latest

    active = int(fetch_df("SELECT COUNT(*) AS n FROM sessions WHERE status = 'Active'").iloc[0]["n"])
    st.markdown(f"<span class='live-dot'>{active} active session(s) · refreshes every 15 seconds</span>",
                unsafe_allow_html=True)
    df = fetch_df("SELECT username, role, login_at, logout_at, device, status FROM sessions "
                  "ORDER BY login_at DESC LIMIT 1000")
    if df.empty:
        st.info("No sign-ins recorded yet.")
        return
    df = df.fillna("—")
    df.columns = ["Username", "Role", "Login time", "Logout time", "Device", "Status"]

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    search = c1.text_input("Search username (press Enter)", key="mon_search")
    role = c2.selectbox("Role", ["All", "customer", "admin"], key="mon_role")
    start = c3.date_input("From", value=date.today() - timedelta(days=30), key="mon_from")
    end = c4.date_input("To", value=date.today(), key="mon_to")
    s1, s2 = st.columns([2, 1])
    sort_by = s1.selectbox("Sort by", df.columns.tolist(), index=2, key="mon_sort")
    order = s2.radio("Order", ["Newest first", "Oldest first"], horizontal=True, key="mon_order")

    view = df[df["Username"].str.contains(search, case=False, na=False)]
    if role != "All":
        view = view[view["Role"] == role]
    day = view["Login time"].astype(str).str[:10]
    view = view[(day >= start.isoformat()) & (day <= end.isoformat())]
    view = view.sort_values(sort_by, ascending=(order == "Oldest first"))
    st.dataframe(view.head(PAGE_SIZE * 10), use_container_width=True, hide_index=True)
    st.download_button("Export login history (CSV)", data=view.to_csv(index=False).encode("utf-8"),
                       file_name="login_history.csv", mime="text/csv")


def page_login_monitor() -> None:
    require_role(ROLE_ADMIN)
    st.markdown("<h2 class='section-title'>Login monitor</h2>", unsafe_allow_html=True)
    login_monitor()


def page_customer_history() -> None:
    require_role(ROLE_ADMIN)
    admin = st.session_state.user
    st.markdown("<h2 class='section-title'>Customer history & analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sub'>Every sign-in, upload, result, and activity for one customer, with export.</p>",
                unsafe_allow_html=True)

    customers = fetch_df("SELECT id, username, full_name, status, last_active, created_at FROM users "
                         "WHERE role = 'customer' ORDER BY username")
    if customers.empty:
        st.info("No customers yet.")
        return
    search = st.text_input("Find a customer (press Enter)", key="ch_search")
    shown = customers[customers["username"].str.contains(search, case=False, na=False)
                      | customers["full_name"].str.contains(search, case=False, na=False)]
    if shown.empty:
        st.info("No customers match that search.")
        return
    options = {f"{r.full_name} (@{r.username})": r for r in shown.itertuples()}
    choice = st.selectbox("Customer", list(options.keys()), key="ch_pick")
    profile = options[choice]
    uid = int(profile.id)

    summary = fetch_df("""
        SELECT
          (SELECT COUNT(*) FROM sessions WHERE user_id = ?) AS logins,
          (SELECT MAX(login_at) FROM sessions WHERE user_id = ?) AS last_login,
          (SELECT MAX(logout_at) FROM sessions WHERE user_id = ?) AS last_logout,
          (SELECT COUNT(*) FROM audit_log WHERE username = ? AND action = 'upload') AS uploads,
          (SELECT COUNT(*) FROM scans WHERE user_id = ?) AS results,
          (SELECT ROUND(AVG(confidence), 1) FROM scans WHERE user_id = ?) AS avg_conf
        """, (uid, uid, uid, profile.username, uid, uid)).iloc[0]

    st.markdown(compact(f"""
        <div class="glass" style="margin-bottom:1rem">
          <h3>{html.escape(profile.full_name)}</h3>
          <p class="sub" style="margin:0">@{html.escape(profile.username)} · Account {html.escape(str(profile.status))} ·
          Joined {html.escape(str(profile.created_at)[:10])} · Last active {html.escape(str(profile.last_active or 'never'))}</p>
        </div>"""), unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Logins", int(summary["logins"]))
    m2.metric("Last login", str(summary["last_login"] or "—")[5:16])
    m3.metric("Last active", str(profile.last_active or "—")[5:16])
    m4.metric("Uploads", int(summary["uploads"]))
    m5.metric("Results", int(summary["results"]))
    m6.metric("Avg confidence", f"{summary['avg_conf']:.1f}%" if pd.notna(summary["avg_conf"]) else "—")

    tab_sessions, tab_results, tab_analytics, tab_activity, tab_export = st.tabs(
        ["Login sessions", "Uploads & results", "Analytics", "Activity log", "Export"])

    with tab_sessions:
        sessions = fetch_df("SELECT login_at, logout_at, device, status FROM sessions WHERE user_id = ? "
                            "ORDER BY login_at DESC", (uid,))
        if sessions.empty:
            st.info("No sign-ins for this customer yet.")
        else:
            duration = (pd.to_datetime(sessions["logout_at"]) - pd.to_datetime(sessions["login_at"])).dt.total_seconds() / 60
            sessions["Duration (min)"] = duration.round(1).fillna(0)
            sessions = sessions.rename(columns={"login_at": "Login time", "logout_at": "Logout time",
                                                "device": "Device", "status": "Status"}).fillna("—")
            st.dataframe(paginate(sessions, "sessions"), use_container_width=True, hide_index=True)

    with tab_results:
        scans = fetch_df("SELECT id, created_at, track, file_name, label, confidence, risk, severity, details "
                         "FROM scans WHERE user_id = ? ORDER BY created_at DESC", (uid,))
        render_scan_table(scans, "admin_scans")
        st.markdown("<p class='sub' style='margin-top:1rem'>Detailed analysis</p>", unsafe_allow_html=True)
        render_scan_detail(scans, "admin")

    with tab_analytics:
        if scans.empty:
            st.info("Analytics appear after this customer's first scan.")
        else:
            ordered = scans.iloc[::-1]
            pal = current_palette()
            chart_frame(line_config(ordered["created_at"].str[5:16].tolist(),
                                    [("Confidence %", ordered["confidence"].round(1).tolist(), pal["accent"])]),
                        height=260)
            counts = scans["risk"].value_counts()
            names = ["Low", "Medium", "High"]
            chart_frame(doughnut_config(names, [int(counts.get(n, 0)) for n in names],
                                        [pal["accent3"], pal["accent"], pal["danger"]]), height=240)

    with tab_activity:
        activity = fetch_df("SELECT created_at, action, detail FROM audit_log WHERE username = ? "
                            "ORDER BY id DESC LIMIT 1000", (profile.username,))
        if activity.empty:
            st.info("No activity logged for this customer.")
        else:
            activity["action"] = activity["action"].map(lambda a: ACTION_TEXT.get(a, a))
            st.dataframe(paginate(activity.rename(columns={"created_at": "When", "action": "Event", "detail": "Detail"}),
                                  "activity"), use_container_width=True, hide_index=True)

    with tab_export:
        st.markdown("<p class='sub'>Download reports for this customer.</p>", unsafe_allow_html=True)
        sessions_all = fetch_df("SELECT login_at, logout_at, device, status FROM sessions WHERE user_id = ? ORDER BY login_at DESC", (uid,))
        scans_all = fetch_df("SELECT created_at, track, file_name, label, confidence, risk, severity FROM scans "
                             "WHERE user_id = ? ORDER BY created_at DESC", (uid,))
        e1, e2 = st.columns(2)
        e1.download_button("Download sign-in history (CSV)", data=sessions_all.to_csv(index=False).encode("utf-8"),
                           file_name=f"{profile.username}_sessions.csv", mime="text/csv")
        e2.download_button("Download results (CSV)", data=scans_all.to_csv(index=False).encode("utf-8"),
                           file_name=f"{profile.username}_results.csv", mime="text/csv")

    if st.session_state.get("viewed") != uid:
        log_audit(admin["username"], "view_customer_history", profile.username)
        st.session_state.viewed = uid


def page_users() -> None:
    require_role(ROLE_ADMIN)
    admin = st.session_state.user
    st.markdown("<h2 class='section-title'>User management</h2>", unsafe_allow_html=True)
    df = fetch_df("""
        SELECT u.id, u.username, u.full_name, u.role, u.status, u.created_at, u.last_active,
          (SELECT COUNT(*) FROM scans s WHERE s.user_id = u.id) AS scans,
          (SELECT COUNT(*) FROM sessions se WHERE se.user_id = u.id) AS logins
        FROM users u ORDER BY u.created_at DESC
    """)
    search = st.text_input("Search users (press Enter)", key="user_search")
    view = df[df["username"].str.contains(search, case=False, na=False)
              | df["full_name"].str.contains(search, case=False, na=False)]
    st.dataframe(paginate(view.drop(columns=["id"]), "users"), use_container_width=True, hide_index=True)

    st.markdown("<h3 class='section-title' style='font-size:1.3rem'>Edit or disable a user</h3>", unsafe_allow_html=True)
    if df.empty:
        return
    options = {f"{r.username} ({r.role})": r for r in df.itertuples()}
    target = options[st.selectbox("User", list(options.keys()), key="user_pick")]
    is_self = int(target.id) == int(admin["id"])
    if is_self:
        st.caption("You cannot change your own role or status, or delete your own account.")
    c1, c2, c3 = st.columns(3)
    new_name = c1.text_input("Full name", value=target.full_name, key=f"name_{target.id}")
    new_role = c2.selectbox("Role", [ROLE_CUSTOMER, ROLE_ADMIN], index=0 if target.role == ROLE_CUSTOMER else 1,
                            key=f"role_{target.id}", disabled=is_self)
    new_status = c3.selectbox("Account status", ["active", "disabled"], index=0 if target.status == "active" else 1,
                              key=f"status_{target.id}", disabled=is_self)
    if st.button("Save changes", key=f"save_{target.id}"):
        execute("UPDATE users SET full_name=?, role=?, status=? WHERE id=?",
                (new_name.strip() or target.full_name, new_role, new_status, int(target.id)))
        log_audit(admin["username"], "user_updated", f"{target.username}: {new_role}, {new_status}")
        st.toast("User updated", icon="✅")
        st.rerun()
    confirm = st.checkbox(f"I understand deleting {target.username} also removes their scans and sessions.",
                          key=f"confirm_{target.id}")
    if st.button("Delete user", key=f"delete_{target.id}", disabled=is_self or not confirm, type="secondary"):
        uid = int(target.id)
        execute("DELETE FROM scans WHERE user_id = ?", (uid,))
        execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
        execute("DELETE FROM users WHERE id = ?", (uid,))
        log_audit(admin["username"], "user_deleted", target.username)
        st.rerun()


def page_results() -> None:
    require_role(ROLE_ADMIN)
    admin = st.session_state.user
    st.markdown("<h2 class='section-title'>Result management</h2>", unsafe_allow_html=True)
    df = fetch_df("SELECT id, created_at, username, track, file_name, label, confidence, risk, severity "
                  "FROM scans ORDER BY created_at DESC")
    if df.empty:
        st.markdown("<div class='empty-state'><div><strong>No results yet</strong>Generated results will appear here.</div></div>",
                    unsafe_allow_html=True)
        return
    c1, c2, c3 = st.columns(3)
    user_filter = c1.selectbox("User", ["All"] + sorted(df["username"].unique().tolist()), key="res_user")
    track_filter = c2.selectbox("Track", ["All"] + sorted(df["track"].unique().tolist()), key="res_track")
    risk_filter = c3.selectbox("Risk", ["All", "Low", "Medium", "High"], key="res_risk")
    view = df
    if user_filter != "All":
        view = view[view["username"] == user_filter]
    if track_filter != "All":
        view = view[view["track"] == track_filter]
    if risk_filter != "All":
        view = view[view["risk"] == risk_filter]
    st.dataframe(paginate(view, "results"), use_container_width=True, hide_index=True)
    st.download_button("Export filtered results (CSV)", data=view.to_csv(index=False).encode("utf-8"),
                       file_name="banana_results.csv", mime="text/csv")

    options = {f"#{r.id} · {r.username} · {r.label}": int(r.id) for r in view.head(200).itertuples()}
    picked = st.multiselect("Select results to delete", list(options.keys()), key="res_delete_pick")
    if st.button("Delete selected", disabled=not picked, type="secondary"):
        ids = tuple(options[label] for label in picked)
        execute(f"DELETE FROM scans WHERE id IN ({','.join('?' for _ in ids)})", ids)
        log_audit(admin["username"], "results_deleted", f"{len(ids)} record(s)")
        st.rerun()


def page_audit() -> None:
    require_role(ROLE_ADMIN)
    st.markdown("<h2 class='section-title'>Audit log</h2>", unsafe_allow_html=True)
    df = fetch_df("SELECT created_at, username, action, detail FROM audit_log ORDER BY id DESC LIMIT 2000")
    if df.empty:
        st.info("No events recorded.")
        return
    choice = st.selectbox("Action", ["All"] + sorted(df["action"].unique().tolist()), key="audit_action")
    if choice != "All":
        df = df[df["action"] == choice]
    st.dataframe(paginate(df, "audit"), use_container_width=True, hide_index=True)


# ---------- Navigation (each page is loaded only when selected) ----------

CUSTOMER_PAGES = [
    st.Page(page_analyse, title="Analyse", icon="🍌", url_path="analyse", default=True),
    st.Page(page_my_history, title="My history", icon="🗂️", url_path="history"),
]
ADMIN_PAGES = [
    st.Page(page_overview, title="Overview", icon="📊", url_path="overview", default=True),
    st.Page(page_analytics, title="Analytics", icon="📈", url_path="analytics"),
    st.Page(page_customer_history, title="Customer history", icon="🧭", url_path="customers"),
    st.Page(page_login_monitor, title="Login monitor", icon="🔐", url_path="logins"),
    st.Page(page_users, title="Users", icon="👥", url_path="users"),
    st.Page(page_results, title="Results", icon="✅", url_path="results"),
    st.Page(page_audit, title="Audit log", icon="📜", url_path="audit"),
]


# ---------- Entry point ----------

setup_database()
enforce_timeout()

user = st.session_state.get("user")
portal = st.session_state.get("portal")
role = user["role"] if user else (portal or ROLE_CUSTOMER)
if "dark_mode" not in st.session_state:
    theme_param = str(st.query_params.get("theme", "dark")).lower()
    st.session_state.dark_mode = theme_param != "light"
light = bool(st.session_state.get("light_mode")) and role == ROLE_CUSTOMER
st.markdown(f"<style>{BASE_CSS}{theme_css(role, light)}</style>", unsafe_allow_html=True)

if user is None:
    if portal is None:
        render_landing()
    elif portal == "register":
        render_registration()
    else:
        render_login(portal)
    st.markdown(compact("""
        <p class="foot">A screening aid for education and field triage. It does not replace a professional agricultural diagnosis.</p>"""),
        unsafe_allow_html=True)
    st.stop()

touch_session()
with st.sidebar:
    st.markdown(compact(f"""
        <span class="admin-badge">{'ADMIN' if user['role'] == ROLE_ADMIN else 'CUSTOMER'}</span>
        <p class="admin-brand">Banana Platform</p>"""), unsafe_allow_html=True)
    st.caption(f"Signed in as {user['full_name']}")
    render_theme_toggle("sidebar_theme", "Dark mode")
    if st.button("Sign out", key="sign_out", type="secondary"):
        end_session("Logged out")
        st.rerun()

navigation = st.navigation(ADMIN_PAGES if user["role"] == ROLE_ADMIN else CUSTOMER_PAGES, position="sidebar")
navigation.run()

st.markdown(compact("""
    <p class="foot">A screening aid for education and field triage. It does not replace a professional agricultural diagnosis.</p>"""),
    unsafe_allow_html=True)
