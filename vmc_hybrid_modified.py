"""
VMC Water Flow — Hybrid Monitor + Analyser (BATCH MODE)
Run:  streamlit run vmc_hybrid.py

Tabs: Live/Batch Feed · EDA · Anomaly Detection · Forecast · Data Table · Pattern Analysis · QoS Trend
Batch mode pulls the full configured window in one API call; live mode polls every second.
Pattern tab fetches Jan+Feb, extracts daily shapes via K-Means, and scores every day vs benchmark.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import requests, urllib3, json, time, sqlite3, io
from datetime import datetime, timedelta
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import pdfplumber
    _PDFPLUMBER = True
except ImportError:
    _PDFPLUMBER = False
try:
    from pypdf import PdfReader as _PdfReader
    _PYPDF = True
except ImportError:
    _PYPDF = False
import os

PATTERN_CACHE_FILE = "vmc_pattern_cache.csv"
CACHE_MAX_AGE_DAYS = 7

def save_pattern_cache(df: pd.DataFrame):
    df.to_csv(PATTERN_CACHE_FILE, index=False)

def load_pattern_cache() -> pd.DataFrame | None:
    if not os.path.exists(PATTERN_CACHE_FILE):
        return None
    file_age_days = (datetime.now() - datetime.fromtimestamp(
        os.path.getmtime(PATTERN_CACHE_FILE))).days
    if file_age_days > CACHE_MAX_AGE_DAYS:
        return None
    return pd.read_csv(PATTERN_CACHE_FILE, parse_dates=["timestamp"])


# ── CONFIG ────────────────────────────────────────────────────────────────────

VMC_BASE    = "https://scph1.vmcsmartwater.in:9090"

HISTORY_API_PATHS = [
    "/ph1/data",
    "/api/history/sensor/Flow/Rate",
    "/api/sensor/Flow/Rate/history",
    "/api/realtime/sensor/Flow/Rate",
]

REALTIME_API_PATH = "/api/realtime/sensor/Flow/Rate"

OBJECT_NAME = "AIB_FT015"
VMC_USER    = "7644881557"
VMC_PASS    = "5678"
DB_PATH     = "vmc_readings.db"
IST_OFFSET  = timedelta(hours=5, minutes=30)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VMC · MJP-4231",
    layout="wide",
    page_icon="💧",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"],.main,.block-container{
    background-color:#0f1117!important;color:#e0e0e0!important;font-family:'Inter',sans-serif!important}
[data-testid="stSidebar"]{background-color:#1a1d27!important;border-right:1px solid #2a2d3a!important}
[data-testid="stSidebar"] *{color:#c0c4d0!important}
.block-container{padding:1rem 2rem 2rem!important;max-width:100%!important}
[data-testid="stTabs"] [role="tab"]{color:#7a8196!important;font-size:0.85rem!important;font-family:'Inter',sans-serif!important}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:#4ecdc4!important;border-bottom:2px solid #4ecdc4!important}
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:#1a1d27!important;border-bottom:1px solid #2a2d3a!important}
.metric-card{background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;padding:16px 20px}
.metric-label{font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.metric-value{font-size:1.9rem;font-weight:600;line-height:1;color:#e0e6f0;font-variant-numeric:tabular-nums}
.metric-value.danger{color:#ff6b6b}
.log-card{background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;padding:16px 20px}
.log-title{font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px;font-weight:600}
.log-row{display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1e2130}
.log-row:last-child{border-bottom:none}
.log-time{font-size:.78rem;color:#7a8196;font-variant-numeric:tabular-nums}
.log-badge{background:rgba(255,107,107,.13);color:#ff6b6b;border:1px solid rgba(255,107,107,.28);border-radius:20px;padding:2px 10px;font-size:.72rem}
.live-pill{display:inline-flex;align-items:center;gap:6px;background:rgba(78,205,196,.12);color:#4ecdc4;border:1px solid rgba(78,205,196,.3);border-radius:20px;padding:4px 12px;font-size:.72rem;font-weight:500;animation:pulse 1.5s ease-in-out infinite}
.live-dot{width:7px;height:7px;border-radius:50%;background:#4ecdc4}
.batch-pill{display:inline-flex;align-items:center;gap:6px;background:rgba(78,145,217,.12);color:#4a90d9;border:1px solid rgba(78,145,217,.3);border-radius:20px;padding:4px 12px;font-size:.72rem;font-weight:500}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
h1,h2,h3{color:#c8cde0!important}
.stAlert{border-radius:8px!important}
label,.stSlider label,.stNumberInput label{color:#888ea0!important;font-size:.8rem!important}
[data-testid="stHorizontalBlock"]{gap:12px!important}
div[data-testid="stMarkdownContainer"] p{margin:0}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#1a1d27","axes.facecolor":"#1a1d27",
    "axes.edgecolor":"#2a2d3a","axes.labelcolor":"#7a8196",
    "xtick.color":"#555d6e","ytick.color":"#555d6e",
    "grid.color":"#23263a","text.color":"#c8cde0",
    "legend.facecolor":"#1a1d27","legend.edgecolor":"#2a2d3a",
    "font.family":"sans-serif","font.size":9,
})

# ── SQLITE ────────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            flow_rate REAL NOT NULL,
            is_anomaly INTEGER DEFAULT 0
        )
    """)
    con.commit()
    con.close()

def db_insert(ts: datetime, flow: float, anom: int):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO readings (timestamp,flow_rate,is_anomaly) VALUES (?,?,?)",
                (ts.isoformat(), flow, anom))
    con.commit()
    con.close()

def db_insert_batch(rows: list):
    if not rows:
        return
    con = sqlite3.connect(DB_PATH)
    con.executemany(
        "INSERT OR IGNORE INTO readings (timestamp,flow_rate,is_anomaly) VALUES (?,?,?)",
        rows,
    )
    con.commit()
    con.close()

def db_load(hours_back: int = 24) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    since = (datetime.now() - timedelta(hours=hours_back)).isoformat()
    df = pd.read_sql(
        "SELECT timestamp,flow_rate,is_anomaly FROM readings WHERE timestamp>=? ORDER BY timestamp",
        con, params=(since,)
    )
    con.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.rename(columns={"flow_rate":"flow_rate_m3hr"})
    return df

def db_count() -> int:
    con = sqlite3.connect(DB_PATH)
    n = con.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    con.close()
    return n

def db_clear():
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM readings")
    con.commit()
    con.close()

init_db()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [("live_rows",[]),("anom_log",[]),("last_raw",""),
             ("last_error",""),("token",None),("field_map",{}),
             ("batch_done", False), ("batch_count", 0),
             ("pattern_df", None), ("benchmark_curve", None),
             ("benchmark_windows", None),
             ("curves_df", None), ("all_curves", None),
             ("centroids", None), ("modal_idx", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── HTTP SESSION ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_session():
    s = requests.Session()
    s.verify = False
    s.headers.update({
        "User-Agent":"Mozilla/5.0","Accept":"application/json, text/plain, */*",
        "Referer":f"{VMC_BASE}/dashboard","Origin":VMC_BASE,
    })
    return s

SESSION = get_session()

# ── LOGIN ─────────────────────────────────────────────────────────────────────
def try_login():
    if st.session_state.token: return True
    try: SESSION.get(f"{VMC_BASE}/login", timeout=8)
    except: pass
    for path in ["/login","/api/login","/api/auth","/api/token","/dashboard/login"]:
        try:
            r = SESSION.post(f"{VMC_BASE}{path}",
                data={"username":VMC_USER,"password":VMC_PASS},
                headers={"Content-Type":"application/x-www-form-urlencoded"},
                timeout=8, allow_redirects=True)
            if r.status_code==200 and "<title>login</title>" not in r.text.lower():
                st.session_state.token="session"; return True
            r2 = SESSION.post(f"{VMC_BASE}{path}",
                json={"username":VMC_USER,"password":VMC_PASS},timeout=8)
            if r2.status_code==200:
                d=r2.json()
                tok=d.get("token") or d.get("access_token") or d.get("jwt")
                if tok: SESSION.headers["Authorization"]=f"Bearer {tok}"
                st.session_state.token=tok or "session"; return True
        except: pass
    return False

# ── FIELD EXTRACTOR ───────────────────────────────────────────────────────────
def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None) + IST_OFFSET
        return ts
    except Exception:
        return None

def _extract(row: dict, fallback_ts: datetime):
    ts = fallback_ts
    for tk in ["DateTime","dateTime","timestamp","time","Timestamp","ts","date"]:
        raw = row.get(tk)
        if raw:
            parsed = _parse_ts(str(raw)[:25])
            if parsed:
                ts = parsed; break
            try: ts=datetime.fromisoformat(str(raw)[:19]); break
            except: pass
    numeric = {}
    for k, v in row.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lk = k.lower()
            if not any(x in lk for x in ["id","time","stamp","index","seq","row","count","num"]):
                numeric[k] = float(v)
    st.session_state.field_map = numeric
    if not numeric: return None, ts
    for pk in ["Value","value","flow","Flow","flowRate","flow_rate","reading",
               "val","data","Flow_Rate","FlowRate","instantaneous","rate","FLOW"]:
        if pk in numeric: return numeric[pk], ts
    nonzero = {k: v for k, v in numeric.items() if v != 0.0}
    if nonzero: return next(iter(nonzero.values())), ts
    return next(iter(numeric.values())), ts

# ── BATCH RESPONSE PARSER ─────────────────────────────────────────────────────
def _parse_batch_response(data, fallback_ts: datetime) -> list[dict]:
    records = []

    # tagname-keyed list (most common VMC format)
    if (isinstance(data, list) and data
            and isinstance(data[0], dict) and "tagname" in data[0]):
        rows = [d for d in data if d.get("tagname") == OBJECT_NAME]
        if not rows:
            rows = [d for d in data if float(d.get("value") or 0) > 0]
        for row in rows:
            try:
                flow = float(row.get("value") or 0)
            except (TypeError, ValueError):
                continue
            ts = None
            for tk in ["updated_at", "created_at", "DateTime", "timestamp"]:
                ts = _parse_ts(str(row.get(tk, "")))
                if ts:
                    break
            if ts is None:
                ts = fallback_ts
            records.append({"timestamp": ts, "flow_rate": flow})

    # [timestamp_ms, value] pairs
    elif (isinstance(data, list) and data
          and isinstance(data[0], (list, tuple))):
        for pt in data:
            try:
                ts   = datetime.utcfromtimestamp(float(pt[0]) / 1000) + IST_OFFSET
                flow = float(pt[1])
                records.append({"timestamp": ts, "flow_rate": flow})
            except Exception:
                continue

    # {"data": [...]} wrapper
    elif isinstance(data, dict) and "data" in data:
        pts = data["data"]
        if pts and isinstance(pts[0], dict):
            for row in pts:
                flow, ts = _extract(row, fallback_ts)
                if flow is not None:
                    records.append({"timestamp": ts, "flow_rate": flow})
        elif pts:
            for pt in pts:
                try:
                    ts   = datetime.utcfromtimestamp(float(pt[0]) / 1000) + IST_OFFSET
                    flow = float(pt[1])
                    records.append({"timestamp": ts, "flow_rate": flow})
                except Exception:
                    continue

    elif isinstance(data, list) and data and isinstance(data[0], dict):
        for row in data:
            flow, ts = _extract(row, fallback_ts)
            if flow is not None:
                records.append({"timestamp": ts, "flow_rate": flow})

    elif isinstance(data, dict):
        flow, ts = _extract(data, fallback_ts)
        if flow is not None:
            records.append({"timestamp": ts, "flow_rate": flow})

    # deduplicate and sort
    seen = set()
    unique = []
    for rec in records:
        key = rec["timestamp"].isoformat()
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    unique.sort(key=lambda x: x["timestamp"])
    return unique

# ── BATCH FETCH ───────────────────────────────────────────────────────────────
def fetch_batch(hours: int = 24) -> list[dict]:
    now   = datetime.now()
    start = now - timedelta(hours=hours)
    st.session_state.last_error = ""

    for path in HISTORY_API_PATHS:
        try:
            r = SESSION.get(
                f"{VMC_BASE}{path}",
                params={
                    "objectname": OBJECT_NAME,
                    "startTime":  start.strftime("%Y-%m-%d %H:%M:%S"),
                    "endTime":    now.strftime("%Y-%m-%d %H:%M:%S"),
                },
                timeout=60,
            )
        except Exception as e:
            st.session_state.last_error = f"[{path}] {e}"
            continue

        if "<title>login</title>" in r.text.lower():
            st.session_state.token = None
            continue

        st.session_state.last_raw = (
            f"HTTP {r.status_code} | path={path} | window={hours}h"
            f"\nURL: {r.url}\n\n{r.text[:3000]}")

        if r.status_code != 200:
            continue

        try:
            data = r.json()
        except Exception:
            st.session_state.last_raw = f"[{path}] Non-JSON: {r.text[:500]}"
            continue

        records = _parse_batch_response(data, now)

        if len(records) > 1:
            return records
        st.session_state.last_raw += (
            f"\n\n⚠️ [{path}] returned only {len(records)} row(s) — "
            f"likely a realtime-only endpoint. Trying next...")

    return []

# ── SINGLE READING — live poll ────────────────────────────────────────────────
def fetch_reading():
    now = datetime.now()
    for delta in [timedelta(hours=1), timedelta(hours=6), timedelta(hours=24)]:
        start = now - delta
        try:
            r = SESSION.get(f"{VMC_BASE}{REALTIME_API_PATH}",
                params={"objectname":OBJECT_NAME,
                        "startTime":start.strftime("%Y-%m-%d %H:%M:%S"),
                        "endTime":now.strftime("%Y-%m-%d %H:%M:%S")}, timeout=10)
        except Exception as e:
            st.session_state.last_error = str(e); return None
        if "<title>login</title>" in r.text.lower():
            st.session_state.token = None; return None
        try: data = r.json()
        except:
            st.session_state.last_raw = f"Non-JSON: {r.text[:500]}"; return None
        st.session_state.last_raw = (
            f"HTTP {r.status_code} | window={delta}\nURL: {r.url}\n\n{r.text[:3000]}")
        if r.status_code != 200: continue

        flow, ts = None, now

        if isinstance(data, list) and data and isinstance(data[0], dict) and "tagname" in data[0]:
            row = next((d for d in data if d.get("tagname") == OBJECT_NAME), None)
            if row is None or float(row.get("value") or 0) == 0.0:
                candidates = [d for d in data if float(d.get("value") or 0) > 0]
                candidates.sort(key=lambda d: d.get("updated_at",""), reverse=True)
                if candidates: row = candidates[0]
            if row:
                try: flow = float(row["value"])
                except: flow = None
                for tk in ["updated_at","created_at"]:
                    raw = row.get(tk,"")
                    if raw:
                        parsed = _parse_ts(raw)
                        if parsed:
                            ts = parsed; break
        elif isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
            ts = datetime.utcfromtimestamp(float(data[-1][0])/1000)+IST_OFFSET
            flow = float(data[-1][1])
        elif isinstance(data, dict) and "data" in data:
            pts = data["data"]
            if pts and isinstance(pts[0], dict): flow, ts = _extract(pts[-1], now)
            elif pts:
                ts = datetime.utcfromtimestamp(float(pts[-1][0])/1000)+IST_OFFSET
                flow = float(pts[-1][1])
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            flow, ts = _extract(data[-1], now)
        elif isinstance(data, dict):
            flow, ts = _extract(data, now)
        else:
            ts = datetime.now()

        if flow is not None:
            return {"timestamp": ts, "flow_rate_m3hr": float(flow)}
    return None

# ── ANOMALY — live tab ────────────────────────────────────────────────────────
def is_anomaly_live(val, history, spike_thresh, z_thresh):
    if val < 0 or val > spike_thresh:
        return True
    if val < 5 and len(history) >= 5 and np.mean(history[-5:]) > 50:
        return True
    if len(history) < 10:
        return False
    arr = np.array(history[-60:])
    std = arr.std()
    if std < 1e-6:
        return False
    if abs(val - arr.mean()) / std > z_thresh:
        return True
    recent_mean = np.mean(history[-10:])
    if recent_mean > 100 and val < recent_mean * 0.4:
        return True
    return False

def tag_anomalies_batch(records, spike_thresh, z_thresh, night_start, night_end):
    if not records:
        return records

    flows = np.array([r["flow_rate"] for r in records])
    active_mask = flows > 0
    z_flags = np.zeros(len(flows), dtype=bool)
    if active_mask.sum() > 10:
        active_z = np.abs(stats.zscore(flows[active_mask]))
        active_indices = np.where(active_mask)[0]
        z_flags[active_indices[active_z > z_thresh]] = True

    roll_mean = pd.Series(flows).rolling(10, min_periods=3).mean().values

    for i, rec in enumerate(records):
        flow = rec["flow_rate"]
        hour = rec["timestamp"].hour
        is_night = hour >= night_start or hour <= night_end

        prev_mean  = roll_mean[i - 1] if i > 0 else 0
        supply_cut = (flow < 5) and (prev_mean is not None) and (prev_mean > 100)
        sudden_drop = (
            prev_mean is not None
            and prev_mean > 100
            and flow < prev_mean * 0.4
            and flow > 5
        )

        anom = (
            flow < 0
            or flow > spike_thresh
            or (is_night and flow > 5)
            or z_flags[i]
            or supply_cut
            or sudden_drop
        )
        rec["is_anomaly"] = int(anom)

    return records

# ── FULL DETECTOR — analysis tabs ─────────────────────────────────────────────
def run_detectors(df, sensitivity, contamination, spike_threshold, night_start, night_end):
    df = df.copy()
    df["hour"]        = df["timestamp"].dt.hour
    df["dow"]         = df["timestamp"].dt.dayofweek
    df["date"]        = df["timestamp"].dt.date
    df["roll_mean_10"]= df["flow_rate_m3hr"].rolling(10, min_periods=1).mean()
    df["roll_std_10"] = df["flow_rate_m3hr"].rolling(10, min_periods=1).std().fillna(0)
    df["roll_mean_30"]= df["flow_rate_m3hr"].rolling(30, min_periods=1).mean()
    df["flow_diff"]   = df["flow_rate_m3hr"].diff().fillna(0)
    df["lag_1"]       = df["flow_rate_m3hr"].shift(1).fillna(0)
    df["deviation"]   = df["flow_rate_m3hr"] - df["roll_mean_30"]
    df["in_supply"]   = df["hour"].between(8, 10).astype(int)
    df["is_night"]    = ((df["hour"] >= night_start) | (df["hour"] <= night_end)).astype(int)

    df["anom_spike"]    = (df["flow_rate_m3hr"] > spike_threshold).astype(int)
    df["anom_negative"] = (df["flow_rate_m3hr"] < 0).astype(int)
    NIGHT_FLOW_LIMIT    = spike_threshold * 0.8
    df["anom_night"]    = ((df["is_night"]==1) & (df["flow_rate_m3hr"] > NIGHT_FLOW_LIMIT)).astype(int)

    active = df["flow_rate_m3hr"] > 0
    dfa    = df[active].copy()

    supply_hours = dfa[~((dfa["hour"] >= night_start) | (dfa["hour"] <= night_end))]
    if len(supply_hours) > 10:
        z_vals = np.abs(stats.zscore(supply_hours["flow_rate_m3hr"]))
        supply_hours = supply_hours.copy()
        supply_hours["anom_z"] = (z_vals > sensitivity).astype(int)
        dfa["anom_z"] = 0
        dfa.loc[supply_hours.index, "anom_z"] = supply_hours["anom_z"]
    else:
        dfa["anom_z"] = 0
        dfa["anom_z"] = (dfa["z"] > sensitivity).astype(int) if "z" in dfa.columns else 0

    df["anom_zscore"] = 0
    df.loc[dfa.index, "anom_zscore"] = dfa["anom_z"]

    if len(dfa) > 3:
        Q1, Q3 = dfa["flow_rate_m3hr"].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        IQR_FENCE = 2.5
        dfa["anom_iqr_f"] = ((dfa["flow_rate_m3hr"] < Q1 - IQR_FENCE * IQR) |
                              (dfa["flow_rate_m3hr"] > Q3 + IQR_FENCE * IQR)).astype(int)
    else:
        dfa["anom_iqr_f"] = 0
    df["anom_iqr"] = 0
    df.loc[dfa.index, "anom_iqr"] = dfa["anom_iqr_f"]

    FEATS = ["flow_rate_m3hr","roll_mean_10","roll_std_10","flow_diff","lag_1","deviation","hour","in_supply"]
    df["anom_iforest"] = 0; df["iforest_score"] = 0.0
    if len(dfa) >= 20:
        sc    = StandardScaler()
        Xsc   = sc.fit_transform(dfa[FEATS])
        ifor  = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
        preds = ifor.fit_predict(Xsc)
        dfa["anom_if"] = (preds == -1).astype(int)
        dfa["if_score"] = -ifor.decision_function(Xsc)
        df.loc[dfa.index, "anom_iforest"]   = dfa["anom_if"]
        df.loc[dfa.index, "iforest_score"]  = dfa["if_score"]

    df["anom_pca"] = 0; df["pca_score"] = 0.0
    if len(dfa) >= 20:
        mms = MinMaxScaler()
        Xn  = mms.fit_transform(dfa[FEATS])
        pca = PCA(n_components=min(3, len(FEATS)), random_state=42)
        Xp  = pca.fit_transform(Xn)
        Xr  = pca.inverse_transform(Xp)
        err = np.mean((Xn - Xr) ** 2, axis=1)
        thr = err.mean() + 3 * err.std()
        dfa["anom_pca_f"] = (err > thr).astype(int)
        dfa["pca_sc"]     = err
        df.loc[dfa.index, "anom_pca"]   = dfa["anom_pca_f"]
        df.loc[dfa.index, "pca_score"]  = dfa["pca_sc"]

    df["prev_flow"]       = df["flow_rate_m3hr"].shift(1).fillna(0)
    df["anom_supply_cut"] = ((df["flow_rate_m3hr"] < 5) & (df["prev_flow"] > 100)).astype(int)
    df["anom_sudden_drop"]= ((df["flow_rate_m3hr"] > 5) & (df["roll_mean_10"] > 100) &
                              (df["flow_rate_m3hr"] < df["roll_mean_10"] * 0.4)).astype(int)
    df["model_vote"]      = df["anom_zscore"] + df["anom_iqr"] + df["anom_iforest"] + df["anom_pca"]
    df["final_anomaly"]   = ((df["anom_negative"] == 1) | (df["anom_spike"] == 1) |
                             (df["anom_night"] == 1)    | (df["anom_supply_cut"] == 1) |
                             (df["anom_sudden_drop"] == 1) | (df["model_vote"] >= 3)).astype(int)
    return df

# ── FORECAST ──────────────────────────────────────────────────────────────────
def forecast(df, steps):
    active = df[df["flow_rate_m3hr"] > 0]["flow_rate_m3hr"].values
    if len(active) < 10: return None, None, None, None, None
    alpha = 0.3
    sm = [active[0]]
    for v in active[1:]: sm.append(alpha * v + (1 - alpha) * sm[-1])
    sm = np.array(sm)
    n = min(20, len(sm))
    trend = (sm[-1] - sm[-n]) / n
    fcast = np.array([sm[-1] + trend * i for i in range(1, steps + 1)])
    std   = np.std(active[-30:]) if len(active) >= 30 else np.std(active)
    diffs = df["timestamp"].diff().dt.total_seconds().median() / 60
    freq  = max(1, int(diffs)) if not np.isnan(diffs) else 3
    fts   = pd.date_range(start=df["timestamp"].iloc[-1] + pd.Timedelta(minutes=freq),
                           periods=steps, freq=f"{freq}min")
    return fcast, fcast - 1.96 * std, fcast + 1.96 * std, fts, sm


# ── PATTERN ANALYSIS HELPERS ──────────────────────────────────────────────────

def fetch_two_months(year: int = 2025) -> pd.DataFrame:
    """
    Pull Jan + Feb in weekly chunks to avoid silent API truncation.
    9 chunks cover the full two months with minimal round-trips.
    Tracks which weeks fail and warns the user before returning.
    """
    failed_chunks = []
    jan_start = datetime(year, 1, 1, 0, 0, 0)
    feb_end   = datetime(year, 2, 28, 23, 59, 59)

    all_records: list[dict] = []
    chunk_start = jan_start

    while chunk_start <= feb_end:
        chunk_end       = min(chunk_start + timedelta(days=7), feb_end)
        chunk_succeeded = False

        for path in HISTORY_API_PATHS:
            try:
                r = SESSION.get(
                    f"{VMC_BASE}{path}",
                    params={
                        "objectname": OBJECT_NAME,
                        "startTime":  chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                        "endTime":    chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    timeout=10,
                )
            except Exception as e:
                st.session_state.last_error = f"[{path}] {e}"
                continue

            if r.status_code != 200:
                continue
            if "<title>login</title>" in r.text.lower():
                st.session_state.token = None
                continue

            try:
                data = r.json()
            except Exception:
                continue

            records = _parse_batch_response(data, chunk_end)
            if len(records) > 1:
                all_records.extend(records)
                chunk_succeeded = True
                break

        if not chunk_succeeded:
            failed_chunks.append(
                f"{chunk_start.strftime('%b %d')}–{chunk_end.strftime('%b %d')}"
            )

        chunk_start = chunk_end + timedelta(seconds=1)

    if failed_chunks:
        st.warning(
            f"⚠️ These date ranges returned no data from the API: "
            f"**{', '.join(failed_chunks)}**\n\n"
            f"Your benchmark may be incomplete. "
            f"Check your connection or try fetching again."
        )

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.rename(columns={"flow_rate": "flow_rate_m3hr"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def normalize_daily_curve(day_df: pd.DataFrame) -> np.ndarray | None:
    """
    Collapse one day → 24-point hourly vector, min-max scaled to [0,1].
    Comparing shape rather than volume lets different-volume days match.
    Returns None if fewer than 6 hours have data (not enough shape to use).
    """
    day_df = day_df.copy()
    day_df["hour"] = day_df["timestamp"].dt.hour
    hourly = day_df.groupby("hour")["flow_rate_m3hr"].mean()
    curve  = hourly.reindex(range(24), fill_value=0.0).values.astype(float)

    if (curve > 0).sum() < 6:
        return None

    mn, mx = curve.min(), curve.max()
    if mx - mn < 1e-6:
        return None  # flat line — no shape info
    return (curve - mn) / (mx - mn)


def curve_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance between two 24-point normalised curves.
    We skip DTW here because curves are already clock-aligned — time-shifting
    would mask real supply-timing differences we actually care about.
    """
    return float(np.sqrt(np.sum((a - b) ** 2)))


def find_benchmark_pattern(df: pd.DataFrame, n_clusters: int = 6):
    """
    K-Means on daily shape curves → modal cluster centroid = benchmark.
    Returns benchmark curve, per-day summary df, raw curves dict,
    labels, all centroids, and modal cluster index.
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    all_curves: dict  = {}
    valid_dates: list = []

    for date, group in df.groupby("date"):
        curve = normalize_daily_curve(group)
        if curve is not None:
            all_curves[str(date)] = curve
            valid_dates.append(str(date))

    if len(valid_dates) < n_clusters:
        n_clusters = max(2, len(valid_dates) // 2)

    X  = np.array([all_curves[d] for d in valid_dates])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels    = km.fit_predict(X)
    centroids = km.cluster_centers_

    cluster_sizes = np.bincount(labels)
    modal_idx     = int(np.argmax(cluster_sizes))
    benchmark     = centroids[modal_idx]

    rows = []
    for i, date_str in enumerate(valid_dates):
        dist = curve_distance(all_curves[date_str], benchmark)
        # max Euclidean distance for 24-dim unit-range vectors ≈ 4.899
        similarity = max(0.0, 100.0 * (1.0 - dist / np.sqrt(24)))
        rows.append({
            "date":       date_str,
            "cluster":    int(labels[i]),
            "similarity": round(similarity, 1),
            "distance":   round(dist, 4),
            "is_benchmark_cluster": int(labels[i]) == modal_idx,
        })

    return benchmark, pd.DataFrame(rows), all_curves, labels, centroids, modal_idx


# ── BOX-METHOD HELPERS (supply-window matching, PDF §2.2–2.4) ─────────────────

def detect_supply_windows_df(day_df, threshold=1.0, min_duration_min=5):
    """Detect active supply windows from one day's flow. Returns list of window dicts."""
    df = day_df.copy().sort_values("timestamp").reset_index(drop=True)
    col = "flow_rate_m3hr" if "flow_rate_m3hr" in df.columns else "flow_rate"
    windows = []; in_window = False; start_idx = None

    for i, row in df.iterrows():
        if row[col] >= threshold and not in_window:
            in_window = True; start_idx = i
        elif row[col] < threshold and in_window:
            in_window = False
            wdf = df.iloc[start_idx:i]
            dur = (wdf["timestamp"].iloc[-1] - wdf["timestamp"].iloc[0]).total_seconds() / 60
            if dur >= min_duration_min:
                windows.append({
                    "start": wdf["timestamp"].iloc[0],
                    "end":   wdf["timestamp"].iloc[-1],
                    "duration": dur,
                    "peak":  wdf[col].max(),
                    "avg":   wdf[col].mean(),
                    "start_hour_frac": wdf["timestamp"].iloc[0].hour + wdf["timestamp"].iloc[0].minute / 60,
                    "end_hour_frac":   wdf["timestamp"].iloc[-1].hour + wdf["timestamp"].iloc[-1].minute / 60,
                })

    # catch window still open at end of day
    if in_window and start_idx is not None:
        wdf = df.iloc[start_idx:]
        dur = (wdf["timestamp"].iloc[-1] - wdf["timestamp"].iloc[0]).total_seconds() / 60
        if dur >= min_duration_min:
            windows.append({
                "start": wdf["timestamp"].iloc[0],
                "end":   wdf["timestamp"].iloc[-1],
                "duration": dur,
                "peak":  wdf[col].max(),
                "avg":   wdf[col].mean(),
                "start_hour_frac": wdf["timestamp"].iloc[0].hour + wdf["timestamp"].iloc[0].minute / 60,
                "end_hour_frac":   wdf["timestamp"].iloc[-1].hour + wdf["timestamp"].iloc[-1].minute / 60,
            })
    return windows


def build_benchmark_from_windows(all_windows, n_clusters=6):
    """
    Cluster all supply windows by start time (Ward linkage).
    Dominant cluster median values become the benchmark profile.
    """
    if not all_windows:
        return None, {}
    wdf    = pd.DataFrame(all_windows)
    starts = wdf["start_hour_frac"].values.reshape(-1, 1)
    try:
        Z      = linkage(starts, method="ward")
        labels = fcluster(Z, t=1.0, criterion="distance")
    except Exception:
        labels = np.ones(len(wdf), dtype=int)

    wdf["cluster"]   = labels
    cluster_sizes    = wdf.groupby("cluster").size()
    dominant_cluster = cluster_sizes.idxmax()
    dominant_wdf     = wdf[wdf["cluster"] == dominant_cluster]

    benchmark = {
        "start_hour": float(np.median(dominant_wdf["start_hour_frac"])),
        "end_hour":   float(np.median(dominant_wdf["end_hour_frac"])),
        "duration":   float(np.median(dominant_wdf["duration"])),
        "peak":       float(np.median(dominant_wdf["peak"])),
        "avg":        float(np.median(dominant_wdf["avg"])),
        "peak_std":   float(dominant_wdf["peak"].std() or 1),
        "avg_std":    float(dominant_wdf["avg"].std() or 1),
        "start_std":  float(dominant_wdf["start_hour_frac"].std() or 0.25),
        "samples":    len(dominant_wdf),
        "cluster_id": int(dominant_cluster),
        "all_clusters": {int(c): int(s) for c, s in cluster_sizes.items()},
    }
    return benchmark, wdf


def score_day_vs_benchmark(day_windows, benchmark, time_tol_min=30, flow_tol=0.20):
    """
    Score one day's windows against the benchmark (PDF §2.4).
    Returns (qos 0–100, anomaly list, matched window).
    Timing counts 50%, flow deviation counts 50%.
    """
    if not day_windows:
        return 0.0, ["No supply windows detected"], None
    if benchmark is None:
        return 50.0, ["Benchmark not available"], None

    bm_start_h = benchmark["start_hour"]
    bm_end_h   = benchmark["end_hour"]
    bm_dur     = benchmark["duration"]
    bm_peak    = benchmark["peak"]
    bm_avg     = benchmark["avg"]

    best_win = min(day_windows, key=lambda w: abs(w["start_hour_frac"] - bm_start_h))
    anomalies = []

    start_dev_min = abs(best_win["start_hour_frac"] - bm_start_h) * 60
    end_dev_min   = abs(best_win["end_hour_frac"]   - bm_end_h)   * 60
    dur_dev_min   = abs(best_win["duration"]         - bm_dur)

    if start_dev_min > time_tol_min:
        h, m = int(bm_start_h), int((bm_start_h % 1) * 60)
        anomalies.append(f"Start time off by {start_dev_min:.0f} min (benchmark: {h:02d}:{m:02d})")
    if end_dev_min > time_tol_min:
        h, m = int(bm_end_h), int((bm_end_h % 1) * 60)
        anomalies.append(f"End time off by {end_dev_min:.0f} min (benchmark: {h:02d}:{m:02d})")
    if dur_dev_min > time_tol_min:
        anomalies.append(f"Duration deviated by {dur_dev_min:.0f} min (benchmark: {bm_dur:.0f} min)")

    peak_dev = abs(best_win["peak"] - bm_peak) / max(bm_peak, 1e-6)
    avg_dev  = abs(best_win["avg"]  - bm_avg)  / max(bm_avg,  1e-6)
    if peak_dev > flow_tol:
        anomalies.append(f"Peak flow deviated by {peak_dev*100:.0f}% (benchmark: {bm_peak:.1f} m³/hr)")
    if avg_dev > flow_tol:
        anomalies.append(f"Avg flow deviated by {avg_dev*100:.0f}% (benchmark: {bm_avg:.1f} m³/hr)")

    t_score = max(0, 1 - (start_dev_min + end_dev_min) / (2 * time_tol_min * 3))
    f_score = max(0, 1 - (peak_dev + avg_dev) / (2 * flow_tol * 3))
    qos     = min(100, max(0, (t_score * 0.5 + f_score * 0.5) * 100))
    return round(qos, 1), anomalies, best_win


def find_benchmark_pattern_kmeans(df, n_clusters=6):
    """K-Means on normalised daily curves — thin wrapper used by pattern tab."""
    df = df.copy(); df["date"] = df["timestamp"].dt.date
    all_curves_km = {}; valid_dates = []
    for date, group in df.groupby("date"):
        curve = normalize_daily_curve(group)
        if curve is not None:
            all_curves_km[str(date)] = curve; valid_dates.append(str(date))
    if len(valid_dates) < 2:
        return None, pd.DataFrame(), all_curves_km, np.array([]), np.array([]), 0
    n_clusters = min(n_clusters, len(valid_dates))
    X  = np.array([all_curves_km[d] for d in valid_dates])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(X); centroids_km = km.cluster_centers_
    cluster_sizes = np.bincount(labels); modal_idx_km = int(np.argmax(cluster_sizes))
    benchmark_curve_km = centroids_km[modal_idx_km]
    rows = []
    for i, d in enumerate(valid_dates):
        dist = float(np.sqrt(np.sum((all_curves_km[d] - benchmark_curve_km) ** 2)))
        sim  = max(0.0, 100.0 * (1.0 - dist / np.sqrt(24)))
        rows.append({"date": d, "cluster": int(labels[i]),
                     "similarity": round(sim, 1), "distance": round(dist, 4),
                     "is_benchmark_cluster": int(labels[i]) == modal_idx_km})
    return benchmark_curve_km, pd.DataFrame(rows), all_curves_km, labels, centroids_km, modal_idx_km


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1rem;font-weight:600;color:#c8cde0'>💧 VMC Monitor</div>"
        "<div style='font-size:.72rem;color:#555d6e'>MJP-4231 · Vadodara</div>",
        unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Fetch mode</div>", unsafe_allow_html=True)
    fetch_mode = st.radio("Fetch mode", ["📦 Batch (single call)", "🔴 Live (per-second)"],
                          index=0, label_visibility="collapsed")
    batch_mode = fetch_mode.startswith("📦")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Clear live"):
            st.session_state.live_rows = []; st.session_state.anom_log = []
            st.session_state.batch_done = False; st.rerun()
    with col_b:
        if st.button("🗑 Clear DB"):
            db_clear(); st.rerun()

    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>", unsafe_allow_html=True)

    if batch_mode:
        st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Batch settings</div>", unsafe_allow_html=True)
        batch_hours = st.slider("Fetch window (hours)", 1, 24, 24)
    else:
        st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Live settings</div>", unsafe_allow_html=True)
        poll_interval = st.slider("Poll interval (s)", 1, 30, 1)
        window_mins   = st.slider("Chart window (min)", 1, 60, 5)

    spike_threshold = st.number_input("Spike threshold (m³/hr)", 100, 2000, 600, 50)
    z_sensitivity   = st.slider("Z-score sensitivity", 1.5, 5.0, 3.0, 0.1)

    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Analysis settings</div>", unsafe_allow_html=True)
    contamination  = st.slider("IF contamination", 0.01, 0.15, 0.05, 0.01)
    night_start    = st.slider("Night start (hr)", 18, 23, 23)
    night_end      = st.slider("Night end (hr)", 0, 8, 5)
    forecast_steps = st.slider("Forecast horizon", 10, 60, 30)
    db_hours       = st.slider("DB history (hrs)", 1, 168, 24)

    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Pattern analysis</div>", unsafe_allow_html=True)
    pattern_year  = st.number_input("Jan–Feb year", 2023, 2026, 2025, 1)
    pattern_k     = st.slider("K-Means clusters (k)", 2, 10, 6, 1)
    sim_threshold = st.slider("Match threshold (%)", 50, 95, 75, 5)
    time_tol_min  = st.slider("Timing tolerance (min)", 15, 60, 30, 5)
    flow_tol_pct  = st.slider("Flow tolerance (%)", 10, 40, 20, 5)

    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>CSV upload (optional)</div>", unsafe_allow_html=True)
    file_fr  = st.file_uploader("Flow rate CSV", type="csv")
    file_tf  = st.file_uploader("Cumulative volume CSV", type="csv")
    file_pdf = st.file_uploader("Flow data PDF", type="pdf")

    n_db = db_count()
    st.markdown(f"<div style='font-size:.7rem;color:#555d6e;margin-top:8px'>DB: {n_db:,} readings stored</div>", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([5, 1])
with hc1:
    st.markdown(
        "<h1 style='font-size:1.35rem;font-weight:600;margin:0;color:#c8cde0'>💧 VMC Water Flow — Live + Analysis</h1>"
        "<p style='color:#555d6e;font-size:.75rem;margin:2px 0 12px'>MJP-4231 · Vadodara Municipal Corporation</p>",
        unsafe_allow_html=True)
with hc2:
    if batch_mode:
        st.markdown("<br><span class='batch-pill'>📦 BATCH</span>", unsafe_allow_html=True)
    else:
        st.markdown("<br><span class='live-pill'><span class='live-dot'></span>LIVE</span>", unsafe_allow_html=True)

# ── DATA SOURCE ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(fr_bytes, tf_bytes):
    fr = pd.read_csv(io.BytesIO(fr_bytes), parse_dates=["DateTime"])
    tf = pd.read_csv(io.BytesIO(tf_bytes), parse_dates=["DateTime"])
    fr.columns = ["timestamp", "flow_rate_m3hr"]
    tf.columns = ["timestamp", "cumulative_flow_m3"]
    df = pd.merge(fr, tf, on="timestamp", how="inner")
    return df.sort_values("timestamp").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    buf = io.BytesIO(pdf_bytes)
    if _PDFPLUMBER:
        import pdfplumber
        all_tables: list[pd.DataFrame] = []
        with pdfplumber.open(buf) as pdf:
            for page in pdf.pages:
                for tbl in page.extract_tables():
                    if not tbl or len(tbl) < 2:
                        continue
                    try:
                        df_t = pd.DataFrame(tbl[1:], columns=tbl[0])
                        all_tables.append(df_t)
                    except Exception:
                        continue
        for df_t in all_tables:
            result = _coerce_pdf_table(df_t)
            if result is not None:
                return result
    buf.seek(0)
    if _PYPDF:
        from pypdf import PdfReader as _PR
        reader = _PR(buf)
        lines  = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            lines.extend(txt.splitlines())
        result = _parse_pdf_text_lines(lines)
        if result is not None:
            return result
    raise ValueError("Could not extract a flow-rate table from this PDF.")

def _coerce_pdf_table(df_t):
    if df_t is None or df_t.empty: return None
    df_t.columns = [str(c).strip() if c else f"col_{i}" for i, c in enumerate(df_t.columns)]
    ts_col = next((c for c in df_t.columns
                   if any(kw in c.lower() for kw in ["datetime","date","time","timestamp","ts"])), None)
    flow_col = None
    for col in df_t.columns:
        if col == ts_col: continue
        if any(kw in col.lower() for kw in ["flow","rate","value","m3","reading","val"]):
            flow_col = col; break
    if ts_col is None: ts_col = df_t.columns[0]
    if flow_col is None:
        for col in df_t.columns:
            if col == ts_col: continue
            try: pd.to_numeric(df_t[col].dropna(), errors="raise"); flow_col = col; break
            except: continue
    if flow_col is None: return None
    try:
        out = pd.DataFrame()
        out["timestamp"]     = pd.to_datetime(df_t[ts_col], infer_datetime_format=True, errors="coerce")
        out["flow_rate_m3hr"]= pd.to_numeric(df_t[flow_col], errors="coerce")
        out = out.dropna(subset=["timestamp","flow_rate_m3hr"])
        if len(out) < 2: return None
        return out.sort_values("timestamp").reset_index(drop=True)
    except: return None

def _parse_pdf_text_lines(lines):
    import re
    date_pat = re.compile(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}[T ]\d{1,2}:\d{2}(?::\d{2})?)[\s,;|]+([\d.]+)")
    records  = []
    for line in lines:
        m = date_pat.search(line)
        if m:
            try:
                ts   = datetime.fromisoformat(m.group(1).replace("/", "-"))
                flow = float(m.group(2))
                records.append({"timestamp": ts, "flow_rate_m3hr": flow})
            except: continue
    if len(records) < 2: return None
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

def get_analysis_df():
    db_df  = db_load(db_hours)
    frames = []
    if file_fr and file_tf:
        try: csv_df = load_csv(file_fr.read(), file_tf.read()); frames.append(csv_df)
        except Exception as e: st.warning(f"CSV load error: {e}")
    if file_pdf:
        if not (_PDFPLUMBER or _PYPDF):
            st.warning("PDF parsing requires pdfplumber or pypdf.")
        else:
            try:
                pdf_df = load_pdf(file_pdf.read()); frames.append(pdf_df)
                st.sidebar.success(f"PDF: {len(pdf_df):,} rows loaded")
            except ValueError as e: st.warning(str(e))
            except Exception as e: st.warning(f"PDF load error: {e}")
    if frames:
        if not db_df.empty: frames.append(db_df)
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.sort_values("timestamp").drop_duplicates("timestamp")
        return merged.reset_index(drop=True)
    return db_df

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_live, tab_eda, tab_anom, tab_fcast, tab_data, tab_pattern, tab_qos = st.tabs([
    "📦 Live / Batch Feed", "📊 EDA", "🔍 Anomaly Detection",
    "📈 Forecast", "📋 Data Table", "📐 Pattern Analysis", "📉 QoS Trend"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE / BATCH FEED
# ═════════════════════════════════════════════════════════════════════════════
with tab_live:
    rows     = st.session_state.live_rows
    anom_log = st.session_state.anom_log

    if rows:
        cur      = rows[-1]["flow_rate_m3hr"]
        hist     = [r["flow_rate_m3hr"] for r in rows]
        avg_f    = np.mean(hist); max_f = np.max(hist)
        is_anom_now = is_anomaly_live(cur, hist[:-1], spike_threshold, z_sensitivity)
        cur_cls  = "danger" if is_anom_now else ""
    else:
        cur = avg_f = max_f = None; cur_cls = ""

    def mc(label, value, cls=""):
        val_s = f"{value:.1f}" if value is not None else "—"
        return (f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                f"<div class='metric-value {cls}'>{val_s}</div></div>")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(mc("Current (m³/hr)", cur, cur_cls), unsafe_allow_html=True)
    c2.markdown(mc("Average (m³/hr)", avg_f), unsafe_allow_html=True)
    c3.markdown(mc("Peak (m³/hr)", max_f), unsafe_allow_html=True)
    c4.markdown(mc("Readings", float(len(rows)) if rows else None), unsafe_allow_html=True)
    c5.markdown(
        f"<div class='metric-card'><div class='metric-label'>Anomalies</div>"
        f"<div class='metric-value {'danger' if anom_log else ''}'>{len(anom_log)}</div></div>",
        unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    chart_ph  = st.empty()
    status_ph = st.empty()

    def draw_live(rows, wsecs, spike, z):
        if not rows: return
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").tail(wsecs)
        hist  = df["flow_rate_m3hr"].tolist()
        flags = [is_anomaly_live(v, hist[:i], spike, z) for i, v in enumerate(hist)]
        df["is_anom"] = flags
        fig, ax = plt.subplots(figsize=(12, 3.6))
        ax.plot(df["timestamp"], df["flow_rate_m3hr"], color="#4a90d9", linewidth=1.3, alpha=.95, label="Flow rate")
        ax.fill_between(df["timestamp"], df["flow_rate_m3hr"], alpha=.07, color="#4a90d9")
        anoms = df[df["is_anom"]]
        if not anoms.empty:
            for _, row_ in anoms.iterrows():
                ax.annotate("", xy=(row_["timestamp"], row_["flow_rate_m3hr"]+5),
                    xytext=(row_["timestamp"], row_["flow_rate_m3hr"]+55),
                    arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.4))
            ax.scatter(anoms["timestamp"], anoms["flow_rate_m3hr"],
                color="#ff6b6b", s=40, zorder=7, label=f"Anomaly ({len(anoms)})")
        ax.axhline(spike, color="#ffa94d", lw=.8, linestyle="--", alpha=.7, label=f"Spike limit ({spike})")
        ax.set_ylabel("m³/hr", fontsize=8, color="#555d6e")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=.4, lw=.5)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S" if wsecs <= 600 else "%H:%M"))
        leg = ax.legend(fontsize=7.5, loc="upper right", framealpha=.85, edgecolor="#2a2d3a")
        for t in leg.get_texts(): t.set_color("#9aa0b0")
        fig.autofmt_xdate(rotation=20)
        fig.tight_layout(pad=.5)
        chart_ph.pyplot(fig); plt.close(fig)

    bl, br = st.columns(2, gap="small")
    with bl:
        log_items = ""
        for e in reversed(anom_log[-20:]):
            log_items += (f"<div class='log-row'><span class='log-time'>{e['time']}</span>"
                          f"<span class='log-badge'>{e['val']:.1f} m³/hr</span></div>")
        if not log_items:
            log_items = "<div style='color:#555d6e;font-size:.8rem;padding:12px 0'>No anomalies yet</div>"
        st.markdown(f"<div class='log-card'><div class='log-title'>Anomaly log</div>{log_items}</div>",
                    unsafe_allow_html=True)

    with br:
        mode_label = "Batch" if batch_mode else "Live poll"
        st.markdown(
            f"<div class='log-card'><div class='log-title'>Session info</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>"
            f"<div><div class='metric-label'>Mode</div><div style='font-size:1.1rem;font-weight:500;color:#c8cde0'>{mode_label}</div></div>"
            f"<div><div class='metric-label'>Spike limit</div><div style='font-size:1.3rem;font-weight:500;color:#c8cde0'>{spike_threshold}</div></div>"
            f"<div><div class='metric-label'>Readings</div><div style='font-size:1.1rem;font-weight:500;color:#c8cde0'>{len(rows)}</div></div>"
            f"<div><div class='metric-label'>DB total</div><div style='font-size:1.1rem;font-weight:500;color:#c8cde0'>{db_count():,}</div></div>"
            f"</div></div>", unsafe_allow_html=True)

    debug_ph = st.empty()

    if batch_mode:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        btn_col, info_col = st.columns([1, 3])
        with btn_col:
            do_fetch = st.button("📦 Fetch batch now", type="primary", width="stretch")
        with info_col:
            if st.session_state.batch_done:
                st.markdown(
                    f"<div style='padding:8px 12px;background:#1a1d27;border:1px solid #2a2d3a;"
                    f"border-radius:8px;font-size:.8rem;color:#4ecdc4'>"
                    f"✅ Last batch: <b>{st.session_state.batch_count:,}</b> readings loaded "
                    f"for the past <b>{batch_hours}h</b>. "
                    f"Go to EDA / Anomaly / Forecast tabs to analyse.</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='padding:8px 12px;background:#1a1d27;border:1px solid #2a2d3a;"
                    f"border-radius:8px;font-size:.8rem;color:#7a8196'>"
                    f"Click <b>Fetch batch now</b> to pull the last <b>{batch_hours}h</b> of data "
                    f"in a single API call, store to DB, and render the chart.</div>",
                    unsafe_allow_html=True)

        if do_fetch:
            try_login()
            with st.spinner(f"Fetching {batch_hours}h batch from VMC API…"):
                records = fetch_batch(batch_hours)

            with debug_ph.expander("🔍 Debug", expanded=(not records)):
                if st.session_state.last_error:
                    st.error(f"Last error: {st.session_state.last_error}")
                if st.session_state.field_map:
                    st.json(st.session_state.field_map)
                st.code(st.session_state.last_raw or "No response", language="json")

            if not records:
                status_ph.error("❌ No data returned — expand Debug above")
            else:
                records = tag_anomalies_batch(records, spike_threshold, z_sensitivity, night_start, night_end)
                db_rows = [
                    (r["timestamp"].isoformat(), r["flow_rate"], r.get("is_anomaly", 0))
                    for r in records
                ]
                db_insert_batch(db_rows)
                st.session_state.live_rows = [
                    {"timestamp": r["timestamp"], "flow_rate_m3hr": r["flow_rate"]}
                    for r in records
                ]
                st.session_state.anom_log = [
                    {"time": r["timestamp"].strftime("%H:%M:%S"), "val": r["flow_rate"]}
                    for r in records if r.get("is_anomaly")
                ]
                st.session_state.batch_done  = True
                st.session_state.batch_count = len(records)
                n_anom = sum(1 for r in records if r.get("is_anomaly"))
                status_ph.success(
                    f"✅ Loaded {len(records):,} readings ({batch_hours}h) · "
                    f"{n_anom} anomalies · stored to DB")
                st.rerun()

        if st.session_state.live_rows:
            draw_live(st.session_state.live_rows,
                      len(st.session_state.live_rows),
                      spike_threshold, z_sensitivity)
        else:
            if not do_fetch:
                st.info("Click **Fetch batch now** to load data.")

    else:
        run_live = True
        if run_live:
            try_login()
            reading = fetch_reading()

            with debug_ph.expander("🔍 Debug", expanded=(reading is None)):
                if st.session_state.last_error:
                    st.error(f"Last error: {st.session_state.last_error}")
                if st.session_state.field_map:
                    st.json(st.session_state.field_map)
                st.code(st.session_state.last_raw or "No response", language="json")

            if reading is None:
                status_ph.error("❌ No reading — expand Debug above")
            else:
                st.session_state.live_rows.append(reading)
                max_rows = max(600, window_mins * 60 * 2)
                st.session_state.live_rows = st.session_state.live_rows[-max_rows:]
                hist = [r["flow_rate_m3hr"] for r in st.session_state.live_rows]
                anom = is_anomaly_live(reading["flow_rate_m3hr"], hist[:-1], spike_threshold, z_sensitivity)
                if anom:
                    st.session_state.anom_log.append({
                        "time": reading["timestamp"].strftime("%H:%M:%S"),
                        "val":  reading["flow_rate_m3hr"]
                    })
                    status_ph.warning(f"⚠️ Anomaly {reading['timestamp'].strftime('%H:%M:%S')} — {reading['flow_rate_m3hr']:.1f} m³/hr")
                else:
                    status_ph.success(f"✅ {reading['timestamp'].strftime('%H:%M:%S')} — {reading['flow_rate_m3hr']:.1f} m³/hr")
                db_insert(reading["timestamp"], reading["flow_rate_m3hr"], int(anom))

            draw_live(st.session_state.live_rows, window_mins * 60, spike_threshold, z_sensitivity)
            time.sleep(poll_interval)
            st.rerun()
        else:
            draw_live(st.session_state.live_rows, window_mins * 60, spike_threshold, z_sensitivity)
            if not st.session_state.live_rows:
                st.info("Switch to **Live (per-second)** mode in the sidebar to start streaming.")

# ── SHARED DATA for tabs 2-5 ──────────────────────────────────────────────────
ana_df_raw = get_analysis_df()

def analysis_ready():
    return ana_df_raw is not None and not ana_df_raw.empty and len(ana_df_raw) >= 5

def get_processed():
    return run_detectors(ana_df_raw.copy(), z_sensitivity, contamination,
                         spike_threshold, night_start, night_end)

# ── QoS DB LOADERS ────────────────────────────────────────────────────────────
def load_qos_history() -> pd.DataFrame:
    """Load QoS scores written by vmc_worker.py — returns empty df if none yet."""
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM qos_scores ORDER BY date ASC", con)
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df

def load_benchmark_snapshots() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT * FROM benchmark_snapshot ORDER BY saved_at DESC LIMIT 30", con)
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═════════════════════════════════════════════════════════════════════════════
with tab_eda:
    if not analysis_ready():
        st.info("Fetch a batch or start live feed to collect data, or upload CSV files in the sidebar.")
        st.stop()

    df = ana_df_raw.copy()
    df["hour"]         = df["timestamp"].dt.hour
    df["date"]         = df["timestamp"].dt.date
    df["roll_mean_10"] = df["flow_rate_m3hr"].rolling(10, min_periods=1).mean()
    df["roll_std_10"]  = df["flow_rate_m3hr"].rolling(10, min_periods=1).std().fillna(0)

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(df["timestamp"], df["flow_rate_m3hr"], color="#4a90d9", lw=.7, alpha=.85)
    ax.fill_between(df["timestamp"], df["flow_rate_m3hr"], alpha=.06, color="#4a90d9")
    ax.axhline(0, color="#ff6b6b", lw=.6, linestyle="--", alpha=.4)
    ax.set_ylabel("Flow rate (m³/hr)"); ax.set_title("Full flow rate time series")
    ax.grid(True, alpha=.3); ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    ca, cb = st.columns(2)
    with ca:
        hourly = df[df["flow_rate_m3hr"] > 0].groupby("hour")["flow_rate_m3hr"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        colors = ["#ffa94d" if (h >= night_start or h <= night_end)
                  else "#3fb950" if 8 <= h <= 10 else "#4a90d9"
                  for h in hourly.index]
        ax.bar(hourly.index, hourly.values, color=colors, width=.7, zorder=3)
        ax.set_xlabel("Hour"); ax.set_ylabel("Avg m³/hr"); ax.set_title("Average flow by hour")
        ax.set_xticks(range(0, 24, 2)); ax.grid(True, alpha=.3, axis="y")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with cb:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.hist(df[df["flow_rate_m3hr"] > 0]["flow_rate_m3hr"], bins=50,
                color="#4a90d9", alpha=.8, density=True, label="Normal")
        ax.set_xlabel("Flow rate (m³/hr)"); ax.set_ylabel("Density")
        ax.set_title("Flow distribution"); ax.grid(True, alpha=.3)
        ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(df["timestamp"], df["flow_rate_m3hr"], color="#4a90d9", lw=.5, alpha=.5, label="Flow")
    ax.plot(df["timestamp"], df["roll_mean_10"], color="#c8cde0", lw=1.0, label="Rolling mean (10)")
    ax.fill_between(df["timestamp"],
                    df["roll_mean_10"] - 2 * df["roll_std_10"],
                    df["roll_mean_10"] + 2 * df["roll_std_10"],
                    alpha=.12, color="#4a90d9", label="±2σ band")
    ax.set_ylabel("Flow rate (m³/hr)"); ax.set_title("Rolling mean ± 2σ confidence band")
    ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=.3)
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    if "cumulative_flow_m3" in df.columns:
        daily_max = df.groupby("date")["cumulative_flow_m3"].max()
        typical   = daily_max.median()
        fig, ax   = plt.subplots(figsize=(13, 3.5))
        clrs = ["#ff6b6b" if v < typical * .7 else "#4a90d9" for v in daily_max.values]
        ax.bar(range(len(daily_max)), daily_max.values, color=clrs, width=.72, zorder=3)
        ax.axhline(typical, color="#ffa94d", lw=1.2, linestyle="--", label=f"Median {typical:.0f} m³")
        ax.set_xticks(range(len(daily_max)))
        ax.set_xticklabels([str(d)[5:] for d in daily_max.index], rotation=45, fontsize=7)
        ax.set_ylabel("m³ / day"); ax.set_title("Daily cumulative supply")
        ax.legend(fontsize=8); ax.grid(True, alpha=.3, axis="y")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        low = (daily_max < typical * .7).sum()
        if low: st.warning(f"⚠️ {low} low-supply day(s) detected (below 70% of median {typical:.0f} m³)")
        else:   st.success("✅ No low-supply days detected")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════
with tab_anom:
    if not analysis_ready():
        st.info("Fetch a batch or start live feed to collect data, or upload CSV files in the sidebar.")
        st.stop()

    with st.spinner("Running 4 detection models…"):
        df = get_processed()

    total = int(df["final_anomaly"].sum())
    st.markdown(f"<div style='font-size:.8rem;color:#555d6e;margin-bottom:12px'>{len(df):,} readings analysed · {total} anomalies found</div>", unsafe_allow_html=True)

    mcounts = {
        "Z-score":         int(df["anom_zscore"].sum()),
        "IQR":             int(df["anom_iqr"].sum()),
        "Isolation Forest":int(df["anom_iforest"].sum()),
        "PCA Autoencoder": int(df["anom_pca"].sum()),
        "Final (3+ / rule)":int(df["final_anomaly"].sum()),
    }
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bclrs   = ["#9b8ec4","#9b8ec4","#c4736b","#6bab7a","#4a90d9"]
    bars    = ax.bar(list(mcounts.keys()), list(mcounts.values()), color=bclrs, width=.55, zorder=3)
    for bar, v in zip(bars, mcounts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .3, str(v),
                ha="center", va="bottom", fontsize=9, color="#c8cde0")
    ax.set_ylabel("Count"); ax.set_title("Anomalies per model")
    ax.grid(True, alpha=.3, axis="y"); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(df["timestamp"], df["flow_rate_m3hr"], color="#4a90d9", lw=.7, alpha=.7, label="Flow")
    fa = df[df["final_anomaly"] == 1]
    ax.scatter(fa["timestamp"], fa["flow_rate_m3hr"], color="#ff6b6b", s=30, zorder=6,
               marker="^", label=f"Final anomaly ({len(fa)})")
    ax.axhline(spike_threshold, color="#ffa94d", lw=.8, linestyle="--", alpha=.6, label="Spike limit")
    ax.set_ylabel("m³/hr"); ax.set_title("Final anomaly flags (3+ models / rule-based)")
    ax.legend(fontsize=8); ax.grid(True, alpha=.3); ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("<div style='font-size:.85rem;color:#c8cde0;margin:12px 0 6px'>Model-by-model overlay</div>", unsafe_allow_html=True)
    model_cols   = ["anom_zscore","anom_iqr","anom_iforest","anom_pca"]
    model_labels = ["Z-score","IQR","Isolation Forest","PCA Autoencoder"]
    model_colors = ["#ffa94d","#ff6b6b","#8b949e","#3fb950"]
    fig, axes    = plt.subplots(4, 1, figsize=(13, 11), sharex=True, gridspec_kw={"hspace":.45})
    for ax, col, lbl, clr in zip(axes, model_cols, model_labels, model_colors):
        ax.plot(df["timestamp"], df["flow_rate_m3hr"], color="#4a90d9", lw=.4, alpha=.5)
        fl = df[df[col] == 1]
        ax.scatter(fl["timestamp"], fl["flow_rate_m3hr"], color=clr, s=18, zorder=6, label=f"{lbl} ({len(fl)})")
        ax.set_ylabel("m³/hr", fontsize=8); ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=.2); ax.spines[["top","right"]].set_visible(False)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    if "iforest_score" in df.columns and df["iforest_score"].sum() > 0:
        df["day_label"] = df["timestamp"].dt.strftime("%d %b")
        df["hour_col"]  = df["timestamp"].dt.hour
        pivot = df.pivot_table(index="day_label", columns="hour_col",
                               values="iforest_score", aggfunc="max").fillna(0)
        fig, ax = plt.subplots(figsize=(13, max(4, len(pivot) * 0.5 + 2)))
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=.15, linecolor="#0f1117",
                    cbar_kws={"label":"IF score"}, annot=False)
        ax.set_title("Anomaly score heatmap — day × hour")
        ax.set_xlabel("Hour"); ax.set_ylabel("")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("<div style='font-size:.85rem;color:#c8cde0;margin:12px 0 6px'>Anomaly events</div>", unsafe_allow_html=True)
    dcols = [c for c in ["timestamp","flow_rate_m3hr","roll_mean_10","deviation",
                          "anom_zscore","anom_iqr","anom_iforest","anom_pca","model_vote"] if c in df.columns]
    st.dataframe(df[df["final_anomaly"] == 1][dcols].reset_index(drop=True), width="content", height=280)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — FORECAST
# ═════════════════════════════════════════════════════════════════════════════
with tab_fcast:
    if not analysis_ready():
        st.info("Fetch a batch or start live feed to collect data, or upload CSV files in the sidebar.")
        st.stop()

    with st.spinner("Building forecast…"):
        df = get_processed()
        fc, lo, hi, fts, sm = forecast(df, forecast_steps)

    if fc is None:
        st.warning("Not enough active readings for forecast (need ≥10).")
        st.stop()

    freq = max(1, int(df["timestamp"].diff().dt.total_seconds().median() / 60))
    st.markdown(f"<div style='font-size:.8rem;color:#555d6e;margin-bottom:12px'>Horizon: {forecast_steps} readings × {freq} min = {forecast_steps*freq} min ahead</div>", unsafe_allow_html=True)

    lookback    = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=2)]
    active_look = lookback[lookback["flow_rate_m3hr"] > 0]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(lookback["timestamp"], lookback["flow_rate_m3hr"], color="#4a90d9", lw=1.0, label="Actual", alpha=.85)
    n_sm = len(active_look)
    if n_sm > 0:
        ax.plot(active_look["timestamp"], sm[-n_sm:], color="#3fb950", lw=1.0,
                linestyle="--", label="Fitted", alpha=.8)
    ax.plot(fts, fc, color="#ffa94d", lw=1.5, label="Forecast", zorder=5)
    ax.fill_between(fts, lo, hi, alpha=.18, color="#ffa94d", label="95% CI")
    fa_look = lookback[lookback["final_anomaly"] == 1]
    ax.scatter(fa_look["timestamp"], fa_look["flow_rate_m3hr"], color="#ff6b6b", s=28, zorder=7, label="Anomaly")
    ax.axvline(df["timestamp"].max(), color="#555d6e", lw=.8, linestyle=":", label="Now")
    ax.set_ylabel("Flow rate (m³/hr)"); ax.set_title("Last 48 hrs + Forecast")
    ax.legend(fontsize=8, ncol=5); ax.grid(True, alpha=.25)
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=30); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    if n_sm > 0:
        act_v  = active_look["flow_rate_m3hr"].values
        fit_v  = sm[-n_sm:]
        resids = act_v - fit_v
        rthresh= np.std(resids) * 2
        fig, ax= plt.subplots(figsize=(13, 2.8))
        rclrs  = ["#ff6b6b" if abs(r) > rthresh else "#4a90d9" for r in resids]
        ax.bar(active_look["timestamp"].values, resids, color=rclrs,
               width=pd.Timedelta(minutes=freq * .8))
        ax.axhline(0, color="#555d6e", lw=.6)
        ax.axhline(rthresh,  color="#ff6b6b", lw=.7, linestyle="--", alpha=.6)
        ax.axhline(-rthresh, color="#ff6b6b", lw=.7, linestyle="--", alpha=.6)
        ax.set_ylabel("Residual (m³/hr)"); ax.set_title("Forecast residuals (red = large error)")
        ax.grid(True, alpha=.2); ax.spines[["top","right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        fig.autofmt_xdate(rotation=30); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    fcast_df = pd.DataFrame({"timestamp": fts, "forecast_m3hr": np.round(fc, 2),
                              "upper": np.round(hi, 2), "lower": np.round(lo, 2)})
    st.dataframe(fcast_df, width="stretch", height=260)
    st.download_button("⬇️ Download forecast CSV",
                       data=fcast_df.to_csv(index=False).encode(),
                       file_name="vmc_forecast.csv", mime="text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA TABLE
# ═════════════════════════════════════════════════════════════════════════════
with tab_data:
    if not analysis_ready():
        st.info("Fetch a batch or start live feed to collect data, or upload CSV files in the sidebar.")
        st.stop()

    with st.spinner("Processing…"):
        df = get_processed()

    tc1, tc2 = st.columns([3, 1])
    with tc1:
        only_anom = st.checkbox("Show anomaly rows only", value=False)

    dcols = [c for c in ["timestamp","flow_rate_m3hr","roll_mean_10","deviation",
                          "anom_zscore","anom_iqr","anom_iforest","anom_pca",
                          "anom_negative","final_anomaly"] if c in df.columns]
    tbl = df[dcols].reset_index(drop=True)
    if only_anom: tbl = tbl[tbl["final_anomaly"] == 1].reset_index(drop=True)
    st.dataframe(tbl, width="stretch", height=460)
    st.download_button("⬇️ Download results CSV",
                       data=df[dcols].to_csv(index=False).encode(),
                       file_name="vmc_full_results.csv", mime="text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — PATTERN ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_pattern:

    st.markdown(
        "<div style='font-size:.8rem;color:#555d6e;margin-bottom:14px'>"
        "Fetches Jan + Feb from VMC API → overlays ALL daily curves together in one graph "
        "(multi-day overlay) → computes median profile as benchmark → identifies best "
        "repeated shape via K-Means → scores every day against the benchmark."
        "</div>",
        unsafe_allow_html=True,
    )

    p_col1, p_col2 = st.columns([1, 3])
    with p_col1:
        do_pattern = st.button(f"📥 Fetch Jan–Feb {pattern_year}", type="primary", width="stretch")
    with p_col2:
        if st.session_state.pattern_df is not None:
            pdf = st.session_state.pattern_df
            st.markdown(
                f"<div style='padding:8px 12px;background:#1a1d27;border:1px solid #2a2d3a;"
                f"border-radius:8px;font-size:.8rem;color:#4ecdc4'>"
                f"✅ Loaded: <b>{len(pdf):,}</b> readings across "
                f"<b>{pdf['timestamp'].dt.date.nunique()}</b> days "
                f"(Jan–Feb {pattern_year}). Benchmark set. Scroll down for charts.</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='padding:8px 12px;background:#1a1d27;border:1px solid #2a2d3a;"
                "border-radius:8px;font-size:.8rem;color:#7a8196'>"
                "Click the button to fetch 2 months of data and run pattern analysis.</div>",
                unsafe_allow_html=True)

    pat_debug = st.empty()

    if do_pattern:
        try_login()
        with st.spinner(f"Fetching Jan + Feb {pattern_year} in weekly chunks…"):
            cached = load_pattern_cache()
            if cached is not None:
                pat_df = cached
                st.info(f"✅ Loaded from local cache ({len(pat_df):,} rows). "
                        f"Cache is less than {CACHE_MAX_AGE_DAYS} days old.")
            else:
                pat_df = fetch_two_months(year=int(pattern_year))
                if not pat_df.empty:
                    save_pattern_cache(pat_df)
                    st.success("Data fetched from API and saved to local cache.")

        if pat_df.empty:
            st.warning("⚠️ No data from VMC API for that period — falling back to DB data.")
            pat_df = db_load(hours_back=168)
            if "flow_rate_m3hr" not in pat_df.columns and "flow_rate" in pat_df.columns:
                pat_df = pat_df.rename(columns={"flow_rate": "flow_rate_m3hr"})

        if pat_df.empty:
            st.error("❌ No data available at all. Fetch a batch first from the Live/Batch tab.")
            with pat_debug.expander("Debug", expanded=True):
                st.code(st.session_state.last_raw or "No response")
        else:
            st.session_state.pattern_df = pat_df
            with st.spinner("Computing benchmark pattern…"):
                bench, curves_df, all_curves, labels, centroids, modal_idx = \
                    find_benchmark_pattern(pat_df, n_clusters=int(pattern_k))
                # also build box-method benchmark for supply-window scoring
                all_wins   = []
                pat_df_cp  = pat_df.copy()
                pat_df_cp["date_"] = pat_df_cp["timestamp"].dt.date
                for date_, group in pat_df_cp.groupby("date_"):
                    wins = detect_supply_windows_df(group)
                    for w in wins: w["date"] = str(date_)
                    all_wins.extend(wins)
                benchmark_box, _ = build_benchmark_from_windows(all_wins, n_clusters=int(pattern_k))

            st.session_state.benchmark_curve   = bench
            st.session_state.benchmark_windows = benchmark_box
            st.session_state.curves_df         = curves_df
            st.session_state.all_curves        = all_curves
            st.session_state.centroids         = centroids
            st.session_state.modal_idx         = modal_idx
            st.rerun()

    if st.session_state.pattern_df is None:
        st.info("Press **Fetch Jan–Feb** to start.")
        st.stop()

    pat_df    = st.session_state.pattern_df
    bench     = st.session_state.benchmark_curve
    bench_box = st.session_state.benchmark_windows
    curves_df = st.session_state.curves_df
    all_curves= st.session_state.all_curves
    centroids = st.session_state.centroids
    modal_idx = st.session_state.modal_idx

    if bench is None or all_curves is None or curves_df is None or centroids is None:
        st.warning("Benchmark not computed yet — press the fetch button.")
        st.stop()

    bm_start_str = (f"{int(bench_box['start_hour']):02d}:"
                    f"{int((bench_box['start_hour']%1)*60):02d}") if bench_box else "N/A"
    bm_end_str   = (f"{int(bench_box['end_hour']):02d}:"
                    f"{int((bench_box['end_hour']%1)*60):02d}") if bench_box else "N/A"

    hours_axis = np.arange(24)

    # ① Jan + Feb all-day overlay — PDF Figure 1 style
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:16px 0 6px'>"
        "① Jan + Feb — all days overlaid with Benchmark Median</div>",
        unsafe_allow_html=True)

    jan_curves, feb_curves = [], []
    for date_str, curve in all_curves.items():
        month = int(date_str[5:7])
        if month == 1: jan_curves.append(curve)
        elif month == 2: feb_curves.append(curve)

    all_day_curves = jan_curves + feb_curves
    n_total        = len(all_day_curves)

    fig, ax = plt.subplots(figsize=(13, 5))
    for c in all_day_curves:
        ax.plot(hours_axis, c, color="#4a90d9", lw=0.6, alpha=0.20)
    benchmark_median = np.median(all_day_curves, axis=0)
    ax.plot(hours_axis, benchmark_median, color="#e74c3c", lw=2.8,
            label=f"Benchmark Median — {n_total} days (Jan+Feb {pattern_year})", zorder=5)
    ax.set_xlabel("Hour of Day", fontsize=9)
    ax.set_ylabel("Normalised Flow (0=min, 1=max)", fontsize=9)
    ax.set_title(
        f"Multi-Day Overlay: Flow Pattern (Jan–Feb {pattern_year})\n"
        f"{n_total} Days Overlaid with Benchmark Median  "
        f"[Jan: {len(jan_curves)} days | Feb: {len(feb_curves)} days]", fontsize=10)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ② K-Means cluster centroids — benchmark highlighted
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "② All discovered daily shapes — benchmark highlighted</div>",
        unsafe_allow_html=True)

    cluster_sizes = np.bincount(curves_df["cluster"].values.astype(int), minlength=len(centroids))
    fig, ax = plt.subplots(figsize=(13, 4))
    palette = ["#555d6e"] * len(centroids)
    palette[modal_idx] = "#ffa94d"
    for i, centroid in enumerate(centroids):
        n   = cluster_sizes[i]
        lw  = 2.5 if i == modal_idx else 0.9
        alpha = 1.0 if i == modal_idx else 0.55
        label = (f"Cluster {i} — {n} days  ← BENCHMARK (most frequent)"
                 if i == modal_idx else f"Cluster {i} — {n} days")
        ax.plot(hours_axis, centroid, color=palette[i], lw=lw, alpha=alpha, label=label)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Normalised flow")
    ax.set_title(f"K-Means cluster centroids (k={pattern_k}) — modal = benchmark")
    ax.set_xticks(range(0, 24, 2)); ax.legend(fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ③ Daily similarity bar chart
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        f"③ Daily similarity to benchmark  (threshold = {sim_threshold}%)</div>",
        unsafe_allow_html=True)

    curves_df_sorted = curves_df.sort_values("date").reset_index(drop=True)
    bar_colors = ["#3fb950" if s >= sim_threshold else "#ff6b6b"
                  for s in curves_df_sorted["similarity"]]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(range(len(curves_df_sorted)), curves_df_sorted["similarity"],
           color=bar_colors, width=0.75, zorder=3)
    ax.axhline(sim_threshold, color="#ffa94d", lw=1.2,
               linestyle="--", label=f"Threshold {sim_threshold}%")
    ax.axhline(100, color="#555d6e", lw=0.5, linestyle=":")
    ax.set_xticks(range(len(curves_df_sorted)))
    ax.set_xticklabels([d[5:] for d in curves_df_sorted["date"]], rotation=60, fontsize=6.5)
    ax.set_ylabel("Similarity to benchmark (%)")
    ax.set_title("Daily pattern similarity — green ≥ threshold, red = deviant")
    ax.set_ylim(0, 105); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y"); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ④ Best vs worst matching days overlay
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "④ Best-match vs worst-match days vs benchmark</div>",
        unsafe_allow_html=True)

    ranked = curves_df_sorted.sort_values("similarity", ascending=False)
    top5   = ranked.head(5)["date"].tolist()
    bot5   = ranked.tail(5)["date"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for date_str in top5:
        if date_str in all_curves:
            axes[0].plot(hours_axis, all_curves[date_str], color="#3fb950", lw=1.0, alpha=0.65,
                         label=f"{date_str[5:]} ({curves_df_sorted[curves_df_sorted['date']==date_str]['similarity'].values[0]:.0f}%)")
    axes[0].plot(hours_axis, bench, color="#ffa94d", lw=2.2, linestyle="--", label="Benchmark")
    axes[0].set_title("Top 5 closest days to benchmark")
    axes[0].set_xlabel("Hour"); axes[0].set_ylabel("Normalised flow")
    axes[0].legend(fontsize=7.5); axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 24, 2)); axes[0].spines[["top","right"]].set_visible(False)

    for date_str in bot5:
        if date_str in all_curves:
            axes[1].plot(hours_axis, all_curves[date_str], color="#ff6b6b", lw=1.0, alpha=0.65,
                         label=f"{date_str[5:]} ({curves_df_sorted[curves_df_sorted['date']==date_str]['similarity'].values[0]:.0f}%)")
    axes[1].plot(hours_axis, bench, color="#ffa94d", lw=2.2, linestyle="--", label="Benchmark")
    axes[1].set_title("Bottom 5 most deviant days")
    axes[1].set_xlabel("Hour")
    axes[1].legend(fontsize=7.5); axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(0, 24, 2)); axes[1].spines[["top","right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ⑤ Summary metrics
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "⑤ Summary</div>", unsafe_allow_html=True)

    n_match   = (curves_df_sorted["similarity"] >= sim_threshold).sum()
    n_deviant = len(curves_df_sorted) - n_match
    avg_sim   = curves_df_sorted["similarity"].mean()
    best_day  = curves_df_sorted.loc[curves_df_sorted["similarity"].idxmax(), "date"]
    worst_day = curves_df_sorted.loc[curves_df_sorted["similarity"].idxmin(), "date"]

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col, label, val in [
        (mc1, "Total days",     f"{len(curves_df_sorted)}"),
        (mc2, "Match ✅",       f"{n_match}"),
        (mc3, "Deviant ❌",     f"{n_deviant}"),
        (mc4, "Avg similarity", f"{avg_sim:.1f}%"),
        (mc5, "Best match",     best_day[5:]),
    ]:
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>{label}</div>"
            f"<div class='metric-value' style='font-size:1.3rem'>{val}</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:.85rem;color:#c8cde0;margin:16px 0 6px'>Full similarity table</div>", unsafe_allow_html=True)
    st.dataframe(
        curves_df_sorted[["date","cluster","similarity","distance","is_benchmark_cluster"]].reset_index(drop=True),
        width="stretch", height=340)
    st.download_button("⬇️ Download similarity CSV",
                       data=curves_df_sorted.to_csv(index=False).encode(),
                       file_name=f"vmc_pattern_similarity_{pattern_year}.csv", mime="text/csv")

    # ⑥ Today vs Benchmark — PDF §5 comparison
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "⑥ Today's Flow vs Benchmark (PDF §5 comparison methodology)</div>",
        unsafe_allow_html=True)

    today_df = db_load(hours_back=24)
    if today_df.empty:
        st.info("No today data in DB yet — fetch a batch from the Live tab first.")
        today_windows = []; today_qos = 0.0; today_anomalies = []; matched_win = None
        status_str = "N/A"
    else:
        today_col = "flow_rate_m3hr" if "flow_rate_m3hr" in today_df.columns else "flow_rate"
        if today_col == "flow_rate":
            today_df  = today_df.rename(columns={"flow_rate": "flow_rate_m3hr"})
            today_col = "flow_rate_m3hr"
        today_df["hour_frac"] = today_df["timestamp"].dt.hour + today_df["timestamp"].dt.minute / 60

        today_windows = detect_supply_windows_df(today_df)
        if bench_box:
            today_qos, today_anomalies, matched_win = score_day_vs_benchmark(
                today_windows, bench_box,
                time_tol_min=time_tol_min, flow_tol=flow_tol_pct / 100)
        else:
            today_qos, today_anomalies, matched_win = 0.0, ["Benchmark not available"], None

        qos_color  = "#3fb950" if today_qos >= 85 else "#ffa94d" if today_qos >= 70 else "#ff6b6b"
        status_str = "EXCELLENT" if today_qos >= 85 else "GOOD" if today_qos >= 70 else "⚠️ POOR"

        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        today_peak = today_df[today_col].max()
        for col_, label, val, cls in [
            (kc1, "Today QoS",       f"{today_qos:.1f}%",       "danger" if today_qos < 70 else ""),
            (kc2, "Status",           status_str,                "danger" if today_qos < 70 else ""),
            (kc3, "Supply Windows",   str(len(today_windows)),   ""),
            (kc4, "Today Peak Flow",  f"{today_peak:.1f} m³/hr", ""),
            (kc5, "Anomalies",        str(len(today_anomalies)), "danger" if today_anomalies else ""),
        ]:
            col_.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                f"<div class='metric-value {cls}' style='font-size:1.3rem'>{val}</div></div>",
                unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.plot(today_df["hour_frac"], today_df[today_col],
                color="#4a90d9", lw=1.4, label="Today's flow")
        ax.fill_between(today_df["hour_frac"], today_df[today_col], alpha=0.08, color="#4a90d9")
        if bench_box:
            bx_s = bench_box["start_hour"]; bx_e = bench_box["end_hour"]
            ax.axvspan(bx_s, bx_e, ymin=0, ymax=0.95, alpha=0.10, color="#e74c3c")
            ax.axhline(bench_box["peak"], color="#e74c3c", lw=1.0, linestyle="--",
                       alpha=0.8, label=f"Benchmark peak ({bench_box['peak']:.1f})")
            ax.axhline(bench_box["avg"],  color="#ffa94d", lw=0.8, linestyle=":",
                       alpha=0.8, label=f"Benchmark avg ({bench_box['avg']:.1f})")
            ax.axvline(bx_s, color="#e74c3c", lw=1.0, linestyle="--",
                       alpha=0.7, label=f"Bm start {bm_start_str}")
            ax.axvline(bx_e, color="#e74c3c", lw=1.0, linestyle="--", alpha=0.7)
        for i, w in enumerate(today_windows):
            ax.axvspan(w["start_hour_frac"], w["end_hour_frac"], alpha=0.12, color="#3fb950",
                       label="Today window" if i == 0 else "")
        if today_anomalies and matched_win:
            for idx_a, a_text in enumerate(today_anomalies[:3]):
                ax.text(0.02, 0.97 - idx_a * 0.08, f"⚠ {a_text}",
                        transform=ax.transAxes, fontsize=7, color="#ff6b6b", va="top")
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2))
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Flow rate (m³/hr)")
        ax.set_title(f"Today's Flow vs Benchmark | QoS: {today_qos:.1f}% ({status_str})")
        ax.legend(fontsize=7.5, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        if today_anomalies:
            st.markdown(
                "<div style='font-size:.85rem;font-weight:500;color:#ff6b6b;margin:10px 0 4px'>"
                "⚠️ Today's Anomaly Details</div>", unsafe_allow_html=True)
            anom_rows = [[i + 1, a] for i, a in enumerate(today_anomalies)]
            st.dataframe(pd.DataFrame(anom_rows, columns=["#", "Description"]),
                         hide_index=True, height=min(250, len(anom_rows) * 38 + 40))
        else:
            st.success("✅ Today's distribution matches the benchmark — no anomalies detected.")

        if today_windows:
            st.markdown(
                "<div style='font-size:.85rem;font-weight:500;color:#c8cde0;margin:10px 0 4px'>"
                "Today's Supply Windows</div>", unsafe_allow_html=True)
            win_rows = []
            for i, w in enumerate(today_windows):
                bm_ok = bench_box and abs(w["start_hour_frac"] - bench_box["start_hour"]) * 60 <= time_tol_min
                win_rows.append({
                    "#":              i + 1,
                    "Start":          w["start"].strftime("%H:%M"),
                    "End":            w["end"].strftime("%H:%M"),
                    "Duration (min)": f"{w['duration']:.0f}",
                    "Peak (m³/hr)":   f"{w['peak']:.1f}",
                    "Avg (m³/hr)":    f"{w['avg']:.1f}",
                    "vs Benchmark":   "✅ Normal" if bm_ok else "⚠️ Deviated",
                })
            st.dataframe(pd.DataFrame(win_rows), hide_index=True)

    # ⑦ Flow rate heatmap — all Jan+Feb days (PDF Figure 5)
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "⑦ Flow Rate Heatmap — Jan+Feb All Days (matches PDF Figure 5)</div>",
        unsafe_allow_html=True)

    pat_df_h = pat_df.copy()
    pat_df_h["date_str"] = pat_df_h["timestamp"].dt.strftime("%m-%d")
    pat_df_h["hour"]     = pat_df_h["timestamp"].dt.hour
    hcol_    = "flow_rate_m3hr" if "flow_rate_m3hr" in pat_df_h.columns else "flow_rate"
    pivot_hm = pat_df_h.pivot_table(
        index="date_str", columns="hour", values=hcol_, aggfunc="mean").fillna(0)

    if not pivot_hm.empty:
        fig_h = max(6, len(pivot_hm) * 0.18 + 2)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        sns.heatmap(pivot_hm, ax=ax, cmap="YlOrRd", linewidths=0.05,
                    linecolor="#0f1117", cbar_kws={"label": "Flow rate (m³/hr)"},
                    annot=False, xticklabels=2)
        if curves_df is not None and not curves_df.empty:
            deviant_dates = set(curves_df[curves_df["similarity"] < sim_threshold]["date"].str[5:])
            for lbl in ax.get_yticklabels():
                if lbl.get_text() in deviant_dates:
                    lbl.set_color("#ffa94d"); lbl.set_fontweight("bold")
        ax.set_title(f"Flow Rate Heatmap — Jan+Feb {pattern_year} (all days × 24 hours)")
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Date (MM-DD)")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ⑧ Executive Dashboard — 4-panel (PDF Figure 6)
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "⑧ Executive Dashboard (matches PDF Figure 6 layout)</div>",
        unsafe_allow_html=True)

    if curves_df is not None and not curves_df.empty:
        n_match_ex   = (curves_df["similarity"] >= sim_threshold).sum()
        n_deviant_ex = len(curves_df) - n_match_ex
        avg_sim_ex   = curves_df["similarity"].mean()
        best_day_ex  = curves_df.loc[curves_df["similarity"].idxmax(), "date"]
        worst_day_ex = curves_df.loc[curves_df["similarity"].idxmin(), "date"]

        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
        for col_, label, val in [
            (ec1, "Total Days",     f"{len(curves_df)}"),
            (ec2, "Match ✅",       f"{n_match_ex}"),
            (ec3, "Deviant ❌",     f"{n_deviant_ex}"),
            (ec4, "Avg Similarity", f"{avg_sim_ex:.1f}%"),
            (ec5, "Best Match",     best_day_ex[5:]),
        ]:
            col_.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='font-size:1.3rem'>{val}</div>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="#1a1d27")
        for axi in axes.flat:
            axi.set_facecolor("#1a1d27"); axi.spines[["top","right"]].set_visible(False)

        # top-left: similarity distribution
        ax1 = axes[0, 0]
        ax1.hist(curves_df["similarity"], bins=20, color="#4a90d9", alpha=0.75, edgecolor="#0f1117")
        ax1.axvline(sim_threshold, color="#ffa94d", lw=1.2, linestyle="--", label=f"Threshold {sim_threshold}%")
        ax1.axvline(avg_sim_ex, color="#e74c3c", lw=1.0, linestyle=":", label=f"Avg {avg_sim_ex:.1f}%")
        ax1.set_xlabel("Similarity (%)"); ax1.set_ylabel("Number of Days")
        ax1.set_title("Similarity Score Distribution")
        ax1.legend(fontsize=7.5); ax1.grid(True, alpha=0.25)

        # top-right: anomaly type breakdown
        ax2 = axes[0, 1]
        if bench_box:
            type_counts = {"Start Time": 0, "End Time": 0, "Duration": 0, "Peak Flow": 0, "Avg Flow": 0}
            pat_cp = pat_df.copy(); pat_cp["date_"] = pat_cp["timestamp"].dt.date
            for date_, grp in pat_cp.groupby("date_"):
                wins_ = detect_supply_windows_df(grp)
                _, anoms_, _ = score_day_vs_benchmark(wins_, bench_box,
                                                      time_tol_min=time_tol_min,
                                                      flow_tol=flow_tol_pct / 100)
                for a in anoms_:
                    if "Start time"  in a: type_counts["Start Time"] += 1
                    elif "End time"  in a: type_counts["End Time"]   += 1
                    elif "Duration"  in a: type_counts["Duration"]   += 1
                    elif "Peak"      in a: type_counts["Peak Flow"]  += 1
                    elif "Avg"       in a: type_counts["Avg Flow"]   += 1
            bar_clrs2 = ["#ff6b6b","#ffa94d","#ffd700","#4a90d9","#3fb950"]
            ax2.barh(list(type_counts.keys()), list(type_counts.values()),
                     color=bar_clrs2, height=0.55, zorder=3)
            for i, (k, v) in enumerate(type_counts.items()):
                if v > 0: ax2.text(v + 0.1, i, str(v), va="center", fontsize=9)
            ax2.set_xlabel("Count"); ax2.set_title("Anomaly Types Breakdown")
            ax2.grid(True, alpha=0.25, axis="x")

        # bottom-left: supply start/end time scatter
        ax3 = axes[1, 0]
        if bench_box:
            pat_cp2 = pat_df.copy(); pat_cp2["date_"] = pat_cp2["timestamp"].dt.date
            day_starts_sc = []; day_ends_sc = []
            for date_, grp in sorted(pat_cp2.groupby("date_")):
                wins_ = detect_supply_windows_df(grp)
                if wins_:
                    best_ = min(wins_, key=lambda w: abs(w["start_hour_frac"] - bench_box["start_hour"]))
                    day_starts_sc.append(best_["start_hour_frac"])
                    day_ends_sc.append(best_["end_hour_frac"])
            x_sc = range(len(day_starts_sc))
            ax3.scatter(x_sc, day_starts_sc, color="#4a90d9", s=20, label="Supply Start", zorder=5)
            ax3.scatter(x_sc, day_ends_sc,   color="#3fb950", s=20, label="Supply End",   zorder=5)
            ax3.axhline(bench_box["start_hour"], color="#4a90d9", lw=0.9, linestyle="--",
                        alpha=0.7, label=f"Bm start ({bm_start_str})")
            ax3.axhline(bench_box["end_hour"],   color="#3fb950", lw=0.9, linestyle="--",
                        alpha=0.7, label=f"Bm end ({bm_end_str})")
            ax3.set_xlabel("Day index"); ax3.set_ylabel("Hour of Day")
            ax3.set_title("Supply Start & End Time Consistency\n(Dashed = median)")
            ax3.legend(fontsize=7, ncol=2); ax3.grid(True, alpha=0.2)

        # bottom-right: text summary panel
        ax4 = axes[1, 1]; ax4.axis("off")
        summary_text = (
            f"WATER DISTRIBUTION QUALITY SUMMARY\n"
            f"{'='*40}\n\n"
            f"Data Period: Jan 1 – Feb 28, {pattern_year}\n"
            f"Total Days Analysed: {len(curves_df)}\n\n"
            f"PATTERN STATISTICS:\n"
            f"  Benchmark match (≥{sim_threshold}%):  "
            f"{n_match_ex}/{len(curves_df)} ({n_match_ex*100//max(len(curves_df),1)}%)\n"
            f"  Deviant days (<{sim_threshold}%):       "
            f"{n_deviant_ex}/{len(curves_df)} ({n_deviant_ex*100//max(len(curves_df),1)}%)\n"
            f"  Average similarity:    {avg_sim_ex:.1f}%\n"
            f"  Best matching day:     {best_day_ex[5:]}\n"
            f"  Worst matching day:    {worst_day_ex[5:]}\n\n"
        )
        if bench_box:
            summary_text += (
                f"BENCHMARK PROFILE:\n"
                f"  Supply start:  ~{bm_start_str}\n"
                f"  Supply end:    ~{bm_end_str}\n"
                f"  Duration:      ~{bench_box['duration']:.0f} min\n"
                f"  Peak flow:     {bench_box['peak']:.1f} m³/hr\n"
                f"  Avg flow:      {bench_box['avg']:.1f} m³/hr\n"
                f"  Sample size:   {bench_box['samples']} windows\n\n"
            )
        if not today_df.empty:
            summary_text += (
                f"TODAY vs BENCHMARK:\n"
                f"  QoS Score:      {today_qos:.1f}%\n"
                f"  Status:         {status_str}\n"
                f"  Supply windows: {len(today_windows)}\n"
                f"  Anomalies:      {len(today_anomalies)}\n"
            )
        ax4.text(0.05, 0.97, summary_text, transform=ax4.transAxes,
                 fontsize=7.5, va="top", ha="left", color="#c8cde0",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e2130",
                           edgecolor="#2a2d3a", alpha=0.9))
        fig.tight_layout(pad=1.5); st.pyplot(fig); plt.close(fig)

    # ⑨ Median Curve + Margin Band — Today vs 2-Month Baseline
    #
    # How this works:
    #   1. For each of the 24 hours, compute median across all Jan+Feb days → typical curve.
    #   2. 25th–75th percentile per hour → "normal range" band (IQR).
    #      e.g. if most days have 150–300 m³/hr at 8am, that's the expected band.
    #   3. Plot today on top — any hour outside the band gets flagged.
    #
    #   Median is used instead of mean so outlier days (e.g. pipe burst) don't skew baseline.
    #   25th–75th percentile covers the middle 50% of historical days; points outside
    #   are rarer than 1 in 4 days and worth flagging.
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:28px 0 6px'>"
        "⑨ Median Curve + Margin Band — Today vs 2-Month Baseline</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.78rem;color:#555d6e;margin-bottom:12px'>"
        "The <span style='color:#e74c3c'>red line</span> is the median hourly flow "
        "across all Jan+Feb days. The <span style='color:#4a90d9'>shaded blue band</span> "
        "is the normal margin (25th–75th percentile). "
        "Today's curve in <span style='color:#3fb950'>green</span> is compared against "
        "this band — <span style='color:#ff6b6b'>red dots mark anomaly hours</span> "
        "where today is outside the margin.</div>",
        unsafe_allow_html=True)

    if all_curves and len(all_curves) >= 3:
        weekday_curves = []; weekend_curves = []
        for date_str, curve in all_curves.items():
            if pd.to_datetime(date_str).dayofweek >= 5:
                weekend_curves.append(curve)
            else:
                weekday_curves.append(curve)

        today_dow = datetime.now().weekday()
        if today_dow >= 5 and len(weekend_curves) >= 3:
            all_curve_matrix = np.array(weekend_curves); baseline_label = "Weekend baseline"
        elif len(weekday_curves) >= 3:
            all_curve_matrix = np.array(weekday_curves); baseline_label = "Weekday baseline"
        else:
            all_curve_matrix = np.array(list(all_curves.values())); baseline_label = "All-days baseline"

        median_curve = np.median(all_curve_matrix, axis=0)

        # Supply hours: 6–10 AM and 17–21 (5–9 PM) — adjust as needed
        SUPPLY_HOURS = list(range(6, 11)) + list(range(17, 22))

        supply_avg = np.mean(all_curve_matrix[:, SUPPLY_HOURS])  # avg across supply hours & all days
        margin_10pct = 0.10 * supply_avg                          # 10% of that avg

        lower_band = median_curve - margin_10pct
        upper_band = median_curve + margin_10pct
        lower_band = np.clip(lower_band, 0, None)                 # flow can't go negative
    today_norm_curve = None
    if not today_df.empty:
        today_norm_curve = normalize_daily_curve(today_df)
        if today_norm_curve is None and len(today_df) >= 2:
            today_df_temp = today_df.copy()
            today_df_temp["hour"] = today_df_temp["timestamp"].dt.hour
            hourly = today_df_temp.groupby("hour")["flow_rate_m3hr"].mean()
            curve  = hourly.reindex(range(24), fill_value=0.0).values.astype(float)
            mn, mx = curve.min(), curve.max()
            if mx - mn > 1e-6:
                today_norm_curve = (curve - mn) / (mx - mn)

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.fill_between(hours_axis, lower_band, upper_band, alpha=0.22, color="#4a90d9",
                label=f"Normal margin (±10% of supply-hour avg = ±{margin_10pct:.3f})")
        ax.plot(hours_axis, upper_band, color="#4a90d9", lw=0.9, linestyle="--", alpha=0.7, label="Upper margin (75th %ile)")
        ax.plot(hours_axis, lower_band, color="#4a90d9", lw=0.9, linestyle="--", alpha=0.7, label="Lower margin (25th %ile)")
        ax.plot(hours_axis, median_curve, color="#e74c3c", lw=2.5,
                label=f"Median (typical day) — {len(all_curves)} days", zorder=5)

        anomaly_hours = []
        if today_norm_curve is not None:
            ax.plot(hours_axis, today_norm_curve, color="#3fb950", lw=2.0,
                    label="Today's flow (normalised)", zorder=6)
            above_margin  = today_norm_curve > upper_band
            below_margin  = today_norm_curve < lower_band
            anomaly_mask  = above_margin | below_margin
            anomaly_hours = hours_axis[anomaly_mask].tolist()
            if anomaly_hours:
                ax.scatter(hours_axis[anomaly_mask], today_norm_curve[anomaly_mask],
                           color="#ff6b6b", s=80, zorder=8,
                           label=f"Today anomaly hours ({len(anomaly_hours)})",
                           edgecolors="#c0392b", linewidths=1.0)
                for h in anomaly_hours:
                    ax.axvline(h, color="#ff6b6b", lw=0.5, alpha=0.3, linestyle=":")
        else:
            ax.text(0.5, 0.5, "No today data in DB\n(fetch a batch from the Live tab first)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#555d6e", fontsize=11)

        ax.set_xlim(-0.5, 23.5); ax.set_xticks(range(0, 24, 1))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7.5)
        ax.set_xlabel("Hour of Day (00 = midnight, 12 = noon)", fontsize=9)
        ax.set_ylabel("Normalised Flow Rate  (0 = daily min, 1 = daily max)", fontsize=9)
        ax.set_ylim(-0.05, 1.15)

        title_str = f"2-Month Baseline vs Today  |  {len(all_curves)} reference days (Jan–Feb {pattern_year})"
        if today_norm_curve is not None and anomaly_hours:
            title_str += f"\n⚠️  Today is OUTSIDE the normal margin at hours: {anomaly_hours}"
        elif today_norm_curve is not None:
            title_str += "\n✅  Today stays within the normal margin — no anomaly"
        ax.set_title(title_str, fontsize=10)
        ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.85)
        ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        if today_norm_curve is not None:
            if anomaly_hours:
                st.markdown(
                    f"<div style='background:#1e1215;border:1px solid #ff6b6b;"
                    f"border-radius:8px;padding:12px 16px;margin-top:8px'>"
                    f"<span style='color:#ff6b6b;font-weight:600;font-size:.85rem'>"
                    f"⚠️  Today has {len(anomaly_hours)} anomaly hour(s) outside the 2-month margin</span>"
                    f"<br><span style='color:#8b949e;font-size:.78rem'>"
                    f"Anomaly hours: {', '.join(f'{h:02d}:00' for h in anomaly_hours)}<br>"
                    f"What this means: at these hours, today's flow pattern is "
                    f"significantly different from what was typical in Jan–Feb. "
                    f"Could be a supply disruption, leak, or demand surge.</span></div>",
                    unsafe_allow_html=True)

                # per-hour anomaly detail table
                anomaly_details = []
                for h in anomaly_hours:
                    h          = int(h)
                    today_val  = float(today_norm_curve[h])
                    median_val = float(median_curve[h])
                    upper_val  = float(upper_band[h])
                    lower_val  = float(lower_band[h])
                    direction  = "↑ ABOVE normal" if today_val > upper_val else "↓ BELOW normal"
                    pct_diff   = abs(today_val - median_val) / max(median_val, 0.01) * 100
                    anomaly_details.append({
                        "Hour"          : f"{h:02d}:00",
                        "Direction"     : direction,
                        "% from median" : f"{pct_diff:.0f}%",
                        "Today (norm)"  : f"{today_val:.3f}",
                        "Median (norm)" : f"{median_val:.3f}",
                        "Normal range"  : f"{lower_val:.3f} – {upper_val:.3f}",
                    })
                st.markdown(
                    "<div style='font-size:.82rem;font-weight:500;color:#ff6b6b;"
                    "margin:12px 0 4px'>Anomaly breakdown by hour</div>",
                    unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(anomaly_details), hide_index=True,
                             height=min(320, len(anomaly_details) * 38 + 40))
            else:
                st.markdown(
                    "<div style='background:#0d1a12;border:1px solid #3fb950;"
                    "border-radius:8px;padding:12px 16px;margin-top:8px'>"
                    "<span style='color:#3fb950;font-weight:600;font-size:.85rem'>"
                    "✅  Today's flow pattern is within the normal 2-month margin — no anomaly detected</span>"
                    "<br><span style='color:#8b949e;font-size:.78rem'>"
                    "Today stays inside the 25th–75th percentile band for all 24 hours.</span></div>",
                    unsafe_allow_html=True)

            with st.expander("💡 How the margin is calculated"):
                st.markdown(f"""
            **Currently using: ±10% of supply-hour average (normalised)**

            - Supply hours considered: 06:00–10:00 and 17:00–21:00
            - Average normalised flow during supply hours across all Jan+Feb days: `{supply_avg:.3f}`
            - 10% margin applied: `±{margin_10pct:.3f}`
            - An hour is flagged anomaly if today's flow deviates more than ±10% from the median at that hour.

            **To change supply hours:** Edit `SUPPLY_HOURS` list in the code.
            **To change margin %:** Change `0.10` to e.g. `0.15` for 15% margin.
            """)
    else:
        st.info("Not enough historical curves — fetch Jan+Feb data first using the button above.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — QoS TREND  (reads worker DB scores)
# ═════════════════════════════════════════════════════════════════════════════
with tab_qos:

    st.markdown(
        "<div style='font-size:.8rem;color:#555d6e;margin-bottom:14px'>"
        "Reads QoS scores and benchmark snapshots written automatically by "
        "<b>vmc_worker.py</b> each day. Run the worker at least once to populate."
        "</div>",
        unsafe_allow_html=True)

    qos_df = load_qos_history()
    bm_df  = load_benchmark_snapshots()

    if qos_df.empty:
        st.info("No QoS data yet. Start **vmc_worker.py** — it writes a score to the DB after each daily batch run.")
        st.stop()

    latest    = qos_df.iloc[-1]
    avg_qos   = qos_df["qos"].mean()
    best_day  = qos_df.loc[qos_df["qos"].idxmax()]
    worst_day = qos_df.loc[qos_df["qos"].idxmin()]
    days_poor = (qos_df["qos"] < 70).sum()

    def qos_cls(q):
        return "" if q >= 85 else "" if q >= 70 else "danger"

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    for col, label, val, cls in [
        (kc1, "Latest QoS",       f"{latest['qos']:.1f}%",                    qos_cls(latest["qos"])),
        (kc2, "Avg QoS",          f"{avg_qos:.1f}%",                           qos_cls(avg_qos)),
        (kc3, "Best day",         f"{best_day['date'][5:]} {best_day['qos']:.0f}%",   ""),
        (kc4, "Worst day",        f"{worst_day['date'][5:]} {worst_day['qos']:.0f}%", "danger"),
        (kc5, "Poor days (<70%)", str(int(days_poor)),                         "danger" if days_poor else ""),
    ]:
        col.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value {cls}' style='font-size:1.3rem'>{val}</div></div>",
            unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ① QoS trend
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:8px 0 6px'>"
        "① Daily QoS score trend</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 3.8))
    qos_df["date_dt"] = pd.to_datetime(qos_df["date"])
    clrs = ["#3fb950" if q >= 85 else "#ffa94d" if q >= 70 else "#ff6b6b" for q in qos_df["qos"]]
    ax.bar(qos_df["date_dt"], qos_df["qos"], color=clrs, width=0.7, zorder=3)
    ax.plot(qos_df["date_dt"], qos_df["qos"],
            color="#c8cde0", lw=1.2, zorder=4, marker="o", markersize=3)
    ax.axhline(85, color="#3fb950", lw=0.8, linestyle="--", alpha=0.7, label="Excellent (85%)")
    ax.axhline(70, color="#ffa94d", lw=0.8, linestyle="--", alpha=0.7, label="Good (70%)")
    ax.set_ylabel("QoS Score (%)"); ax.set_title("Daily Quality of Service — worker-computed scores")
    ax.set_ylim(0, 105); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y"); ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate(rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ② Anomaly breakdown
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "② Daily anomaly breakdown</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.bar(qos_df["date_dt"], qos_df["spike_anomalies"],
           color="#ff6b6b", width=0.6, label="Spike", zorder=3)
    ax.bar(qos_df["date_dt"], qos_df["night_anomalies"],
           color="#ffa94d", width=0.6, bottom=qos_df["spike_anomalies"], label="Night", zorder=3)
    z_anoms = (qos_df["total_anomalies"] - qos_df["spike_anomalies"] - qos_df["night_anomalies"]).clip(lower=0)
    ax.bar(qos_df["date_dt"], z_anoms, color="#9b8ec4", width=0.6,
           bottom=qos_df["spike_anomalies"] + qos_df["night_anomalies"],
           label="Z-score/other", zorder=3)
    ax.set_ylabel("Anomaly count"); ax.set_title("Anomaly breakdown by type per day")
    ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3, axis="y")
    ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate(rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ③ Avg + peak flow trend
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "③ Average and peak flow trend</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(qos_df["date_dt"], qos_df["avg_flow"],
            color="#4a90d9", lw=1.5, marker="o", markersize=3, label="Avg flow")
    ax.plot(qos_df["date_dt"], qos_df["peak_flow"],
            color="#ffa94d", lw=1.2, linestyle="--", marker="^", markersize=3, label="Peak flow")
    ax.set_ylabel("Flow rate (m³/hr)"); ax.set_title("Daily average and peak flow")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate(rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ④ Benchmark snapshots
    if not bm_df.empty:
        st.markdown(
            "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
            "④ Benchmark snapshots (worker-computed)</div>", unsafe_allow_html=True)
        bm_display = bm_df.copy()
        bm_display["saved_at"] = pd.to_datetime(bm_display["saved_at"]).dt.strftime("%d %b %Y %H:%M")
        for col in ["start_min","end_min"]:
            if col in bm_display.columns:
                bm_display[col] = bm_display[col].apply(
                    lambda x: f"{int(x)//60:02d}:{int(x)%60:02d}" if pd.notna(x) else "—")
        st.dataframe(bm_display, width="stretch", height=220)

    st.markdown(
        "<div style='font-size:.85rem;color:#c8cde0;margin:16px 0 6px'>Full QoS history table</div>",
        unsafe_allow_html=True)
    st.dataframe(
        qos_df.drop(columns=["date_dt"], errors="ignore").reset_index(drop=True),
        width="stretch", height=300)
    st.download_button(
        "⬇️ Download QoS history CSV",
        data=qos_df.drop(columns=["date_dt"], errors="ignore").to_csv(index=False).encode(),
        file_name="vmc_qos_history.csv", mime="text/csv")
    #