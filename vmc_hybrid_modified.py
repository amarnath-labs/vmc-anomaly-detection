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
CACHE_MAX_AGE_DAYS = 7


def _pattern_cache_path() -> str:
    meter = st.session_state.get("object_name", "default")
    safe  = meter.replace("/", "_").replace("\\", "_")
    return f"vmc_pattern_cache_{safe}.csv"

def save_pattern_cache(df: pd.DataFrame):
    df.to_csv(_pattern_cache_path(), index=False)

def load_pattern_cache() -> pd.DataFrame | None:
    path = _pattern_cache_path()
    if not os.path.exists(path):
        return None
    age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
    if age > CACHE_MAX_AGE_DAYS:
        return None
    return pd.read_csv(path, parse_dates=["timestamp"])


# ── CONFIG ────────────────────────────────────────────────────────────────────

VMC_BASE    = "https://scph1.vmcsmartwater.in:9090"
USE_DIRECT_API = True
METER_MAP = {
    # ── DLP1 / Phase-1 meters ─────────────────────────────────────────────
    "MJP-5917-A": {
        "dlp": "4797",
        "flowmeter": "A",
        "channel": "AI1",
        "tag_match": "MJP-5917",
        "aliases": ["MJP-5917", "AIB_FT015", "DLP1.AI1"],
    },
    "MJP-5917-B": {
        "dlp": "4797",
        "flowmeter": "B",
        "channel": "BI2",
        "tag_match": "FMB",
        "aliases": ["FMB", "DLP1.BI2"],
    },
    "MJP-4730": {
        "dlp": "4730",
        "flowmeter": "B",
        "channel": "B1",
        "tag_match": "MJP-4730",
        "aliases": ["MJP-4730", "DLP4730.B1"],
    },
    "AIB_FT015": {
        "dlp": "4797",
        "flowmeter": "A",
        "channel": "AIB_FT015",
        "tag_match": "AIB_FT015",
        "aliases": ["AIB_FT015", "DLP1.AIB"],
    },
    "KRL-5751": {
        "dlp": "5751",
        "flowmeter": "B",
        "channel": "BI2",
        "tag_match": "KRL-5751",
        "aliases": ["KRL-5751", "KRL-5917", "DLP5751.BI2"],
    },

    # ── DLP2 / Phase-2 meters (from dashboard screenshots) ────────────────
    "MJP-4684": {
        "dlp": "4684",
        "flowmeter": "A",
        "channel": "AI1",
        "tag_match": "FMA.AI1",
        "aliases": ["MJP-4684", "FMA", "DLP2.AI1", "VMC.DLP2.MJP.MJP-4684"],
        "flow_rate_max": 100,
    },

    "MJP-4738": {
        "dlp": "4738",
        "flowmeter": "B",
        "channel": "BI1",
        "tag_match": "MJP-4738",
        "aliases": ["MJP-4738", "FMB", "DLP2.BI1"],
    },
    "KRL-6136": {
        "dlp": "4797",
        "flowmeter": "A",
        "channel": "AI1",
        "tag_match": "KRL-6136",
        "aliases": ["KRL-6136", "VMC.DLP1.KRL.KRL-6136.Tags.FMA.AI1", "FMA.AI1"],
    },
    "KRL-5528": {
        "dlp": "5528",
        "flowmeter": "A",
        "channel": "AI2",
        "tag_match": "KRL-5528",
        "aliases": ["KRL-5528", "VMC.DLP1.KRL.KRL-5528.Tags.FMA", "FMA.AI2"],
        "flow_rate_max": 150,
    },
}


def get_meter_runtime_config():
    selected_meter = st.session_state.get("object_name", "MJP-5917").strip()
    meter_cfg = METER_MAP.get(selected_meter, {})

    raw_terms = [selected_meter]
    raw_terms.extend(meter_cfg.get("aliases", []))
    raw_terms.extend([
        meter_cfg.get("tag_match"),
        meter_cfg.get("channel"),
    ])

    match_terms = []
    seen = set()
    for term in raw_terms:
        term = str(term or "").strip()
        if term and term not in seen:
            match_terms.append(term)
            seen.add(term)

    return selected_meter, meter_cfg, match_terms


def tag_matches_meter(tag: str, match_terms: list[str]) -> bool:
    tag = str(tag or "")
    return any(term in tag for term in match_terms)


HISTORY_API_PATHS = [
    "/ph1/data",
    "/api/history/sensor/Flow/Rate",
    "/api/sensor/Flow/Rate/history",
    "/api/realtime/sensor/Flow/Rate",
]

REALTIME_API_PATH = "/api/realtime/sensor/Flow/Rate"

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
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            meter_id   TEXT NOT NULL DEFAULT 'MJP-5917',
            timestamp  TEXT NOT NULL,
            flow_rate  REAL NOT NULL,
            is_anomaly INTEGER DEFAULT 0
        )
    """)

    cols = [row[1] for row in con.execute("PRAGMA table_info(readings)").fetchall()]
    if "meter_id" not in cols:
        con.execute("ALTER TABLE readings ADD COLUMN meter_id TEXT DEFAULT 'MJP-5917'")

    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_readings_meter_ts_flow
        ON readings (meter_id, timestamp, flow_rate)
    """)

    con.commit()
    con.close()


def db_insert(ts: datetime, flow: float, anom: int, meter_id: str | None = None):
    meter_id = meter_id or st.session_state.get("object_name", "MJP-5917")
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR IGNORE INTO readings (meter_id, timestamp, flow_rate, is_anomaly) VALUES (?, ?, ?, ?)",
        (meter_id, ts.isoformat(), flow, anom)
    )
    con.commit(); con.close()

def db_insert_batch(rows: list, meter_id: str | None = None):
    if not rows: return
    meter_id = meter_id or st.session_state.get("object_name", "MJP-5917")
    rows_with_meter = [
        (meter_id, ts, flow, anom)
        for ts, flow, anom in rows
    ]
    con = sqlite3.connect(DB_PATH)
    con.executemany(
        "INSERT OR IGNORE INTO readings (meter_id, timestamp, flow_rate, is_anomaly) VALUES (?, ?, ?, ?)",
        rows_with_meter,
    )
    con.commit(); con.close()

def db_sanitize(max_flow: float = 800.0, meter_id: str | None = None) -> int:
    meter_id = meter_id or st.session_state.get("object_name", "MJP-5917")
    con = sqlite3.connect(DB_PATH)
    deleted = con.execute(
        "DELETE FROM readings WHERE meter_id = ? AND flow_rate > ?",
        (meter_id, max_flow)
    ).rowcount
    con.commit()
    con.close()
    return deleted


def db_load(hours_back: int = 24, meter_id: str | None = None) -> pd.DataFrame:
    meter_id = meter_id or st.session_state.get("object_name", "MJP-5917")

    con = sqlite3.connect(DB_PATH)
    since = (datetime.now() - timedelta(hours=hours_back)).isoformat()

    df = pd.read_sql(
        """
        SELECT meter_id, timestamp, flow_rate, is_anomaly
        FROM readings
        WHERE timestamp >= ?
          AND meter_id = ?
        ORDER BY timestamp
        """,
        con,
        params=(since, meter_id)
    )

    con.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.rename(columns={"flow_rate": "flow_rate_m3hr"})
    return df

def db_count(meter_id: str | None = None) -> int:
    meter_id = meter_id or st.session_state.get("object_name", "MJP-5917")

    con = sqlite3.connect(DB_PATH)
    n = con.execute(
        "SELECT COUNT(*) FROM readings WHERE meter_id = ?",
        (meter_id,)
    ).fetchone()[0]
    con.close()
    return n


def db_clear(meter_id: str | None = None):
    meter_id = meter_id or st.session_state.get("object_name", "MJP-5917")

    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM readings WHERE meter_id = ?", (meter_id,))
    con.commit()
    con.close()

def db_clear_all():
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM readings")
    con.commit()
    con.close()

init_db()

def db_count_all() -> int:
    con = sqlite3.connect(DB_PATH)
    n = con.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    con.close()
    return n


# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [("live_rows",[]),("anom_log",[]),("last_raw",""),
             ("last_error",""),("token",None),("field_map",{}),
             ("batch_done", False), ("batch_count", 0),
             ("pattern_df", None), ("benchmark_curve", None),
             ("benchmark_windows", None),
             ("curves_df", None), ("all_curves", None),
             ("centroids", None), ("modal_idx", None),
             ("object_name", "MJP-5917")]:
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
def build_benchmark_from_windows(all_windows, n_clusters=6):
    if not all_windows:
        return None, {}
    wdf = pd.DataFrame(all_windows)
    starts = wdf["start_hour_frac"].values.reshape(-1, 1)

    if len(wdf) < 3:
        benchmark = {
            "start_hour": float(np.median(wdf["start_hour_frac"])),
            "end_hour":   float(np.median(wdf["end_hour_frac"])),
            "duration":   float(np.median(wdf["duration"])),
            "peak":       float(np.median(wdf["peak"])),
            "avg":        float(np.median(wdf["avg"])),
            "peak_std":   float(wdf["peak"].std() or 1),
            "avg_std":    float(wdf["avg"].std() or 1),
            "start_std":  float(wdf["start_hour_frac"].std() or 0.25),
            "samples":    len(wdf),
            "cluster_id": 0,
            "all_clusters": {0: len(wdf)},
        }
        benchmark["start_hour"] = max(0.0, min(23.99, benchmark["start_hour"]))
        benchmark["end_hour"]   = max(0.0, min(23.99, benchmark["end_hour"]))
        benchmark["duration"]   = max(0.0, benchmark["duration"])
        return benchmark, wdf

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

    benchmark["start_hour"] = max(0.0, min(23.99, benchmark["start_hour"]))
    benchmark["end_hour"]   = max(0.0, min(23.99, benchmark["end_hour"]))
    benchmark["duration"]   = max(0.0, benchmark["duration"])
    return benchmark, wdf

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

    FLOW_RATE_MAX = 800

    for pk in ["Value","value","flow","Flow","flowRate","flow_rate","reading",
               "val","data","Flow_Rate","FlowRate","instantaneous","rate","FLOW"]:
        if pk in numeric:
            # SAFE FIX: removed abs() — preserve real signed flow value
            flow = float(numeric[pk])
            if abs(flow) > FLOW_RATE_MAX:
                return None, ts
            return flow, ts

    nonzero = {k: v for k, v in numeric.items() if v != 0.0}
    if nonzero:
        # SAFE FIX: removed abs() — preserve real signed flow value
        flow = float(next(iter(nonzero.values())))
        if abs(flow) > FLOW_RATE_MAX:
            return None, ts
        return flow, ts

    # SAFE FIX: removed abs() — preserve real signed flow value
    flow = float(next(iter(numeric.values())))
    if abs(flow) > FLOW_RATE_MAX:
        return None, ts
    return flow, ts

# ── BATCH RESPONSE PARSER ─────────────────────────────────────────────────────
def _parse_batch_response(data, fallback_ts: datetime) -> list[dict]:
    records = []

    if (isinstance(data, list) and data
            and isinstance(data[0], dict) and "tagname" in data[0]):
        _, _, match_terms = get_meter_runtime_config()

        rows = [
            d for d in data
            if tag_matches_meter(d.get("tagname") or d.get("tagName") or "", match_terms)
        ]

        if not rows:
            _meter = st.session_state.get("object_name", "unknown")
            st.session_state.last_error = f"No rows matched selected meter: {_meter}"
            return []

        for row in rows:
            try:
                # SAFE FIX: removed abs() — preserve real signed flow value
                flow = float(row.get("value") or 0)
            except (TypeError, ValueError):
                continue
            if abs(flow) > FLOW_RATE_MAX:
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
                # SAFE FIX: removed abs() — preserve real signed flow value
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
                    # SAFE FIX: removed abs() — preserve real signed flow value
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

    seen = set()
    unique = []
    for rec in records:
        key = f"{rec['timestamp'].isoformat()}_{rec['flow_rate']:.3f}"
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    unique.sort(key=lambda x: x["timestamp"])
    return unique


# ── BATCH FETCH ───────────────────────────────────────────────────────────────
def fetch_real_data(hours: int = 24) -> list[dict]:
    now = datetime.now()
    start = now - timedelta(hours=hours)
    selected_meter, meter_cfg, match_terms = get_meter_runtime_config()

    if not meter_cfg:
        st.session_state.last_error = (
            f"No direct /data mapping for meter '{selected_meter}'. "
            f"Trying generic objectname API instead."
        )
        return []

    url = f"{VMC_BASE}/data"
    all_records = []

    chunk_start = start
    while chunk_start < now:
        chunk_end = min(chunk_start + timedelta(hours=24), now)

        params = {
            "dlp": meter_cfg["dlp"],
            "flowmeter": meter_cfg["flowmeter"],
            "startTime": chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
            "endTime":   chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            r = SESSION.get(url, params=params, timeout=30)
            st.session_state.last_raw = (
                f"HTTP {r.status_code} | chunk {chunk_start.date()}→{chunk_end.date()} | "
                f"meter={selected_meter}\nURL: {r.url}\n\n{r.text[:1000]}"
            )
            r.raise_for_status()
            payload = r.json()
            data = payload.get("data", [])
        except Exception as e:
            st.session_state.last_error = f"[DIRECT API chunk {chunk_start.date()}] {e}"
            chunk_start = chunk_end
            continue

        channel    = meter_cfg.get("channel", "AI1")
        tag_match  = meter_cfg.get("tag_match", selected_meter)

        for row in data:
            tag = str(row.get("tagName") or row.get("tagname") or "")
            if not tag_matches_meter(tag, match_terms):
                continue
            try:
                ts   = pd.to_datetime(row["timestamp"])
                # SAFE FIX: removed abs() — preserve real signed flow value
                flow = float(row["value"])
                meter_max = meter_cfg.get("flow_rate_max", FLOW_RATE_MAX)
                # SAFE FIX: use abs() only for range guard, not for assignment
                if abs(flow) > meter_max:
                    continue
                all_records.append({
                    "timestamp": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    "flow_rate": flow,
                })
            except Exception:
                continue

        chunk_start = chunk_end

    if all_records:
        df = pd.DataFrame(all_records)
        df = df.sort_values("timestamp")
        df["_flow_round"] = df["flow_rate"].round(3)
        df = df.drop_duplicates(["timestamp", "_flow_round"])
        df = df.drop(columns=["_flow_round"])
        return df.to_dict("records")

    st.session_state.last_error = (
        f"No rows found for meter={selected_meter} across {hours}h range "
        f"(dlp={meter_cfg['dlp']}, flowmeter={meter_cfg['flowmeter']})"
    )
    return []

def fetch_data(hours):
    selected_meter = st.session_state.get("object_name", "").strip()

    if not selected_meter:
        st.session_state.last_error = "Enter a flow meter ID first."
        return []

    if USE_DIRECT_API and selected_meter in METER_MAP:
        data = fetch_real_data(hours)

        if data:
            return data

        return fetch_batch_old(hours)

    return fetch_batch_old(hours)

def fetch_batch_old(hours: int = 24) -> list[dict]:
    now   = datetime.now()
    start = now - timedelta(hours=hours)
    st.session_state.last_error = ""
    selected_meter, meter_cfg, match_terms = get_meter_runtime_config()

    query_names = []
    for name in [selected_meter, meter_cfg.get("tag_match"), meter_cfg.get("channel"), *meter_cfg.get("aliases", [])]:
        name = str(name or "").strip()
        if name and name not in query_names:
            query_names.append(name)

    for object_query in query_names or [OBJECT_NAME]:
        for path in HISTORY_API_PATHS:
            try:
                r = SESSION.get(
                    f"{VMC_BASE}{path}",
                    params={
                        "objectname": object_query,
                        "startTime": start.strftime("%Y-%m-%d %H:%M:%S"),
                        "endTime": now.strftime("%Y-%m-%d %H:%M:%S"),
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
            f"HTTP {r.status_code} | path={path} | objectname={object_query} | window={hours}h"
            f"\nURL: {r.url}\n\n{r.text[:3000]}"
        )

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
    selected_meter, meter_cfg, match_terms = get_meter_runtime_config()

    for delta in [timedelta(hours=1), timedelta(hours=6), timedelta(hours=24)]:
        start = now - delta
        try:
            r = SESSION.get(f"{VMC_BASE}{REALTIME_API_PATH}",
                params={"objectname": selected_meter,
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
            row = next(
                (d for d in data if tag_matches_meter(d.get("tagname") or d.get("tagName") or "", match_terms)),
                None
            )

            if row is None or float(row.get("value") or 0) == 0.0:
                candidates = [d for d in data if float(d.get("value") or 0) != 0.0]
                candidates.sort(key=lambda d: d.get("updated_at",""), reverse=True)
                if candidates: row = candidates[0]
            if row:
                try:
                    # SAFE FIX: removed abs() — preserve real signed flow value
                    flow = float(row["value"])
                except: flow = None
                for tk in ["updated_at","created_at"]:
                    raw = row.get(tk,"")
                    if raw:
                        parsed = _parse_ts(raw)
                        if parsed:
                            ts = parsed; break
        elif isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
            ts = datetime.utcfromtimestamp(float(data[-1][0])/1000)+IST_OFFSET
            # SAFE FIX: removed abs() — preserve real signed flow value
            flow = float(data[-1][1])
        elif isinstance(data, dict) and "data" in data:
            pts = data["data"]
            if pts and isinstance(pts[0], dict): flow, ts = _extract(pts[-1], now)
            elif pts:
                ts = datetime.utcfromtimestamp(float(pts[-1][0])/1000)+IST_OFFSET
                # SAFE FIX: removed abs() — preserve real signed flow value
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
    # SAFE FIX: use abs() only for magnitude spike check, keep sign for negative detection
    if val < 0 or abs(val) > spike_thresh:
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
    active_mask = flows > 5
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
            flow < 0                            # SAFE FIX: negative = real anomaly (no abs)
            or abs(flow) > spike_thresh         # SAFE FIX: magnitude check only
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

    # SAFE FIX: use abs() only for magnitude spike check
    df["anom_spike"]    = (df["flow_rate_m3hr"].abs() > spike_threshold).astype(int)
    df["anom_negative"] = (df["flow_rate_m3hr"] < 0).astype(int)
    NIGHT_FLOW_LIMIT    = spike_threshold * 0.8
    df["anom_night"]    = ((df["is_night"]==1) & (df["flow_rate_m3hr"] > NIGHT_FLOW_LIMIT)).astype(int)

    active = df["flow_rate_m3hr"] > 5
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
    active = df[df["flow_rate_m3hr"] > 5]["flow_rate_m3hr"].values
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
    selected_meter, meter_cfg, match_terms = get_meter_runtime_config()

    jan_start = datetime(year, 1, 1, 0, 0, 0)
    end_date  = datetime.now()
    all_records: list[dict] = []
    failed_chunks = []

    if meter_cfg:
        url = f"{VMC_BASE}/data"
        chunk_start = jan_start

        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=1), end_date)

            params = {
                "dlp":       meter_cfg["dlp"],
                "flowmeter": meter_cfg["flowmeter"],
                "startTime": chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                "endTime":   chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
            }

            try:
                r = SESSION.get(url, params=params, timeout=30)
                r.raise_for_status()
                payload = r.json()
                data = payload.get("data", [])

                chunk_records = []
                for row in data:
                    tag = str(row.get("tagName") or row.get("tagname") or "")
                    if not tag_matches_meter(tag, match_terms):
                        continue
                    try:
                        ts   = pd.to_datetime(row["timestamp"])
                        # SAFE FIX: removed abs() — preserve real signed flow value
                        flow = float(row["value"])
                    except Exception:
                        continue
                    # SAFE FIX: use abs() only for range guard, not for assignment
                    if abs(flow) > FLOW_RATE_MAX:
                        continue
                    chunk_records.append({
                        "timestamp": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        "flow_rate": flow,
                    })

                if chunk_records:
                    all_records.extend(chunk_records)
                else:
                    failed_chunks.append(chunk_start.strftime('%b %d'))

            except Exception as e:
                failed_chunks.append(chunk_start.strftime('%b %d'))
                st.session_state.last_error = f"[fetch_two_months chunk {chunk_start.date()}] {e}"

            chunk_start = chunk_end + timedelta(seconds=1)

    else:
        chunk_start = jan_start
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=7), end_date)
            chunk_ok  = False

            for path in HISTORY_API_PATHS:
                try:
                    r = SESSION.get(
                        f"{VMC_BASE}{path}",
                        params={
                            "objectname": selected_meter,
                            "startTime":  chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                            "endTime":    chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        timeout=10,
                    )
                    if r.status_code != 200:
                        continue
                    records = _parse_batch_response(r.json(), chunk_end)
                    if records:
                        all_records.extend(records)
                        chunk_ok = True
                        break
                except Exception:
                    continue

            if not chunk_ok:
                failed_chunks.append(chunk_start.strftime('%b %d'))
            chunk_start = chunk_end + timedelta(seconds=1)

    if failed_chunks:
        st.warning(
            f"⚠️ No data for: **{', '.join(failed_chunks[:10])}**"
            + (f" ... and {len(failed_chunks)-10} more" if len(failed_chunks) > 10 else "")
        )

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.rename(columns={"flow_rate": "flow_rate_m3hr"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["_fr"]       = df["flow_rate_m3hr"].round(3)
    df = df.drop_duplicates(["timestamp", "_fr"]).drop(columns=["_fr"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def normalize_daily_curve(day_df: pd.DataFrame) -> np.ndarray | None:
    """
    Collapses one day into a 24-point hourly-mean curve.
    SHAPE-SAFE: returns raw m³/hr values — NO min-max normalization.
    Used only for clustering/pattern detection, NOT for plotting raw signal.
    """
    day_df = day_df.copy()
    day_df["hour"] = day_df["timestamp"].dt.hour

    # Per-hour IQR clip — removes intra-hour sensor spikes only
    def _iqr_clip_hour(s):
        if len(s) < 4:
            return s
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        fence = 3.0
        lo, hi = q1 - fence * iqr, q3 + fence * iqr
        return s.clip(lo, hi)

    day_df["flow_clipped"] = day_df.groupby("hour")["flow_rate_m3hr"].transform(_iqr_clip_hour)

    # Raw hourly mean (unchanged original signal)
    hourly_raw = day_df.groupby("hour")["flow_rate_m3hr"].mean()
    raw_curve  = hourly_raw.reindex(range(24), fill_value=np.nan).values.astype(float)

    # Processed hourly mean (on IQR-clipped values)
    hourly_proc = day_df.groupby("hour")["flow_clipped"].mean()
    proc_curve  = hourly_proc.reindex(range(24), fill_value=np.nan).values.astype(float)

    n_real = np.sum(~np.isnan(raw_curve))
    if n_real < 2:
        return None

    # SAFE FIX: fill gaps with 0 — no interpolation to fabricate values
    def _fill_curve(arr):
        s = pd.Series(arr)
        s = s.fillna(0)   # SAFE FIX: removed interpolate() — no fabrication
        return s.values

    raw_curve  = _fill_curve(raw_curve)
    proc_curve = _fill_curve(proc_curve)

    # Reject dead days
    if raw_curve.max() < 1.0:
        return None

    # Reject cumulative-volume contamination
    diffs = np.diff(raw_curve)
    rising_streak = 0
    max_streak = 0
    for d in diffs:
        if d > 0.5:
            rising_streak += 1
            max_streak = max(max_streak, rising_streak)
        else:
            rising_streak = 0
    if max_streak >= 8:
        return None

    # Shape-preservation validation
    valid_mask = raw_curve > 0.5
    if valid_mask.sum() >= 4:
        corr = float(np.corrcoef(raw_curve[valid_mask], proc_curve[valid_mask])[0, 1])
        raw_peak_hr  = int(np.argmax(raw_curve))
        proc_peak_hr = int(np.argmax(proc_curve))
        peak_ok  = abs(raw_peak_hr - proc_peak_hr) <= 1
        shape_ok = (corr >= 0.98) and peak_ok
    else:
        shape_ok = False

    # SAFE FIX: return raw when shape validation fails — never distort
    return proc_curve if shape_ok else raw_curve


def curve_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))

@st.cache_data(show_spinner=False)
def find_benchmark_pattern(df: pd.DataFrame, n_clusters: int = 6):
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    all_curves: dict  = {}
    valid_dates: list = []

    for date, group in df.groupby("date"):
        curve = normalize_daily_curve(group)
        if curve is not None:
            all_curves[str(date)] = curve
            valid_dates.append(str(date))

    X = np.array([all_curves[d] for d in valid_dates])

    if len(X) == 0:
        st.error("❌ No valid daily curves found — API returned no usable data. "
                 "Clear pattern cache and try fetching again.")
        st.stop()

    if len(X) == 1:
        benchmark = X[0]
        rows = [{
            "date":       valid_dates[0],
            "cluster":    0,
            "similarity": 100.0,
            "distance":   0.0,
            "is_benchmark_cluster": True,
        }]
        dummy_centroid = X
        return benchmark, pd.DataFrame(rows), all_curves, np.array([0]), dummy_centroid, 0

    n_clusters = min(n_clusters, max(1, len(valid_dates) // 2))
    n_clusters = max(2, n_clusters)

    if len(X) < 2:
        st.error("❌ Need at least 2 days of data for pattern analysis.")
        st.stop()

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels    = km.fit_predict(X)
    centroids = km.cluster_centers_

    cluster_sizes = np.bincount(labels)
    modal_idx     = int(np.argmax(cluster_sizes))

    modal_mask = labels == modal_idx
    modal_curves = X[modal_mask]
    benchmark = np.median(modal_curves, axis=0)
    rows = []
    for i, date_str in enumerate(valid_dates):
        dist = curve_distance(all_curves[date_str], benchmark)
        similarity = max(0.0, 100.0 * (1.0 - dist / np.sqrt(24)))
        rows.append({
            "date":       date_str,
            "cluster":    int(labels[i]),
            "similarity": round(similarity, 1),
            "distance":   round(dist, 4),
            "is_benchmark_cluster": int(labels[i]) == modal_idx,
        })

    return benchmark, pd.DataFrame(rows), all_curves, labels, centroids, modal_idx


# ── BOX-METHOD HELPERS ────────────────────────────────────────────────────────

def detect_supply_windows_df(day_df, threshold=5.0,
                              min_duration_min=5,
                              min_gap_min=30):
    df = day_df.copy().sort_values("timestamp").reset_index(drop=True)
    col = "flow_rate_m3hr" if "flow_rate_m3hr" in df.columns else "flow_rate"
    windows = []
    in_window = False
    start_idx = None
    zero_start = None

    for i, row in df.iterrows():
        if row[col] >= threshold and not in_window:
            in_window = True
            start_idx = i
            zero_start = None
        elif row[col] < threshold and in_window:
            if zero_start is None:
                zero_start = row["timestamp"]
            gap_min = (row["timestamp"] - zero_start).total_seconds() / 60
            if gap_min >= min_gap_min:
                in_window = False
                wdf = df.iloc[start_idx:i]
                dur = (wdf["timestamp"].iloc[-1] -
                       wdf["timestamp"].iloc[0]).total_seconds() / 60
                if dur >= min_duration_min:
                    windows.append({
                        "start": wdf["timestamp"].iloc[0],
                        "end":   wdf["timestamp"].iloc[-1],
                        "duration": dur,
                        "peak":  wdf[col].max(),
                        "avg":   wdf[col].mean(),
                        "start_hour_frac": (wdf["timestamp"].iloc[0].hour +
                                           wdf["timestamp"].iloc[0].minute / 60),
                        "end_hour_frac":   (wdf["timestamp"].iloc[-1].hour +
                                           wdf["timestamp"].iloc[-1].minute / 60),
                    })
                zero_start = None
        elif row[col] >= threshold and in_window:
            zero_start = None

    if in_window and start_idx is not None:
        wdf = df.iloc[start_idx:]
        dur = (wdf["timestamp"].iloc[-1] -
               wdf["timestamp"].iloc[0]).total_seconds() / 60
        if dur >= min_duration_min:
            windows.append({
                "start": wdf["timestamp"].iloc[0],
                "end":   wdf["timestamp"].iloc[-1],
                "duration": dur,
                "peak":  wdf[col].max(),
                "avg":   wdf[col].mean(),
                "start_hour_frac": (wdf["timestamp"].iloc[0].hour +
                                   wdf["timestamp"].iloc[0].minute / 60),
                "end_hour_frac":   (wdf["timestamp"].iloc[-1].hour +
                                   wdf["timestamp"].iloc[-1].minute / 60),
            })
    return windows


def score_day_vs_benchmark(day_windows, benchmark, time_tol_min=30, flow_tol=0.20):
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
    df = df.copy(); df["date"] = df["timestamp"].dt.date
    all_curves_km = {}; valid_dates = []
    for date, group in df.groupby("date"):
        curve = normalize_daily_curve(group)
        if curve is not None:
            all_curves_km[str(date)] = curve; valid_dates.append(str(date))
    if len(valid_dates) < 2:
        if len(valid_dates) == 1:
            only_curve = all_curves_km[valid_dates[0]]
            rows = [{"date": valid_dates[0], "cluster": 0,
                    "similarity": 100.0, "distance": 0.0,
                    "is_benchmark_cluster": True}]
            return only_curve, pd.DataFrame(rows), all_curves_km, np.array([0]), np.array([only_curve]), 0
        return None, pd.DataFrame(), all_curves_km, np.array([]), np.array([]), 0

    n_clusters = min(n_clusters, max(1, len(valid_dates) // 2))
    n_clusters = max(2, n_clusters)
    X  = np.array([all_curves_km[d] for d in valid_dates])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels    = km.fit_predict(X)
    centroids = km.cluster_centers_

    cluster_sizes = np.bincount(labels)
    modal_idx_km  = int(np.argmax(cluster_sizes))

    modal_mask_km    = labels == modal_idx_km
    modal_curves_km  = X[modal_mask_km]
    benchmark_curve_km = np.median(modal_curves_km, axis=0)
    rows = []
    for i, d in enumerate(valid_dates):
        dist = float(np.sqrt(np.sum((all_curves_km[d] - benchmark_curve_km) ** 2)))
        sim  = max(0.0, 100.0 * (1.0 - dist / np.sqrt(24)))
        rows.append({"date": d, "cluster": int(labels[i]),
                     "similarity": round(sim, 1), "distance": round(dist, 4),
                     "is_benchmark_cluster": int(labels[i]) == modal_idx_km})
    return benchmark_curve_km, pd.DataFrame(rows), all_curves_km, labels, centroids, modal_idx_km


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1rem;font-weight:600;color:#c8cde0'>💧 VMC Monitor</div>"
        "<div style='font-size:.72rem;color:#555d6e'>MJP-4231 · Vadodara</div>",
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;"
        "letter-spacing:.07em;margin-bottom:6px'>Flow Meter</div>",
        unsafe_allow_html=True
    )

    _meter_input = st.text_input(
        "Flow meter ID",
        value=st.session_state.get("object_name", "MJP-5917"),
        placeholder="Enter any meter ID, e.g. KRL-5751, MJP-5917, MJP-5432",
        label_visibility="collapsed"
    ).strip()

    if _meter_input != st.session_state.get("object_name", ""):
        st.session_state.object_name = _meter_input
        st.session_state.live_rows = []
        st.session_state.anom_log = []
        st.session_state.batch_done = False
        st.session_state.batch_count = 0
        st.session_state.last_raw = ""
        st.session_state.last_error = ""
        st.session_state.pattern_df = None
        st.session_state.benchmark_curve = None
        st.session_state.benchmark_windows = None
        st.session_state.curves_df = None
        st.session_state.all_curves = None
        st.session_state.centroids = None
        st.session_state.modal_idx = None

        cache_path = _pattern_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)

        st.rerun()

    st.markdown(
        "<div style='font-size:.65rem;color:#4a90d9;margin-top:4px;margin-bottom:4px'>"
        "ℹ️ Bidirectional meter — negative readings preserved as-is (real signal)."
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;"
        "letter-spacing:.07em;margin-bottom:6px'>Fetch mode</div>",
        unsafe_allow_html=True
    )

    fetch_mode = st.radio(
        "Fetch mode",
        ["📦 Batch (single call)", "🔴 Live (per-second)"],
        index=0,
        label_visibility="collapsed"
    )

    batch_mode = fetch_mode.startswith("📦")

    if st.button("💥 FULL RESET (all data)", type="primary"):
        db_clear_all()
        cache_path = _pattern_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Clear live"):
            st.session_state.live_rows = []; st.session_state.anom_log = []
            st.session_state.batch_done = False; st.rerun()
        if st.button("🧹 Sanitize DB"):
            n_deleted = db_sanitize(max_flow=800.0)
            st.toast(f"✅ Removed {n_deleted} corrupt readings (>800 m³/hr)", icon="🧹")
            st.rerun()
    with col_b:
        if st.button("🗑 Clear DB"):
            db_clear(); st.rerun()

        if st.button("🗑 Clear Pattern Cache"):
            cache_path = _pattern_cache_path()
            if os.path.exists(cache_path):
                os.remove(cache_path)
            st.session_state.pattern_df = None
            st.session_state.benchmark_curve = None
            st.session_state.all_curves = None
            st.session_state.curves_df = None
            st.session_state.centroids = None
            st.session_state.modal_idx = None
            st.rerun()

    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>", unsafe_allow_html=True)

    if batch_mode:
        st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Batch settings</div>", unsafe_allow_html=True)
        batch_hours = st.slider("Fetch window (hours)", 1, 192, 192)
    else:
        st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Live settings</div>", unsafe_allow_html=True)
        poll_interval = st.slider("Poll interval (s)", 1, 30, 1)
        window_mins   = st.slider("Chart window (min)", 1, 60, 5)

    spike_threshold = st.number_input("Spike threshold (m³/hr)", 1, 5000, 1500, 50)
    z_sensitivity   = st.slider("Z-score sensitivity", 1.5, 5.0, 3.0, 0.1)

    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;"
                "letter-spacing:.07em;margin-bottom:6px'>Analysis settings</div>", unsafe_allow_html=True)
    contamination  = st.slider("IF contamination", 0.01, 0.15, 0.05, 0.01)

    night_start    = st.slider("Night start (hr)", 0, 23, 23)
    night_end      = st.slider("Night end (hr)", 0, 23, 16)

    forecast_steps = st.slider("Forecast horizon", 10, 60, 30)
    db_hours = st.slider("DB history (hrs)", 1, 192, 192)

    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Pattern analysis</div>", unsafe_allow_html=True)
    pattern_year  = st.number_input("Jan–Feb year", 2023, 2026, 2026, 1)
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
n_db_all = db_count_all()
active_meter = st.session_state.get("object_name", "MJP-5917")
meter_id = st.session_state.get("object_name", "MJP-5917")

st.markdown(
    f"<div style='font-size:.7rem;color:#555d6e;margin-top:8px'>"
    f"DB for {active_meter}: {n_db:,} readings · All meters: {n_db_all:,}"
    f"</div>",
    unsafe_allow_html=True
)

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
OBJECT_NAME = st.session_state.get("object_name", "MJP-5917").strip()

FLOW_RATE_MAX = 800

tab_live, tab_eda, tab_anom, tab_pattern, tab_qos = st.tabs([
    "📦 Live / Batch Feed", "📊 EDA", "🔍 Anomaly Detection",
      "📐 Pattern Analysis", "📉 QoS Trend"])

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
        # SAFE FIX: plot raw flow_rate_m3hr directly — no transformation applied
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
        ax.axhline(0, color="#555d6e", lw=.6, linestyle=":", alpha=.5)  # SAFE FIX: show zero line for negative visibility
        ax.set_ylabel("m³/hr", fontsize=8, color="#555d6e")
        ax.grid(True, alpha=.4, lw=.5)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        span_hours = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
        if span_hours <= 0.1:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        elif span_hours <= 6:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        elif span_hours <= 48:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
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
                records = fetch_data(batch_hours)

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
                db_insert_batch(db_rows, meter_id=OBJECT_NAME)

                # SAFE FIX: store raw signed flow_rate directly — no abs()
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
                db_insert(reading["timestamp"], reading["flow_rate_m3hr"], int(anom), meter_id=OBJECT_NAME)

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
    df = ana_df_raw.copy()
    # SAFE FIX: snapshot raw signal before any detector enrichment
    df["raw_flow_rate"] = df["flow_rate_m3hr"].copy()
    return run_detectors(df, z_sensitivity, contamination,
                         spike_threshold, night_start, night_end)

# ── QoS DB LOADERS ────────────────────────────────────────────────────────────
def load_qos_history() -> pd.DataFrame:
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
    # SAFE FIX: plot raw flow_rate_m3hr — includes negatives, no transformation
    ax.plot(df["timestamp"], df["flow_rate_m3hr"], color="#4a90d9", lw=.7, alpha=.85,
            label="Flow rate (raw)")
    try:
        from scipy.signal import savgol_filter as _savgol
        _n = len(df)
        _win = min(11, _n if _n % 2 == 1 else _n - 1)
        if _win >= 5 and _n >= _win:
            _sg = _savgol(df["flow_rate_m3hr"].values, window_length=_win, polyorder=2)
            ax.plot(df["timestamp"], _sg, color="#c8cde0", lw=.9, alpha=.55,
                    linestyle="--", label="Light smooth (SG-11)")
    except Exception:
        pass
    ax.fill_between(df["timestamp"], df["flow_rate_m3hr"], alpha=.06, color="#4a90d9")
    ax.axhline(0, color="#ff6b6b", lw=.6, linestyle="--", alpha=.4)
    ax.set_ylabel("Flow rate (m³/hr)"); ax.set_title("Full flow rate time series")
    ax.grid(True, alpha=.3); ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax.legend(fontsize=7.5, loc="upper right", framealpha=.7)
    fig.autofmt_xdate(rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    ca, cb = st.columns(2)
    with ca:
        hourly = df[df["flow_rate_m3hr"] > 5].groupby("hour")["flow_rate_m3hr"].mean()
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
        ax.hist(df[df["flow_rate_m3hr"] > 5]["flow_rate_m3hr"], bins=50,
                color="#4a90d9", alpha=.8, density=True, label="Normal")
        ax.set_xlabel("Flow rate (m³/hr)"); ax.set_ylabel("Density")
        ax.set_title("Flow distribution (excl. near-zero noise)")
        ax.grid(True, alpha=.3)
        ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 3.5))
    # SAFE FIX: primary signal is raw flow_rate_m3hr throughout
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
    # SAFE FIX: plot raw_flow_rate (pre-detector snapshot) for anomaly overlay
    ax.plot(df["timestamp"], df["raw_flow_rate"], color="#4a90d9", lw=.7, alpha=.7, label="Flow (raw)")
    fa = df[df["final_anomaly"] == 1]
    ax.scatter(fa["timestamp"], fa["raw_flow_rate"], color="#ff6b6b", s=30, zorder=6,
               marker="^", label=f"Final anomaly ({len(fa)})")
    ax.axhline(spike_threshold, color="#ffa94d", lw=.8, linestyle="--", alpha=.6, label="Spike limit")
    ax.axhline(0, color="#555d6e", lw=.5, linestyle=":", alpha=.4)
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
        # SAFE FIX: use raw_flow_rate for all model overlay plots
        ax.plot(df["timestamp"], df["raw_flow_rate"], color="#4a90d9", lw=.4, alpha=.5)
        fl = df[df[col] == 1]
        ax.scatter(fl["timestamp"], fl["raw_flow_rate"], color=clr, s=18, zorder=6, label=f"{lbl} ({len(fl)})")
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
    dcols = [c for c in ["timestamp","raw_flow_rate","roll_mean_10","deviation",
                          "anom_zscore","anom_iqr","anom_iforest","anom_pca","model_vote"] if c in df.columns]
    st.dataframe(df[df["final_anomaly"] == 1][dcols].reset_index(drop=True), width="content", height=280)




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

    st.caption(f"🔧 Active meter: `{OBJECT_NAME}` — if this changed recently, clear pattern cache first.")

    p_col1, p_col2 = st.columns([1, 3])

    with p_col1:
        do_pattern = st.button(
            f"📥 Fetch Jan {pattern_year}–Today",
            type="primary",
            width="stretch"
        )

    with p_col2:
        if st.session_state.pattern_df is not None:
            pdf = st.session_state.pattern_df
            st.markdown(
                f"<div style='padding:8px 12px;background:#1a1d27;border:1px solid #2a2d3a;"
                f"border-radius:8px;font-size:.8rem;color:#4ecdc4'>"
                f"✅ Loaded: <b>{len(pdf):,}</b> readings across "
                f"<b>{pdf['timestamp'].dt.date.nunique()}</b> days "
                f"(Jan {pattern_year}–Today). Benchmark set. Scroll down for charts.</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='padding:8px 12px;background:#1a1d27;border:1px solid #2a2d3a;"
                "border-radius:8px;font-size:.8rem;color:#7a8196'>"
                "Click the button to fetch 2 months of data and run pattern analysis.</div>",
                unsafe_allow_html=True
            )

    pat_debug = st.empty()

    if do_pattern:
        try_login()

        with st.spinner(f"Fetching Jan {pattern_year}–Today in weekly chunks…"):
            cached = load_pattern_cache()

            if cached is not None:
                pat_df = cached
                st.warning(
                    f"⚠️ Loaded from **local cache** ({len(pat_df):,} rows). "
                    f"If you recently changed the meter (`{OBJECT_NAME}`), "
                    f"click **'🗑 Clear Pattern Cache'** in the sidebar and re-fetch."
                )
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
                bench, curves_df, all_curves, labels, centroids, modal_idx = find_benchmark_pattern(
                    pat_df,
                    n_clusters=int(pattern_k)
                )

                all_wins = []
                pat_df_cp = pat_df.copy()
                pat_df_cp["date_"] = pat_df_cp["timestamp"].dt.date

                for date_, group in pat_df_cp.groupby("date_"):
                    wins = detect_supply_windows_df(group)
                    for w in wins:
                        w["date"] = str(date_)
                    all_wins.extend(wins)

                benchmark_box, _ = build_benchmark_from_windows(
                    all_wins,
                    n_clusters=int(pattern_k)
                )

            st.session_state.benchmark_curve = bench
            st.session_state.benchmark_windows = benchmark_box
            st.session_state.curves_df = curves_df
            st.session_state.all_curves = all_curves
            st.session_state.centroids = centroids
            st.session_state.modal_idx = modal_idx

            st.rerun()

    if st.session_state.pattern_df is None:
        st.info("No Jan-Feb API data fetched yet — checking DB for accumulated readings...")
        fallback_df = db_load(hours_back=720)

        if not fallback_df.empty:
            if "flow_rate" in fallback_df.columns:
                fallback_df = fallback_df.rename(
                    columns={"flow_rate": "flow_rate_m3hr"}
                )

            st.session_state.pattern_df = fallback_df

            with st.spinner("Computing benchmark pattern from DB fallback..."):
                bench, curves_df, all_curves, labels, centroids, modal_idx = find_benchmark_pattern(
                    fallback_df,
                    n_clusters=int(pattern_k)
                )

                all_wins = []
                fallback_cp = fallback_df.copy()
                fallback_cp["date_"] = fallback_cp["timestamp"].dt.date

                for date_, group in fallback_cp.groupby("date_"):
                    wins = detect_supply_windows_df(group)
                    for w in wins:
                        w["date"] = str(date_)
                    all_wins.extend(wins)

                benchmark_box, _ = build_benchmark_from_windows(
                    all_wins,
                    n_clusters=int(pattern_k)
                )

            st.session_state.benchmark_curve = bench
            st.session_state.benchmark_windows = benchmark_box
            st.session_state.curves_df = curves_df
            st.session_state.all_curves = all_curves
            st.session_state.centroids = centroids
            st.session_state.modal_idx = modal_idx

            st.warning(
                f"Using last 30 days from DB "
                f"({len(fallback_df):,} readings across "
                f"{fallback_df['timestamp'].dt.date.nunique()} days) "
                f"as pattern baseline.\n\n"
                f"Fetch more daily batches to improve benchmark quality. "
                f"Or click Fetch Jan-Feb button above for historical data."
            )

            st.rerun()

        else:
            st.error(
                "No data in DB either. "
                "Go to Live / Batch Feed tab and fetch a batch first."
            )
            st.stop()

    pat_df = st.session_state.pattern_df
    bench = st.session_state.benchmark_curve
    bench_box = st.session_state.benchmark_windows
    curves_df = st.session_state.curves_df
    all_curves = st.session_state.all_curves
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

    # ── Build raw_curves_9 in RAW m³/hr (no normalization, no interpolation) ──
    raw_curves_9 = {}
    pat_df_9 = pat_df.copy()
    pat_df_9["date"] = pat_df_9["timestamp"].dt.date

    for date_, grp in pat_df_9.groupby("date"):
        grp = grp.copy()
        grp["hour"] = grp["timestamp"].dt.hour
        hourly = (grp.groupby("hour")["flow_rate_m3hr"]
                     .mean()                              # SAFE FIX: mean preserves magnitude
                     .reindex(range(24), fill_value=0.0)) # SAFE FIX: fill 0, no interpolation
        curve = hourly.values.astype(float)
        if curve.max() >= 1.0:
            raw_curves_9[str(date_)] = curve

    st.caption(f"Section ⑨ using {len(raw_curves_9)} raw daily curves "
               f"(max value: {max(c.max() for c in raw_curves_9.values()):.1f} m³/hr)"
               if raw_curves_9 else "No curves available")

    if raw_curves_9 and len(raw_curves_9) >= 1:

        if len(raw_curves_9) < 10:
            db_supplement = db_load(hours_back=168)
            if not db_supplement.empty:
                db_supplement["date"] = db_supplement["timestamp"].dt.date
                for date_, grp in db_supplement.groupby("date"):
                    date_str = str(date_)
                    if date_str not in raw_curves_9:
                        grp = grp.copy()
                        grp["hour"] = grp["timestamp"].dt.hour
                        hourly = (grp.groupby("hour")["flow_rate_m3hr"]
                                    .mean()
                                    .reindex(range(24), fill_value=0.0))  # SAFE FIX: no interpolation
                        curve = hourly.values.astype(float)
                        if curve.max() >= 1.0:
                            raw_curves_9[date_str] = curve
                st.info(f"ℹ️ Only {len(raw_curves_9)} API reference days — "
                        f"supplemented with last 7 days from DB.")

        # ⑥ Today vs Benchmark
        st.markdown(
            "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
            "⑥ Today's Flow vs Benchmark (PDF §5 comparison methodology)</div>",
            unsafe_allow_html=True)

    def db_load_recent_day(min_readings: int = 20) -> tuple[pd.DataFrame, str]:
        meter_id  = st.session_state.get("object_name", "MJP-5917")
        now       = datetime.now()
        since_24h = now - timedelta(hours=24)

        con   = sqlite3.connect(DB_PATH)
        df_24 = pd.read_sql(
            """
            SELECT meter_id, timestamp, flow_rate, is_anomaly
            FROM readings
            WHERE timestamp >= ? AND meter_id = ?
            ORDER BY timestamp
            """,
            con,
            params=(since_24h.isoformat(), meter_id),
        )
        con.close()

        if df_24.empty:
            return pd.DataFrame(), "No data"

        df_24["timestamp"]    = pd.to_datetime(df_24["timestamp"], format="mixed")
        df_24                 = df_24.rename(columns={"flow_rate": "flow_rate_m3hr"})
        df_24                 = df_24.sort_values("timestamp").reset_index(drop=True)

        df_24["hour"]      = df_24["timestamp"].dt.hour
        df_24["hour_frac"] = (
            df_24["timestamp"].dt.hour
            + df_24["timestamp"].dt.minute / 60
            + df_24["timestamp"].dt.second / 3600
        )

        label = f"Last 24h ({since_24h.strftime('%d %b %H:%M')} → now, {len(df_24):,} readings)"
        return df_24, label

    bench_box = st.session_state.get("benchmark_windows", None)
    today_df, _today_label = db_load_recent_day(min_readings=20)

    if _today_label != "Today":
        st.info(f"📅 {_today_label} — used to build a complete 24h comparison curve.")

    if not today_df.empty:
        today_df["hour"] = today_df["timestamp"].dt.hour
        today_df = today_df.sort_values("timestamp").reset_index(drop=True)
        today_df["hour_frac"] = (
            today_df["timestamp"].dt.hour +
            today_df["timestamp"].dt.minute / 60 +
            today_df["timestamp"].dt.second / 3600
        )

    if today_df.empty:
        st.info("No today data in DB yet — fetch a batch from the Live tab first.")
        today_windows = []
        today_qos = (0.0, ["Benchmark not available"], None)
        today_anomalies = []
        matched_win = None
        status_str = "N/A"
    else:
        today_col = "flow_rate_m3hr"

        today_windows = detect_supply_windows_df(today_df)

        if bench_box:
            today_qos, today_anomalies, matched_win = score_day_vs_benchmark(
                today_windows,
                bench_box,
                time_tol_min=time_tol_min,
                flow_tol=flow_tol_pct / 100
            )
        else:
            today_qos, today_anomalies, matched_win = (
                0.0,
                ["Benchmark not available"],
                None
            )

        qos_color = "#3fb950" if today_qos >= 85 else "#ffa94d" if today_qos >= 70 else "#ff6b6b"
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
        # SAFE FIX: plot raw flow_rate_m3hr directly — no transformation
        ax.plot(today_df["hour_frac"], today_df["flow_rate_m3hr"],
                color="#4a90d9", lw=1.0, alpha=0.85, label="Today's flow (m³/hr)")
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

    # ⑦ Flow rate heatmap
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

    # ⑨ Median Curve + Margin Band — Today vs 2-Month Baseline
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:28px 0 6px'>"
        "⑨ Median Curve + Margin Band — Today vs 2-Month Baseline</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.78rem;color:#555d6e;margin-bottom:12px'>"
        "The <span style='color:#e74c3c'>red line</span> is the median hourly flow "
        "across all Jan+Feb days. The <span style='color:#4a90d9'>shaded blue band</span> "
        "is the normal margin (±20% of supply-hour avg). "
        "Today's curve in <span style='color:#3fb950'>green</span> is compared against "
        "this band — <span style='color:#ff6b6b'>red dots mark anomaly hours</span> "
        "where today is outside the margin.</div>",
        unsafe_allow_html=True)

    if raw_curves_9 and len(raw_curves_9) >= 1:
        # ── STEP 1: Separate evening-supply days from no-supply days ──────────
        all_curve_matrix = np.array(list(raw_curves_9.values()))
        hours_axis = np.arange(24)

        EVENING_HOURS = slice(18, 24)
        evening_flow_per_day = all_curve_matrix[:, EVENING_HOURS].max(axis=1)
        has_evening_supply = evening_flow_per_day > 5

        if has_evening_supply.sum() >= 5:
            reference_matrix = all_curve_matrix[has_evening_supply]
            baseline_label = f"Evening-supply days ({has_evening_supply.sum()} days)"
        else:
            reference_matrix = all_curve_matrix
            baseline_label = f"All-days baseline ({len(all_curve_matrix)} days)"

        # ── STEP 2: Compute per-hour median ignoring zero-supply hours ────────
        hourly_medians = []
        for h in range(24):
            col = reference_matrix[:, h]
            active = col[col > 5]
            if len(active) >= 2:
                hourly_medians.append(float(np.median(active)))
            elif len(active) == 1:
                hourly_medians.append(float(active[0]))
            else:
                hourly_medians.append(0.0)
        median_curve = np.array(hourly_medians)

        # ── STEP 4: Build today's RAW hourly curve ────────────────────────────
        today_raw_curve = None
        if not today_df.empty:
            today_df_temp = today_df.copy()
            today_df_temp["hour"] = today_df_temp["timestamp"].dt.hour
            # SAFE FIX: use mean, fill 0, no interpolation — preserve raw signal shape
            hourly_today = (today_df_temp
                            .groupby("hour")["flow_rate_m3hr"]
                            .mean()
                            .reindex(range(24), fill_value=0.0))  # SAFE FIX: no interpolation
            today_raw_curve = hourly_today.values.astype(float)

        # ── STEP 3: Margin based on supply hours ──────────────────────────────
        non_zero_mask = median_curve > 5

        if non_zero_mask.any():
            supply_avg = np.mean(reference_matrix[:, non_zero_mask])
        else:
            supply_avg = np.mean(reference_matrix)

        if today_raw_curve is not None:
            today_active = today_raw_curve[today_raw_curve > 5]
            if len(today_active) > 0:
                today_supply_avg = float(np.mean(today_active))
                supply_avg = max(supply_avg, today_supply_avg)

        margin = 0.20 * supply_avg
        lower_band = np.clip(median_curve - margin, 0, None)
        upper_band = median_curve + margin

        # ── STEP 5: Detect anomaly hours ──────────────────────────────────────
        anomaly_hours = []
        if today_raw_curve is not None:
            above_margin = today_raw_curve > upper_band
            below_margin = (today_raw_curve < lower_band) | (
                (today_raw_curve < 5) & (median_curve > 5)
            )
            supply_hour_mask = median_curve > 5
            anomaly_mask = (above_margin | below_margin) & supply_hour_mask
            anomaly_hours = hours_axis[anomaly_mask].tolist()

        # ── STEP 6: Y-axis scale from actual data ─────────────────────────────
        y_max_ref   = float(np.nanmax(reference_matrix)) if len(reference_matrix) else 60.0
        y_max_today = float(np.nanmax(today_raw_curve))  if today_raw_curve is not None else 0.0
        y_max = max(y_max_ref, y_max_today) * 1.15
        y_max = max(y_max, 10.0)

        # ── STEP 7: Plot ───────────────────────────────────────────────────────
        current_hour = datetime.now().hour
        x_max = 23 if today_raw_curve is None else min(23, current_hour + 1)

        fig, ax = plt.subplots(figsize=(13, 5))

        ax.fill_between(hours_axis, lower_band, upper_band,
                        alpha=0.35, color="#4a90d9",
                        label=f"Normal margin (±20% of supply avg = ±{margin:.1f} m³/hr)")
        ax.plot(hours_axis, upper_band, color="#4a90d9",
                lw=0.9, linestyle="--", alpha=0.7, label=f"Upper margin (+20%)")
        ax.plot(hours_axis, lower_band, color="#4a90d9",
                lw=0.9, linestyle="--", alpha=0.7, label=f"Lower margin (−20%)")

        ax.plot(hours_axis, median_curve, color="#e74c3c", lw=2.5,
                label=f"Median — {len(reference_matrix)} days ({baseline_label})",
                zorder=5)

        # SAFE FIX: plot raw today curve directly — no normalization
        if today_raw_curve is not None:
            ax.plot(hours_axis[:x_max+1], today_raw_curve[:x_max+1],
                    color="#3fb950", lw=2.0,
                    label="Today's flow (m³/hr)", zorder=6)

            if anomaly_hours:
                anom_idx = [int(h) for h in anomaly_hours if int(h) <= x_max]
                ax.scatter(anom_idx,
                           today_raw_curve[anom_idx],
                           color="#ff6b6b", s=90, zorder=8,
                           label=f"Anomaly hours ({len(anom_idx)})",
                           edgecolors="#c0392b", linewidths=1.2)
                for h in anom_idx:
                    ax.axvline(h, color="#ff6b6b", lw=0.5, alpha=0.25, linestyle=":")
        else:
            ax.text(0.5, 0.5,
                    "No today data in DB\n(fetch a batch from Live tab first)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#555d6e", fontsize=11)

        ax.set_xlim(-0.5, x_max + 0.5)
        ax.set_xticks(range(0, x_max + 1, 1))
        ax.set_xticklabels([f"{h:02d}" for h in range(x_max + 1)], fontsize=7.5)
        ax.set_ylim(-1, y_max)
        ax.set_xlabel("Hour of Day (00 = midnight, 12 = noon)", fontsize=9)
        ax.set_ylabel("Flow Rate (m³/hr)", fontsize=9)

        if today_raw_curve is not None and anomaly_hours:
            title_str = (f"2-Month Baseline vs Today  |  {len(reference_matrix)} "
                         f"reference days (Jan–Feb {pattern_year})\n"
                         f"⚠️  Today OUTSIDE normal margin at hours: {anomaly_hours}")
        elif today_raw_curve is not None:
            title_str = (f"2-Month Baseline vs Today  |  {len(reference_matrix)} "
                         f"reference days (Jan–Feb {pattern_year})\n"
                         f"✅  Today stays within normal margin")
        else:
            title_str = (f"2-Month Baseline  |  {len(reference_matrix)} "
                         f"reference days (Jan–Feb {pattern_year})")

        ax.set_title(title_str, fontsize=10)
        ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.85)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── STEP 8: Status banner ──────────────────────────────────────────────
        if today_raw_curve is not None:
            if anomaly_hours:
                st.markdown(
                    f"<div style='background:#1e1215;border:1px solid #ff6b6b;"
                    f"border-radius:8px;padding:12px 16px;margin-top:8px'>"
                    f"<span style='color:#ff6b6b;font-weight:600;font-size:.85rem'>"
                    f"⚠️  Today has {len(anomaly_hours)} anomaly hour(s) outside "
                    f"the 2-month margin</span><br>"
                    f"<span style='color:#8b949e;font-size:.78rem'>"
                    f"Anomaly hours: "
                    f"{', '.join(f'{int(h):02d}:00' for h in anomaly_hours)}<br>"
                    f"Could be a supply disruption, leak, or demand surge."
                    f"</span></div>",
                    unsafe_allow_html=True)

                anomaly_details = []
                for h in anomaly_hours:
                    h         = int(h)
                    today_val = float(today_raw_curve[h])
                    med_val   = float(median_curve[h])
                    up_val    = float(upper_band[h])
                    lo_val    = float(lower_band[h])
                    direction = "↑ ABOVE normal" if today_val > up_val else "↓ BELOW normal"
                    pct_diff  = abs(today_val - med_val) / max(med_val, 0.01) * 100
                    anomaly_details.append({
                        "Hour"          : f"{h:02d}:00",
                        "Direction"     : direction,
                        "% from median" : f"{pct_diff:.0f}%",
                        "Today (m³/hr)" : f"{today_val:.2f}",
                        "Median (m³/hr)": f"{med_val:.2f}",
                        "Normal range"  : f"{lo_val:.1f} – {up_val:.1f} m³/hr",
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
                    "✅  Today's flow pattern is within the normal 2-month margin"
                    "</span></div>",
                    unsafe_allow_html=True)

                with st.expander("💡 How the margin is calculated"):
                    st.markdown(f"""
        **Currently using: ±20% of supply-hour average flow rate (m³/hr)**
        - Reference days used: `{len(reference_matrix)}` ({baseline_label})
        - Supply-hour avg flow rate: `{supply_avg:.2f} m³/hr`
        - Margin: `±{margin:.2f} m³/hr`
        - Supply hours identified: hours where median > 5 m³/hr
                    """)

    else:
        st.info("Not enough historical curves — fetch data first using the button above.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — QoS TREND
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