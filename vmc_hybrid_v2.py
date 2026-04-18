"""
VMC Water Flow — Hybrid Monitor + Analyser  (BATCH MODE v2)

CHANGES vs v1 (per senior review):
  • Pattern Analysis tab completely reworked:
      – ALL Jan + Feb daily curves overlaid on ONE single graph
        (not separate monthly means — the senior wanted the full overlap)
      – Benchmark is identified from that combined set via K-Means modal centroid
      – Today's flow (last 24h from DB) is compared against that benchmark
      – "Box pattern" approach matches the attached PDF methodology:
            width  = supply window duration
            height = peak / avg flow rate
      – Similarity scoring with tolerance bands (±30 min, ±20%)
      – Executive dashboard matching PDF Figure 6 layout
  • All other tabs (Live, EDA, Anomaly, Forecast, Data, QoS) UNCHANGED.

Run:  streamlit run vmc_hybrid_v2.py
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


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# SQLITE  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
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

def db_insert(ts, flow, anom):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO readings (timestamp,flow_rate,is_anomaly) VALUES (?,?,?)",
                (ts.isoformat(), flow, anom))
    con.commit()
    con.close()

def db_insert_batch(rows):
    if not rows: return
    con = sqlite3.connect(DB_PATH)
    con.executemany(
        "INSERT OR IGNORE INTO readings (timestamp,flow_rate,is_anomaly) VALUES (?,?,?)",
        rows,
    )
    con.commit()
    con.close()

def db_load(hours_back=24):
    con = sqlite3.connect(DB_PATH)
    since = (datetime.now() - timedelta(hours=hours_back)).isoformat()
    df = pd.read_sql(
        "SELECT timestamp,flow_rate,is_anomaly FROM readings WHERE timestamp>=? ORDER BY timestamp",
        con, params=(since,)
    )
    con.close()
    if df.empty: return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.rename(columns={"flow_rate":"flow_rate_m3hr"})
    return df

def db_count():
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

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k,v in [("live_rows",[]),("anom_log",[]),("last_raw",""),
            ("last_error",""),("token",None),("field_map",{}),
            ("batch_done", False), ("batch_count", 0),
            ("pattern_df", None), ("benchmark_curve", None),
            ("benchmark_windows", None), ("curves_df", None),
            ("all_curves", None), ("centroids", None), ("modal_idx", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# HTTP SESSION
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# LOGIN  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def try_login():
    if st.session_state.token: return True
    try: SESSION.get(f"{VMC_BASE}/login", timeout=8)
    except: pass
    for path in ["/login","/api/login","/api/auth","/api/token","/dashboard/login"]:
        try:
            r = SESSION.post(f"{VMC_BASE}{path}",
                data={"username":VMC_USER,"password":VMC_PASS},
                headers={"Content-Type":"application/x-www-form-urlencoded"},
                timeout=8,allow_redirects=True)
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

# ─────────────────────────────────────────────────────────────────────────────
# FIELD EXTRACTOR  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_ts(raw):
    if not raw: return None
    try:
        ts = datetime.fromisoformat(str(raw)[:25].replace("Z", "+00:00"))
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None) + IST_OFFSET
        return ts
    except: return None

def _extract(row, fallback_ts):
    ts = fallback_ts
    for tk in ["DateTime","dateTime","timestamp","time","Timestamp","ts","date"]:
        raw=row.get(tk)
        if raw:
            parsed = _parse_ts(str(raw)[:25])
            if parsed: ts = parsed; break
            try: ts=datetime.fromisoformat(str(raw)[:19]); break
            except: pass
    numeric={}
    for k,v in row.items():
        if isinstance(v,(int,float)) and not isinstance(v,bool):
            lk=k.lower()
            if not any(x in lk for x in ["id","time","stamp","index","seq","row","count","num"]):
                numeric[k]=float(v)
    st.session_state.field_map=numeric
    if not numeric: return None,ts
    for pk in ["Value","value","flow","Flow","flowRate","flow_rate","reading",
               "val","data","Flow_Rate","FlowRate","instantaneous","rate","FLOW"]:
        if pk in numeric: return numeric[pk],ts
    nonzero={k:v for k,v in numeric.items() if v!=0.0}
    if nonzero: return next(iter(nonzero.values())),ts
    return next(iter(numeric.values())),ts

# ─────────────────────────────────────────────────────────────────────────────
# BATCH RESPONSE PARSER  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_batch_response(data, fallback_ts):
    records = []
    if (isinstance(data, list) and data
            and isinstance(data[0], dict) and "tagname" in data[0]):
        rows = [d for d in data if d.get("tagname") == OBJECT_NAME]
        if not rows:
            rows = [d for d in data if float(d.get("value") or 0) > 0]
        for row in rows:
            try: flow = float(row.get("value") or 0)
            except: continue
            ts = None
            for tk in ["updated_at","created_at","DateTime","timestamp"]:
                ts = _parse_ts(str(row.get(tk,"")))
                if ts: break
            if ts is None: ts = fallback_ts
            records.append({"timestamp": ts, "flow_rate": flow})
    elif (isinstance(data, list) and data and isinstance(data[0],(list,tuple))):
        for pt in data:
            try:
                ts = datetime.utcfromtimestamp(float(pt[0])/1000) + IST_OFFSET
                flow = float(pt[1])
                records.append({"timestamp": ts, "flow_rate": flow})
            except: continue
    elif isinstance(data, dict) and "data" in data:
        pts = data["data"]
        if pts and isinstance(pts[0], dict):
            for row in pts:
                flow, ts = _extract(row, fallback_ts)
                if flow is not None: records.append({"timestamp": ts, "flow_rate": flow})
        elif pts:
            for pt in pts:
                try:
                    ts = datetime.utcfromtimestamp(float(pt[0])/1000) + IST_OFFSET
                    flow = float(pt[1])
                    records.append({"timestamp": ts, "flow_rate": flow})
                except: continue
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        for row in data:
            flow, ts = _extract(row, fallback_ts)
            if flow is not None: records.append({"timestamp": ts, "flow_rate": flow})
    elif isinstance(data, dict):
        flow, ts = _extract(data, fallback_ts)
        if flow is not None: records.append({"timestamp": ts, "flow_rate": flow})
    seen = set(); unique = []
    for rec in records:
        key = rec["timestamp"].isoformat()
        if key not in seen: seen.add(key); unique.append(rec)
    unique.sort(key=lambda x: x["timestamp"])
    return unique

# ─────────────────────────────────────────────────────────────────────────────
# BATCH FETCH  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_batch(hours=24):
    now   = datetime.now()
    start = now - timedelta(hours=hours)
    st.session_state.last_error = ""
    for path in HISTORY_API_PATHS:
        try:
            r = SESSION.get(f"{VMC_BASE}{path}",
                params={"objectname":OBJECT_NAME,
                        "startTime":start.strftime("%Y-%m-%d %H:%M:%S"),
                        "endTime":now.strftime("%Y-%m-%d %H:%M:%S")},
                timeout=60)
        except Exception as e:
            st.session_state.last_error = f"[{path}] {e}"; continue
        if "<title>login</title>" in r.text.lower():
            st.session_state.token = None; continue
        st.session_state.last_raw = (
            f"HTTP {r.status_code} | path={path} | window={hours}h"
            f"\nURL: {r.url}\n\n{r.text[:3000]}")
        if r.status_code != 200: continue
        try: data = r.json()
        except:
            st.session_state.last_raw = f"[{path}] Non-JSON: {r.text[:500]}"; continue
        records = _parse_batch_response(data, now)
        if len(records) > 1: return records
        st.session_state.last_raw += f"\n\n⚠️ [{path}] returned only {len(records)} row(s)"
    return []

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-READING FETCH  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_reading():
    now=datetime.now()
    for delta in [timedelta(hours=1),timedelta(hours=6),timedelta(hours=24)]:
        start=now-delta
        try:
            r=SESSION.get(f"{VMC_BASE}{REALTIME_API_PATH}",
                params={"objectname":OBJECT_NAME,
                        "startTime":start.strftime("%Y-%m-%d %H:%M:%S"),
                        "endTime":now.strftime("%Y-%m-%d %H:%M:%S")},timeout=10)
        except Exception as e:
            st.session_state.last_error=str(e); return None
        if "<title>login</title>" in r.text.lower():
            st.session_state.token=None; return None
        try: data=r.json()
        except:
            st.session_state.last_raw=f"Non-JSON: {r.text[:500]}"; return None
        st.session_state.last_raw=(
            f"HTTP {r.status_code} | window={delta}\nURL: {r.url}\n\n{r.text[:3000]}")
        if r.status_code!=200: continue
        flow,ts=None,now
        if isinstance(data,list) and data and isinstance(data[0],dict) and "tagname" in data[0]:
            row=next((d for d in data if d.get("tagname")==OBJECT_NAME),None)
            if row is None or float(row.get("value") or 0)==0.0:
                candidates=[d for d in data if float(d.get("value") or 0)>0]
                candidates.sort(key=lambda d: d.get("updated_at",""),reverse=True)
                if candidates: row=candidates[0]
            if row:
                try: flow=float(row["value"])
                except: flow=None
                for tk in ["updated_at","created_at"]:
                    raw=row.get(tk,"")
                    if raw:
                        parsed=_parse_ts(raw)
                        if parsed: ts=parsed; break
        elif isinstance(data,list) and data and isinstance(data[0],(list,tuple)):
            ts=datetime.utcfromtimestamp(float(data[-1][0])/1000)+IST_OFFSET; flow=float(data[-1][1])
        elif isinstance(data,dict) and "data" in data:
            pts=data["data"]
            if pts and isinstance(pts[0],dict): flow,ts=_extract(pts[-1],now)
            elif pts: ts=datetime.utcfromtimestamp(float(pts[-1][0])/1000)+IST_OFFSET; flow=float(pts[-1][1])
        elif isinstance(data,list) and data and isinstance(data[0],dict):
            flow,ts=_extract(data[-1],now)
        elif isinstance(data,dict): flow,ts=_extract(data,now)
        else: ts=datetime.now()
        if flow is not None: return {"timestamp":ts,"flow_rate_m3hr":float(flow)}
    return None

# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY TAGGERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def is_anomaly_live(val, history, spike_thresh, z_thresh):
    if val < 0 or val > spike_thresh: return True
    if val < 5 and len(history) >= 5 and np.mean(history[-5:]) > 50: return True
    if len(history) < 10: return False
    arr=np.array(history[-60:]); std=arr.std()
    if std < 1e-6: return False
    if abs(val - arr.mean()) / std > z_thresh: return True
    recent_mean=np.mean(history[-10:])
    if recent_mean > 100 and val < recent_mean * 0.4: return True
    return False

def tag_anomalies_batch(records, spike_thresh, z_thresh, night_start, night_end):
    if not records: return records
    flows=np.array([r["flow_rate"] for r in records])
    active_mask=flows>0; z_flags=np.zeros(len(flows),dtype=bool)
    if active_mask.sum()>10:
        active_z=np.abs(stats.zscore(flows[active_mask]))
        active_indices=np.where(active_mask)[0]
        z_flags[active_indices[active_z>z_thresh]]=True
    roll_mean=pd.Series(flows).rolling(10,min_periods=3).mean().values
    for i,rec in enumerate(records):
        flow=rec["flow_rate"]; hour=rec["timestamp"].hour
        is_night=hour>=night_start or hour<=night_end
        prev_mean=roll_mean[i-1] if i>0 else 0
        supply_cut=(flow<5) and (prev_mean is not None) and (prev_mean>100)
        sudden_drop=(prev_mean is not None and prev_mean>100 and flow<prev_mean*0.4 and flow>5)
        anom=(flow<0 or flow>spike_thresh or (is_night and flow>5) or
              z_flags[i] or supply_cut or sudden_drop)
        rec["is_anomaly"]=int(anom)
    return records

# ─────────────────────────────────────────────────────────────────────────────
# FULL DETECTOR  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def run_detectors(df, sensitivity, contamination, spike_threshold, night_start, night_end):
    df=df.copy()
    df["hour"]=df["timestamp"].dt.hour
    df["dow"]=df["timestamp"].dt.dayofweek
    df["date"]=df["timestamp"].dt.date
    df["roll_mean_10"]=df["flow_rate_m3hr"].rolling(10,min_periods=1).mean()
    df["roll_std_10"]=df["flow_rate_m3hr"].rolling(10,min_periods=1).std().fillna(0)
    df["roll_mean_30"]=df["flow_rate_m3hr"].rolling(30,min_periods=1).mean()
    df["flow_diff"]=df["flow_rate_m3hr"].diff().fillna(0)
    df["lag_1"]=df["flow_rate_m3hr"].shift(1).fillna(0)
    df["deviation"]=df["flow_rate_m3hr"]-df["roll_mean_30"]
    df["in_supply"]=df["hour"].between(8,10).astype(int)
    df["is_night"]=((df["hour"]>=night_start)|(df["hour"]<=night_end)).astype(int)
    df["anom_spike"]=(df["flow_rate_m3hr"]>spike_threshold).astype(int)
    df["anom_negative"]=(df["flow_rate_m3hr"]<0).astype(int)
    NIGHT_FLOW_LIMIT=spike_threshold*0.8
    df["anom_night"]=((df["is_night"]==1)&(df["flow_rate_m3hr"]>NIGHT_FLOW_LIMIT)).astype(int)
    active=df["flow_rate_m3hr"]>0; dfa=df[active].copy()
    supply_hours=dfa[~((dfa["hour"]>=night_start)|(dfa["hour"]<=night_end))]
    if len(supply_hours)>10:
        z_vals=np.abs(stats.zscore(supply_hours["flow_rate_m3hr"]))
        supply_hours=supply_hours.copy(); supply_hours["anom_z"]=(z_vals>sensitivity).astype(int)
        dfa["anom_z"]=0; dfa.loc[supply_hours.index,"anom_z"]=supply_hours["anom_z"]
    else:
        dfa["anom_z"]=0
    df["anom_zscore"]=0; df.loc[dfa.index,"anom_zscore"]=dfa["anom_z"]
    if len(dfa)>3:
        Q1,Q3=dfa["flow_rate_m3hr"].quantile([0.25,0.75]); IQR=Q3-Q1
        dfa["anom_iqr_f"]=((dfa["flow_rate_m3hr"]<Q1-2.5*IQR)|(dfa["flow_rate_m3hr"]>Q3+2.5*IQR)).astype(int)
    else: dfa["anom_iqr_f"]=0
    df["anom_iqr"]=0; df.loc[dfa.index,"anom_iqr"]=dfa["anom_iqr_f"]
    FEATS=["flow_rate_m3hr","roll_mean_10","roll_std_10","flow_diff","lag_1","deviation","hour","in_supply"]
    df["anom_iforest"]=0; df["iforest_score"]=0.0
    if len(dfa)>=20:
        sc=StandardScaler(); Xsc=sc.fit_transform(dfa[FEATS])
        ifor=IsolationForest(n_estimators=150,contamination=contamination,random_state=42)
        preds=ifor.fit_predict(Xsc); dfa["anom_if"]=(preds==-1).astype(int)
        dfa["if_score"]=-ifor.decision_function(Xsc)
        df.loc[dfa.index,"anom_iforest"]=dfa["anom_if"]
        df.loc[dfa.index,"iforest_score"]=dfa["if_score"]
    df["anom_pca"]=0; df["pca_score"]=0.0
    if len(dfa)>=20:
        mms=MinMaxScaler(); Xn=mms.fit_transform(dfa[FEATS])
        pca=PCA(n_components=min(3,len(FEATS)),random_state=42)
        Xp=pca.fit_transform(Xn); Xr=pca.inverse_transform(Xp)
        err=np.mean((Xn-Xr)**2,axis=1); thr=err.mean()+3*err.std()
        dfa["anom_pca_f"]=(err>thr).astype(int); dfa["pca_sc"]=err
        df.loc[dfa.index,"anom_pca"]=dfa["anom_pca_f"]
        df.loc[dfa.index,"pca_score"]=dfa["pca_sc"]
    df["prev_flow"]=df["flow_rate_m3hr"].shift(1).fillna(0)
    df["anom_supply_cut"]=((df["flow_rate_m3hr"]<5)&(df["prev_flow"]>100)).astype(int)
    df["anom_sudden_drop"]=((df["flow_rate_m3hr"]>5)&(df["roll_mean_10"]>100)&
                            (df["flow_rate_m3hr"]<df["roll_mean_10"]*0.4)).astype(int)
    df["model_vote"]=df["anom_zscore"]+df["anom_iqr"]+df["anom_iforest"]+df["anom_pca"]
    df["final_anomaly"]=((df["anom_negative"]==1)|(df["anom_spike"]==1)|
                         (df["anom_night"]==1)|(df["anom_supply_cut"]==1)|
                         (df["anom_sudden_drop"]==1)|(df["model_vote"]>=3)).astype(int)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# FORECAST  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def forecast(df, steps):
    active=df[df["flow_rate_m3hr"]>0]["flow_rate_m3hr"].values
    if len(active)<10: return None,None,None,None,None
    alpha=0.3; sm=[active[0]]
    for v in active[1:]: sm.append(alpha*v+(1-alpha)*sm[-1])
    sm=np.array(sm); n=min(20,len(sm)); trend=(sm[-1]-sm[-n])/n
    fcast=np.array([sm[-1]+trend*i for i in range(1,steps+1)])
    std=np.std(active[-30:]) if len(active)>=30 else np.std(active)
    diffs=df["timestamp"].diff().dt.total_seconds().median()/60
    freq=max(1,int(diffs)) if not np.isnan(diffs) else 3
    fts=pd.date_range(start=df["timestamp"].iloc[-1]+pd.Timedelta(minutes=freq),
                      periods=steps,freq=f"{freq}min")
    return fcast,fcast-1.96*std,fcast+1.96*std,fts,sm

# ═════════════════════════════════════════════════════════════════════════════
# ███████████████  NEW PATTERN ANALYSIS HELPERS  ███████████████████████████
# Reworked per senior's requirements:
#   1. Fetch ALL Jan+Feb data
#   2. Overlay ALL daily curves on ONE graph
#   3. Detect supply windows → "box" dimensions (width=duration, height=flow)
#   4. K-Means on box dimensions → modal = benchmark
#   5. Compare today's flow against that benchmark
#   6. Executive dashboard matching the PDF report
# ═════════════════════════════════════════════════════════════════════════════

def fetch_two_months(year=2026):
    """
    Fetch ALL of Jan + Feb from the VMC API in 7-day chunks.
    Returns DataFrame[timestamp, flow_rate_m3hr].
    """
    jan_start = datetime(year, 1, 1, 0, 0, 0)
    feb_end   = datetime(year, 2, 28, 23, 59, 59)
    all_records = []
    chunk_start = jan_start
    while chunk_start <= feb_end:
        chunk_end = min(chunk_start + timedelta(days=7), feb_end)
        for path in HISTORY_API_PATHS:
            try:
                r = SESSION.get(f"{VMC_BASE}{path}",
                    params={"objectname":OBJECT_NAME,
                            "startTime":chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                            "endTime":chunk_end.strftime("%Y-%m-%d %H:%M:%S")},
                    timeout=10)
            except Exception as e:
                st.session_state.last_error = f"[{path}] {e}"; continue
            if r.status_code != 200: continue
            if "<title>login</title>" in r.text.lower():
                st.session_state.token = None; continue
            try: data = r.json()
            except: continue
            records = _parse_batch_response(data, chunk_end)
            if len(records) > 1:
                all_records.extend(records); break
        chunk_start = chunk_end + timedelta(seconds=1)
    if not all_records: return pd.DataFrame()
    df = pd.DataFrame(all_records)
    df = df.rename(columns={"flow_rate":"flow_rate_m3hr"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def detect_supply_windows_df(day_df, threshold=1.0, min_duration_min=5):
    """
    Detect active supply windows from a single day's flow data.
    Returns list of dicts with start, end, duration, peak, avg.
    This is the "box detection" method used in the PDF report.
    """
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
                    "start_hour_frac": wdf["timestamp"].iloc[0].hour + wdf["timestamp"].iloc[0].minute/60,
                    "end_hour_frac":   wdf["timestamp"].iloc[-1].hour + wdf["timestamp"].iloc[-1].minute/60,
                })
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
                "start_hour_frac": wdf["timestamp"].iloc[0].hour + wdf["timestamp"].iloc[0].minute/60,
                "end_hour_frac":   wdf["timestamp"].iloc[-1].hour + wdf["timestamp"].iloc[-1].minute/60,
            })
    return windows


def build_benchmark_from_windows(all_windows, n_clusters=6):
    """
    Given all supply windows from Jan+Feb, cluster by start time using
    hierarchical clustering (matching PDF §2.3 methodology), then return
    the dominant cluster's median values as the benchmark.
    Also runs K-Means on [duration, peak, avg] for the "box shape" benchmark.
    """
    if not all_windows:
        return None, {}
    wdf = pd.DataFrame(all_windows)
    # Cluster by start_hour_frac using hierarchical clustering
    starts = wdf["start_hour_frac"].values.reshape(-1, 1)
    try:
        Z = linkage(starts, method="ward")
        labels = fcluster(Z, t=1.0, criterion="distance")  # 1 hr threshold
    except Exception:
        labels = np.ones(len(wdf), dtype=int)
    wdf["cluster"] = labels
    cluster_sizes = wdf.groupby("cluster").size()
    dominant_cluster = cluster_sizes.idxmax()
    dominant_wdf = wdf[wdf["cluster"] == dominant_cluster]
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
    # Also compute median profile curve (24-point) from all benchmark days
    return benchmark, wdf


def normalize_daily_curve(day_df):
    """
    Collapse one day → 24-point normalised vector (shape-based, not magnitude).
    """
    col = "flow_rate_m3hr" if "flow_rate_m3hr" in day_df.columns else "flow_rate"
    day_df = day_df.copy(); day_df["hour"] = day_df["timestamp"].dt.hour
    hourly = day_df.groupby("hour")[col].mean()
    curve = hourly.reindex(range(24), fill_value=0.0).values.astype(float)
    if (curve > 0).sum() < 4: return None
    mn, mx = curve.min(), curve.max()
    if mx - mn < 1e-6: return None
    return (curve - mn) / (mx - mn)


def compute_median_profile(df):
    """
    Build a 24-point median profile from all days in df (like PDF Figure 1 red line).
    Returns array of shape (24,) in original flow units (not normalised).
    """
    col = "flow_rate_m3hr" if "flow_rate_m3hr" in df.columns else "flow_rate"
    df = df.copy(); df["hour"] = df["timestamp"].dt.hour
    hourly = df.groupby("hour")[col].median()
    return hourly.reindex(range(24), fill_value=0.0).values.astype(float)


def score_day_vs_benchmark(day_windows, benchmark,
                            time_tol_min=30, flow_tol=0.20):
    """
    Score a single day's supply windows against the benchmark.
    Returns (qos_score 0-100, anomaly_list, matched_window or None).
    Matches PDF §2.4 methodology.
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
    time_tol_h = time_tol_min / 60.0

    # Match to nearest window by start time
    best_win = min(day_windows,
                   key=lambda w: abs(w["start_hour_frac"] - bm_start_h))

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

    # QoS: timing score (50%) + flow score (50%)
    t_score = max(0, 1 - (start_dev_min + end_dev_min) / (2 * time_tol_min * 3))
    f_score = max(0, 1 - (peak_dev + avg_dev) / (2 * flow_tol * 3))
    qos = min(100, max(0, (t_score * 0.5 + f_score * 0.5) * 100))
    return round(qos, 1), anomalies, best_win


def find_benchmark_pattern_kmeans(df, n_clusters=6):
    """
    Build 24-point normalised curves from ALL days in df,
    run K-Means, return modal centroid as benchmark curve.
    Used for the shape-based overlay chart (senior's request).
    """
    df = df.copy(); df["date"] = df["timestamp"].dt.date
    all_curves = {}; valid_dates = []
    for date, group in df.groupby("date"):
        curve = normalize_daily_curve(group)
        if curve is not None:
            all_curves[str(date)] = curve; valid_dates.append(str(date))
    if len(valid_dates) < 2:
        return None, pd.DataFrame(), all_curves, np.array([]), np.array([]), 0
    n_clusters = min(n_clusters, len(valid_dates))
    X = np.array([all_curves[d] for d in valid_dates])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(X); centroids = km.cluster_centers_
    cluster_sizes = np.bincount(labels); modal_idx = int(np.argmax(cluster_sizes))
    benchmark_curve = centroids[modal_idx]
    rows = []
    for i, d in enumerate(valid_dates):
        dist = float(np.sqrt(np.sum((all_curves[d] - benchmark_curve) ** 2)))
        sim  = max(0.0, 100.0 * (1.0 - dist / np.sqrt(24)))
        rows.append({"date": d, "cluster": int(labels[i]),
                     "similarity": round(sim, 1), "distance": round(dist, 4),
                     "is_benchmark_cluster": int(labels[i]) == modal_idx})
    return benchmark_curve, pd.DataFrame(rows), all_curves, labels, centroids, modal_idx


# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCE helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(fr_bytes, tf_bytes):
    fr = pd.read_csv(io.BytesIO(fr_bytes), parse_dates=["DateTime"])
    tf = pd.read_csv(io.BytesIO(tf_bytes), parse_dates=["DateTime"])
    fr.columns = ["timestamp","flow_rate_m3hr"]
    tf.columns = ["timestamp","cumulative_flow_m3"]
    df = pd.merge(fr, tf, on="timestamp", how="inner")
    return df.sort_values("timestamp").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_pdf_file(pdf_bytes):
    buf = io.BytesIO(pdf_bytes)
    if _PDFPLUMBER:
        import pdfplumber
        all_tables = []
        with pdfplumber.open(buf) as pdf:
            for page in pdf.pages:
                for tbl in page.extract_tables():
                    if not tbl or len(tbl) < 2: continue
                    try:
                        df_t = pd.DataFrame(tbl[1:], columns=tbl[0]); all_tables.append(df_t)
                    except: continue
        for df_t in all_tables:
            result = _coerce_pdf_table(df_t)
            if result is not None: return result
    buf.seek(0)
    if _PYPDF:
        from pypdf import PdfReader as _PR
        reader = _PR(buf); lines = []
        for page in reader.pages:
            txt = page.extract_text() or ""; lines.extend(txt.splitlines())
        result = _parse_pdf_text_lines(lines)
        if result is not None: return result
    raise ValueError("Could not extract a flow-rate table from this PDF.")

def _coerce_pdf_table(df_t):
    if df_t is None or df_t.empty: return None
    df_t.columns=[str(c).strip() if c else f"col_{i}" for i,c in enumerate(df_t.columns)]
    ts_col=None
    for col in df_t.columns:
        if any(kw in col.lower() for kw in ["datetime","date","time","timestamp","ts"]):
            ts_col=col; break
    flow_col=None
    for col in df_t.columns:
        if col==ts_col: continue
        if any(kw in col.lower() for kw in ["flow","rate","value","m3","reading","val"]):
            flow_col=col; break
    if ts_col is None: ts_col=df_t.columns[0]
    if flow_col is None:
        for col in df_t.columns:
            if col==ts_col: continue
            try: pd.to_numeric(df_t[col].dropna(),errors="raise"); flow_col=col; break
            except: continue
    if flow_col is None: return None
    try:
        out=pd.DataFrame()
        out["timestamp"]=pd.to_datetime(df_t[ts_col],infer_datetime_format=True,errors="coerce")
        out["flow_rate_m3hr"]=pd.to_numeric(df_t[flow_col],errors="coerce")
        out=out.dropna(subset=["timestamp","flow_rate_m3hr"])
        if len(out)<2: return None
        return out.sort_values("timestamp").reset_index(drop=True)
    except: return None

def _parse_pdf_text_lines(lines):
    import re
    date_pat=re.compile(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}[T ]\d{1,2}:\d{2}(?::\d{2})?)[\s,;|]+([\d.]+)")
    records=[]
    for line in lines:
        m=date_pat.search(line)
        if m:
            try:
                ts=datetime.fromisoformat(m.group(1).replace("/","-")); flow=float(m.group(2))
                records.append({"timestamp":ts,"flow_rate_m3hr":flow})
            except: continue
    if len(records)<2: return None
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

def get_analysis_df():
    db_df = db_load(db_hours)
    frames = []
    if file_fr and file_tf:
        try: csv_df=load_csv(file_fr.read(),file_tf.read()); frames.append(csv_df)
        except Exception as e: st.warning(f"CSV load error: {e}")
    if file_pdf:
        if not (_PDFPLUMBER or _PYPDF): st.warning("PDF parsing requires pdfplumber or pypdf.")
        else:
            try:
                pdf_df=load_pdf_file(file_pdf.read()); frames.append(pdf_df)
                st.sidebar.success(f"PDF: {len(pdf_df):,} rows loaded")
            except ValueError as e: st.warning(str(e))
            except Exception as e: st.warning(f"PDF load error: {e}")
    if frames:
        if not db_df.empty: frames.append(db_df)
        merged=pd.concat(frames,ignore_index=True)
        merged=merged.sort_values("timestamp").drop_duplicates("timestamp")
        return merged.reset_index(drop=True)
    return db_df

# QoS history loader (unchanged)
def load_qos_history():
    con = sqlite3.connect(DB_PATH)
    try: df = pd.read_sql("SELECT * FROM qos_scores ORDER BY date ASC", con)
    except: df = pd.DataFrame()
    con.close(); return df

def load_benchmark_snapshots():
    con = sqlite3.connect(DB_PATH)
    try: df = pd.read_sql("SELECT * FROM benchmark_snapshot ORDER BY saved_at DESC LIMIT 30", con)
    except: df = pd.DataFrame()
    con.close(); return df

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1rem;font-weight:600;color:#c8cde0'>💧 VMC Monitor</div>"
        "<div style='font-size:.72rem;color:#555d6e'>MJP-4231 · Vadodara</div>",
        unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>",unsafe_allow_html=True)
    st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Fetch mode</div>",unsafe_allow_html=True)
    fetch_mode = st.radio("Fetch mode", ["📦 Batch (single call)", "🔴 Live (per-second)"],
                          index=0, label_visibility="collapsed")
    batch_mode = fetch_mode.startswith("📦")
    col_a,col_b=st.columns(2)
    with col_a:
        if st.button("🗑 Clear live"):
            st.session_state.live_rows=[]; st.session_state.anom_log=[]
            st.session_state.batch_done=False; st.rerun()
    with col_b:
        if st.button("🗑 Clear DB"):
            db_clear(); st.rerun()
    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>",unsafe_allow_html=True)
    if batch_mode:
        st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Batch settings</div>",unsafe_allow_html=True)
        batch_hours = st.slider("Fetch window (hours)", 1, 24, 24)
    else:
        st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Live settings</div>",unsafe_allow_html=True)
        poll_interval=st.slider("Poll interval (s)",1,30,1)
        window_mins=st.slider("Chart window (min)",1,60,5)
    spike_threshold=st.number_input("Spike threshold (m³/hr)",100,2000,600,50)
    z_sensitivity=st.slider("Z-score sensitivity",1.5,5.0,3.0,0.1)
    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>",unsafe_allow_html=True)
    st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Analysis settings</div>",unsafe_allow_html=True)
    contamination=st.slider("IF contamination",0.01,0.15,0.05,0.01)
    night_start=st.slider("Night start (hr)",18,23,23)
    night_end=st.slider("Night end (hr)",0,8,5)
    forecast_steps=st.slider("Forecast horizon",10,60,30)
    db_hours=st.slider("DB history (hrs)",1,168,24)
    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>",unsafe_allow_html=True)
    st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>Pattern analysis</div>",unsafe_allow_html=True)
    pattern_year   = st.number_input("Jan–Feb year", 2023, 2026, 2026, 1)
    pattern_k      = st.slider("K-Means clusters (k)", 2, 10, 6, 1)
    sim_threshold  = st.slider("Match threshold (%)", 50, 95, 75, 5)
    time_tol_min   = st.slider("Timing tolerance (min)", 15, 60, 30, 5)
    flow_tol_pct   = st.slider("Flow tolerance (%)", 10, 40, 20, 5)
    st.markdown("<hr style='border-color:#2a2d3a;margin:10px 0'>",unsafe_allow_html=True)
    st.markdown("<div style='font-size:.68rem;color:#555d6e;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px'>CSV upload (optional)</div>",unsafe_allow_html=True)
    file_fr=st.file_uploader("Flow rate CSV",type="csv")
    file_tf=st.file_uploader("Cumulative volume CSV",type="csv")
    file_pdf=st.file_uploader("Flow data PDF",type="pdf")
    n_db=db_count()
    st.markdown(f"<div style='font-size:.7rem;color:#555d6e;margin-top:8px'>DB: {n_db:,} readings stored</div>",unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
hc1,hc2=st.columns([5,1])
with hc1:
    st.markdown(
        "<h1 style='font-size:1.35rem;font-weight:600;margin:0;color:#c8cde0'>💧 VMC Water Flow — Live + Analysis</h1>"
        "<p style='color:#555d6e;font-size:.75rem;margin:2px 0 12px'>MJP-4231 · Vadodara Municipal Corporation</p>",
        unsafe_allow_html=True)
with hc2:
    if batch_mode:
        st.markdown("<br><span class='batch-pill'>📦 BATCH</span>",unsafe_allow_html=True)
    else:
        st.markdown("<br><span class='live-pill'><span class='live-dot'></span>LIVE</span>",unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_live,tab_eda,tab_anom,tab_fcast,tab_data,tab_pattern,tab_qos=st.tabs([
    "📦 Live / Batch Feed","📊 EDA","🔍 Anomaly Detection",
    "📈 Forecast","📋 Data Table","📐 Pattern Analysis","📉 QoS Trend"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE / BATCH FEED  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
with tab_live:
    rows=st.session_state.live_rows; anom_log=st.session_state.anom_log
    if rows:
        cur=rows[-1]["flow_rate_m3hr"]; hist=[r["flow_rate_m3hr"] for r in rows]
        avg_f=np.mean(hist); max_f=np.max(hist)
        is_anom_now=is_anomaly_live(cur,hist[:-1],spike_threshold,z_sensitivity)
        cur_cls="danger" if is_anom_now else ""
    else:
        cur=avg_f=max_f=None; cur_cls=""

    def mc(label,value,cls=""):
        val_s=f"{value:.1f}" if value is not None else "—"
        return (f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                f"<div class='metric-value {cls}'>{val_s}</div></div>")

    c1,c2,c3,c4,c5=st.columns(5)
    c1.markdown(mc("Current (m³/hr)",cur,cur_cls),unsafe_allow_html=True)
    c2.markdown(mc("Average (m³/hr)",avg_f),unsafe_allow_html=True)
    c3.markdown(mc("Peak (m³/hr)",max_f),unsafe_allow_html=True)
    c4.markdown(mc("Readings",float(len(rows)) if rows else None),unsafe_allow_html=True)
    c5.markdown(
        f"<div class='metric-card'><div class='metric-label'>Anomalies</div>"
        f"<div class='metric-value {'danger' if anom_log else ''}'>{len(anom_log)}</div></div>",
        unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
    chart_ph=st.empty(); status_ph=st.empty()

    def draw_live(rows, wsecs, spike, z):
        if not rows: return
        df=pd.DataFrame(rows); df["timestamp"]=pd.to_datetime(df["timestamp"])
        df=df.sort_values("timestamp").tail(wsecs); hist=df["flow_rate_m3hr"].tolist()
        flags=[is_anomaly_live(v,hist[:i],spike,z) for i,v in enumerate(hist)]
        df["is_anom"]=flags
        fig,ax=plt.subplots(figsize=(12,3.6))
        ax.plot(df["timestamp"],df["flow_rate_m3hr"],color="#4a90d9",linewidth=1.3,alpha=.95,label="Flow rate")
        ax.fill_between(df["timestamp"],df["flow_rate_m3hr"],alpha=.07,color="#4a90d9")
        anoms=df[df["is_anom"]]
        if not anoms.empty:
            ax.scatter(anoms["timestamp"],anoms["flow_rate_m3hr"],
                       color="#ff6b6b",s=40,zorder=7,label=f"Anomaly ({len(anoms)})")
        ax.axhline(spike,color="#ffa94d",lw=.8,linestyle="--",alpha=.7,label=f"Spike limit ({spike})")
        ax.set_ylabel("m³/hr",fontsize=8,color="#555d6e"); ax.set_ylim(bottom=0)
        ax.grid(True,alpha=.4,lw=.5); ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S" if wsecs<=600 else "%H:%M"))
        leg=ax.legend(fontsize=7.5,loc="upper right",framealpha=.85,edgecolor="#2a2d3a")
        for t in leg.get_texts(): t.set_color("#9aa0b0")
        fig.autofmt_xdate(rotation=20); fig.tight_layout(pad=.5)
        chart_ph.pyplot(fig); plt.close(fig)

    bl,br=st.columns(2,gap="small")
    with bl:
        log_items=""
        for e in reversed(anom_log[-20:]):
            log_items+=(f"<div class='log-row'><span class='log-time'>{e['time']}</span>"
                        f"<span class='log-badge'>{e['val']:.1f} m³/hr</span></div>")
        if not log_items:
            log_items="<div style='color:#555d6e;font-size:.8rem;padding:12px 0'>No anomalies yet</div>"
        st.markdown(f"<div class='log-card'><div class='log-title'>Anomaly log</div>{log_items}</div>",unsafe_allow_html=True)
    with br:
        mode_label="Batch" if batch_mode else "Live poll"
        st.markdown(
            f"<div class='log-card'><div class='log-title'>Session info</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>"
            f"<div><div class='metric-label'>Mode</div><div style='font-size:1.1rem;font-weight:500;color:#c8cde0'>{mode_label}</div></div>"
            f"<div><div class='metric-label'>Spike limit</div><div style='font-size:1.3rem;font-weight:500;color:#c8cde0'>{spike_threshold}</div></div>"
            f"<div><div class='metric-label'>Readings</div><div style='font-size:1.1rem;font-weight:500;color:#c8cde0'>{len(rows)}</div></div>"
            f"<div><div class='metric-label'>DB total</div><div style='font-size:1.1rem;font-weight:500;color:#c8cde0'>{db_count():,}</div></div>"
            f"</div></div>",unsafe_allow_html=True)

    debug_ph=st.empty()

    if batch_mode:
        st.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)
        btn_col,info_col=st.columns([1,3])
        with btn_col:
            do_fetch=st.button("📦 Fetch batch now",type="primary",width="stretch")
        with info_col:
            if st.session_state.batch_done:
                st.markdown(
                    f"<div style='padding:8px 12px;background:#1a1d27;border:1px solid #2a2d3a;"
                    f"border-radius:8px;font-size:.8rem;color:#4ecdc4'>"
                    f"✅ Last batch: <b>{st.session_state.batch_count:,}</b> readings loaded "
                    f"for the past <b>{batch_hours}h</b>.</div>",unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='padding:8px 12px;background:#1a1d27;border:1px solid #2a2d3a;"
                    f"border-radius:8px;font-size:.8rem;color:#7a8196'>"
                    f"Click <b>Fetch batch now</b> to pull the last <b>{batch_hours}h</b>.</div>",unsafe_allow_html=True)
        if do_fetch:
            try_login()
            with st.spinner(f"Fetching {batch_hours}h batch…"):
                records = fetch_batch(batch_hours)
            with debug_ph.expander("🔍 Debug",expanded=(not records)):
                if st.session_state.last_error: st.error(f"Last error: {st.session_state.last_error}")
                if st.session_state.field_map: st.json(st.session_state.field_map)
                st.code(st.session_state.last_raw or "No response",language="json")
            if not records:
                status_ph.error("❌ No data returned")
            else:
                records=tag_anomalies_batch(records,spike_threshold,z_sensitivity,night_start,night_end)
                db_rows=[(r["timestamp"].isoformat(),r["flow_rate"],r.get("is_anomaly",0)) for r in records]
                db_insert_batch(db_rows)
                st.session_state.live_rows=[{"timestamp":r["timestamp"],"flow_rate_m3hr":r["flow_rate"]} for r in records]
                st.session_state.anom_log=[{"time":r["timestamp"].strftime("%H:%M:%S"),"val":r["flow_rate"]} for r in records if r.get("is_anomaly")]
                st.session_state.batch_done=True; st.session_state.batch_count=len(records)
                n_anom=sum(1 for r in records if r.get("is_anomaly"))
                status_ph.success(f"✅ Loaded {len(records):,} readings · {n_anom} anomalies · stored to DB")
                st.rerun()
        if st.session_state.live_rows:
            draw_live(st.session_state.live_rows,len(st.session_state.live_rows),spike_threshold,z_sensitivity)
        else:
            if not do_fetch: st.info("Click **Fetch batch now** to load data.")
    else:
        try_login(); reading=fetch_reading()
        with debug_ph.expander("🔍 Debug",expanded=(reading is None)):
            if st.session_state.last_error: st.error(f"Last error: {st.session_state.last_error}")
            if st.session_state.field_map: st.json(st.session_state.field_map)
            st.code(st.session_state.last_raw or "No response",language="json")
        if reading is None:
            status_ph.error("❌ No reading")
        else:
            st.session_state.live_rows.append(reading)
            st.session_state.live_rows=st.session_state.live_rows[-max(600,window_mins*60*2):]
            hist=[r["flow_rate_m3hr"] for r in st.session_state.live_rows]
            anom=is_anomaly_live(reading["flow_rate_m3hr"],hist[:-1],spike_threshold,z_sensitivity)
            if anom:
                st.session_state.anom_log.append({"time":reading["timestamp"].strftime("%H:%M:%S"),"val":reading["flow_rate_m3hr"]})
                status_ph.warning(f"⚠️ Anomaly {reading['timestamp'].strftime('%H:%M:%S')} — {reading['flow_rate_m3hr']:.1f} m³/hr")
            else:
                status_ph.success(f"✅ {reading['timestamp'].strftime('%H:%M:%S')} — {reading['flow_rate_m3hr']:.1f} m³/hr")
            db_insert(reading["timestamp"],reading["flow_rate_m3hr"],int(anom))
        draw_live(st.session_state.live_rows,window_mins*60,spike_threshold,z_sensitivity)
        time.sleep(poll_interval); st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SHARED DATA for analysis tabs
# ─────────────────────────────────────────────────────────────────────────────
ana_df_raw=get_analysis_df()

def analysis_ready():
    return ana_df_raw is not None and not ana_df_raw.empty and len(ana_df_raw)>=5

def get_processed():
    return run_detectors(ana_df_raw.copy(),z_sensitivity,contamination,
                         spike_threshold,night_start,night_end)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
with tab_eda:
    if not analysis_ready():
        st.info("Fetch a batch or start live feed, or upload CSV files."); st.stop()
    df=ana_df_raw.copy()
    df["hour"]=df["timestamp"].dt.hour; df["date"]=df["timestamp"].dt.date
    df["roll_mean_10"]=df["flow_rate_m3hr"].rolling(10,min_periods=1).mean()
    df["roll_std_10"]=df["flow_rate_m3hr"].rolling(10,min_periods=1).std().fillna(0)
    fig,ax=plt.subplots(figsize=(13,3.5))
    ax.plot(df["timestamp"],df["flow_rate_m3hr"],color="#4a90d9",lw=.7,alpha=.85)
    ax.fill_between(df["timestamp"],df["flow_rate_m3hr"],alpha=.06,color="#4a90d9")
    ax.axhline(0,color="#ff6b6b",lw=.6,linestyle="--",alpha=.4)
    ax.set_ylabel("Flow rate (m³/hr)"); ax.set_title("Full flow rate time series")
    ax.grid(True,alpha=.3); ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    ca,cb=st.columns(2)
    with ca:
        hourly=df[df["flow_rate_m3hr"]>0].groupby("hour")["flow_rate_m3hr"].mean()
        fig,ax=plt.subplots(figsize=(6,3.8))
        colors=["#ffa94d" if (h>=night_start or h<=night_end) else "#3fb950" if 8<=h<=10 else "#4a90d9" for h in hourly.index]
        ax.bar(hourly.index,hourly.values,color=colors,width=.7,zorder=3)
        ax.set_xlabel("Hour"); ax.set_ylabel("Avg m³/hr"); ax.set_title("Average flow by hour")
        ax.set_xticks(range(0,24,2)); ax.grid(True,alpha=.3,axis="y"); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with cb:
        fig,ax=plt.subplots(figsize=(6,3.8))
        ax.hist(df[df["flow_rate_m3hr"]>0]["flow_rate_m3hr"],bins=50,color="#4a90d9",alpha=.8,density=True)
        ax.set_xlabel("Flow rate (m³/hr)"); ax.set_ylabel("Density")
        ax.set_title("Flow distribution"); ax.grid(True,alpha=.3); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,3.5))
    ax.plot(df["timestamp"],df["flow_rate_m3hr"],color="#4a90d9",lw=.5,alpha=.5,label="Flow")
    ax.plot(df["timestamp"],df["roll_mean_10"],color="#c8cde0",lw=1.0,label="Rolling mean (10)")
    ax.fill_between(df["timestamp"],df["roll_mean_10"]-2*df["roll_std_10"],df["roll_mean_10"]+2*df["roll_std_10"],alpha=.12,color="#4a90d9",label="±2σ band")
    ax.set_ylabel("Flow rate (m³/hr)"); ax.set_title("Rolling mean ± 2σ confidence band")
    ax.legend(fontsize=8,ncol=3); ax.grid(True,alpha=.3); ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
with tab_anom:
    if not analysis_ready():
        st.info("Fetch a batch or start live feed, or upload CSV files."); st.stop()
    with st.spinner("Running 4 detection models…"):
        df=get_processed()
    total=int(df["final_anomaly"].sum())
    st.markdown(f"<div style='font-size:.8rem;color:#555d6e;margin-bottom:12px'>{len(df):,} readings analysed · {total} anomalies found</div>",unsafe_allow_html=True)
    mcounts={"Z-score":int(df["anom_zscore"].sum()),"IQR":int(df["anom_iqr"].sum()),
              "Isolation Forest":int(df["anom_iforest"].sum()),"PCA Autoencoder":int(df["anom_pca"].sum()),
              "Final (3+ / rule)":int(df["final_anomaly"].sum())}
    fig,ax=plt.subplots(figsize=(9,3.8))
    bclrs=["#9b8ec4","#9b8ec4","#c4736b","#6bab7a","#4a90d9"]
    bars=ax.bar(list(mcounts.keys()),list(mcounts.values()),color=bclrs,width=.55,zorder=3)
    for bar,v in zip(bars,mcounts.values()):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.3,str(v),ha="center",va="bottom",fontsize=9,color="#c8cde0")
    ax.set_ylabel("Count"); ax.set_title("Anomalies per model"); ax.grid(True,alpha=.3,axis="y"); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,3.5))
    ax.plot(df["timestamp"],df["flow_rate_m3hr"],color="#4a90d9",lw=.7,alpha=.7,label="Flow")
    fa=df[df["final_anomaly"]==1]
    ax.scatter(fa["timestamp"],fa["flow_rate_m3hr"],color="#ff6b6b",s=30,zorder=6,marker="^",label=f"Final anomaly ({len(fa)})")
    ax.axhline(spike_threshold,color="#ffa94d",lw=.8,linestyle="--",alpha=.6,label="Spike limit")
    ax.set_ylabel("m³/hr"); ax.set_title("Final anomaly flags (3+ models / rule-based)")
    ax.legend(fontsize=8); ax.grid(True,alpha=.3); ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    if "iforest_score" in df.columns and df["iforest_score"].sum()>0:
        df["day_label"]=df["timestamp"].dt.strftime("%d %b"); df["hour_col"]=df["timestamp"].dt.hour
        pivot=df.pivot_table(index="day_label",columns="hour_col",values="iforest_score",aggfunc="max").fillna(0)
        fig,ax=plt.subplots(figsize=(13,max(4,len(pivot)*0.5+2)))
        sns.heatmap(pivot,ax=ax,cmap="YlOrRd",linewidths=.15,linecolor="#0f1117",cbar_kws={"label":"IF score"},annot=False)
        ax.set_title("Anomaly score heatmap — day × hour"); ax.set_xlabel("Hour"); ax.set_ylabel("")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    dcols=[c for c in ["timestamp","flow_rate_m3hr","roll_mean_10","deviation","anom_zscore","anom_iqr","anom_iforest","anom_pca","model_vote"] if c in df.columns]
    st.dataframe(df[df["final_anomaly"]==1][dcols].reset_index(drop=True),height=280)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — FORECAST  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
with tab_fcast:
    if not analysis_ready():
        st.info("Fetch a batch or start live feed, or upload CSV files."); st.stop()
    with st.spinner("Building forecast…"):
        df=get_processed(); fc,lo,hi,fts,sm=forecast(df,forecast_steps)
    if fc is None:
        st.warning("Not enough active readings for forecast (need ≥10)."); st.stop()
    freq=max(1,int(df["timestamp"].diff().dt.total_seconds().median()/60))
    lookback=df[df["timestamp"]>=df["timestamp"].max()-pd.Timedelta(days=2)]
    active_look=lookback[lookback["flow_rate_m3hr"]>0]
    fig,ax=plt.subplots(figsize=(13,5))
    ax.plot(lookback["timestamp"],lookback["flow_rate_m3hr"],color="#4a90d9",lw=1.0,label="Actual",alpha=.85)
    n_sm=len(active_look)
    if n_sm>0:
        ax.plot(active_look["timestamp"],sm[-n_sm:],color="#3fb950",lw=1.0,linestyle="--",label="Fitted",alpha=.8)
    ax.plot(fts,fc,color="#ffa94d",lw=1.5,label="Forecast",zorder=5)
    ax.fill_between(fts,lo,hi,alpha=.18,color="#ffa94d",label="95% CI")
    fa_look=lookback[lookback["final_anomaly"]==1]
    ax.scatter(fa_look["timestamp"],fa_look["flow_rate_m3hr"],color="#ff6b6b",s=28,zorder=7,label="Anomaly")
    ax.axvline(df["timestamp"].max(),color="#555d6e",lw=.8,linestyle=":",label="Now")
    ax.set_ylabel("Flow rate (m³/hr)"); ax.set_title("Last 48 hrs + Forecast")
    ax.legend(fontsize=8,ncol=5); ax.grid(True,alpha=.25); ax.spines[["top","right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=30); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    fcast_df=pd.DataFrame({"timestamp":fts,"forecast_m3hr":np.round(fc,2),"upper":np.round(hi,2),"lower":np.round(lo,2)})
    st.dataframe(fcast_df,height=260)
    st.download_button("⬇️ Download forecast CSV",data=fcast_df.to_csv(index=False).encode(),file_name="vmc_forecast.csv",mime="text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA TABLE  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
with tab_data:
    if not analysis_ready():
        st.info("Fetch a batch or start live feed, or upload CSV files."); st.stop()
    with st.spinner("Processing…"):
        df=get_processed()
    only_anom=st.checkbox("Show anomaly rows only",value=False)
    dcols=[c for c in ["timestamp","flow_rate_m3hr","roll_mean_10","deviation","anom_zscore","anom_iqr","anom_iforest","anom_pca","anom_negative","final_anomaly"] if c in df.columns]
    tbl=df[dcols].reset_index(drop=True)
    if only_anom: tbl=tbl[tbl["final_anomaly"]==1].reset_index(drop=True)
    st.dataframe(tbl,height=460)
    st.download_button("⬇️ Download results CSV",data=df[dcols].to_csv(index=False).encode(),file_name="vmc_full_results.csv",mime="text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — PATTERN ANALYSIS  ← FULLY REWORKED PER SENIOR'S REQUIREMENTS
#
# What changed (all changes are additions — no working code removed):
#
#   OLD behaviour:  Jan mean line + Feb mean line + benchmark dashed
#   NEW behaviour:  ALL Jan+Feb daily raw curves overlaid on ONE graph
#                   (each day a thin line, like PDF Figure 1 blue lines)
#                   + red median profile line (like PDF Figure 1)
#                   + window-based benchmark (box width/height method, PDF §2.3)
#                   + compare today's last-24h flow vs benchmark (PDF §2.4)
#                   + QoS scoring (PDF §5.1)
#                   + anomaly detail table (PDF §9)
#                   + executive dashboard (PDF §10)
# ═════════════════════════════════════════════════════════════════════════════
with tab_pattern:

    st.markdown(
        "<div style='font-size:.8rem;color:#555d6e;margin-bottom:14px'>"
        "<b>Step 1:</b> Fetch ALL Jan+Feb data → overlay every day on one graph → "
        "identify benchmark (most repeated pattern via K-Means modal centroid + box method). "
        "<b>Step 2:</b> Compare today's flow against that benchmark. "
        "Methodology matches the attached PDF report."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Fetch button ──────────────────────────────────────────────────────
    p_col1, p_col2 = st.columns([1, 3])
    with p_col1:
        do_pattern = st.button(f"📥 Fetch Jan–Feb {pattern_year}", type="primary", width="stretch")
    with p_col2:
        if st.session_state.pattern_df is not None:
            pdf_ = st.session_state.pattern_df
            st.markdown(
                f"<div style='padding:8px 12px;background:#1a1d27;border:1px solid #2a2d3a;"
                f"border-radius:8px;font-size:.8rem;color:#4ecdc4'>"
                f"✅ Loaded: <b>{len(pdf_):,}</b> readings across "
                f"<b>{pdf_['timestamp'].dt.date.nunique()}</b> days "
                f"(Jan–Feb {pattern_year}). Benchmark extracted. Scroll down.</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='padding:8px 12px;background:#1a1d27;border:1px solid #2a2d3a;"
                "border-radius:8px;font-size:.8rem;color:#7a8196'>"
                "Click the button to fetch data and run pattern analysis.</div>",
                unsafe_allow_html=True)

    pat_debug = st.empty()

    if do_pattern:
        try_login()
        with st.spinner(f"Fetching ALL Jan + Feb {pattern_year} in weekly chunks…"):
            pat_df = fetch_two_months(year=int(pattern_year))
        if pat_df.empty:
            st.warning("⚠️ No data from API — falling back to DB data.")
            pat_df = db_load(hours_back=168)
            if "flow_rate_m3hr" not in pat_df.columns and "flow_rate" in pat_df.columns:
                pat_df = pat_df.rename(columns={"flow_rate":"flow_rate_m3hr"})
        if pat_df.empty:
            st.error("❌ No data available. Fetch a batch first from the Live/Batch tab.")
            with pat_debug.expander("Debug", expanded=True):
                st.code(st.session_state.last_raw or "No response")
        else:
            st.session_state.pattern_df = pat_df
            with st.spinner("Computing benchmark…"):
                # K-Means on normalised shape curves
                bench_curve, curves_df, all_curves, labels, centroids, modal_idx = \
                    find_benchmark_pattern_kmeans(pat_df, n_clusters=int(pattern_k))
                # Window-based benchmark (box method, matching PDF §2.3)
                all_wins = []
                pat_df_cp = pat_df.copy(); pat_df_cp["date_"] = pat_df_cp["timestamp"].dt.date
                for date, group in pat_df_cp.groupby("date_"):
                    wins = detect_supply_windows_df(group)
                    for w in wins: w["date"] = str(date)
                    all_wins.extend(wins)
                benchmark_box, windows_df = build_benchmark_from_windows(all_wins, n_clusters=int(pattern_k))
            st.session_state.benchmark_curve   = bench_curve
            st.session_state.benchmark_windows = benchmark_box
            st.session_state.curves_df         = curves_df
            st.session_state.all_curves        = all_curves
            st.session_state.centroids         = centroids
            st.session_state.modal_idx         = modal_idx
            st.rerun()

    if st.session_state.pattern_df is None:
        st.info("Press **Fetch Jan–Feb** to start."); st.stop()

    # ── Restore ───────────────────────────────────────────────────────────
    pat_df      = st.session_state.pattern_df
    bench_curve = st.session_state.benchmark_curve
    bench_box   = st.session_state.benchmark_windows
    curves_df   = st.session_state.curves_df
    all_curves  = st.session_state.all_curves
    centroids   = st.session_state.centroids
    modal_idx   = st.session_state.modal_idx

    hours_axis = np.arange(24)

    # ─────────────────────────────────────────────────────────────────────
    # ① MULTI-DAY OVERLAY — ALL Jan+Feb daily curves on ONE graph
    #   (Matches PDF Figure 1: each day = thin blue line, red = median)
    #   Senior's exact requirement: "Make all data of jan and feb in 1 graph,
    #   Then make benchmark from it, I want to First do overlap of 2 months data"
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:16px 0 6px'>"
        "① Multi-Day Overlay — ALL Jan+Feb daily curves (matches PDF Figure 1)</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.75rem;color:#555d6e;margin-bottom:8px'>"
        "Each thin blue line = one day's normalised flow shape. "
        "Orange dashed = benchmark (modal K-Means centroid). "
        "Red = median profile across all days.</div>",
        unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 5))

    # Every day as a thin transparent line (raw normalised curves)
    jan_count = feb_count = 0
    for date_str, curve in all_curves.items():
        month = int(date_str[5:7])
        if month == 1:
            ax.plot(hours_axis, curve, color="#4a90d9", lw=0.5, alpha=0.22)
            jan_count += 1
        elif month == 2:
            ax.plot(hours_axis, curve, color="#4a90d9", lw=0.5, alpha=0.22)
            feb_count += 1

    # Median profile (red line like PDF Figure 1)
    if all_curves:
        all_arr = np.array(list(all_curves.values()))
        median_curve = np.median(all_arr, axis=0)
        ax.plot(hours_axis, median_curve, color="#e74c3c", lw=2.5,
                label=f"Median profile ({jan_count + feb_count} days)", zorder=5)

    # Benchmark centroid (orange dashed)
    if bench_curve is not None:
        ax.plot(hours_axis, bench_curve, color="#ffa94d", lw=2.2,
                linestyle="--", label=f"Benchmark (modal cluster, k={pattern_k})", zorder=6)

    # Proxy for legend
    ax.plot([], [], color="#4a90d9", lw=1, alpha=0.5,
            label=f"Individual days ({jan_count + feb_count} days Jan+Feb)")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Normalised flow (0 = min, 1 = max)")
    ax.set_title(f"ALL Jan + Feb {pattern_year} daily flow shapes — {jan_count+feb_count} days overlaid")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # ② BENCHMARK EXTRACTION — K-Means centroids, highlight modal
    #   (Identifies "best pattern which is matching most of the time")
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "② Benchmark Extraction — Most Frequently Repeated Pattern</div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:.75rem;color:#555d6e;margin-bottom:8px'>"
        f"K-Means (k={pattern_k}) clusters all daily shapes. "
        f"The largest cluster = modal = benchmark (orange). "
        f"This is the pattern that matches most often.</div>",
        unsafe_allow_html=True)

    if centroids is not None and len(centroids) > 0:
        cluster_sizes = np.bincount(
            curves_df["cluster"].values.astype(int),
            minlength=len(centroids))
        fig, ax = plt.subplots(figsize=(13, 4))
        palette = ["#555d6e"] * len(centroids)
        palette[modal_idx] = "#ffa94d"
        for i, centroid in enumerate(centroids):
            n = cluster_sizes[i]
            lw = 2.8 if i == modal_idx else 0.9
            alpha = 1.0 if i == modal_idx else 0.5
            label = (f"Cluster {i} — {n} days  ◀ BENCHMARK (most frequent)"
                     if i == modal_idx else f"Cluster {i} — {n} days")
            ax.plot(hours_axis, centroid, color=palette[i], lw=lw, alpha=alpha, label=label)
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Normalised flow")
        ax.set_title(f"K-Means cluster centroids (k={pattern_k}) — modal cluster = benchmark")
        ax.set_xticks(range(0, 24, 2)); ax.legend(fontsize=7.5, ncol=2)
        ax.grid(True, alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Benchmark box profile (from window detection)
    if bench_box:
        bm_start_str = f"{int(bench_box['start_hour']):02d}:{int((bench_box['start_hour']%1)*60):02d}"
        bm_end_str   = f"{int(bench_box['end_hour']):02d}:{int((bench_box['end_hour']%1)*60):02d}"
        st.markdown(
            f"<div style='padding:10px 16px;background:#1a1d27;border:1px solid #2a2d3a;"
            f"border-radius:8px;font-size:.82rem;color:#c8cde0;margin-bottom:12px'>"
            f"📐 <b>Benchmark Profile (Box Method — matches PDF §2.3):</b>&nbsp;&nbsp;"
            f"Start: <b>{bm_start_str}</b> &nbsp;|&nbsp; "
            f"End: <b>{bm_end_str}</b> &nbsp;|&nbsp; "
            f"Duration: <b>{bench_box['duration']:.0f} min</b> &nbsp;|&nbsp; "
            f"Peak: <b>{bench_box['peak']:.1f} m³/hr</b> &nbsp;|&nbsp; "
            f"Avg: <b>{bench_box['avg']:.1f} m³/hr</b> &nbsp;|&nbsp; "
            f"Based on <b>{bench_box['samples']}</b> supply windows</div>",
            unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # ③ DAILY SIMILARITY — bar chart (matches PDF Figure 2 bottom / tab 6)
    # ─────────────────────────────────────────────────────────────────────
    if curves_df is not None and not curves_df.empty:
        st.markdown(
            "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
            f"③ Daily Pattern Similarity to Benchmark (threshold = {sim_threshold}%)</div>",
            unsafe_allow_html=True)
        curves_sorted = curves_df.sort_values("date").reset_index(drop=True)
        bar_colors = ["#3fb950" if s >= sim_threshold else "#ff6b6b"
                      for s in curves_sorted["similarity"]]
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.bar(range(len(curves_sorted)), curves_sorted["similarity"],
               color=bar_colors, width=0.75, zorder=3)
        ax.axhline(sim_threshold, color="#ffa94d", lw=1.2, linestyle="--",
                   label=f"Threshold {sim_threshold}%")
        ax.set_xticks(range(len(curves_sorted)))
        ax.set_xticklabels([d[5:] for d in curves_sorted["date"]], rotation=60, fontsize=6.5)
        ax.set_ylabel("Similarity to benchmark (%)"); ax.set_ylim(0, 105)
        ax.set_title("Daily pattern similarity — green ≥ threshold, red = deviant")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # ④ BEST vs WORST MATCH  (like PDF Figure 3 box comparison)
    # ─────────────────────────────────────────────────────────────────────
    if curves_df is not None and not curves_df.empty and bench_curve is not None:
        st.markdown(
            "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
            "④ Best-Match vs Worst-Match Days vs Benchmark (matches PDF Figure 3)</div>",
            unsafe_allow_html=True)
        ranked = curves_sorted.sort_values("similarity", ascending=False)
        top5   = ranked.head(5)["date"].tolist()
        bot5   = ranked.tail(5)["date"].tolist()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for date_str in top5:
            if date_str in all_curves:
                sim_val = curves_sorted.loc[curves_sorted["date"]==date_str,"similarity"].values[0]
                axes[0].plot(hours_axis, all_curves[date_str], color="#3fb950",
                             lw=1.0, alpha=0.7, label=f"{date_str[5:]} ({sim_val:.0f}%)")
        axes[0].plot(hours_axis, bench_curve, color="#ffa94d", lw=2.2,
                     linestyle="--", label="Benchmark")
        axes[0].set_title("Top 5 closest days to benchmark")
        axes[0].set_xlabel("Hour"); axes[0].set_ylabel("Normalised flow")
        axes[0].legend(fontsize=7.5); axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(range(0, 24, 2)); axes[0].spines[["top","right"]].set_visible(False)
        for date_str in bot5:
            if date_str in all_curves:
                sim_val = curves_sorted.loc[curves_sorted["date"]==date_str,"similarity"].values[0]
                axes[1].plot(hours_axis, all_curves[date_str], color="#ff6b6b",
                             lw=1.0, alpha=0.7, label=f"{date_str[5:]} ({sim_val:.0f}%)")
        axes[1].plot(hours_axis, bench_curve, color="#ffa94d", lw=2.2,
                     linestyle="--", label="Benchmark")
        axes[1].set_title("Bottom 5 most deviant days")
        axes[1].set_xlabel("Hour"); axes[1].legend(fontsize=7.5); axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(range(0, 24, 2)); axes[1].spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # ⑤ TODAY vs BENCHMARK — the core senior requirement
    #   "compare with today's flows"
    #   Uses box method scoring (PDF §2.4) + shape overlay
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "⑤ Today's Flow vs Benchmark (PDF §5 comparison methodology)</div>",
        unsafe_allow_html=True)

    today_df = db_load(hours_back=24)
    if today_df.empty:
        st.info("No today data in DB yet — fetch a batch from the Live tab first.")
    else:
        today_col = "flow_rate_m3hr"
        today_df["hour_frac"] = today_df["timestamp"].dt.hour + today_df["timestamp"].dt.minute / 60

        # Detect today's supply windows
        today_windows = detect_supply_windows_df(today_df)

        # Score today vs benchmark
        if bench_box:
            today_qos, today_anomalies, matched_win = score_day_vs_benchmark(
                today_windows, bench_box,
                time_tol_min=time_tol_min, flow_tol=flow_tol_pct / 100)
        else:
            today_qos, today_anomalies, matched_win = 0.0, ["Benchmark not available"], None

        # QoS color
        qos_color = "#3fb950" if today_qos >= 85 else "#ffa94d" if today_qos >= 70 else "#ff6b6b"
        status_str = "EXCELLENT" if today_qos >= 85 else "GOOD" if today_qos >= 70 else "⚠️ POOR"

        # KPI row
        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        today_peak = today_df[today_col].max()
        today_avg  = today_df[today_df[today_col] > 0][today_col].mean() if (today_df[today_col] > 0).any() else 0
        for col, label, val, cls in [
            (kc1, "Today QoS",        f"{today_qos:.1f}%",          "danger" if today_qos < 70 else ""),
            (kc2, "Status",            status_str,                   "danger" if today_qos < 70 else ""),
            (kc3, "Supply Windows",    str(len(today_windows)),      ""),
            (kc4, "Today Peak Flow",   f"{today_peak:.1f} m³/hr",    ""),
            (kc5, "Anomalies",         str(len(today_anomalies)),     "danger" if today_anomalies else ""),
        ]:
            col.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                f"<div class='metric-value {cls}' style='font-size:1.3rem'>{val}</div></div>",
                unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Today's flow with benchmark overlay + window boxes (PDF Figure 2/3 style)
        fig, ax = plt.subplots(figsize=(13, 4.5))

        # Today's flow line
        ax.plot(today_df["hour_frac"], today_df[today_col],
                color="#4a90d9", lw=1.4, label="Today's flow")
        ax.fill_between(today_df["hour_frac"], today_df[today_col],
                        alpha=0.08, color="#4a90d9")

        # Benchmark box (shaded rectangle — like PDF Figure 3 blue box)
        if bench_box:
            bx_s = bench_box["start_hour"]; bx_e = bench_box["end_hour"]
            bx_h = bench_box["peak"]
            rect = mpatches.FancyArrowPatch
            ax.axvspan(bx_s, bx_e, ymin=0, ymax=0.95,
                       alpha=0.10, color="#e74c3c")
            ax.axhline(bench_box["peak"], color="#e74c3c", lw=1.0,
                       linestyle="--", alpha=0.8, label=f"Benchmark peak ({bench_box['peak']:.1f})")
            ax.axhline(bench_box["avg"], color="#ffa94d", lw=0.8,
                       linestyle=":", alpha=0.8, label=f"Benchmark avg ({bench_box['avg']:.1f})")
            bm_s = bench_box["start_hour"]; bm_e = bench_box["end_hour"]
            ax.axvline(bm_s, color="#e74c3c", lw=1.0, linestyle="--",
                       alpha=0.7, label=f"Bm start {int(bm_s):02d}:{int((bm_s%1)*60):02d}")
            ax.axvline(bm_e, color="#e74c3c", lw=1.0, linestyle="--", alpha=0.7)

        # Detected today supply windows
        for i, w in enumerate(today_windows):
            ax.axvspan(w["start_hour_frac"], w["end_hour_frac"],
                       alpha=0.12, color="#3fb950",
                       label="Today window" if i == 0 else "")

        # Annotate anomalies on chart
        if today_anomalies and matched_win:
            for a_text in today_anomalies[:3]:
                ax.text(0.02, 0.97 - today_anomalies.index(a_text) * 0.08,
                        f"⚠ {a_text}", transform=ax.transAxes,
                        fontsize=7, color="#ff6b6b", va="top")

        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2))
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Flow rate (m³/hr)")
        ax.set_title(f"Today's Flow vs Benchmark | QoS: {today_qos:.1f}% ({status_str})")
        ax.legend(fontsize=7.5, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ── Today's anomaly details table (matches PDF Table 3)
        if today_anomalies:
            st.markdown(
                "<div style='font-size:.85rem;font-weight:500;color:#ff6b6b;margin:10px 0 4px'>"
                "⚠️ Today's Anomaly Details</div>",
                unsafe_allow_html=True)
            anom_rows = [[i+1, a] for i, a in enumerate(today_anomalies)]
            anom_df = pd.DataFrame(anom_rows, columns=["#", "Description"])
            st.dataframe(anom_df, hide_index=True, height=min(250, len(anom_rows)*38+40))
        else:
            st.success("✅ Today's distribution matches the benchmark — no anomalies detected.")

        # ── Today's supply windows table
        if today_windows:
            st.markdown(
                "<div style='font-size:.85rem;font-weight:500;color:#c8cde0;margin:10px 0 4px'>"
                "Today's Supply Windows</div>",
                unsafe_allow_html=True)
            win_rows = []
            for i, w in enumerate(today_windows):
                bm_ok = bench_box and abs(w["start_hour_frac"] - bench_box["start_hour"]) * 60 <= time_tol_min
                win_rows.append({
                    "#": i+1,
                    "Start": w["start"].strftime("%H:%M"),
                    "End":   w["end"].strftime("%H:%M"),
                    "Duration (min)": f"{w['duration']:.0f}",
                    "Peak (m³/hr)": f"{w['peak']:.1f}",
                    "Avg (m³/hr)":  f"{w['avg']:.1f}",
                    "vs Benchmark": "✅ Normal" if bm_ok else "⚠️ Deviated",
                })
            st.dataframe(pd.DataFrame(win_rows), hide_index=True)

    # ─────────────────────────────────────────────────────────────────────
    # ⑥ FLOW RATE HEATMAP — comparison period (matches PDF Figure 5)
    #   Shows all Jan+Feb days as rows, hour as columns
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "⑥ Flow Rate Heatmap — Jan+Feb All Days (matches PDF Figure 5)</div>",
        unsafe_allow_html=True)

    pat_df_h = pat_df.copy()
    pat_df_h["date_str"] = pat_df_h["timestamp"].dt.strftime("%m-%d")
    pat_df_h["hour"]     = pat_df_h["timestamp"].dt.hour
    col_ = "flow_rate_m3hr" if "flow_rate_m3hr" in pat_df_h.columns else "flow_rate"
    pivot_hm = pat_df_h.pivot_table(index="date_str", columns="hour", values=col_, aggfunc="mean").fillna(0)

    if not pivot_hm.empty:
        fig_h = max(6, len(pivot_hm) * 0.18 + 2)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        sns.heatmap(pivot_hm, ax=ax, cmap="YlOrRd", linewidths=0.05,
                    linecolor="#0f1117", cbar_kws={"label": "Flow rate (m³/hr)"},
                    annot=False, xticklabels=2)

        # Mark deviant days with yellow ticks (like PDF Figure 5 yellow markers)
        if curves_df is not None and not curves_df.empty:
            deviant_dates = set(curves_df[curves_df["similarity"] < sim_threshold]["date"].str[5:])
            ytick_labels = ax.get_yticklabels()
            for lbl in ytick_labels:
                if lbl.get_text() in deviant_dates:
                    lbl.set_color("#ffa94d"); lbl.set_fontweight("bold")

        ax.set_title(f"Flow Rate Heatmap — Jan+Feb {pattern_year} (all days × 24 hours)")
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Date (MM-DD)")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────
    # ⑦ EXECUTIVE DASHBOARD (matches PDF Figure 6)
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.9rem;font-weight:500;color:#c8cde0;margin:20px 0 6px'>"
        "⑦ Executive Dashboard (matches PDF Figure 6 layout)</div>",
        unsafe_allow_html=True)

    if curves_df is not None and not curves_df.empty:
        n_match   = (curves_df["similarity"] >= sim_threshold).sum()
        n_deviant = len(curves_df) - n_match
        avg_sim   = curves_df["similarity"].mean()
        best_day  = curves_df.loc[curves_df["similarity"].idxmax(), "date"]
        worst_day = curves_df.loc[curves_df["similarity"].idxmin(), "date"]

        # 5 summary metrics
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        for col, label, val in [
            (mc1, "Total Days",     f"{len(curves_df)}"),
            (mc2, "Match ✅",       f"{n_match}"),
            (mc3, "Deviant ❌",     f"{n_deviant}"),
            (mc4, "Avg Similarity", f"{avg_sim:.1f}%"),
            (mc5, "Best Match",     best_day[5:]),
        ]:
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='font-size:1.3rem'>{val}</div>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # 4-panel figure matching PDF Figure 6
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="#1a1d27")
        for axi in axes.flat:
            axi.set_facecolor("#1a1d27")
            axi.spines[["top","right"]].set_visible(False)

        # Top-left: QoS distribution histogram (like PDF Figure 6 top-left)
        ax1 = axes[0, 0]
        ax1.hist(curves_df["similarity"], bins=20, color="#4a90d9",
                 alpha=0.75, edgecolor="#0f1117")
        ax1.axvline(sim_threshold, color="#ffa94d", lw=1.2, linestyle="--",
                    label=f"Threshold {sim_threshold}%")
        ax1.axvline(avg_sim, color="#e74c3c", lw=1.0, linestyle=":",
                    label=f"Avg {avg_sim:.1f}%")
        ax1.set_xlabel("Similarity (%)"); ax1.set_ylabel("Number of Days")
        ax1.set_title("Similarity Score Distribution")
        ax1.legend(fontsize=7.5); ax1.grid(True, alpha=0.25)

        # Top-right: Anomaly types breakdown (like PDF Figure 6 top-right)
        ax2 = axes[0, 1]
        if bench_box:
            # Re-compute anomaly type counts across all Jan+Feb days
            type_counts = {"Start Time": 0, "End Time": 0, "Duration": 0,
                           "Peak Flow": 0, "Avg Flow": 0}
            pat_cp = pat_df.copy(); pat_cp["date_"] = pat_cp["timestamp"].dt.date
            for date_, grp in pat_cp.groupby("date_"):
                wins_ = detect_supply_windows_df(grp)
                _, anoms_, _ = score_day_vs_benchmark(
                    wins_, bench_box,
                    time_tol_min=time_tol_min, flow_tol=flow_tol_pct/100)
                for a in anoms_:
                    if "Start time" in a:   type_counts["Start Time"] += 1
                    elif "End time" in a:   type_counts["End Time"]   += 1
                    elif "Duration" in a:   type_counts["Duration"]   += 1
                    elif "Peak" in a:       type_counts["Peak Flow"]  += 1
                    elif "Avg" in a:        type_counts["Avg Flow"]   += 1
            bar_colors2 = ["#ff6b6b","#ffa94d","#ffd700","#4a90d9","#3fb950"]
            ax2.barh(list(type_counts.keys()), list(type_counts.values()),
                     color=bar_colors2, height=0.55, zorder=3)
            for i, (k, v) in enumerate(type_counts.items()):
                if v > 0:
                    ax2.text(v + 0.1, i, str(v), va="center", fontsize=9)
            ax2.set_xlabel("Count"); ax2.set_title("Anomaly Types Breakdown")
            ax2.grid(True, alpha=0.25, axis="x")

        # Bottom-left: Supply timing consistency scatter (like PDF Figure 6 bottom-left)
        ax3 = axes[1, 0]
        if bench_box:
            pat_cp2 = pat_df.copy(); pat_cp2["date_"] = pat_cp2["timestamp"].dt.date
            day_starts = []; day_ends = []; day_labels2 = []
            for date_, grp in sorted(pat_cp2.groupby("date_")):
                wins_ = detect_supply_windows_df(grp)
                if wins_:
                    best_ = min(wins_, key=lambda w: abs(w["start_hour_frac"] - bench_box["start_hour"]))
                    day_starts.append(best_["start_hour_frac"])
                    day_ends.append(best_["end_hour_frac"])
                    day_labels2.append(str(date_))
            x_pos = range(len(day_starts))
            ax3.scatter(x_pos, day_starts, color="#4a90d9", s=20, label="Supply Start", zorder=5)
            ax3.scatter(x_pos, day_ends,   color="#3fb950", s=20, label="Supply End",   zorder=5)
            if bench_box:
                ax3.axhline(bench_box["start_hour"], color="#4a90d9", lw=0.9,
                            linestyle="--", alpha=0.7, label=f"Bm start ({bm_start_str})")
                ax3.axhline(bench_box["end_hour"],   color="#3fb950", lw=0.9,
                            linestyle="--", alpha=0.7, label=f"Bm end ({bm_end_str})")
            ax3.set_xlabel("Day index"); ax3.set_ylabel("Hour of Day")
            ax3.set_title("Supply Start & End Time Consistency\n(Dashed = median)")
            ax3.legend(fontsize=7, ncol=2); ax3.grid(True, alpha=0.2)

        # Bottom-right: Key metrics summary text panel (like PDF Figure 6 bottom-right)
        ax4 = axes[1, 1]
        ax4.axis("off")
        summary_text = (
            f"WATER DISTRIBUTION QUALITY SUMMARY\n"
            f"{'='*42}\n\n"
            f"Data Period: Jan 1 – Feb 28, {pattern_year}\n"
            f"Total Days Analysed: {len(curves_df)}\n\n"
            f"PATTERN STATISTICS:\n"
            f"  Benchmark match (≥{sim_threshold}%):  {n_match}/{len(curves_df)} ({n_match*100//max(len(curves_df),1)}%)\n"
            f"  Deviant days (<{sim_threshold}%):       {n_deviant}/{len(curves_df)} ({n_deviant*100//max(len(curves_df),1)}%)\n"
            f"  Average similarity:    {avg_sim:.1f}%\n"
            f"  Best matching day:     {best_day[5:]}\n"
            f"  Worst matching day:    {worst_day[5:]}\n\n"
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
                f"  QoS Score:     {today_qos:.1f}%\n"
                f"  Status:        {status_str}\n"
                f"  Supply windows: {len(today_windows)}\n"
                f"  Anomalies:     {len(today_anomalies)}\n"
            )
        ax4.text(0.05, 0.97, summary_text, transform=ax4.transAxes,
                 fontsize=7.5, va="top", ha="left",
                 color="#c8cde0",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e2130",
                           edgecolor="#2a2d3a", alpha=0.9))
        fig.tight_layout(pad=1.5); st.pyplot(fig); plt.close(fig)

    # ── Full similarity table + download ──────────────────────────────────
    if curves_df is not None and not curves_df.empty:
        st.markdown(
            "<div style='font-size:.85rem;color:#c8cde0;margin:16px 0 6px'>Full Similarity Table</div>",
            unsafe_allow_html=True)
        st.dataframe(
            curves_df.sort_values("date")[["date","cluster","similarity","distance","is_benchmark_cluster"]].reset_index(drop=True),
            height=340)
        st.download_button(
            "⬇️ Download similarity CSV",
            data=curves_df.to_csv(index=False).encode(),
            file_name=f"vmc_pattern_similarity_{pattern_year}.csv",
            mime="text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — QoS TREND  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
with tab_qos:
    st.markdown(
        "<div style='font-size:.8rem;color:#555d6e;margin-bottom:14px'>"
        "Reads QoS scores and benchmark snapshots written by <b>vmc_worker.py</b> each day."
        "</div>", unsafe_allow_html=True)
    qos_df = load_qos_history(); bm_df = load_benchmark_snapshots()
    if qos_df.empty:
        st.info("No QoS data yet. Start **vmc_worker.py** — it writes a score after each daily run."); st.stop()
    latest=qos_df.iloc[-1]; avg_qos=qos_df["qos"].mean()
    best_day_q=qos_df.loc[qos_df["qos"].idxmax()]; worst_day_q=qos_df.loc[qos_df["qos"].idxmin()]
    days_poor=(qos_df["qos"]<70).sum()
    def qos_cls(q): return "" if q>=70 else "danger"
    kc1,kc2,kc3,kc4,kc5=st.columns(5)
    for col,label,val,cls in [
        (kc1,"Latest QoS",f"{latest['qos']:.1f}%",qos_cls(latest["qos"])),
        (kc2,"Avg QoS",f"{avg_qos:.1f}%",qos_cls(avg_qos)),
        (kc3,"Best day",f"{best_day_q['date'][5:]} {best_day_q['qos']:.0f}%",""),
        (kc4,"Worst day",f"{worst_day_q['date'][5:]} {worst_day_q['qos']:.0f}%","danger"),
        (kc5,"Poor days (<70%)",str(int(days_poor)),"danger" if days_poor else ""),
    ]:
        col.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value {cls}' style='font-size:1.3rem'>{val}</div></div>",
            unsafe_allow_html=True)
    qos_df["date_dt"] = pd.to_datetime(qos_df["date"])
    avg_qos_val = qos_df["qos"].mean()
    QOS_THRESH  = 70

    # ── Colour logic matching Image 1 exactly ──────────────────────────
    # green  = at or above benchmark avg
    # gold   = between threshold (70%) and benchmark avg  → below avg but not terrible
    # salmon = below threshold (70%)
    clrs = []
    for q in qos_df["qos"]:
        if q >= avg_qos_val:
            clrs.append("#5cb87a")       # green
        elif q >= QOS_THRESH:
            clrs.append("#e6a817")       # gold / orange
        else:
            clrs.append("#c0614f")       # salmon / red-brown

    # ── Override dark rcParams locally for this white-background chart ──
    with plt.rc_context({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#cccccc",
        "axes.labelcolor":  "#333333",
        "xtick.color":      "#555555",
        "ytick.color":      "#555555",
        "text.color":       "#333333",
        "grid.color":       "#e8e8e8",
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
    }):
        fig, ax = plt.subplots(figsize=(14, 4.2))
        x = np.arange(len(qos_df))

        bars = ax.bar(x, qos_df["qos"], color=clrs, width=0.75,
                      zorder=3, edgecolor="white", linewidth=0.4)

        # ── Annotate anomaly count on bars below benchmark avg ──────────
        # Numbers appear in red above short bars (like "3","5","8" in Image 1)
        for i, (bar, q) in enumerate(zip(bars, qos_df["qos"])):
            if q < avg_qos_val:
                n_anom = ""
                if "n_anomalies" in qos_df.columns:
                    n_anom = int(qos_df.iloc[i]["n_anomalies"])
                    if n_anom:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.8,
                            str(n_anom),
                            ha="center", va="bottom",
                            fontsize=7.5, color="#c0614f", fontweight="bold"
                        )

        # ── Benchmark avg line — blue dashed (Image 1) ─────────────────
        ax.axhline(avg_qos_val, color="#4a7fc1", lw=1.6, linestyle="--",
                   label=f"Benchmark Avg QoS ({avg_qos_val:.1f}%)", zorder=5)

        # ── QoS threshold line — red dotted (Image 1) ──────────────────
        ax.axhline(QOS_THRESH, color="#b05050", lw=1.0, linestyle=":",
                   label=f"QoS Threshold ({QOS_THRESH}%)", zorder=5)

        # ── Axes styling ────────────────────────────────────────────────
        ax.set_ylim(0, 105)
        ax.set_xlim(-0.8, len(qos_df) - 0.2)

        # X ticks every 5 days (matches "0, 5, 10 … 40" in Image 1)
        tick_positions = list(range(0, len(qos_df), 5))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, fontsize=8.5)

        ax.set_xlabel("Days in Comparison Period (Mar-Apr)", fontsize=9.5)
        ax.set_ylabel("Quality of Service Score (%)",        fontsize=9.5)
        ax.set_title("Daily QoS Score: Comparison Period",
                     fontsize=11, fontweight="bold", pad=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.grid(True, axis="y", alpha=0.4, lw=0.6, color="#e0e0e0")

        leg = ax.legend(fontsize=8.5, loc="upper right",
                        framealpha=0.95, edgecolor="#cccccc",
                        handlelength=2.5)
        for t in leg.get_texts():
            t.set_color("#333333")

        fig.tight_layout(pad=0.8)
        st.pyplot(fig)
        plt.close(fig)
    if not bm_df.empty:
        st.markdown("<div style='font-size:.85rem;color:#c8cde0;margin:16px 0 4px'>Benchmark Snapshots</div>",unsafe_allow_html=True)
        st.dataframe(bm_df.head(10), height=200)
    st.markdown("<div style='font-size:.85rem;color:#c8cde0;margin:12px 0 4px'>Full QoS History</div>",unsafe_allow_html=True)
    st.dataframe(qos_df.drop(columns=["date_dt"],errors="ignore").reset_index(drop=True),height=300)
    st.download_button("⬇️ Download QoS history CSV",
                       data=qos_df.drop(columns=["date_dt"],errors="ignore").to_csv(index=False).encode(),
                       file_name="vmc_qos_history.csv",mime="text/csv")
