"""
VMC Water Flow — Background Worker (Telegram Edition) — BATCH MODE

Run:  python vmc_worker.py

KEY CHANGE from v1:
  • No per-minute polling. Instead, once every 24 hours the worker
    fetches the full 24-hour batch from the VMC API in a single call,
    stores ALL rows into SQLite, runs analysis, and sends the Telegram
    report + chart.
  • An optional lightweight "heartbeat" poll (configurable, default OFF)
    can still store a single reading hourly if you want the Streamlit
    dashboard to stay warm — set HEARTBEAT_ENABLED = True.
  • PDF report is now generated and sent via Telegram daily.

KEY ADDITIONS (matching Streamlit dashboard — zero original code removed):
  • run_full_detectors()   — 4-model anomaly detector (Z-score + IQR +
                             Isolation Forest + PCA), same logic as tab 3.
  • forecast_flow()        — exponential smoothing + 95% CI, same as tab 4.
  • make_eda_charts()      — time series, rolling mean ±2σ, hourly bar,
                             distribution histogram (tab 2).
  • make_anomaly_charts()  — model comparison bar, timeline overlay,
                             model-by-model panel, anomaly heatmap (tab 3).
  • make_forecast_chart()  — forecast + residuals chart (tab 4).
  • Pattern analysis block — fetch_two_months(), normalize_daily_curve(),
                             curve_distance(), find_benchmark_pattern(),
                             Jan/Feb overlap, K-Means centroids, similarity
                             bar, best/worst overlay (tab 6).
  • All new charts are embedded in the PDF and sent via Telegram.

Setup (5 minutes):
  1. Open Telegram → search @BotFather
  2. Send:  /newbot
  3. Give it a name: VMC Monitor Bot
  4. Give it a username: vmc_mjp4231_bot
  5. Copy the token → paste in BOT_TOKEN below
  6. Create a group "VMC MJP-4231 Alerts"
  7. Add your bot to that group
  8. Run:  python get_chat_id.py   ← (separate helper script)
  9. Paste the chat_id in TELEGRAM_CHAT_IDS below
  10. Done — run this worker!

Dependencies:
  pip install requests urllib3 apscheduler scipy numpy pandas tzdata reportlab scikit-learn
"""

import sqlite3
import time
import logging
import signal
import sys
import os
import io
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import urllib3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — fill these in
# ─────────────────────────────────────────────────────────────────────────────

# --- Telegram ---
BOT_TOKEN = "8651505010:AAFez7NjjAp31nOSADop22tUAxy3WyYrxdY"

TELEGRAM_CHAT_IDS = [
    "871141199",      # ✅ your personal chat (amar nath)
    #"-5296731596",    # ✅ Anamoly report group
]

# --- VMC API ---
VMC_BASE         = "https://scph1.vmcsmartwater.in:9090"

HISTORY_API_PATHS = [
    "/ph1/data",
    "/api/history/sensor/Flow/Rate",
    "/api/sensor/Flow/Rate/history",
    "/api/realtime/sensor/Flow/Rate",
]

REALTIME_API_PATH = "/api/realtime/sensor/Flow/Rate"

METER_LABEL  = "AIB_FT015"          # human-readable label for PDF/reports
STATION_NAME = "VMC MJP-5445"      # used in Telegram messages
OBJECT_NAME = "AIB_FT015"
VMC_USER    = "7644881557"
VMC_PASS    = "5678"

# --- SQLite (shared with Streamlit dashboard) ---
DB_PATH = "vmc_readings.db"

# --- Batch fetch window (hours) ---
BATCH_WINDOW_HOURS = 24

# --- Optional hourly heartbeat ---
HEARTBEAT_ENABLED     = False
HEARTBEAT_INTERVAL_HR = 1

# --- Anomaly thresholds ---
SPIKE_THRESHOLD  = 6000   # adjust based on FMA_5445 max expected flow
NIGHT_FLOW_LIMIT = 500    # adjust for FMA_5445 night baseline
Z_THRESHOLD      = 3.0
NIGHT_START_HR   = 23
NIGHT_END_HR     = 5


# --- Full-detector settings (mirrors Streamlit sidebar defaults) ---
CONTAMINATION    = 0.05   # Isolation Forest contamination
Z_SENSITIVITY    = 3.0    # Z-score threshold for full detector
FORECAST_STEPS   = 30     # Exponential smoothing steps ahead

# --- Pattern analysis settings ---
PATTERN_YEAR     = 2026   # Year to fetch Jan–Feb for pattern analysis
PATTERN_K        = 6      # K-Means clusters
SIM_THRESHOLD    = 75     # Similarity % threshold for match/deviant

# --- Daily report time (IST) ---
REPORT_HOUR   = 6
REPORT_MINUTE = 0


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("vmc_worker.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("vmc_worker")
IST = ZoneInfo("Asia/Kolkata")
IST_OFFSET = timedelta(hours=5, minutes=30)

plt.rcParams.update({
    "figure.facecolor": "#1a1d27", "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#2a2d3a",   "axes.labelcolor": "#7a8196",
    "xtick.color": "#555d6e",      "ytick.color": "#555d6e",
    "grid.color": "#23263a",       "text.color": "#c8cde0",
    "font.family": "sans-serif",   "font.size": 9,
})

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"


# ─────────────────────────────────────────────────────────────────────────────
# SQLITE
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT NOT NULL,
            flow_rate  REAL NOT NULL,
            is_anomaly INTEGER DEFAULT 0
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS sent_reports (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            sent_at  TEXT NOT NULL,
            report   TEXT NOT NULL,
            type     TEXT NOT NULL
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS qos_scores (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL UNIQUE,
            qos             REAL NOT NULL,
            total_readings  INTEGER NOT NULL,
            total_anomalies INTEGER NOT NULL,
            spike_anomalies INTEGER NOT NULL,
            night_anomalies INTEGER NOT NULL,
            supply_windows  INTEGER NOT NULL,
            avg_flow        REAL NOT NULL,
            peak_flow       REAL NOT NULL,
            benchmark_used  INTEGER NOT NULL DEFAULT 0,
            status          TEXT NOT NULL
        )""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_snapshot (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            saved_at     TEXT NOT NULL,
            start_min    REAL,
            end_min      REAL,
            duration_min REAL,
            peak_flow    REAL,
            avg_flow     REAL,
            samples      INTEGER,
            method       TEXT NOT NULL
        )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON readings(timestamp)")
    con.commit()
    con.close()
    log.info("DB ready: %s", DB_PATH)


def db_insert_batch(rows: list):
    if not rows:
        return
    con = sqlite3.connect(DB_PATH)
    con.executemany(
        "INSERT OR IGNORE INTO readings (timestamp, flow_rate, is_anomaly) VALUES (?,?,?)",
        rows,
    )
    con.commit()
    con.close()
    log.info("Batch inserted %d rows into DB", len(rows))


def db_insert(ts: datetime, flow: float, anom: int):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO readings (timestamp, flow_rate, is_anomaly) VALUES (?,?,?)",
        (ts.isoformat(), flow, anom),
    )
    con.commit()
    con.close()


def db_load_hours(hours: int) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    df = pd.read_sql(
        "SELECT timestamp, flow_rate, is_anomaly FROM readings "
        "WHERE timestamp >= ? ORDER BY timestamp",
        con, params=(since,),
    )
    con.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df


def db_log_report(text: str, rtype: str):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO sent_reports (sent_at, report, type) VALUES (?,?,?)",
        (datetime.now().isoformat(), text, rtype),
    )
    con.commit()
    con.close()


def db_last_report_time(rtype: str) -> datetime | None:
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT sent_at FROM sent_reports WHERE type=? ORDER BY id DESC LIMIT 1",
        (rtype,),
    ).fetchone()
    con.close()
    return datetime.fromisoformat(row[0]) if row else None


def db_save_qos(date_str: str, qos: float, total_readings: int,
                total_anomalies: int, spike_anomalies: int,
                night_anomalies: int, supply_windows: int,
                avg_flow: float, peak_flow: float,
                benchmark_used: bool, status: str):
    """Persist daily QoS score so Streamlit can read trend over time."""
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT OR REPLACE INTO qos_scores
        (date, qos, total_readings, total_anomalies, spike_anomalies,
         night_anomalies, supply_windows, avg_flow, peak_flow,
         benchmark_used, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (date_str, qos, total_readings, total_anomalies, spike_anomalies,
          night_anomalies, supply_windows, avg_flow, peak_flow,
          int(benchmark_used), status))
    con.commit()
    con.close()
    log.info("QoS score saved: %s = %.1f%%", date_str, qos)


def db_save_benchmark_snapshot(benchmark: dict, method: str = "supply_window"):
    """Save the current benchmark so Streamlit can display it."""
    if not benchmark:
        return
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT INTO benchmark_snapshot
        (saved_at, start_min, end_min, duration_min,
         peak_flow, avg_flow, samples, method)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(),
        benchmark.get("start_min"),
        benchmark.get("end_min"),
        benchmark.get("duration_min"),
        benchmark.get("peak_flow"),
        benchmark.get("avg_flow"),
        benchmark.get("samples"),
        method,
    ))
    con.commit()
    con.close()
    log.info("Benchmark snapshot saved (%s, %d samples)",
             method, benchmark.get("samples", 0))
    
def db_load_7day_trend() -> pd.DataFrame:
    """Load last 7 days of QoS scores for trend table in PDF."""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """SELECT date, qos, avg_flow, peak_flow, total_anomalies,
                spike_anomalies, night_anomalies, status
        FROM qos_scores
        ORDER BY date DESC LIMIT 7""",
        con
    )
    con.close()
    return df


def build_benchmark(con) -> dict | None:
    """Build benchmark from data older than 30 days stored in DB."""
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    df = pd.read_sql(
        "SELECT timestamp, flow_rate FROM readings WHERE timestamp < ? ORDER BY timestamp",
        con, params=(cutoff,)
    )
    if df.empty or len(df) < 50:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    all_windows = []
    for date, group in df.groupby(df["timestamp"].dt.date):
        wins = detect_supply_windows(group)
        all_windows.extend(wins)
    if not all_windows:
        return None
    starts = [w["start"].hour * 60 + w["start"].minute for w in all_windows]
    ends   = [w["end"].hour   * 60 + w["end"].minute   for w in all_windows]
    return {
        "start_min":    np.median(starts),
        "end_min":      np.median(ends),
        "duration_min": np.median([w["duration"] for w in all_windows]),
        "peak_flow":    np.median([w["peak"]     for w in all_windows]),
        "avg_flow":     np.median([w["avg"]      for w in all_windows]),
        "samples":      len(all_windows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM SENDERS
# ─────────────────────────────────────────────────────────────────────────────
def send_message(text: str, chat_ids: list = None) -> bool:
    if chat_ids is None:
        chat_ids = TELEGRAM_CHAT_IDS
    if not chat_ids:
        log.warning("No Telegram chat IDs configured")
        return False
    if "YOUR_BOT_TOKEN_HERE" in BOT_TOKEN:
        log.warning("Bot token not set — message NOT sent:\n%s", text)
        return False
    success = True
    for cid in chat_ids:
        try:
            r = requests.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": cid, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            if r.status_code == 200:
                log.info("Telegram message sent to %s", cid)
            else:
                log.error("Telegram error %s: %s", r.status_code, r.text[:200])
                success = False
        except Exception as e:
            log.error("Telegram send failed: %s", e)
            success = False
    return success


def send_photo(buf: io.BytesIO, caption: str, chat_ids: list = None) -> bool:
    if chat_ids is None:
        chat_ids = TELEGRAM_CHAT_IDS
    if not chat_ids or "YOUR_BOT_TOKEN_HERE" in BOT_TOKEN:
        return False
    success = True
    for cid in chat_ids:
        try:
            buf.seek(0)
            r = requests.post(
                f"{TELEGRAM_API}/sendPhoto",
                data={"chat_id": cid, "caption": caption, "parse_mode": "Markdown"},
                files={"photo": ("chart.png", buf, "image/png")},
                timeout=20,
            )
            if r.status_code == 200:
                log.info("Telegram photo sent to %s", cid)
            else:
                log.error("Telegram photo error %s: %s", r.status_code, r.text[:200])
                success = False
        except Exception as e:
            log.error("Telegram photo send failed: %s", e)
            success = False
    return success


def send_pdf(buf: io.BytesIO, filename: str, caption: str,
             chat_ids: list = None) -> bool:
    """Send PDF document via Telegram."""
    if chat_ids is None:
        chat_ids = TELEGRAM_CHAT_IDS
    if not chat_ids or "YOUR_BOT_TOKEN_HERE" in BOT_TOKEN:
        return False
    success = True
    for cid in chat_ids:
        try:
            buf.seek(0)
            r = requests.post(
                f"{TELEGRAM_API}/sendDocument",
                data={"chat_id": cid, "caption": caption},
                files={"document": (filename, buf, "application/pdf")},
                timeout=30,
            )
            if r.status_code == 200:
                log.info("PDF sent to %s", cid)
            else:
                log.error("PDF send error %s: %s", r.status_code, r.text[:200])
                success = False
        except Exception as e:
            log.error("PDF send failed: %s", e)
            success = False
    return success


# ─────────────────────────────────────────────────────────────────────────────
# FULL 4-MODEL ANOMALY DETECTOR  (mirrors Streamlit tab 3 run_detectors())
# ─────────────────────────────────────────────────────────────────────────────
def run_full_detectors(df: pd.DataFrame,
                       sensitivity: float = Z_SENSITIVITY,
                       contamination: float = CONTAMINATION,
                       spike_threshold: float = SPIKE_THRESHOLD,
                       night_start: int = NIGHT_START_HR,
                       night_end: int = NIGHT_END_HR) -> pd.DataFrame:
    """
    Runs Z-score, IQR, Isolation Forest, and PCA anomaly detection on df.
    Identical logic to Streamlit run_detectors() — returns enriched DataFrame
    with columns: anom_zscore, anom_iqr, anom_iforest, anom_pca, model_vote,
    final_anomaly, iforest_score, pca_score.
    """
    df = df.copy()
    df["hour"]         = df["timestamp"].dt.hour
    df["dow"]          = df["timestamp"].dt.dayofweek
    df["date"]         = df["timestamp"].dt.date
    df["roll_mean_10"] = df["flow_rate"].rolling(10, min_periods=1).mean()
    df["roll_std_10"]  = df["flow_rate"].rolling(10, min_periods=1).std().fillna(0)
    df["roll_mean_30"] = df["flow_rate"].rolling(30, min_periods=1).mean()
    df["flow_diff"]    = df["flow_rate"].diff().fillna(0)
    df["lag_1"]        = df["flow_rate"].shift(1).fillna(0)
    df["deviation"]    = df["flow_rate"] - df["roll_mean_30"]
    df["in_supply"]    = df["hour"].between(8, 10).astype(int)
    df["is_night"]     = ((df["hour"] >= night_start) | (df["hour"] <= night_end)).astype(int)

    df["anom_spike"]    = (df["flow_rate"] > spike_threshold).astype(int)
    df["anom_negative"] = (df["flow_rate"] < 0).astype(int)
    NIGHT_FLOW_LIMIT_   = spike_threshold * 0.8
    df["anom_night"]    = ((df["is_night"] == 1) &
                           (df["flow_rate"] > NIGHT_FLOW_LIMIT_)).astype(int)

    active = df["flow_rate"] > 0
    dfa = df[active].copy()

    # Z-score (supply hours only)
    supply_hours = dfa[~((dfa["hour"] >= night_start) | (dfa["hour"] <= night_end))]
    if len(supply_hours) > 10:
        z_vals = np.abs(stats.zscore(supply_hours["flow_rate"]))
        supply_hours = supply_hours.copy()
        supply_hours["anom_z"] = (z_vals > sensitivity).astype(int)
        dfa["anom_z"] = 0
        dfa.loc[supply_hours.index, "anom_z"] = supply_hours["anom_z"]
    else:
        dfa["anom_z"] = 0
    df["anom_zscore"] = 0
    df.loc[dfa.index, "anom_zscore"] = dfa["anom_z"]

    # IQR
    if len(dfa) > 3:
        Q1, Q3 = dfa["flow_rate"].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        IQR_FENCE = 2.5
        dfa["anom_iqr_f"] = (
            (dfa["flow_rate"] < Q1 - IQR_FENCE * IQR) |
            (dfa["flow_rate"] > Q3 + IQR_FENCE * IQR)
        ).astype(int)
    else:
        dfa["anom_iqr_f"] = 0
    df["anom_iqr"] = 0
    df.loc[dfa.index, "anom_iqr"] = dfa["anom_iqr_f"]

    FEATS = ["flow_rate", "roll_mean_10", "roll_std_10",
             "flow_diff", "lag_1", "deviation", "hour", "in_supply"]

    # Isolation Forest
    df["anom_iforest"]  = 0
    df["iforest_score"] = 0.0
    if len(dfa) >= 20:
        sc   = StandardScaler()
        Xsc  = sc.fit_transform(dfa[FEATS])
        ifor = IsolationForest(n_estimators=150, contamination=contamination,
                               random_state=42)
        preds = ifor.fit_predict(Xsc)
        dfa["anom_if"]  = (preds == -1).astype(int)
        dfa["if_score"] = -ifor.decision_function(Xsc)
        df.loc[dfa.index, "anom_iforest"]  = dfa["anom_if"]
        df.loc[dfa.index, "iforest_score"] = dfa["if_score"]

    # PCA autoencoder
    df["anom_pca"]  = 0
    df["pca_score"] = 0.0
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
        df.loc[dfa.index, "anom_pca"]  = dfa["anom_pca_f"]
        df.loc[dfa.index, "pca_score"] = dfa["pca_sc"]

    df["prev_flow"]        = df["flow_rate"].shift(1).fillna(0)
    df["anom_supply_cut"]  = ((df["flow_rate"] < 5) &
                               (df["prev_flow"] > 100)).astype(int)
    df["anom_sudden_drop"] = ((df["flow_rate"] > 5) &
                               (df["roll_mean_10"] > 100) &
                               (df["flow_rate"] < df["roll_mean_10"] * 0.4)).astype(int)
    df["model_vote"]    = (df["anom_zscore"] + df["anom_iqr"] +
                           df["anom_iforest"] + df["anom_pca"])
    df["final_anomaly"] = ((df["anom_negative"]   == 1) |
                            (df["anom_spike"]      == 1) |
                            (df["anom_night"]      == 1) |
                            (df["anom_supply_cut"] == 1) |
                            (df["anom_sudden_drop"]== 1) |
                            (df["model_vote"]      >= 3)).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST  (mirrors Streamlit tab 4 forecast())
# ─────────────────────────────────────────────────────────────────────────────
def forecast_flow(df: pd.DataFrame,
                  steps: int = FORECAST_STEPS):
    """
    Exponential smoothing forecast with trend + 95% CI.
    Returns (fcast, lo, hi, fts, sm) or (None,)*5 if insufficient data.
    Identical logic to Streamlit forecast().
    """
    active = df[df["flow_rate"] > 0]["flow_rate"].values
    if len(active) < 10:
        return None, None, None, None, None
    alpha = 0.3
    sm = [active[0]]
    for v in active[1:]:
        sm.append(alpha * v + (1 - alpha) * sm[-1])
    sm = np.array(sm)
    n  = min(20, len(sm))
    trend = (sm[-1] - sm[-n]) / n
    fcast = np.array([sm[-1] + trend * i for i in range(1, steps + 1)])
    std   = np.std(active[-30:]) if len(active) >= 30 else np.std(active)
    diffs = df["timestamp"].diff().dt.total_seconds().median() / 60
    freq  = max(1, int(diffs)) if not np.isnan(diffs) else 3
    fts   = pd.date_range(
        start=df["timestamp"].iloc[-1] + pd.Timedelta(minutes=freq),
        periods=steps, freq=f"{freq}min",
    )
    return fcast, fcast - 1.96 * std, fcast + 1.96 * std, fts, sm


# ═════════════════════════════════════════════════════════════════════════════
# PATTERN ANALYSIS HELPERS  (mirrors Streamlit tab 6 — identical logic)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_two_months(year: int = PATTERN_YEAR) -> pd.DataFrame:
    """
    Fetch January + February from the VMC API in 7-day chunks.
    Returns a DataFrame with columns [timestamp, flow_rate].
    Identical logic to Streamlit fetch_two_months().
    """
    jan_start = datetime(year, 1, 1, 0, 0, 0)
    feb_end   = datetime(year, 2, 28, 23, 59, 59)

    all_records: list[dict] = []
    chunk_start = jan_start

    while chunk_start <= feb_end:
        chunk_end = min(chunk_start + timedelta(days=7), feb_end)

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
                log.debug("Pattern fetch [%s]: %s", path, e)
                continue

            if r.status_code != 200:
                continue
            if "<title>login</title>" in r.text.lower():
                global _token
                _token = None
                try_login()
                continue

            try:
                data = r.json()
            except Exception:
                continue

            records = _parse_batch_response(data, chunk_end)
            if len(records) > 1:
                all_records.extend(records)
                break   # this path worked — move to next chunk

        chunk_start = chunk_end + timedelta(seconds=1)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (df.drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True))
    return df


def normalize_daily_curve(day_df: pd.DataFrame) -> np.ndarray | None:
    """
    Collapse one day's readings into a 24-point normalised vector.
    Identical logic to Streamlit normalize_daily_curve().
    """
    day_df = day_df.copy()
    day_df["hour"] = day_df["timestamp"].dt.hour
    hourly = day_df.groupby("hour")["flow_rate"].mean()
    curve  = hourly.reindex(range(24), fill_value=0.0).values.astype(float)

    if (curve > 0).sum() < 6:
        return None

    mn, mx = curve.min(), curve.max()
    if mx - mn < 1e-6:
        return None
    return (curve - mn) / (mx - mn)


def curve_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 24-point normalised curves."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def find_benchmark_pattern(df: pd.DataFrame,
                            n_clusters: int = PATTERN_K):
    """
    Identify the most frequently repeated daily flow shape via K-Means.
    Identical logic to Streamlit find_benchmark_pattern().
    Returns: benchmark, curves_df, all_curves, labels, centroids, modal_idx
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    all_curves: dict  = {}
    valid_dates: list = []

    for date, group in df.groupby("date"):
        curve = normalize_daily_curve(group)
        if curve is not None:
            all_curves[str(date)]  = curve
            valid_dates.append(str(date))

    if len(valid_dates) < n_clusters:
        n_clusters = max(2, len(valid_dates) // 2)

    X = np.array([all_curves[d] for d in valid_dates])

    km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels   = km.fit_predict(X)
    centroids = km.cluster_centers_

    cluster_sizes = np.bincount(labels)
    modal_idx     = int(np.argmax(cluster_sizes))
    benchmark     = centroids[modal_idx]

    rows = []
    for i, date_str in enumerate(valid_dates):
        dist       = curve_distance(all_curves[date_str], benchmark)
        similarity = max(0.0, 100.0 * (1.0 - dist / np.sqrt(24)))
        rows.append({
            "date":                  date_str,
            "cluster":               int(labels[i]),
            "similarity":            round(similarity, 1),
            "distance":              round(dist, 4),
            "is_benchmark_cluster":  int(labels[i]) == modal_idx,
        })

    curves_df = pd.DataFrame(rows)
    return benchmark, curves_df, all_curves, labels, centroids, modal_idx


# ─────────────────────────────────────────────────────────────────────────────
# CHART GENERATORS — ORIGINAL (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def make_daily_chart(df: pd.DataFrame) -> io.BytesIO:
    """Generate 24-hr flow chart with anomaly markers. Returns PNG buffer."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df["flow_rate"],
            color="#4a90d9", lw=0.8, alpha=0.85, label="Flow rate")
    ax.fill_between(df["timestamp"], df["flow_rate"],
                    alpha=0.07, color="#4a90d9")
    anoms = df[df["is_anomaly"] == 1]
    if not anoms.empty:
        ax.scatter(anoms["timestamp"], anoms["flow_rate"],
                   color="#ff6b6b", s=35, zorder=7, marker="^",
                   label=f"Anomaly ({len(anoms)})")
    ax.axhline(SPIKE_THRESHOLD, color="#ffa94d", lw=0.8,
               linestyle="--", alpha=0.7, label=f"Spike limit ({SPIKE_THRESHOLD})")
    ax.set_ylabel("Flow rate (m\u00b3/hr)")
    ax.set_title(f"{STATION_NAME} — Last 24 hrs", color="#c8cde0", fontsize=10)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    leg = ax.legend(fontsize=8, loc="upper right",
                    framealpha=0.85, edgecolor="#2a2d3a")
    for t in leg.get_texts():
        t.set_color("#9aa0b0")
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#1a1d27")
    plt.close(fig)
    return buf


def make_hourly_bar_chart(df: pd.DataFrame) -> io.BytesIO:
    """Hourly average flow bar chart."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    hourly = df[df["flow_rate"] > 0].groupby("hour")["flow_rate"].mean()
    if hourly.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.5))
    clrs = [
        "#ffa94d" if (h >= NIGHT_START_HR or h <= NIGHT_END_HR)
        else "#3fb950" if 8 <= h <= 10
        else "#4a90d9"
        for h in hourly.index
    ]
    ax.bar(hourly.index, hourly.values, color=clrs, width=0.7, zorder=3)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Avg m\u00b3/hr")
    ax.set_title("Average flow by hour (24 hr batch)", color="#c8cde0", fontsize=10)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1d27")
    plt.close(fig)
    return buf


def make_pdf_chart(df: pd.DataFrame, benchmark: dict,
                   windows: list, qos: float) -> io.BytesIO:
    """4-panel chart for embedding in PDF (white background)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="white")
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#333333")
        ax.xaxis.label.set_color("#333333")
        ax.yaxis.label.set_color("#333333")
        ax.title.set_color("#1a3a5c")

    df = df.copy().sort_values("timestamp")
    df["hour_frac"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60

    # Panel 1 — Flow vs benchmark
    ax1 = axes[0, 0]
    ax1.plot(df["hour_frac"], df["flow_rate"],
             color="#2a6496", lw=1.2, label="Today's flow")
    if benchmark:
        bm_s = benchmark["start_min"] / 60
        bm_e = benchmark["end_min"]   / 60
        ax1.axvspan(bm_s, bm_e, alpha=0.12, color="#27ae60",
                    label="Benchmark window")
        ax1.axhline(benchmark["peak_flow"], color="#e67e22",
                    lw=0.9, linestyle="--",
                    label=f"Benchmark peak ({benchmark['peak_flow']:.1f})")
    anoms = df[df["is_anomaly"] == 1] if "is_anomaly" in df.columns else pd.DataFrame()
    if not anoms.empty:
        ax1.scatter(anoms["hour_frac"], anoms["flow_rate"],
                    color="#e74c3c", s=25, zorder=5, marker="^",
                    label=f"Anomaly ({len(anoms)})")
    ax1.set_xlim(0, 24)
    ax1.set_xlabel("Hour of day", fontsize=8)
    ax1.set_ylabel("Flow rate (m\u00b3/hr)", fontsize=8)
    ax1.set_title("Flow Rate vs Benchmark", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=6.5)
    ax1.grid(True, alpha=0.3)

    # Panel 2 — QoS gauge
    ax2 = axes[0, 1]
    qos_clr = "#27ae60" if qos >= 85 else "#f39c12" if qos >= 70 else "#e74c3c"
    ax2.barh(["QoS"], [qos],       color=qos_clr,   height=0.5)
    ax2.barh(["QoS"], [100 - qos], color="#ecf0f1", height=0.5, left=qos)
    ax2.axvline(85, color="#27ae60", lw=1, linestyle="--", alpha=0.7)
    ax2.axvline(70, color="#f39c12", lw=1, linestyle="--", alpha=0.7)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Score (%)", fontsize=8)
    ax2.set_title(f"Quality of Service: {qos:.1f}%", fontsize=9, fontweight="bold")
    ax2.text(qos / 2, 0, f"{qos:.1f}%",
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="x")

    # Panel 3 — Hourly bar
    ax3 = axes[1, 0]
    df["hour"] = df["timestamp"].dt.hour
    hourly = df[df["flow_rate"] > 0].groupby("hour")["flow_rate"].mean()
    if not hourly.empty:
        bar_clrs = [
            "#e67e22" if (h >= NIGHT_START_HR or h <= NIGHT_END_HR)
            else "#27ae60" if 8 <= h <= 10
            else "#2a6496"
            for h in hourly.index
        ]
        ax3.bar(hourly.index, hourly.values, color=bar_clrs, width=0.7)
        if benchmark:
            ax3.axhline(benchmark["avg_flow"], color="#e74c3c",
                        lw=0.8, linestyle="--",
                        label=f"Benchmark avg ({benchmark['avg_flow']:.1f})")
            ax3.legend(fontsize=6.5)
    ax3.set_xlabel("Hour of day", fontsize=8)
    ax3.set_ylabel("Avg m\u00b3/hr", fontsize=8)
    ax3.set_title("Hourly Average Flow", fontsize=9, fontweight="bold")
    ax3.set_xticks(range(0, 24, 2))
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4 — Anomaly hourly heatmap
    ax4 = axes[1, 1]
    if "is_anomaly" in df.columns and df["is_anomaly"].sum() > 0:
        anom_hourly = df[df["is_anomaly"] == 1].groupby("hour").size()
        full_hours  = pd.Series(0, index=range(24))
        full_hours.update(anom_hourly)
        bar_clrs4 = ["#e74c3c" if v > 0 else "#ecf0f1"
                     for v in full_hours.values]
        ax4.bar(full_hours.index, full_hours.values, color=bar_clrs4, width=0.8)
        ax4.set_xlabel("Hour of day", fontsize=8)
        ax4.set_ylabel("Anomaly count", fontsize=8)
        ax4.set_title("Anomalies by Hour", fontsize=9, fontweight="bold")
        ax4.set_xticks(range(0, 24, 2))
        ax4.grid(True, alpha=0.3, axis="y")
    else:
        ax4.text(0.5, 0.5, "No anomalies detected",
                 ha="center", va="center", fontsize=11,
                 color="#27ae60", fontweight="bold",
                 transform=ax4.transAxes)
        ax4.set_title("Anomalies by Hour", fontsize=9, fontweight="bold")

    fig.tight_layout(pad=1.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# NEW CHART GENERATORS  (matching Streamlit tabs 2, 3, 4, 6)
# ─────────────────────────────────────────────────────────────────────────────

def make_eda_charts(df: pd.DataFrame) -> list[io.BytesIO]:
    """
    EDA charts matching Streamlit tab 2:
      1. Full time series
      2. Rolling mean ±2σ
      3. Hourly average (already in make_hourly_bar_chart — included here for PDF)
      4. Flow distribution histogram
    Returns list of PNG BytesIO buffers.
    """
    bufs = []
    df = df.copy()
    df["hour"]         = df["timestamp"].dt.hour
    df["roll_mean_10"] = df["flow_rate"].rolling(10, min_periods=1).mean()
    df["roll_std_10"]  = df["flow_rate"].rolling(10, min_periods=1).std().fillna(0)

    # Chart 1 — Full time series
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(df["timestamp"], df["flow_rate"],
            color="#4a90d9", lw=0.7, alpha=0.85)
    ax.fill_between(df["timestamp"], df["flow_rate"],
                    alpha=0.06, color="#4a90d9")
    ax.axhline(0, color="#ff6b6b", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_ylabel("Flow rate (m\u00b3/hr)")
    ax.set_title("Full flow rate time series")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1d27")
    plt.close(fig)
    bufs.append(buf)

    # Chart 2 — Rolling mean ±2σ
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(df["timestamp"], df["flow_rate"],
            color="#4a90d9", lw=0.5, alpha=0.5, label="Flow")
    ax.plot(df["timestamp"], df["roll_mean_10"],
            color="#c8cde0", lw=1.0, label="Rolling mean (10)")
    ax.fill_between(df["timestamp"],
                    df["roll_mean_10"] - 2 * df["roll_std_10"],
                    df["roll_mean_10"] + 2 * df["roll_std_10"],
                    alpha=0.12, color="#4a90d9", label="±2σ band")
    ax.set_ylabel("Flow rate (m\u00b3/hr)")
    ax.set_title("Rolling mean ± 2σ confidence band")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    buf2 = io.BytesIO()
    fig.savefig(buf2, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1d27")
    plt.close(fig)
    bufs.append(buf2)

    # Chart 3 — Flow distribution histogram
    active_flow = df[df["flow_rate"] > 0]["flow_rate"]
    if not active_flow.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(active_flow, bins=50, color="#4a90d9", alpha=0.8,
                density=True, label="Flow distribution")
        mu, sigma = active_flow.mean(), active_flow.std()
        x = np.linspace(active_flow.min(), active_flow.max(), 200)
        from scipy.stats import norm as _norm
        ax.plot(x, _norm.pdf(x, mu, sigma),
                color="#ffa94d", lw=1.5, label=f"Normal fit (μ={mu:.1f})")
        ax.set_xlabel("Flow rate (m\u00b3/hr)")
        ax.set_ylabel("Density")
        ax.set_title("Flow distribution (active readings)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        buf3 = io.BytesIO()
        fig.savefig(buf3, format="png", dpi=120, bbox_inches="tight",
                    facecolor="#1a1d27")
        plt.close(fig)
        bufs.append(buf3)

    return bufs


def make_anomaly_charts(df_full: pd.DataFrame) -> list[io.BytesIO]:
    """
    Anomaly detection charts matching Streamlit tab 3:
      1. Model comparison bar chart
      2. Final anomaly timeline overlay
      3. 4-panel model-by-model overlay
      4. Isolation Forest score heatmap (day × hour)
    Returns list of PNG BytesIO buffers.
    """
    bufs = []

    # Chart 1 — Model comparison bar
    mcounts = {
        "Z-score":          int(df_full.get("anom_zscore",   pd.Series([0])).sum()),
        "IQR":              int(df_full.get("anom_iqr",      pd.Series([0])).sum()),
        "Isolation Forest": int(df_full.get("anom_iforest",  pd.Series([0])).sum()),
        "PCA Autoencoder":  int(df_full.get("anom_pca",      pd.Series([0])).sum()),
        "Final (3+/rule)":  int(df_full.get("final_anomaly", pd.Series([0])).sum()),
    }
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bclrs = ["#9b8ec4", "#9b8ec4", "#c4736b", "#6bab7a", "#4a90d9"]
    bars  = ax.bar(list(mcounts.keys()), list(mcounts.values()),
                   color=bclrs, width=0.55, zorder=3)
    for bar, v in zip(bars, mcounts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(v),
                ha="center", va="bottom", fontsize=9, color="#c8cde0")
    ax.set_ylabel("Count")
    ax.set_title("Anomalies per detection model")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1d27")
    plt.close(fig)
    bufs.append(buf)

    # Chart 2 — Final anomaly timeline
    if "final_anomaly" in df_full.columns:
        fig, ax = plt.subplots(figsize=(13, 3.5))
        ax.plot(df_full["timestamp"], df_full["flow_rate"],
                color="#4a90d9", lw=0.7, alpha=0.7, label="Flow")
        fa = df_full[df_full["final_anomaly"] == 1]
        ax.scatter(fa["timestamp"], fa["flow_rate"],
                   color="#ff6b6b", s=30, zorder=6, marker="^",
                   label=f"Final anomaly ({len(fa)})")
        ax.axhline(SPIKE_THRESHOLD, color="#ffa94d", lw=0.8,
                   linestyle="--", alpha=0.6, label="Spike limit")
        ax.set_ylabel("m\u00b3/hr")
        ax.set_title("Final anomaly flags (3+ models / rule-based)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        fig.autofmt_xdate(rotation=25)
        fig.tight_layout()
        buf2 = io.BytesIO()
        fig.savefig(buf2, format="png", dpi=120, bbox_inches="tight",
                    facecolor="#1a1d27")
        plt.close(fig)
        bufs.append(buf2)

    # Chart 3 — 4-panel model-by-model overlay
    model_cols   = ["anom_zscore", "anom_iqr", "anom_iforest", "anom_pca"]
    model_labels = ["Z-score", "IQR", "Isolation Forest", "PCA Autoencoder"]
    model_colors = ["#ffa94d", "#ff6b6b", "#8b949e", "#3fb950"]
    cols_present = [c for c in model_cols if c in df_full.columns]
    if cols_present:
        fig, axes = plt.subplots(len(cols_present), 1,
                                  figsize=(13, 3 * len(cols_present)),
                                  sharex=True,
                                  gridspec_kw={"hspace": 0.45})
        if len(cols_present) == 1:
            axes = [axes]
        for ax, col, lbl, clr in zip(axes, model_cols, model_labels, model_colors):
            if col not in df_full.columns:
                continue
            ax.plot(df_full["timestamp"], df_full["flow_rate"],
                    color="#4a90d9", lw=0.4, alpha=0.5)
            fl = df_full[df_full[col] == 1]
            ax.scatter(fl["timestamp"], fl["flow_rate"],
                       color=clr, s=18, zorder=6,
                       label=f"{lbl} ({len(fl)})")
            ax.set_ylabel("m\u00b3/hr", fontsize=8)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        fig.autofmt_xdate(rotation=25)
        fig.tight_layout()
        buf3 = io.BytesIO()
        fig.savefig(buf3, format="png", dpi=110, bbox_inches="tight",
                    facecolor="#1a1d27")
        plt.close(fig)
        bufs.append(buf3)

    # Chart 4 — IF score heatmap (day × hour)
    if "iforest_score" in df_full.columns and df_full["iforest_score"].sum() > 0:
        df_h = df_full.copy()
        df_h["day_label"] = df_h["timestamp"].dt.strftime("%d %b")
        df_h["hour_col"]  = df_h["timestamp"].dt.hour
        pivot = df_h.pivot_table(index="day_label", columns="hour_col",
                                  values="iforest_score", aggfunc="max").fillna(0)
        fig, ax = plt.subplots(figsize=(13, max(4, len(pivot) * 0.5 + 2)))
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.15,
                    linecolor="#0f1117",
                    cbar_kws={"label": "IF score"}, annot=False)
        ax.set_title("Anomaly score heatmap — day × hour")
        ax.set_xlabel("Hour")
        ax.set_ylabel("")
        fig.tight_layout()
        buf4 = io.BytesIO()
        fig.savefig(buf4, format="png", dpi=110, bbox_inches="tight",
                    facecolor="#1a1d27")
        plt.close(fig)
        bufs.append(buf4)

    return bufs


def make_forecast_chart(df: pd.DataFrame,
                         df_full: pd.DataFrame) -> io.BytesIO | None:
    """
    Forecast chart matching Streamlit tab 4:
      - Last 48 hrs actual + fitted + forecast + 95% CI + anomaly markers.
      - Residuals panel below.
    Returns PNG BytesIO buffer, or None if insufficient data.
    """
    fc, lo, hi, fts, sm = forecast_flow(df)
    if fc is None:
        log.info("Forecast: not enough active data — skipping chart")
        return None

    freq = max(1, int(df["timestamp"].diff().dt.total_seconds().median() / 60))
    lookback  = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=2)]
    active_lk = lookback[lookback["flow_rate"] > 0]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7),
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.35})

    ax = axes[0]
    ax.plot(lookback["timestamp"], lookback["flow_rate"],
            color="#4a90d9", lw=1.0, label="Actual", alpha=0.85)
    n_sm = len(active_lk)
    if n_sm > 0:
        ax.plot(active_lk["timestamp"], sm[-n_sm:],
                color="#3fb950", lw=1.0, linestyle="--",
                label="Fitted", alpha=0.8)
    ax.plot(fts, fc, color="#ffa94d", lw=1.5, label="Forecast", zorder=5)
    ax.fill_between(fts, lo, hi, alpha=0.18, color="#ffa94d", label="95% CI")
    # anomaly markers from full detector if available
    if "final_anomaly" in df_full.columns:
        fa_lk = df_full[
            (df_full["final_anomaly"] == 1) &
            (df_full["timestamp"] >= lookback["timestamp"].min())
        ]
        if not fa_lk.empty:
            ax.scatter(fa_lk["timestamp"], fa_lk["flow_rate"],
                       color="#ff6b6b", s=28, zorder=7, label="Anomaly")
    ax.axvline(df["timestamp"].max(),
               color="#555d6e", lw=0.8, linestyle=":", label="Now")
    ax.set_ylabel("Flow rate (m\u00b3/hr)")
    ax.set_title("Last 48 hrs + Exponential Smoothing Forecast")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))

    # Residuals panel
    ax2 = axes[1]
    if n_sm > 0:
        act_v    = active_lk["flow_rate"].values
        fit_v    = sm[-n_sm:]
        resids   = act_v - fit_v
        rthresh  = np.std(resids) * 2
        rclrs    = ["#ff6b6b" if abs(r) > rthresh else "#4a90d9"
                    for r in resids]
        ax2.bar(active_lk["timestamp"].values, resids, color=rclrs,
                width=pd.Timedelta(minutes=freq * 0.8))
        ax2.axhline(0,        color="#555d6e", lw=0.6)
        ax2.axhline( rthresh, color="#ff6b6b", lw=0.7,
                    linestyle="--", alpha=0.6)
        ax2.axhline(-rthresh, color="#ff6b6b", lw=0.7,
                    linestyle="--", alpha=0.6)
        ax2.set_ylabel("Residual (m\u00b3/hr)")
        ax2.set_title("Forecast residuals (red = large error)")
        ax2.grid(True, alpha=0.2)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1d27")
    plt.close(fig)
    return buf


def make_pattern_charts(pat_df: pd.DataFrame,
                         n_clusters: int = PATTERN_K,
                         sim_threshold: float = SIM_THRESHOLD
                         ) -> tuple[list[io.BytesIO], dict]:
    """
    Pattern analysis charts matching Streamlit tab 6:
      1. Jan + Feb combined multi-day overlay — all days as faint blue, bold red median = benchmark
         (Exactly matches PDF Figure 1: "Multi-Day Overlay with Median Profile")
      2. K-Means cluster centroids (benchmark highlighted)
      3. Daily similarity bar chart
      4. Best-match vs worst-match overlay (side by side)
    Returns (list of PNG BytesIO buffers, summary dict).
    """
    bufs    = []
    summary = {}

    if pat_df.empty:
        log.warning("Pattern charts: empty DataFrame — skipping")
        return bufs, summary

    try:
        bench, curves_df, all_curves, labels, centroids, modal_idx = \
            find_benchmark_pattern(pat_df, n_clusters=n_clusters)
    except Exception as e:
        log.error("find_benchmark_pattern failed: %s", e)
        return bufs, summary

    hours_axis = np.arange(24)

    # ── Chart 1 — Jan + Feb combined multi-day overlay (PDF Figure 1 style) ──
    # ALL days from both months plotted as faint blue lines on the same
    # 24-hour axis.  Bold red line = median across ALL days = benchmark.
    # Matches exactly the "Multi-Day Overlay" chart in the PDF report.
    jan_curves, feb_curves = [], []
    for date_str, curve in all_curves.items():
        month = int(date_str[5:7])
        if month == 1:
            jan_curves.append(curve)
        elif month == 2:
            feb_curves.append(curve)

    all_day_curves = jan_curves + feb_curves   # combined pool
    n_total = len(all_day_curves)

    fig, ax = plt.subplots(figsize=(13, 5))

    # Every individual day as a faint blue line (both months same colour)
    for c in all_day_curves:
        ax.plot(hours_axis, c, color="#4a90d9", lw=0.6, alpha=0.20)

    # Benchmark median — computed from ALL days combined (bold red)
    # Using median for robustness, matching PDF methodology
    benchmark_median = np.median(all_day_curves, axis=0)
    ax.plot(hours_axis, benchmark_median, color="#e74c3c", lw=2.8,
            label=f"Benchmark Median — {n_total} days (Jan+Feb {PATTERN_YEAR})",
            zorder=5)

    ax.set_xlabel("Hour of Day", fontsize=9)
    ax.set_ylabel("Normalised Flow (0=min, 1=max)", fontsize=9)
    ax.set_title(
        f"Multi-Day Overlay: Flow Pattern (Jan–Feb {PATTERN_YEAR})\n"
        f"{n_total} Days Overlaid with Benchmark Median  "
        f"[Jan: {len(jan_curves)} days | Feb: {len(feb_curves)} days]",
        fontsize=10,
    )
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf1 = io.BytesIO()
    fig.savefig(buf1, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1d27")
    plt.close(fig)
    bufs.append(buf1)

    # ── Chart 2 — K-Means centroids ───────────────────────────────────────
    cluster_sizes = np.bincount(curves_df["cluster"].values.astype(int),
                                 minlength=len(centroids))
    fig, ax = plt.subplots(figsize=(13, 4))
    palette = ["#555d6e"] * len(centroids)
    palette[modal_idx] = "#ffa94d"
    for i, centroid in enumerate(centroids):
        n   = cluster_sizes[i]
        lw  = 2.5 if i == modal_idx else 0.9
        alp = 1.0 if i == modal_idx else 0.55
        lbl = (f"Cluster {i} — {n} days  ← BENCHMARK"
               if i == modal_idx
               else f"Cluster {i} — {n} days")
        ax.plot(hours_axis, centroid, color=palette[i],
                lw=lw, alpha=alp, label=lbl)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Normalised flow")
    ax.set_title(f"K-Means cluster centroids (k={n_clusters}) — modal = benchmark")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf2 = io.BytesIO()
    fig.savefig(buf2, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1d27")
    plt.close(fig)
    bufs.append(buf2)

    # ── Chart 3 — Daily similarity bar ────────────────────────────────────
    curves_sorted = curves_df.sort_values("date").reset_index(drop=True)
    bar_colors    = ["#3fb950" if s >= sim_threshold else "#ff6b6b"
                     for s in curves_sorted["similarity"]]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(range(len(curves_sorted)), curves_sorted["similarity"],
           color=bar_colors, width=0.75, zorder=3)
    ax.axhline(sim_threshold, color="#ffa94d", lw=1.2,
               linestyle="--", label=f"Threshold {sim_threshold}%")
    ax.set_xticks(range(len(curves_sorted)))
    ax.set_xticklabels([d[5:] for d in curves_sorted["date"]],
                        rotation=60, fontsize=6.5)
    ax.set_ylabel("Similarity to benchmark (%)")
    ax.set_title("Daily pattern similarity — green ≥ threshold, red = deviant")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf3 = io.BytesIO()
    fig.savefig(buf3, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1d27")
    plt.close(fig)
    bufs.append(buf3)

    # ── Chart 4 — Best-match vs worst-match ──────────────────────────────
    ranked = curves_sorted.sort_values("similarity", ascending=False)
    top5   = ranked.head(5)["date"].tolist()
    bot5   = ranked.tail(5)["date"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for date_str in top5:
        if date_str in all_curves:
            sim_val = curves_sorted.loc[
                curves_sorted["date"] == date_str, "similarity"
            ].values[0]
            axes[0].plot(hours_axis, all_curves[date_str],
                         color="#3fb950", lw=1.0, alpha=0.65,
                         label=f"{date_str[5:]} ({sim_val:.0f}%)")
    axes[0].plot(hours_axis, bench, color="#ffa94d", lw=2.2,
                 linestyle="--", label="Benchmark")
    axes[0].set_title("Top 5 closest days to benchmark")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Normalised flow")
    axes[0].legend(fontsize=7.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].spines[["top", "right"]].set_visible(False)

    for date_str in bot5:
        if date_str in all_curves:
            sim_val = curves_sorted.loc[
                curves_sorted["date"] == date_str, "similarity"
            ].values[0]
            axes[1].plot(hours_axis, all_curves[date_str],
                         color="#ff6b6b", lw=1.0, alpha=0.65,
                         label=f"{date_str[5:]} ({sim_val:.0f}%)")
    axes[1].plot(hours_axis, bench, color="#ffa94d", lw=2.2,
                 linestyle="--", label="Benchmark")
    axes[1].set_title("Bottom 5 most deviant days")
    axes[1].set_xlabel("Hour")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    buf4 = io.BytesIO()
    fig.savefig(buf4, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1d27")
    plt.close(fig)
    bufs.append(buf4)

    # ── Summary dict ─────────────────────────────────────────────────────
    n_match   = (curves_sorted["similarity"] >= sim_threshold).sum()
    n_deviant = len(curves_sorted) - n_match
    avg_sim   = curves_sorted["similarity"].mean()
    best_day  = curves_sorted.loc[curves_sorted["similarity"].idxmax(), "date"]
    worst_day = curves_sorted.loc[curves_sorted["similarity"].idxmin(), "date"]
    summary = {
        "total_days":    len(curves_sorted),
        "n_match":       int(n_match),
        "n_deviant":     int(n_deviant),
        "avg_similarity":round(avg_sim, 1),
        "best_day":      best_day,
        "worst_day":     worst_day,
        "jan_days":      len(jan_curves),
        "feb_days":      len(feb_curves),
        "modal_cluster": int(modal_idx),
    }
    return bufs, summary


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLY WINDOW DETECTION + QoS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def detect_supply_windows(df: pd.DataFrame,
                           threshold=1.0,
                           min_duration_min=5) -> list[dict]:
    """Detect active supply windows from flow data."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    windows   = []
    in_window = False
    start_idx = None

    for i, row in df.iterrows():
        if row["flow_rate"] >= threshold and not in_window:
            in_window = True
            start_idx = i
        elif row["flow_rate"] < threshold and in_window:
            in_window = False
            wdf = df.iloc[start_idx:i]
            dur = (wdf["timestamp"].iloc[-1] -
                   wdf["timestamp"].iloc[0]).total_seconds() / 60
            if dur >= min_duration_min:
                windows.append({
                    "start":    wdf["timestamp"].iloc[0],
                    "end":      wdf["timestamp"].iloc[-1],
                    "duration": dur,
                    "peak":     wdf["flow_rate"].max(),
                    "avg":      wdf["flow_rate"].mean(),
                })
    if in_window and start_idx is not None:
        wdf = df.iloc[start_idx:]
        dur = (wdf["timestamp"].iloc[-1] -
               wdf["timestamp"].iloc[0]).total_seconds() / 60
        if dur >= min_duration_min:
            windows.append({
                "start":    wdf["timestamp"].iloc[0],
                "end":      wdf["timestamp"].iloc[-1],
                "duration": dur,
                "peak":     wdf["flow_rate"].max(),
                "avg":      wdf["flow_rate"].mean(),
            })
    return windows
def detect_supply_windows_df(day_df, threshold=1.0, min_duration_min=5):
    """
    Mirrors Streamlit detect_supply_windows_df — includes start/end_hour_frac.
    Used by build_benchmark_from_windows and score_day_vs_benchmark.
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
                    "start":           wdf["timestamp"].iloc[0],
                    "end":             wdf["timestamp"].iloc[-1],
                    "duration":        dur,
                    "peak":            wdf[col].max(),
                    "avg":             wdf[col].mean(),
                    "start_hour_frac": wdf["timestamp"].iloc[0].hour + wdf["timestamp"].iloc[0].minute / 60,
                    "end_hour_frac":   wdf["timestamp"].iloc[-1].hour + wdf["timestamp"].iloc[-1].minute / 60,
                })
    if in_window and start_idx is not None:
        wdf = df.iloc[start_idx:]
        dur = (wdf["timestamp"].iloc[-1] - wdf["timestamp"].iloc[0]).total_seconds() / 60
        if dur >= min_duration_min:
            windows.append({
                "start":           wdf["timestamp"].iloc[0],
                "end":             wdf["timestamp"].iloc[-1],
                "duration":        dur,
                "peak":            wdf[col].max(),
                "avg":             wdf[col].mean(),
                "start_hour_frac": wdf["timestamp"].iloc[0].hour + wdf["timestamp"].iloc[0].minute / 60,
                "end_hour_frac":   wdf["timestamp"].iloc[-1].hour + wdf["timestamp"].iloc[-1].minute / 60,
            })
    return windows


def build_benchmark_from_windows(all_windows, n_clusters=6):
    """
    Mirrors Streamlit build_benchmark_from_windows.
    Clusters supply windows by start time; dominant cluster median = benchmark.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    if not all_windows:
        return None, {}
    wdf = pd.DataFrame(all_windows)
    starts = wdf["start_hour_frac"].values.reshape(-1, 1)
    try:
        Z = linkage(starts, method="ward")
        labels = fcluster(Z, t=1.0, criterion="distance")
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
        # Keep legacy keys so existing PDF/QoS code still works
        "start_min":    float(np.median(dominant_wdf["start_hour_frac"])) * 60,
        "end_min":      float(np.median(dominant_wdf["end_hour_frac"])) * 60,
        "duration_min": float(np.median(dominant_wdf["duration"])),
        "peak_flow":    float(np.median(dominant_wdf["peak"])),
        "avg_flow":     float(np.median(dominant_wdf["avg"])),
    }
    return benchmark, wdf


def score_day_vs_benchmark(day_windows, benchmark,
                            time_tol_min=30, flow_tol=0.20):
    """
    Mirrors Streamlit score_day_vs_benchmark — PDF §2.4 methodology.
    Returns (qos_score 0-100, anomaly_list, matched_window).
    """
    if not day_windows:
        return 0.0, ["No supply windows detected"], None
    if benchmark is None:
        return 50.0, ["Benchmark not available"], None
    bm_start_h = benchmark.get("start_hour", benchmark.get("start_min", 0) / 60)
    bm_end_h   = benchmark.get("end_hour",   benchmark.get("end_min",   0) / 60)
    bm_dur     = benchmark.get("duration",   benchmark.get("duration_min", 0))
    bm_peak    = benchmark.get("peak",       benchmark.get("peak_flow",    0))
    bm_avg     = benchmark.get("avg",        benchmark.get("avg_flow",     0))
    best_win   = min(day_windows, key=lambda w: abs(w["start_hour_frac"] - bm_start_h))
    anomalies  = []
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
    qos = min(100, max(0, (t_score * 0.5 + f_score * 0.5) * 100))
    return round(qos, 1), anomalies, best_win

def compute_qos(windows: list, benchmark: dict,
                time_tol=30, flow_tol=0.20) -> tuple[float, list[str]]:
    """Compute QoS score (0-100%) and list anomalies."""
    if not windows or not benchmark:
        return 0.0, ["No supply windows detected or no benchmark available"]

    anomalies     = []
    timing_scores = []
    flow_scores   = []
    bm_start = benchmark["start_min"]
    bm_end   = benchmark["end_min"]
    bm_dur   = benchmark["duration_min"]
    bm_peak  = benchmark["peak_flow"]
    bm_avg   = benchmark["avg_flow"]

    for w in windows:
        w_start = w["start"].hour * 60 + w["start"].minute
        w_end   = w["end"].hour   * 60 + w["end"].minute

        start_dev = abs(w_start - bm_start)
        end_dev   = abs(w_end   - bm_end)
        dur_dev   = abs(w["duration"] - bm_dur)

        if start_dev > time_tol:
            anomalies.append(
                f"Start time off by {start_dev:.0f} min "
                f"(benchmark: {int(bm_start)//60:02d}:{int(bm_start)%60:02d})")
        if end_dev > time_tol:
            anomalies.append(
                f"End time off by {end_dev:.0f} min "
                f"(benchmark: {int(bm_end)//60:02d}:{int(bm_end)%60:02d})")
        if dur_dev > time_tol:
            anomalies.append(
                f"Duration deviated by {dur_dev:.0f} min "
                f"(benchmark: {bm_dur:.0f} min)")

        peak_dev = abs(w["peak"] - bm_peak) / bm_peak if bm_peak > 0 else 0
        avg_dev  = abs(w["avg"]  - bm_avg)  / bm_avg  if bm_avg  > 0 else 0

        if peak_dev > flow_tol:
            anomalies.append(
                f"Peak flow deviated by {peak_dev*100:.0f}% "
                f"(benchmark: {bm_peak:.1f} m\u00b3/hr)")
        if avg_dev > flow_tol:
            anomalies.append(
                f"Avg flow deviated by {avg_dev*100:.0f}% "
                f"(benchmark: {bm_avg:.1f} m\u00b3/hr)")

        t_score = max(0, 1 - (start_dev + end_dev) / (2 * time_tol * 3))
        f_score = max(0, 1 - (peak_dev  + avg_dev)  / (2 * flow_tol  * 3))
        timing_scores.append(t_score)
        flow_scores.append(f_score)

    qos = (np.mean(timing_scores) * 0.5 + np.mean(flow_scores) * 0.5) * 100
    return round(min(100, max(0, qos)), 1), anomalies


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR  (original + new EDA / anomaly / forecast / pattern
#                         sections appended — nothing removed)
# ─────────────────────────────────────────────────────────────────────────────
def make_pdf_report(df: pd.DataFrame, benchmark: dict,
                    windows: list, qos: float,
                    anomalies: list, total_anom: int,
                    avg_flow: float, peak_flow: float,
                    peak_ts: str, spike_anom: int,
                    night_anom: int,
                    df_full: pd.DataFrame = None,
                    pat_summary: dict = None,
                    pat_chart_bufs: list = None) -> io.BytesIO:
    """
    Generate full professional PDF report.
    NEW parameters (all optional, backward compatible):
      df_full       — enriched DataFrame from run_full_detectors()
      pat_summary   — dict from make_pattern_charts()
      pat_chart_bufs — list of BytesIO pattern chart PNGs
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=1.8*cm,   bottomMargin=1.8*cm,
    )

    styles  = getSampleStyleSheet()
    now_str = datetime.now(IST).strftime("%d %b %Y, %H:%M IST")
    date_str= datetime.now(IST).strftime("%d %b %Y")
    W       = 17.4 * cm

    def qos_color(q):
        if q >= 85: return colors.HexColor("#27ae60")
        if q >= 70: return colors.HexColor("#f39c12")
        return colors.HexColor("#e74c3c")

    def status_str(q):
        if q >= 85: return "EXCELLENT"
        if q >= 70: return "GOOD"
        return "POOR - ACTION REQUIRED"

    title_style    = ParagraphStyle("ReportTitle",
        fontSize=20, textColor=colors.HexColor("#1a3a5c"),
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=2)
    subtitle_style = ParagraphStyle("ReportSub",
        fontSize=9, textColor=colors.HexColor("#7f8c8d"),
        alignment=TA_CENTER, spaceAfter=2)
    h1_style = ParagraphStyle("H1",
        fontSize=12, textColor=colors.HexColor("#1a3a5c"),
        fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=5)
    h2_style = ParagraphStyle("H2",
        fontSize=10, textColor=colors.HexColor("#2a6496"),
        fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3)
    body_style  = ParagraphStyle("Body",
        fontSize=8.5, textColor=colors.HexColor("#2c3e50"),
        leading=13, spaceAfter=3)
    footer_style = ParagraphStyle("Footer",
        fontSize=7, textColor=colors.white,
        alignment=TA_CENTER, fontName="Helvetica")

    story = []

    # ── PAGE 1: HEADER ────────────────────────────────────────────────────
    story.append(Paragraph("Water Distribution", title_style))
    story.append(Paragraph("Anomaly Detection &amp; Quality of Service Report",
                            title_style))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Benchmarking, Pattern Analysis, and Daily Anomaly Scoring", subtitle_style))

    # ← REPLACE THE OLD BENCHMARK LINE WITH THIS:
    using_fallback = benchmark.get("samples", 0) <= 1 if benchmark else True
    if using_fallback:
        bm_label = "Hardcoded fallback (DB building up)"
    else:
        bm_label = f"Rolling 30-day DB average ({benchmark['samples']} windows)"
    story.append(Paragraph(
        f"Data Period: {date_str} | Benchmark: {bm_label}",
        subtitle_style))

    story.append(Paragraph(
            f"Data Source: VMC.DLP3.JAM.{METER_LABEL} (Flow Rate) | "
            f"Meter: {METER_LABEL}",
            subtitle_style))
    story.append(Paragraph(f"Generated: {now_str}", subtitle_style))

    story.append(HRFlowable(width=W, thickness=1.5,
                             color=colors.HexColor("#1a3a5c"), spaceAfter=8))

    # QoS Banner
    qc     = qos_color(qos)
    banner = Table(
        [[Paragraph(
            f"QoS Score: {qos:.1f}%   |   Status: {status_str(qos)}",
            ParagraphStyle("Banner", fontSize=14, textColor=colors.white,
                           fontName="Helvetica-Bold", alignment=TA_CENTER)
        )]],
        colWidths=[W], rowHeights=[1.1*cm]
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), qc),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))

    story.append(banner)
    story.append(Spacer(1, 0.3*cm))

    # ── AT-A-GLANCE 3-BOX STATUS PANEL ───────────────────────────────────
    anom_rate   = (total_anom / max(len(df), 1)) * 100
    supply_hrs  = sum(w["duration"] for w in windows) / 60 if windows else 0
    box1_color  = colors.HexColor("#27ae60") if avg_flow >= 300 else colors.HexColor("#f39c12")
    box2_color  = colors.HexColor("#27ae60") if anom_rate < 5 else (
                  colors.HexColor("#f39c12") if anom_rate < 20 else colors.HexColor("#e74c3c"))
    box3_color  = colors.HexColor("#27ae60") if supply_hrs >= 20 else (
                  colors.HexColor("#f39c12") if supply_hrs >= 10 else colors.HexColor("#e74c3c"))

    box_style = ParagraphStyle("BoxVal", fontSize=16, textColor=colors.white,
                               fontName="Helvetica-Bold", alignment=TA_CENTER)
    box_label  = ParagraphStyle("BoxLbl", fontSize=7.5, textColor=colors.white,
                               alignment=TA_CENTER)

    status_panel = Table([
        [
            Paragraph("FLOW STATUS",        box_label),
            Paragraph("ANOMALY RATE",       box_label),
            Paragraph("SUPPLY HOURS",       box_label),
        ],
        [
            Paragraph(f"{avg_flow:.0f} m³/hr", box_style),
            Paragraph(f"{anom_rate:.1f}%",      box_style),
            Paragraph(f"{supply_hrs:.1f} hrs",  box_style),
        ],
        [
            Paragraph(f"Peak: {peak_flow:.0f} m³/hr", box_label),
            Paragraph(f"{total_anom} flags",           box_label),
            Paragraph(f"{len(windows)} window(s)",     box_label),
        ],
    ], colWidths=[W/3]*3, rowHeights=[0.5*cm, 0.7*cm, 0.45*cm])

    status_panel.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), box1_color),
        ("BACKGROUND",  (1,0), (1,-1), box2_color),
        ("BACKGROUND",  (2,0), (2,-1), box3_color),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("LINEAFTER",   (0,0), (1,-1), 1, colors.white),
    ]))
    story.append(status_panel)
    story.append(Spacer(1, 0.4*cm))

   

    # ── 1. Executive Summary ──────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", h1_style))
    anom_pct = (total_anom / max(len(df), 1)) * 100

    # Full-detector counts (if available)
    if df_full is not None and "model_vote" in df_full.columns:
        z_cnt  = int(df_full.get("anom_zscore",  pd.Series([0])).sum())
        iq_cnt = int(df_full.get("anom_iqr",     pd.Series([0])).sum())
        if_cnt = int(df_full.get("anom_iforest", pd.Series([0])).sum())
        pc_cnt = int(df_full.get("anom_pca",     pd.Series([0])).sum())
        fa_cnt = int(df_full.get("final_anomaly",pd.Series([0])).sum())
        model_note = (
            f" The 4-model ensemble (Z-score: {z_cnt}, IQR: {iq_cnt}, "
            f"Isolation Forest: {if_cnt}, PCA: {pc_cnt}) produced "
            f"{fa_cnt} final anomaly flags.")
    else:
        model_note = ""

    story.append(Paragraph(
            f"This report covers water distribution data for <b>{date_str}</b>. "
            f"A total of <b>{len(df):,}</b> readings were analysed from meter "
            f"<b>{METER_LABEL}</b>. The overall Quality of Service score is "
        f"<b>{qos:.1f}%</b> with <b>{total_anom}</b> anomalies detected "
        f"({anom_pct:.1f}% of readings).{model_note}",
        body_style))
    story.append(Spacer(1, 0.2*cm))

    sum_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Report Date",      date_str,
         "QoS Score",        f"{qos:.1f}%"],
        ["Total Readings",   f"{len(df):,}",
         "Status",           status_str(qos)],
        ["Average Flow",     f"{avg_flow:.1f} m\u00b3/hr",
         "Total Anomalies",  str(total_anom)],
        ["Peak Flow",        f"{peak_flow:.1f} m\u00b3/hr at {peak_ts}",
         "Spike Anomalies",  str(spike_anom)],
        ["Supply Windows",   str(len(windows)),
         "Night Anomalies",  str(night_anom)],
        ["Benchmark Samples",str(benchmark["samples"]) if benchmark else "N/A",
         "Z-score Anomalies",str(max(0, total_anom - spike_anom - night_anom))],
    ]
    col_w   = [W/4]*4
    sum_tbl = Table(sum_data, colWidths=col_w)
    sum_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",      (0,1), (0,-1),  "Helvetica-Bold"),
        ("FONTNAME",      (2,1), (2,-1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),
         [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#bdc3c7")),
        ("ALIGN",         (0,0), (-1,-1), "LEFT"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",       (0,0), (-1,-1), 5),
        ("TEXTCOLOR",     (1,2), (1,2),   qc),
        ("FONTNAME",      (1,2), (1,2),   "Helvetica-Bold"),
    ]))
    story.append(sum_tbl)
    story.append(Spacer(1, 0.4*cm))
    # ── AFTER the summary table (around line 1830) ──
    # Add this block:

    story.append(Paragraph("2. Benchmark Profile vs Today", h1_style))
    story.append(Paragraph(
        "The benchmark was constructed from Jan–Feb supply window data. "
        "Today's detected sessions are compared below.", body_style))
    story.append(Spacer(1, 0.15*cm))

    bm_table_data = [["Session", "Start Time", "End Time", "Duration (min)", 
                    "Peak Flow (m³/hr)", "Avg Flow (m³/hr)", "Source"]]

    # Benchmark row
    if benchmark:
        bm_s = f"{int(benchmark['start_min'])//60:02d}:{int(benchmark['start_min'])%60:02d}"
        bm_e = f"{int(benchmark['end_min'])//60:02d}:{int(benchmark['end_min'])%60:02d}"
        bm_table_data.append([
            "Benchmark", bm_s, bm_e,
            f"{benchmark['duration_min']:.0f}",
            f"{benchmark['peak_flow']:.1f}",
            f"{benchmark['avg_flow']:.1f}",
            f"Jan–Feb ({benchmark['samples']} days)"
        ])

    # Today's windows
    for i, w in enumerate(windows, 1):
        bm_table_data.append([
            f"Today #{i}",
            w["start"].strftime("%H:%M"),
            w["end"].strftime("%H:%M"),
            f"{w['duration']:.0f}",
            f"{w['peak']:.1f}",
            f"{w['avg']:.1f}",
            "Today"
        ])

    bm_tbl = Table(bm_table_data, 
                colWidths=[2.2*cm, 2*cm, 2*cm, 2.8*cm, 3.2*cm, 3.2*cm, 2*cm])
    bm_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#bdc3c7")),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",       (0,0), (-1,-1), 5),
        # Highlight benchmark row in blue
        ("BACKGROUND",    (0,1), (-1,1),  colors.HexColor("#d6eaf8")),
    ]))
    story.append(bm_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── 2. Benchmark Profile ──────────────────────────────────────────────
    story.append(Paragraph("3. Benchmark Parameters", h1_style))

    if benchmark:
        bm_s = (f"{int(benchmark['start_min'])//60:02d}:"
                f"{int(benchmark['start_min'])%60:02d}")
        bm_e = (f"{int(benchmark['end_min'])//60:02d}:"
                f"{int(benchmark['end_min'])%60:02d}")
        story.append(Paragraph(
            f"The benchmark is derived from <b>{benchmark['samples']}</b> "
            f"historical supply windows. The primary supply session begins around "
            f"<b>{bm_s}</b> and ends around <b>{bm_e}</b>, lasting approximately "
            f"<b>{benchmark['duration_min']:.0f} minutes</b> with a peak flow of "
            f"<b>{benchmark['peak_flow']:.1f} m\u00b3/hr</b>.",
            body_style))
        story.append(Spacer(1, 0.2*cm))
        bm_data = [
            ["Parameter",    "Benchmark Value",                       "Tolerance"],
            ["Start Time",   bm_s,                                    "\u00b1 30 min"],
            ["End Time",     bm_e,                                    "\u00b1 30 min"],
            ["Duration",     f"{benchmark['duration_min']:.0f} min",  "\u00b1 30 min"],
            ["Peak Flow",    f"{benchmark['peak_flow']:.1f} m\u00b3/hr", "\u00b1 20%"],
            ["Average Flow", f"{benchmark['avg_flow']:.1f} m\u00b3/hr", "\u00b1 20%"],
            ["Sample Size",  str(benchmark["samples"]),               "—"],
        ]
        bm_tbl = Table(bm_data, colWidths=[W*0.35, W*0.38, W*0.27])
        bm_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#2a6496")),
            ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTNAME",     (0,1), (0,-1),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#bdc3c7")),
            ("ALIGN",        (0,0), (-1,-1), "LEFT"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",      (0,0), (-1,-1), 5),
        ]))
        story.append(bm_tbl)
    else:
        story.append(Paragraph(
            "Insufficient historical data for benchmark (need 30+ days). "
            "QoS timing score unavailable — flow scores only.",
            body_style))
    story.append(Spacer(1, 0.4*cm))

    # ── 3. Today's Supply Windows ─────────────────────────────────────────
    story.append(Paragraph("3. Today's Supply Windows", h1_style))
    if windows:
        win_data = [["#", "Start", "End", "Duration",
                     "Peak Flow", "Avg Flow", "Status"]]
        for i, w in enumerate(windows, 1):
            w_start_m = w["start"].hour * 60 + w["start"].minute
            ok = (benchmark and
                  abs(w_start_m - benchmark["start_min"]) <= 30 and
                  (abs(w["peak"] - benchmark["peak_flow"]) /
                   benchmark["peak_flow"]) <= 0.20)
            win_data.append([
                str(i),
                w["start"].strftime("%H:%M"),
                w["end"].strftime("%H:%M"),
                f"{w['duration']:.0f} min",
                f"{w['peak']:.1f} m\u00b3/hr",
                f"{w['avg']:.1f} m\u00b3/hr",
                "Normal" if ok else "Deviated",
            ])
        win_tbl = Table(win_data,
                        colWidths=[0.7*cm, 2.2*cm, 2.2*cm,
                                   2.4*cm, 3*cm, 3*cm, 3.9*cm])
        st_ok  = colors.HexColor("#d5f5e3")
        st_bad = colors.HexColor("#fadbd8")
        win_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#1a3a5c")),
            ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#bdc3c7")),
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",      (0,0), (-1,-1), 5),
        ]))
        for i, w in enumerate(windows, 1):
            w_start_m = w["start"].hour * 60 + w["start"].minute
            ok = benchmark and abs(w_start_m - benchmark["start_min"]) <= 30
            win_tbl.setStyle(TableStyle([
                ("BACKGROUND", (6, i), (6, i), st_ok if ok else st_bad),
                ("TEXTCOLOR",  (6, i), (6, i),
                 colors.HexColor("#1e8449") if ok else colors.HexColor("#922b21")),
                ("FONTNAME",   (6, i), (6, i), "Helvetica-Bold"),
            ]))
        story.append(win_tbl)
    else:
        story.append(Paragraph(
            "No supply windows detected today (flow stayed below 1.0 m\u00b3/hr threshold).",
            body_style))
    story.append(Spacer(1, 0.4*cm))

    # ── 4. Anomaly Details ────────────────────────────────────────────────
    story.append(Paragraph("4. Anomaly Details", h1_style))
    if anomalies:
        anom_data = [["#", "Description", "Severity"]]
        for i, a in enumerate(anomalies, 1):
            sev = "High" if any(str(x) in a for x in range(60, 999)) else "Medium"
            anom_data.append([str(i), a, sev])
        anom_tbl = Table(anom_data, colWidths=[0.7*cm, 13.7*cm, 3*cm])
        anom_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#c0392b")),
            ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#fdf2f2"), colors.HexColor("#fce8e8")]),
            ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#e8a0a0")),
            ("ALIGN",        (0,0), (0,-1),  "CENTER"),
            ("ALIGN",        (2,0), (2,-1),  "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",      (0,0), (-1,-1), 5),
        ]))
        story.append(anom_tbl)
    else:
        ok_tbl = Table(
            [["No anomalies detected — service operating within benchmark parameters"]],
            colWidths=[W]
        )
        ok_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#d5f5e3")),
            ("TEXTCOLOR",  (0,0), (-1,-1), colors.HexColor("#1e8449")),
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",    (0,0), (-1,-1), 10),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#a9dfbf")),
        ]))
        story.append(ok_tbl)
    story.append(Spacer(1, 0.4*cm))

# ── 5. Multi-Model Anomaly Summary (NEW — from run_full_detectors) ────
    if df_full is not None and "model_vote" in df_full.columns:
        story.append(Paragraph("5. Multi-Model Anomaly Analysis", h1_style))
        story.append(Paragraph(
            "The following table shows anomaly counts per detection method "
            "from the 4-model ensemble applied to today's batch data.",
            body_style))
        story.append(Spacer(1, 0.15*cm))

        mm_data = [
            ["Method",               "Anomalies Flagged", "Description"],
            ["Z-score",
             str(int(df_full.get("anom_zscore",  pd.Series([0])).sum())),
             "Supply-hour readings > 3σ from mean"],
            ["IQR",
             str(int(df_full.get("anom_iqr",     pd.Series([0])).sum())),
             "Beyond Q1 − 2.5×IQR or Q3 + 2.5×IQR fence"],
            ["Isolation Forest",
             str(int(df_full.get("anom_iforest", pd.Series([0])).sum())),
             "Tree-based outlier score (contamination=5%)"],
            ["PCA Autoencoder",
             str(int(df_full.get("anom_pca",     pd.Series([0])).sum())),
             "Reconstruction error > mean + 3σ"],
            ["Final (3+ / rule)",
             str(int(df_full.get("final_anomaly",pd.Series([0])).sum())),
             "≥3 models agree, or rule-based (spike / night / drop)"],
        ]
        mm_tbl = Table(mm_data, colWidths=[3.5*cm, 3.5*cm, W - 7*cm])
        mm_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#1a3a5c")),
            ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTNAME",     (0,1), (0,-1),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#bdc3c7")),
            ("ALIGN",        (1,0), (1,-1),  "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",      (0,0), (-1,-1), 5),
        ]))
        story.append(mm_tbl)
        story.append(Spacer(1, 0.4*cm))

    # ── 6. Top Anomalous Readings ─────────────────────────────────────────
    section_num = 6
    if "is_anomaly" in df.columns and total_anom > 0:
        story.append(Paragraph(f"{section_num}. Top Anomalous Readings", h1_style))
        top_anom = (df[df["is_anomaly"] == 1]
                    .sort_values("flow_rate", ascending=False)
                    .head(10))
        top_data = [["#", "Timestamp", "Flow Rate (m\u00b3/hr)", "Type"]]
        for i, (_, row) in enumerate(top_anom.iterrows(), 1):
            ts_s = pd.Timestamp(row["timestamp"]).strftime("%H:%M:%S")
            fl   = row["flow_rate"]
            typ  = ("Spike" if fl > SPIKE_THRESHOLD else
                    "Night" if (pd.Timestamp(row["timestamp"]).hour >= NIGHT_START_HR or
                                pd.Timestamp(row["timestamp"]).hour <= NIGHT_END_HR)
                    else "Z-score")
            top_data.append([str(i), ts_s, f"{fl:.1f}", typ])
        top_tbl = Table(top_data, colWidths=[0.7*cm, 5*cm, 6*cm, 5.7*cm])
        top_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#922b21")),
            ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#fdf2f2"), colors.HexColor("#fce8e8")]),
            ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#e8a0a0")),
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",      (0,0), (-1,-1), 5),
        ]))
        story.append(top_tbl)
        story.append(Spacer(1, 0.4*cm))
    section_num += 1    # ← outside the if, always runs

    # ── 7. Daily Anomaly Detail ───────────────────────────────────────────
    story.append(Paragraph(f"{section_num}. Daily Anomaly Detail", h1_style))
    section_num += 1

    if anomalies:
        story.append(Paragraph(
            f"The following {len(anomalies)} anomaly/anomalies were detected today "
            f"against the Jan–Feb benchmark.", body_style))
        story.append(Spacer(1, 0.15*cm))

        anom_data = [["#", "Anomaly Description", "Category"]]
        for i, a in enumerate(anomalies, 1):
            cat = ("Timing"    if any(x in a for x in ["Start", "End", "Duration"]) else
                   "Flow Rate" if any(x in a for x in ["Peak", "Avg"]) else
                   "Other")
            anom_data.append([str(i), a, cat])

        anom_tbl = Table(anom_data, colWidths=[0.7*cm, W - 3.5*cm, 2.8*cm])
        anom_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#922b21")),
            ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),
             [colors.HexColor("#fdf2f2"), colors.HexColor("#fce8e8")]),
            ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#e8a0a0")),
            ("ALIGN",         (0,0), (0,-1),  "CENTER"),
            ("ALIGN",         (2,0), (2,-1),  "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",       (0,0), (-1,-1), 5),
        ]))
        story.append(anom_tbl)
    else:
        story.append(Paragraph(
            "No supply-window anomalies detected today. "
            "Service is operating within benchmark parameters.", body_style))
    story.append(Spacer(1, 0.4*cm))

    # ── 7. Forecast Summary (NEW) ─────────────────────────────────────────
    fc, lo, hi, fts, sm_vals = forecast_flow(df)
    if fc is not None:
        story.append(Paragraph(f"{section_num}. Flow Forecast", h1_style))
        section_num += 1
        trend_direction = (
                    '<font color="#27ae60">Increasing</font>'
                    if fc[-1] > fc[0] else
                    '<font color="#e74c3c">Decreasing</font>'
                )
        story.append(Paragraph(
            f"Exponential smoothing (\u03b1=0.3) forecast for the next "
            f"<b>{FORECAST_STEPS} readings</b>. "
            f"Predicted range: <b>{lo.min():.1f} \u2013 {hi.max():.1f} m\u00b3/hr</b> "
            f"(95% CI). Trend direction: {trend_direction}.",
            body_style))
        story.append(Spacer(1, 0.15*cm))
        fc_preview = [["Step", "Forecast (m³/hr)", "Lower 95%", "Upper 95%"]]
        for i in range(min(10, len(fc))):
            fc_preview.append([
                str(i + 1),
                f"{fc[i]:.1f}",
                f"{lo[i]:.1f}",
                f"{hi[i]:.1f}",
            ])
        fc_tbl = Table(fc_preview, colWidths=[W/4]*4)
        fc_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#2a6496")),
            ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#bdc3c7")),
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",      (0,0), (-1,-1), 5),
        ]))
        story.append(fc_tbl)
        story.append(Spacer(1, 0.4*cm))
    section_num += 1 

    # ── 8. Pattern Analysis Summary (NEW) ────────────────────────────────
    if pat_summary:
        story.append(Paragraph(f"{section_num}. Pattern Analysis (Jan–Feb {PATTERN_YEAR})",
                                h1_style))
        section_num += 1
        story.append(Paragraph(
            f"K-Means clustering (k={PATTERN_K}) was applied to Jan–Feb {PATTERN_YEAR} "
            f"daily flow curves. The modal cluster centroid was set as the benchmark "
            f"pattern. Days within {SIM_THRESHOLD}% similarity are marked as matching.",
            body_style))
        story.append(Spacer(1, 0.15*cm))
        pat_data = [
            ["Metric", "Value"],
            ["Total days analysed",     str(pat_summary.get("total_days", "N/A"))],
            ["January days",            str(pat_summary.get("jan_days",   "N/A"))],
            ["February days",           str(pat_summary.get("feb_days",   "N/A"))],
            ["Matching days (≥threshold)", str(pat_summary.get("n_match",  "N/A"))],
            ["Deviant days (<threshold)",  str(pat_summary.get("n_deviant","N/A"))],
            ["Average similarity",      f"{pat_summary.get('avg_similarity', 0):.1f}%"],
            ["Best matching day",       str(pat_summary.get("best_day",  "N/A"))[5:]],
            ["Worst matching day",      str(pat_summary.get("worst_day", "N/A"))[5:]],
            ["Modal cluster index",     str(pat_summary.get("modal_cluster", "N/A"))],
            ["Similarity threshold",    f"{SIM_THRESHOLD}%"],
        ]
        pat_tbl = Table(pat_data, colWidths=[W*0.55, W*0.45])
        pat_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#1a3a5c")),
            ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTNAME",     (0,1), (0,-1),  "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#bdc3c7")),
            ("ALIGN",        (1,0), (1,-1),  "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",      (0,0), (-1,-1), 5),
        ]))
        story.append(pat_tbl)
        story.append(Spacer(1, 0.4*cm))
        # ── Add AFTER the make_pdf_chart() block (around line 2129) ──

    # Embed pattern charts (multi-day overlay, K-Means, similarity bar, best/worst)
    if pat_chart_bufs:
        pat_titles = [
            f"Multi-Day Overlay — Jan–Feb {PATTERN_YEAR} (each line = 1 day, red = median)",
            f"K-Means Cluster Centroids (k={PATTERN_K}) — orange = benchmark cluster",
            f"Daily Pattern Similarity — green ≥ {SIM_THRESHOLD}%, red = deviant",
            "Best-match vs Worst-match Days vs Benchmark",
        ]
        for i, (buf_p, title) in enumerate(zip(pat_chart_bufs, pat_titles)):
            story.append(Paragraph(f"Figure {i+1}. {title}", 
                                    ParagraphStyle("FigCaption", fontSize=7.5,
                                                textColor=colors.HexColor("#7f8c8d"),
                                                alignment=TA_CENTER)))
            buf_p.seek(0)
            img_p = RLImage(buf_p, width=W, height=W * 0.38)
            story.append(img_p)
            story.append(Spacer(1, 0.3*cm))



    # ── 9. Flow Analysis Charts ───────────────────────────────────────────
    story.append(Paragraph(f"{section_num}. Flow Analysis Charts", h1_style))
    section_num += 1
    try:
        chart_buf = make_pdf_chart(df, benchmark, windows, qos)
        chart_buf.seek(0)
        img = RLImage(chart_buf, width=W, height=W * 0.57)
        story.append(img)
    except Exception as e:
        story.append(Paragraph(f"Chart generation error: {e}", body_style))
    story.append(Spacer(1, 0.4*cm))

    # ── 10. Recommendations ───────────────────────────────────────────────
# ── Recommendations ───────────────────────────────────────────────────
    story.append(Paragraph(f"{section_num}. Recommendations", h1_style))
    section_num += 1
    recs = []

    # Anomaly-driven recommendations (check these first)
    if any("Start time" in a for a in anomalies):
        recs.append(("Scheduling",
                      "Supply start time is deviating from schedule. "
                      "Check pump station readiness and valve operations."))
    if any("End time" in a or "Duration" in a for a in anomalies):
        recs.append(("Duration",
                      "Supply duration is inconsistent. "
                      "Investigate upstream pressure or demand changes."))
    if any("Peak flow" in a or "Avg flow" in a for a in anomalies):
        recs.append(("Flow Rate",
                      "Flow rate deviation detected. "
                      "Inspect pump performance and pipe conditions."))
    if spike_anom > 0:
        recs.append(("Spike Events",
                      f"{spike_anom} spike(s) above {SPIKE_THRESHOLD} m\u00b3/hr detected. "
                      "Verify meter calibration and check for pressure surges."))
    if night_anom > 0:
        recs.append(("Night Flow",
                      f"{night_anom} readings above threshold during non-supply hours. "
                      "Check for unauthorized connections or leakage."))
    if pat_summary and pat_summary.get("n_deviant", 0) > 5:
        recs.append(("Pattern Drift",
                      f"{pat_summary['n_deviant']} deviant days found in Jan–Feb "
                      f"{PATTERN_YEAR}. Benchmark may need updating to reflect "
                      "seasonal or infrastructure changes."))

    # Supply / flow gap recommendations
    if len(windows) == 0:
        recs.append(("No Supply",
                      "No supply window detected today. "
                      "Verify pump station is operational and check valve status."))
    if benchmark and avg_flow < benchmark.get("avg_flow", 0) * 0.8:
        recs.append(("Flow Rate Monitoring",
                      f"Average flow ({avg_flow:.1f} m³/hr) is >20% below benchmark "
                      f"({benchmark.get('avg_flow', 0):.1f} m³/hr). "
                      "Inspect pump performance and check for pipe blockages."))

    # All clear — only triggers if nothing above matched
    if not recs:
        recs.append(("All Clear",
                      "No immediate action required. Continue routine monitoring."))

    # Always-present standing recommendations
    recs.append(("Reverse Flow",
                  "Monitor non-supply-hour flow. Any sustained readings >200 units "
                  "during off-hours should trigger investigation for unauthorized "
                  "connections or pressure issues."))
    recs.append(("Benchmark",
                  "Update benchmark monthly using a rolling 60-day window "
                  "to capture seasonal demand variations."))

    rec_data = [["Category", "Recommendation"]]
    for cat, rec in recs:
        rec_data.append([cat, rec])
    rec_tbl = Table(rec_data, colWidths=[3.5*cm, W - 3.5*cm])
    rec_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",     (0,1), (0,-1),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#bdc3c7")),
        ("ALIGN",        (0,0), (0,-1),  "CENTER"),
        ("ALIGN",        (1,0), (1,-1),  "LEFT"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",      (0,0), (-1,-1), 6),
        ("TEXTCOLOR",    (0,-1),(0,-1),  colors.HexColor("#2a6496")),
    ]))
    story.append(rec_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── 7-Day QoS Trend (NEW) ─────────────────────────────────────────────
    story.append(Paragraph(f"{section_num}. 7-Day QoS Trend", h1_style))
    section_num += 1
    trend_df = db_load_7day_trend()
    if not trend_df.empty:
        trend_data = [["Date", "QoS%", "Avg Flow", "Peak Flow",
                       "Anomalies", "Spikes", "Status"]]
        for _, row in trend_df.iterrows():
            trend_data.append([
                str(row["date"]),
                f"{row['qos']:.1f}%",
                f"{row['avg_flow']:.0f}",
                f"{row['peak_flow']:.0f}",
                str(int(row["total_anomalies"])),
                str(int(row["spike_anomalies"])),
                str(row["status"]),
            ])
        trend_tbl = Table(trend_data,
                          colWidths=[2.8*cm, 1.8*cm, 2.4*cm,
                                     2.4*cm, 2.4*cm, 2*cm, 3.6*cm])
        trend_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#1a3a5c")),
            ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),
             [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#bdc3c7")),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",       (0,0), (-1,-1), 5),
        ]))
        # Colour-code QoS% column
        for i, row in enumerate(trend_df.itertuples(), 1):
            q = row.qos
            qclr = (colors.HexColor("#d5f5e3") if q >= 85 else
                    colors.HexColor("#fef9e7") if q >= 70 else
                    colors.HexColor("#fadbd8"))
            trend_tbl.setStyle(TableStyle([
                ("BACKGROUND", (1, i), (1, i), qclr),
            ]))
        story.append(trend_tbl)
    else:
        story.append(Paragraph(
            "No historical QoS data yet — trend will appear after the first full day.",
            body_style))
    story.append(Spacer(1, 0.5*cm))
    # ── Add BEFORE the footer ──

    story.append(Paragraph(f"{section_num}. Today's Service Classification", h1_style))
    section_num += 1

    n_anom = len(anomalies)
    day_class = ("Perfect day (0 anomalies)" if n_anom == 0 else
                f"Minor issues ({n_anom} anomalies)" if n_anom <= 2 else
                f"Critical day ({n_anom} anomalies)")
    qos_band  = ("Excellent (≥85%)" if qos >= 85 else
                "Good (70–85%)"    if qos >= 70 else
                "Poor (<70%)")

    class_data = [
        ["Classification", "Value", "Description"],
        ["Day Category",  day_class,
        "Perfect=0 anomalies, Minor=1-2, Critical=3+"],
        ["QoS Band",      qos_band,
        "Excellent≥85%, Good 70-85%, Poor<70%"],
        ["Anomaly Count", str(n_anom),
        "Total supply-window deviations from benchmark"],
        ["Supply Windows",str(len(windows)),
        "Active supply sessions detected today"],
        ["Benchmark Days",str(benchmark["samples"]) if benchmark else "N/A",
        "Jan–Feb days used to build benchmark"],
    ]

    cls_tbl = Table(class_data, colWidths=[3.5*cm, 4*cm, W - 7.5*cm])
    cls_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",      (0,1), (0,-1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#bdc3c7")),
        ("ALIGN",         (0,0), (-1,-1), "LEFT"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",       (0,0), (-1,-1), 5),
    ]))
    story.append(cls_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Footer ────────────────────────────────────────────────────────────

    # ── Footer ────────────────────────────────────────────────────────────
    foot_tbl = Table(
            [[Paragraph(
                f"VMC Water Monitor  \u00b7  Auto-generated QoS Report  \u00b7  {now_str}  "
                f"\u00b7  Meter: {METER_LABEL}",
                footer_style
            )]],
        colWidths=[W], rowHeights=[0.7*cm]
    )
    foot_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#1a3a5c")),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(foot_tbl)

    doc.build(story)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# VMC API — LOGIN + BATCH FETCH  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.verify = False
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept":     "application/json, text/plain, */*",
    "Referer":    f"{VMC_BASE}/dashboard",
    "Origin":     VMC_BASE,
})
_token = None


def try_login() -> bool:
    global _token
    if _token:
        return True
    try:
        SESSION.get(f"{VMC_BASE}/login", timeout=8)
    except Exception:
        pass
    for path in ["/login", "/api/login", "/api/auth", "/api/token"]:
        try:
            r = SESSION.post(
                f"{VMC_BASE}{path}",
                data={"username": VMC_USER, "password": VMC_PASS},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=8, allow_redirects=True,
            )
            if r.status_code == 200 and "<title>login</title>" not in r.text.lower():
                _token = "session"
                log.info("Login OK via %s", path)
                return True
            r2 = SESSION.post(
                f"{VMC_BASE}{path}",
                json={"username": VMC_USER, "password": VMC_PASS},
                timeout=8,
            )
            if r2.status_code == 200:
                d   = r2.json()
                tok = d.get("token") or d.get("access_token") or d.get("jwt")
                if tok:
                    SESSION.headers["Authorization"] = f"Bearer {tok}"
                _token = tok or "session"
                log.info("Login OK via %s (JSON)", path)
                return True
        except Exception as e:
            log.debug("Login %s failed: %s", path, e)
    log.warning("All login attempts failed")
    return False


def _parse_ts(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if ts.tzinfo is not None:
            # Convert to UTC naive, then add IST offset
            utc_naive = ts.replace(tzinfo=None) - ts.utcoffset()
            return utc_naive + IST_OFFSET
        # No tzinfo — assume already IST, return as-is
        return ts
    except Exception:
        return None


def _extract_field(row: dict, fallback_ts: datetime):
    ts = fallback_ts
    for tk in ["DateTime", "dateTime", "timestamp", "time", "Timestamp", "ts"]:
        raw = row.get(tk)
        if raw:
            parsed = _parse_ts(str(raw)[:25])
            if parsed:
                ts = parsed
                break
    numeric = {}
    for k, v in row.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lk = k.lower()
            if not any(x in lk for x in ["id", "time", "stamp", "index", "seq"]):
                numeric[k] = float(v)
    if not numeric:
        return None, ts
    for pk in ["Value", "value", "flow", "Flow", "flowRate", "flow_rate",
               "reading", "val", "data", "Flow_Rate", "FlowRate", "rate"]:
        if pk in numeric:
            return numeric[pk], ts
    nonzero = {k: v for k, v in numeric.items() if v != 0.0}
    if nonzero:
        return next(iter(nonzero.values())), ts
    return next(iter(numeric.values())), ts


def fetch_batch_24hr() -> list[dict]:
    global _token
    now   = datetime.now()
    start = now - timedelta(hours=BATCH_WINDOW_HOURS)
    log.info("Fetching %dhr batch: %s → %s",
             BATCH_WINDOW_HOURS,
             start.strftime("%Y-%m-%d %H:%M"),
             now.strftime("%Y-%m-%d %H:%M"))
    for path in HISTORY_API_PATHS:
        log.info("Trying endpoint: %s", path)
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
            log.warning("Endpoint %s network error: %s", path, e)
            continue
        if "<title>login</title>" in r.text.lower():
            log.warning("Session expired on %s — re-logging in", path)
            _token = None
            try_login()
            continue
        if r.status_code != 200:
            log.warning("Endpoint %s returned HTTP %s", path, r.status_code)
            continue
        try:
            data = r.json()
        except Exception:
            log.warning("Endpoint %s non-JSON: %s", path, r.text[:200])
            continue
        records = _parse_batch_response(data, now)
        non_zero = [r for r in records if r["flow_rate"] > 0]
        log.info("Endpoint %s — %d records, %d non-zero", path, len(records), len(non_zero))
        if len(non_zero) > 1:
            log.info("Endpoint %s OK — using %d non-zero records", path, len(non_zero))
            return non_zero
        elif len(records) > 1:
            log.warning("Endpoint %s has %d records but ALL ARE ZERO — trying next", path, len(records))
        elif len(records) == 1:
            log.warning("Endpoint %s returned only 1 row — trying next...", path)
        else:
            log.warning("Endpoint %s returned 0 records — trying next...", path)
    log.error("All history endpoints exhausted.")
    return []


def _parse_batch_response(data, fallback_ts: datetime) -> list[dict]:
    records = []
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
    elif (isinstance(data, list) and data
          and isinstance(data[0], (list, tuple))):
        for pt in data:
            try:
                ts   = datetime.utcfromtimestamp(float(pt[0]) / 1000) + IST_OFFSET
                flow = float(pt[1])
                records.append({"timestamp": ts, "flow_rate": flow})
            except Exception:
                continue
    elif isinstance(data, dict) and "data" in data:
        pts = data["data"]
        if pts and isinstance(pts[0], dict):
            for row in pts:
                flow, ts = _extract_field(row, fallback_ts)
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
            flow, ts = _extract_field(row, fallback_ts)
            if flow is not None:
                records.append({"timestamp": ts, "flow_rate": flow})
    elif isinstance(data, dict):
        flow, ts = _extract_field(data, fallback_ts)
        if flow is not None:
            records.append({"timestamp": ts, "flow_rate": flow})
    seen   = set()
    unique = []
    for rec in records:
        key = rec["timestamp"].isoformat()
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    unique.sort(key=lambda x: x["timestamp"])
    return unique


def fetch_single_reading() -> dict | None:
    global _token
    now   = datetime.now()
    start = now - timedelta(hours=1)
    try:
        r = SESSION.get(
            f"{VMC_BASE}{REALTIME_API_PATH}",
            params={
                "objectname": OBJECT_NAME,
                "startTime":  start.strftime("%Y-%m-%d %H:%M:%S"),
                "endTime":    now.strftime("%Y-%m-%d %H:%M:%S"),
            },
            timeout=15,
        )
    except Exception as e:
        log.warning("Heartbeat fetch error: %s", e)
        return None
    if "<title>login</title>" in r.text.lower():
        _token = None
        try_login()
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    records = _parse_batch_response(data, now)
    if not records:
        return None
    latest = records[-1]
    return {"timestamp": latest["timestamp"], "flow_rate": latest["flow_rate"]}


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION — simple tagger  (unchanged, used for initial row storage)
# ─────────────────────────────────────────────────────────────────────────────
def tag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_anomaly"] = 0
    df.loc[df["flow_rate"] > SPIKE_THRESHOLD, "is_anomaly"] = 1
    df.loc[df["flow_rate"] < 0,               "is_anomaly"] = 1
    df["_hour"]   = df["timestamp"].dt.hour
    night_mask    = (df["_hour"] >= NIGHT_START_HR) | (df["_hour"] <= NIGHT_END_HR)
    df.loc[night_mask & (df["flow_rate"] > NIGHT_FLOW_LIMIT), "is_anomaly"] = 1
    active = df["flow_rate"] > 0
    if active.sum() > 10:
        z = np.abs(stats.zscore(df.loc[active, "flow_rate"]))
        df.loc[df.index[active][z > Z_THRESHOLD], "is_anomaly"] = 1
    df = df.drop(columns=["_hour"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM TEXT REPORT BUILDER  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def build_daily_report(df: pd.DataFrame) -> str:
    now_ist  = datetime.now(IST)
    date_str = now_ist.strftime("%d %b %Y, %H:%M IST")
    if df.empty:
        return (
            f"💧 *{STATION_NAME} — Daily Report*\n"
            f"📅 {date_str}\n\n"
            f"⚠️ No readings in the last 24 hours.\n"
            f"Please check the VMC API connection."
        )
    total     = len(df)
    avg_flow  = df["flow_rate"].mean()
    peak_flow = df["flow_rate"].max()
    min_flow  = df[df["flow_rate"] > 0]["flow_rate"].min() if (df["flow_rate"] > 0).any() else 0
    peak_ts   = df.loc[df["flow_rate"].idxmax(), "timestamp"]
    try:
        peak_str = pd.Timestamp(peak_ts).strftime("%H:%M")
    except Exception:
        peak_str = str(peak_ts)[:16]
    total_anom = int(df["is_anomaly"].sum())
    anom_pct   = (total_anom / total * 100) if total > 0 else 0
    df["hour"]  = df["timestamp"].dt.hour
    night_mask  = (df["hour"] >= NIGHT_START_HR) | (df["hour"] <= NIGHT_END_HR)
    night_anom  = int(df[night_mask & (df["is_anomaly"] == 1)].shape[0])
    spike_anom  = int((df["flow_rate"] > SPIKE_THRESHOLD).sum())
    anom_by_hr  = df[df["is_anomaly"] == 1].groupby("hour").size()
    top_hours   = anom_by_hr.nlargest(3)
    top_hrs_str = ""
    for hr, cnt in top_hours.items():
        top_hrs_str += f"  {hr:02d}:00 — {cnt} anomalies\n"
    if not top_hrs_str:
        top_hrs_str = "  ✅ None\n"
    anom_events = df[df["is_anomaly"] == 1].sort_values("flow_rate", ascending=False).head(5)
    events_str  = ""
    for i, (_, row) in enumerate(anom_events.iterrows(), 1):
        try:
            ts_str = pd.Timestamp(row["timestamp"]).strftime("%H:%M")
        except Exception:
            ts_str = str(row["timestamp"])[:16]
        flag = "🔴" if row["flow_rate"] > SPIKE_THRESHOLD else "🟡"
        events_str += f"  {i}. {ts_str} — {row['flow_rate']:.1f} m³/hr {flag}\n"
    if not events_str:
        events_str = "  ✅ None\n"
    if total_anom == 0:          status = "✅ All Clear"
    elif anom_pct < 2:           status = f"⚠️ Minor ({anom_pct:.1f}%)"
    elif anom_pct < 5:           status = f"🔶 Moderate ({anom_pct:.1f}%)"
    else:                        status = f"🔴 High Alert ({anom_pct:.1f}%)"
    return (
        f"💧 *{STATION_NAME} — Daily Report*\n"
        f"📅 {date_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"📊 *Flow Summary (24 hr batch)*\n"
        f"  • Readings: {total:,}\n"
        f"  • Average: {avg_flow:.1f} m³/hr\n"
        f"  • Peak: {peak_flow:.1f} m³/hr at {peak_str}\n"
        f"  • Min (active): {min_flow:.1f} m³/hr\n\n"
        f"🚨 *Anomalies: {total_anom} ({anom_pct:.1f}%)*\n"
        f"  • Spike (>{SPIKE_THRESHOLD}): {spike_anom}\n"
        f"  • Night flow: {night_anom}\n"
        f"  • Z-score/pattern: {max(0, total_anom - spike_anom - night_anom)}\n\n"
        f"⏰ *Peak Anomaly Hours*\n"
        f"{top_hrs_str}\n"
        f"⚠️ *Top Events*\n"
        f"{events_str}\n"
        f"📌 *Status: {status}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_VMC Water Monitor · Auto-generated (batch mode)_"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULED JOBS
# ─────────────────────────────────────────────────────────────────────────────

def job_daily_batch_fetch_and_report():
    """
    THE MAIN JOB — runs once per day at REPORT_HOUR:REPORT_MINUTE IST.

    Steps (original + new):
      1.  Fetch 24-hour batch from VMC API (single call)           [original]
      2.  Simple anomaly tagging + store all rows to SQLite         [original]
      3.  Compute stats (avg, peak, night, spike)                   [original]
      4.  Build supply windows + QoS vs benchmark                   [original]
      5.  Send Telegram text report                                 [original]
      6.  Send 24hr flow chart                                      [original]
      7.  Send hourly bar chart                                     [original]
      ── NEW steps below ───────────────────────────────────────────────────
      8.  Run 4-model full detector (Z-score/IQR/IF/PCA)           [NEW]
      9.  Send EDA charts (time series, rolling mean, histogram)    [NEW]
      10. Send anomaly model charts (bar, timeline, panels, heatmap)[NEW]
      11. Send forecast chart (exp. smoothing + residuals)          [NEW]
      12. Fetch Jan–Feb pattern data + run K-Means analysis         [NEW]
      13. Send pattern charts (overlap, clusters, similarity, best/worst)[NEW]
      14. Generate enhanced PDF (all new sections included)         [enhanced]
      15. Save QoS + benchmark snapshot to DB                       [original]
    """
    log.info("=== Daily batch job started ===")

    # ── 1. Fetch ──────────────────────────────────────────────────────────
    if not try_login():
        log.warning("Login failed — attempting fetch anyway")
    records = fetch_batch_24hr()

    if not records:
        log.error("No data from API — sending empty report")
        report = build_daily_report(pd.DataFrame())
        send_message(report)
        db_log_report(report, "daily")
        return

   
# ── 2. Simple tag + store ─────────────────────────────────────────────
    df = pd.DataFrame(records)
    df = tag_anomalies(df)

    db_rows = [
        (row["timestamp"].isoformat(), row["flow_rate"], int(row["is_anomaly"]))
        for _, row in df.iterrows()
    ]
    db_insert_batch(db_rows)

    # ── STRICT 24-HOUR FILTER — applied immediately after DB insert ───────
    # Insert full data to DB for history, but analyse only today's 24 hrs
    cutoff_24hr = datetime.now() - timedelta(hours=24)
    df = df[df["timestamp"] >= cutoff_24hr].copy().reset_index(drop=True)
    log.info("24hr filter applied: %d rows kept for today's analysis", len(df))

    # ── 3. Compute stats ──────────────────────────────────────────────────
    total_anom = int(df["is_anomaly"].sum())
    avg_flow   = df["flow_rate"].mean()
    peak_flow  = df["flow_rate"].max()
    peak_ts    = df.loc[df["flow_rate"].idxmax(), "timestamp"]
    try:
        peak_str = pd.Timestamp(peak_ts).strftime("%H:%M")
    except Exception:
        peak_str = "N/A"
    df["_h"]   = df["timestamp"].dt.hour
    night_mask = (df["_h"] >= NIGHT_START_HR) | (df["_h"] <= NIGHT_END_HR)
    night_anom = int(df[night_mask & (df["is_anomaly"] == 1)].shape[0])
    spike_anom = int((df["flow_rate"] > SPIKE_THRESHOLD).sum())

 
# ── 4. Benchmark + QoS ───────────────────────────────────────────────
# ── 4. Benchmark + QoS ───────────────────────────────────────────────
    # Try DB-derived benchmark first (30+ days history)
    con       = sqlite3.connect(DB_PATH)
    benchmark = build_benchmark(con)
    con.close()

    # FALLBACK: hardcoded until 30 days of data accumulate
    if benchmark is None:
        benchmark = {
            "start_hour":   0.0,     # midnight — supply runs all day
            "end_hour":     23.9,    # ends near midnight
            "duration":     1380.0,  # ~23 hours in minutes
            "peak":         5500.0,  # slightly above your observed 5364 peak
            "avg":          1000.0,  # matches your observed 1005 avg
            "samples":      1,
            # Legacy keys used by compute_qos() and PDF sections
            "start_min":    0.0,
            "end_min":      1434.0,  # 23 hrs 54 min in minutes
            "duration_min": 1380.0,
            "peak_flow":    5500.0,
            "avg_flow":     1000.0,
        }
        log.warning("No historical benchmark — using hardcoded fallback (24hr continuous supply)")

    # Use new detect_supply_windows_df (mirrors Streamlit) for hour_frac fields
    cutoff_24hr = datetime.now() - timedelta(hours=24)  # ← recalculated
    df_today = df[df["timestamp"] >= cutoff_24hr].copy()
    windows = detect_supply_windows_df(df_today)

    # ── DEEP DIAGNOSTIC — remove after fix ───────────────────────────────
    log.info("=== QoS DIAGNOSTIC ===")
    log.info("Benchmark: %s", benchmark)
    log.info("Windows found: %d", len(windows))
    log.info("Total rows: %d", len(df))
    log.info("Flow max=%.1f  min=%.1f  mean=%.1f",
            df["flow_rate"].max(), df["flow_rate"].min(), df["flow_rate"].mean())
    log.info("Rows with flow > 1.0: %d", (df["flow_rate"] > 1.0).sum())
    log.info("Rows with flow > 0:   %d", (df["flow_rate"] > 0).sum())
    log.info("Rows with flow == 0:  %d", (df["flow_rate"] == 0).sum())
    log.info("Sample timestamps: %s", df["timestamp"].head(3).tolist())
    log.info("Sample flow values: %s", df["flow_rate"].head(10).tolist())
    
    # Print first window if any
    for i, w in enumerate(windows):
        log.info("Window %d: start=%s end=%s dur=%.0fmin peak=%.1f avg=%.1f",
                i+1, w["start"], w["end"], w["duration"], w["peak"], w["avg"])
    log.info("=== END DIAGNOSTIC ===")
    qos, anomalies, _ = score_day_vs_benchmark(
    windows, benchmark, time_tol_min=360, flow_tol=0.50)

    # ── DIAGNOSTIC LOGS ──────────────────────────────────────────────────
    log.info("Benchmark: %s", benchmark)
    log.info("Supply windows found: %d", len(windows))
    log.info("Flow stats: max=%.1f mean=%.1f zeros=%d",
            df["flow_rate"].max(), df["flow_rate"].mean(),
            (df["flow_rate"] == 0).sum())

    # ── 5. Telegram text report ───────────────────────────────────────────
    report = build_daily_report(df)
    send_message(report)
    # ── Send 7-day trend summary to Telegram ─────────────────────────────
    try:
        trend_df = db_load_7day_trend()
        if len(trend_df) >= 2:
            trend_lines = ""
            for _, row in trend_df.iterrows():
                icon = "🟢" if row["qos"] >= 85 else "🟡" if row["qos"] >= 70 else "🔴"
                trend_lines += (f"  {icon} {row['date']}  QoS:{row['qos']:.0f}%"
                                f"  Avg:{row['avg_flow']:.0f}  Anom:{int(row['total_anomalies'])}\n")
            trend_msg = (
                f"📈 *7-Day QoS Trend*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"{trend_lines}"
                f"_🟢≥85%  🟡70-84%  🔴<70%_"
            )
            send_message(trend_msg)
    except Exception as e:
        log.error("7-day trend message failed: %s", e)
    log.info("Daily text report sent")

    # ── 6. Send 24hr flow chart ───────────────────────────────────────────
    try:
        buf = make_daily_chart(df)
        send_photo(buf, caption="📊 VMC MJP-4231 — 24hr flow chart")
    except Exception as e:
        log.error("24hr chart failed: %s", e)

    # ── 7. Send hourly bar chart ──────────────────────────────────────────
    try:
        buf2 = make_hourly_bar_chart(df)
        if buf2:
            send_photo(buf2, caption="📊 VMC MJP-4231 — Avg flow by hour")
    except Exception as e:
        log.error("Hourly bar chart failed: %s", e)

    # ── 8. Run 4-model full detector (NEW) ────────────────────────────────
    df_full = None
    try:
        log.info("Running 4-model anomaly detector...")
        df_full = run_full_detectors(df.copy())
        fa_count = int(df_full["final_anomaly"].sum())
        log.info("Full detector: %d final anomalies (4-model ensemble)", fa_count)
        # Append a Telegram summary of the ensemble result
        z_c  = int(df_full["anom_zscore"].sum())
        iq_c = int(df_full["anom_iqr"].sum())
        if_c = int(df_full["anom_iforest"].sum())
        pc_c = int(df_full["anom_pca"].sum())
        ensemble_msg = (
            f"🔬 *4-Model Anomaly Ensemble*\n"
            f"  • Z-score:          {z_c}\n"
            f"  • IQR:              {iq_c}\n"
            f"  • Isolation Forest: {if_c}\n"
            f"  • PCA Autoencoder:  {pc_c}\n"
            f"  • ✅ Final (3+/rule): {fa_count}\n"
            f"_Matches Streamlit Anomaly tab_"
        )
        send_message(ensemble_msg)
    except Exception as e:
        log.error("Full detector failed: %s", e)

    # ── 9. Send EDA charts (NEW) ──────────────────────────────────────────
    try:
        log.info("Generating EDA charts...")
        eda_bufs = make_eda_charts(df)
        eda_captions = [
            "📈 EDA: Full flow time series",
            "📈 EDA: Rolling mean ± 2σ band",
            "📈 EDA: Flow distribution histogram",
        ]
        for i, (buf_eda, cap) in enumerate(zip(eda_bufs, eda_captions)):
            send_photo(buf_eda, caption=cap)
        log.info("EDA charts sent: %d", len(eda_bufs))
    except Exception as e:
        log.error("EDA charts failed: %s", e)

    # ── 10. Send anomaly model charts (NEW) ───────────────────────────────
    if df_full is not None:
        try:
            log.info("Generating anomaly model charts...")
            anom_bufs = make_anomaly_charts(df_full)
            anom_captions = [
                "🔍 Anomaly: Model comparison",
                "🔍 Anomaly: Final flags timeline",
                "🔍 Anomaly: Model-by-model overlay",
                "🔍 Anomaly: IF score heatmap (day × hour)",
            ]
            for buf_a, cap in zip(anom_bufs, anom_captions):
                send_photo(buf_a, caption=cap)
            log.info("Anomaly charts sent: %d", len(anom_bufs))
        except Exception as e:
            log.error("Anomaly charts failed: %s", e)

    # ── 11. Send forecast chart (NEW) ─────────────────────────────────────
    try:
        log.info("Generating forecast chart...")
        fc_buf = make_forecast_chart(df, df_full if df_full is not None else df)
        if fc_buf:
            send_photo(fc_buf,
                       caption=f"📈 Forecast: Exp. smoothing +{FORECAST_STEPS} steps ahead")
            log.info("Forecast chart sent")
        else:
            log.info("Forecast: insufficient data — skipped")
    except Exception as e:
        log.error("Forecast chart failed: %s", e)

    # ── 12. Fetch Jan–Feb pattern data (NEW) ──────────────────────────────
    pat_summary    = None
    pat_chart_bufs = None
    try:
        log.info("Fetching Jan–Feb %d for pattern analysis...", PATTERN_YEAR)
        pat_df = fetch_two_months(year=PATTERN_YEAR)
        if not pat_df.empty:
            log.info("Pattern data: %d readings, %d days",
                     len(pat_df), pat_df["timestamp"].dt.date.nunique())
            # ── 13. Send pattern charts (NEW) ─────────────────────────────
            pat_chart_bufs, pat_summary = make_pattern_charts(
                            pat_df, n_clusters=PATTERN_K, sim_threshold=SIM_THRESHOLD)
                        # Also build box-method benchmark from Jan-Feb windows (mirrors Streamlit ⑥)
            try:
                all_wins_pat = []
                pat_df_cp = pat_df.copy()
                pat_df_cp["date_"] = pat_df_cp["timestamp"].dt.date
                for date_, group in pat_df_cp.groupby("date_"):
                    wins_p = detect_supply_windows_df(group)
                    for w in wins_p:
                        w["date"] = str(date_)
                    all_wins_pat.extend(wins_p)
                benchmark_box, _ = build_benchmark_from_windows(
                    all_wins_pat, n_clusters=PATTERN_K)
                if benchmark_box:
                    db_save_benchmark_snapshot(benchmark_box, method="supply_window_janfeb")
                    log.info("Jan-Feb box benchmark saved: start=%.2fh end=%.2fh peak=%.1f",
                            benchmark_box["start_hour"], benchmark_box["end_hour"],
                            benchmark_box["peak"])
            except Exception as e:
                log.error("Box-method benchmark from Jan-Feb failed: %s", e)
            pat_captions = [
                f"📐 Pattern: Jan+Feb {PATTERN_YEAR} multi-day overlay — {len(pat_df['timestamp'].dt.date.unique())} days + benchmark median",
                f"📐 Pattern: K-Means clusters (k={PATTERN_K}) + benchmark",
                f"📐 Pattern: Daily similarity (threshold {SIM_THRESHOLD}%)",
                "📐 Pattern: Best-match vs worst-match days",
            ]
            for buf_p, cap in zip(pat_chart_bufs, pat_captions):
                send_photo(buf_p, caption=cap)
            if pat_summary:
                pat_msg = (
                    f"📐 *Pattern Analysis — Jan–Feb {PATTERN_YEAR}*\n"
                    f"  • Total days:     {pat_summary['total_days']}\n"
                    f"  • Matching ✅:    {pat_summary['n_match']}\n"
                    f"  • Deviant ❌:     {pat_summary['n_deviant']}\n"
                    f"  • Avg similarity: {pat_summary['avg_similarity']}%\n"
                    f"  • Best day:       {str(pat_summary['best_day'])[5:]}\n"
                    f"  • Worst day:      {str(pat_summary['worst_day'])[5:]}\n"
                    f"_Benchmark = modal K-Means centroid_"
                )
                send_message(pat_msg)
                log.info("Pattern charts + summary sent")
        else:
            log.warning("Pattern: no data returned from API for Jan–Feb %d", PATTERN_YEAR)
    except Exception as e:
        log.error("Pattern analysis failed: %s", e)

    # ── 14. Generate + send enhanced PDF ─────────────────────────────────
    try:
        log.info("Generating enhanced PDF report...")
        pdf_buf  = make_pdf_report(
            df, benchmark, windows, qos, anomalies,
            total_anom, avg_flow, peak_flow, peak_str,
            spike_anom, night_anom,
            df_full=df_full,
            pat_summary=pat_summary,
            pat_chart_bufs=pat_chart_bufs,
        )
        filename = f"VMC_QoS_{datetime.now().strftime('%Y%m%d')}.pdf"
        send_pdf(pdf_buf, filename,
                        caption=(f"📄 {STATION_NAME} Daily QoS Report — "
                          f"{datetime.now().strftime('%d %b %Y')} — "
                          f"QoS: {qos:.1f}%"))
        log.info("Enhanced PDF report sent")
    except Exception as e:
        log.error("PDF report failed: %s", e)

    # ── 15. Persist QoS + benchmark to DB ────────────────────────────────
    date_str_today = datetime.now(IST).strftime("%Y-%m-%d")
    status_label   = ("EXCELLENT" if qos >= 85 else
                      "GOOD"      if qos >= 70 else
                      "POOR")
    db_save_qos(
        date_str        = date_str_today,
        qos             = qos,
        total_readings  = len(df),
        total_anomalies = total_anom,
        spike_anomalies = spike_anom,
        night_anomalies = night_anom,
        supply_windows  = len(windows),
        avg_flow        = avg_flow,
        peak_flow       = peak_flow,
        benchmark_used  = benchmark is not None,
        status          = status_label,
    )
    if benchmark:
        db_save_benchmark_snapshot(benchmark, method="supply_window")

    db_log_report(report, "daily")
    log.info(
        "=== Daily batch job complete: %d readings, %d anomalies, "
        "QoS %.1f%%, pattern_days=%s ===",
        len(df), total_anom, qos,
        str(pat_summary.get("total_days", "N/A")) if pat_summary else "N/A",
    )


def job_heartbeat():
    """Optional heartbeat — keeps Streamlit live tab warm."""
    reading = fetch_single_reading()
    if reading is None:
        log.debug("Heartbeat: no reading")
        return
    flow = reading["flow_rate"]
    ts   = reading["timestamp"]
    anom = int(flow > SPIKE_THRESHOLD or flow < 0)
    db_insert(ts, flow, anom)
    log.info("Heartbeat: [%s] %.1f m³/hr", ts.strftime("%H:%M"), flow)


# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
_scheduler = None

def _shutdown(signum, frame):
    log.info("Shutting down...")
    if _scheduler:
        _scheduler.shutdown(wait=False)
    sys.exit(0)

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global _scheduler

    log.info("════════════════════════════════════")
    log.info("  VMC Worker (Batch Mode + Full Analysis) starting")
    log.info("  DB         : %s", os.path.abspath(DB_PATH))
    log.info("  Batch fetch: every 24 hr (1 API call/day)")
    log.info("  Report at  : %02d:%02d IST", REPORT_HOUR, REPORT_MINUTE)
    log.info("  Heartbeat  : %s", "ON" if HEARTBEAT_ENABLED else "OFF")
    log.info("  Chats      : %d", len(TELEGRAM_CHAT_IDS))
    log.info("  Pattern yr : %d  k=%d  threshold=%d%%",
             PATTERN_YEAR, PATTERN_K, SIM_THRESHOLD)
    log.info("════════════════════════════════════")

    init_db()

    if not try_login():
        log.warning("Initial VMC login failed — will retry at report time")

    _scheduler = BlockingScheduler(timezone=IST)

    _scheduler.add_job(
        job_daily_batch_fetch_and_report,
        CronTrigger(hour=REPORT_HOUR, minute=REPORT_MINUTE, timezone=IST),
        id="daily_batch",
    )

    if HEARTBEAT_ENABLED:
        _scheduler.add_job(
            job_heartbeat,
            "interval",
            hours=HEARTBEAT_INTERVAL_HR,
            id="heartbeat",
            next_run_time=datetime.now(IST),
        )
        log.info("Heartbeat scheduled every %d hr", HEARTBEAT_INTERVAL_HR)

    log.info("Running initial batch fetch on startup...")
    try:
        job_daily_batch_fetch_and_report()
    except Exception as e:
        log.error("Startup batch run failed: %s", e)

    log.info("Scheduler running. Next daily batch at %02d:%02d IST. Ctrl+C to stop.",
             REPORT_HOUR, REPORT_MINUTE)
    _scheduler.start()


if __name__ == "__main__":
    main()
