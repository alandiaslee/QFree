"""Local dashboard server for QFree (stdlib only).

Serves web/index.html + JSON APIs over existing outputs/*.csv artifacts.
Long tasks (single-day scan, backtest rebuild) run as subprocess jobs.

Usage: .venv/bin/python scripts/dashboard.py [--port 8000] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qfree.data.index_cache import load_index  # noqa: E402
from qfree.data.models import Instrument  # noqa: E402
from qfree.data.store import BarStore  # noqa: E402

WEB = ROOT / "web"
OUT = ROOT / "outputs"
CACHE = OUT / "dashboard_cache"
LOGS = OUT / "dashboard_logs"
VENV_PY = ROOT / ".venv" / "bin" / "python"
DATA_JSON = OUT / "dashboard_data.json"
TRACKING = OUT / "daily_tracking.csv"
BREADTH_CSV = OUT / "full_breadth_history.csv"
SIGNALS_CSV = OUT / "long_period_signals.csv"
INDEX_CSV = ROOT / "data" / "index" / "000300_SH.csv"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
COST = 0.001
TARGET = 0.05
HOLDING = 10
THR = 0.009


# ---------------------------------------------------------------- helpers

def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return clean(float(obj))
    except Exception:
        pass
    return str(obj)


def jdump(handler, obj, status=200):
    body = json.dumps(clean(obj), ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_breadth() -> pd.DataFrame:
    return pd.read_csv(BREADTH_CSV, parse_dates=["cal_date"]).sort_values("cal_date")


def data_last_date() -> str | None:
    try:
        idx = pd.read_csv(INDEX_CSV)
        return str(pd.to_datetime(idx["date"].iloc[-1]).date())
    except Exception:
        return None


def regime_status() -> dict:
    idx = pd.read_csv(INDEX_CSV, parse_dates=["date"]).sort_values("date")
    close = idx["close"]
    ma20 = close.rolling(20).mean()
    return {
        "regime_close": float(close.iloc[-1]),
        "regime_ma": float(ma20.iloc[-1]),
        "regime_pass": bool(close.iloc[-1] > ma20.iloc[-1]),
        "regime_date": str(idx["date"].iloc[-1].date()),
    }


GAP_SKIP_UP = 0.02
GAP_SKIP_DOWN = -0.05
TARGET = 0.05
DISASTER = 0.12
ORDER_CAPITAL = 500_000  # 操作明细参考本金（元）


def _enrich_order_fields(rows: list[dict], date: str) -> list[dict]:
    """Fill signal close (from bars if missing) and next-day order details:
    limit price = close*1.02, ref shares (100-lot, capital/10 each),
    profit target price, disaster line. Mirrors make_order_sheet.py."""
    need = [r for r in rows if r.get("ts_code") and r.get("close") is None]
    if need:
        store = BarStore(str(ROOT / "data"))
        d = pd.Timestamp(date)
        for r in need:
            try:
                b = store.load(Instrument(r["ts_code"]), "2023-01-01", "2100-01-01")
            except FileNotFoundError:
                continue
            b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
            hit = b[b["date"] == d]
            if not hit.empty:
                r["close"] = float(hit["close"].iloc[0])
    n = max(len(rows), 1)
    per = ORDER_CAPITAL / n
    for r in rows:
        close = r.get("close")
        if close is None:
            r.setdefault("limit_price", None)
            continue
        limit = round(close * (1 + GAP_SKIP_UP), 2)
        r["limit_price"] = limit
        r["ref_shares"] = int(per / limit // 100) * 100
        r["profit_target_price"] = round(limit * (1 + TARGET), 2)
        r["disaster_line"] = round(limit * (1 - DISASTER), 2)
    return rows


def load_picks_for_date(date: str) -> tuple[list[dict], str]:
    """Picks for a date. Returns (rows, source) with source live_scan|history."""
    live = OUT / f"daily_{date}_selected.csv"
    if live.exists():
        df = pd.read_csv(live)
        if "selected" in df.columns:
            df = df[df["selected"].astype(str).str.lower().isin(("true", "1"))]
        score_col = "base_total_score" if "base_total_score" in df.columns else "total_score"
        df = df.sort_values(score_col, ascending=False).head(10)
        rows = []
        for r in df.itertuples(index=False):
            d = r._asdict()
            close = d.get("close")
            rows.append({
                "ts_code": d.get("ts_code"), "name": d.get("name"),
                "industry": d.get("industry"),
                "score": d.get("base_total_score", d.get("total_score")),
                "quality_score": d.get("quality_score"),
                "close": close,
                "limit_price": round(close * 1.02, 2) if close else None,
                "ret20": d.get("return_20d"),
                "pos60": d.get("price_position_60d"),
                "amount20": d.get("avg_amount_20d"),
            })
        return _enrich_order_fields(rows, date), "live_scan"
    # history: official top10 for that date; fall back to the breadth-only
    # (yellow-tier) reference top10 when the regime gate blocked the day
    from build_dashboard_data import load_official_picks
    _, picks, _ = load_official_picks()
    sub = picks[picks["date"] == pd.Timestamp(date)]
    source = "history"
    if sub.empty:
        try:
            sig = pd.read_csv(SIGNALS_CSV, parse_dates=["date"])
            bmap = read_breadth().set_index("cal_date")["breadth"]
            sig = sig[(sig["date"] == pd.Timestamp(date))
                      & (sig["date"].map(bmap) >= THR)]
            if not sig.empty:
                sub = (sig.sort_values("base_total_score", ascending=False)
                       .head(10))
                source = "history_ref"
                uni = pd.read_csv(ROOT / "data" / "universe.csv",
                                  usecols=["ts_code", "name", "industry"])
                sub = sub.merge(uni, on="ts_code", how="left")
        except Exception:
            sub = picks.iloc[0:0]
    rows = [{"ts_code": r.ts_code, "name": getattr(r, "name", None),
             "industry": getattr(r, "industry", None),
             "score": float(r.base_total_score), "quality_score": None,
             "close": None, "limit_price": None, "ret20": None, "pos60": None,
             "amount20": None} for r in sub.itertuples(index=False)]
    return _enrich_order_fields(rows, date), source


def forward_for_date(date: str) -> dict:
    """10d/5% forward events for the picks of `date`, with JSON cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"forward_{date}.json"
    last = data_last_date()
    if cache.exists():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("data_last_date") == last:
                return payload
        except Exception:
            pass

    rows, source = load_picks_for_date(date)
    if not rows:
        payload = {"date": date, "source": source, "data_last_date": last,
                   "rows": [], "summary": None}
        cache.write_text(json.dumps(clean(payload), ensure_ascii=False))
        return payload

    from qfree.backtest.event_study import evaluate_events
    hist = read_breadth()
    sessions = pd.DatetimeIndex(hist["cal_date"])
    syms = [r["ts_code"] for r in rows if r["ts_code"]]
    store = BarStore(str(ROOT / "data"))
    bars_map = {}
    for sym in syms:
        try:
            b = store.load(Instrument(sym), "2023-01-01", "2100-01-01").copy()
        except FileNotFoundError:
            continue
        b["date"] = pd.to_datetime(b["date"], errors="coerce").dt.normalize()
        bars_map[sym] = (b.dropna(subset=["date"]).drop_duplicates("date")
                         .sort_values("date"))
    sig = pd.DataFrame({"ts_code": syms, "date": pd.Timestamp(date)})
    ev = evaluate_events(sig.assign(selected=True)[["ts_code", "date", "selected"]],
                         bars_map, (HOLDING,), exit_modes=("profit_target_close",),
                         trade_dates=sessions, profit_target=TARGET)
    by_key = {}
    for r in ev.itertuples(index=False):
        d = r._asdict()
        if str(d.get("status")) not in ("ok", "window_end_exit",
                                        "profit_target_exit", "max_holding_exit"):
            by_key[(d.get("ts_code"))] = {"status": str(d.get("status"))}
            continue
        nr = d.get("net_return")
        nr = float(nr) if pd.notna(nr) else None
        if nr is not None:
            nr = (1 + nr) * (1 - COST) - 1
        by_key[d.get("ts_code")] = {
            "net_return": nr, "status": str(d.get("status")),
            "exit_date": str(d.get("exit_date"))[:10] if d.get("exit_date") else None,
            "days_held": int(d["realized_holding_days"])
            if pd.notna(d.get("realized_holding_days")) else None,
        }
    merged = []
    for r in rows:
        m = dict(r)
        m.update(by_key.get(r["ts_code"], {"status": "no_data"}))
        merged.append(m)
    valid = [m["net_return"] for m in merged if m.get("net_return") is not None]
    summary = {
        "n": len(valid),
        "win": sum(1 for v in valid if v > 0) / len(valid) if valid else None,
        "mean": sum(valid) / len(valid) if valid else None,
    } if valid else None
    payload = {"date": date, "source": source, "data_last_date": last,
               "rows": merged, "summary": summary}
    cache.write_text(json.dumps(clean(payload), ensure_ascii=False))
    return payload


# ---------------------------------------------------------------- jobs

class JobRunner:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()  # scan & backtest are mutually exclusive
        self.write_owner: str | None = None

    def submit(self, kind: str, argv: list[str], env_extra: dict | None = None,
               on_success=None) -> tuple[str | None, str | None]:
        with self.lock:
            if self.write_owner is not None:
                return None, f"作业 {self.write_owner} 正在运行，请等它完成"
            job_id = uuid.uuid4().hex[:8]
            self.write_owner = job_id
            log_path = LOGS / f"{job_id}.log"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src")
            if env_extra:
                env.update(env_extra)
            logf = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(argv, cwd=str(ROOT), env=env,
                                    stdout=logf, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL)
            self.jobs[job_id] = {
                "id": job_id, "kind": kind, "status": "running",
                "started": time.time(), "proc": proc, "log": log_path,
                "returncode": None, "on_success": on_success,
            }

            def watcher() -> None:
                rc = proc.wait()
                with self.lock:
                    job = self.jobs[job_id]
                    job["status"] = "done" if rc == 0 else "failed"
                    job["returncode"] = rc
                    job["elapsed_s"] = round(time.time() - job["started"], 1)
                    cb = job.get("on_success") if rc == 0 else None
                    self.write_owner = None
                if cb:
                    try:
                        cb()
                    except Exception as exc:  # noqa: BLE001
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(f"\n[post-run] {type(exc).__name__}: {exc}\n")

            threading.Thread(target=watcher, daemon=True).start()
            return job_id, None

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            out = {k: v for k, v in job.items()
                   if k not in ("proc", "on_success")}
        try:
            lines = Path(job["log"]).read_text(encoding="utf-8",
                                               errors="replace").splitlines()
            out["log_tail"] = lines[-30:]
        except Exception:
            out["log_tail"] = []
        if job["status"] == "running":
            out["elapsed_s"] = round(time.time() - job["started"], 1)
        return out


JOBS = JobRunner()


def submit_scan(as_of: str) -> tuple[str | None, str | None]:
    argv = [str(VENV_PY), "-m", "qfree.cli", "daily-scan",
            "--recommended", "--offline-scan", "--as-of", as_of,
            "--start", "2023-01-01", "--workers", "12"]
    return JOBS.submit("scan", argv)


def submit_backtest() -> tuple[str | None, str | None]:
    def rebuild_json() -> None:
        import build_dashboard_data
        data = build_dashboard_data.build()
        DATA_JSON.write_text(json.dumps(build_dashboard_data.clean(data),
                                        ensure_ascii=False, allow_nan=False),
                             encoding="utf-8")

    argv = [str(VENV_PY), str(ROOT / "scripts" / "long_period_validation.py")]
    return JOBS.submit("backtest", argv, on_success=rebuild_json)


# ---------------------------------------------------------------- handler

class Handler(BaseHTTPRequestHandler):
    server_version = "QFreeDashboard/1.0"

    def log_message(self, fmt, *args):  # quiet
        pass

    # ---- helpers
    def _serve_file(self, path: Path, ctype: str) -> None:
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            return jdump(self, {"error": "not found"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    # ---- routing
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        try:
            if path in ("/", "/index.html"):
                return self._serve_file(WEB / "index.html", "text/html; charset=utf-8")
            if path == "/echarts.min.js":
                return self._serve_file(WEB / "echarts.min.js",
                                        "application/javascript; charset=utf-8")
            if path == "/api/status/latest":
                return jdump(self, api_status())
            if path == "/api/breadth/history":
                days = None
                raw = q.get("days", [""])[0]
                if raw and raw != "all":
                    try:
                        days = max(5, min(int(raw), 2000))
                    except ValueError:
                        days = None
                return jdump(self, api_breadth_history(days))
            if path == "/api/backtest/summary":
                return jdump(self, api_backtest("summary"))
            if path == "/api/backtest/equity":
                return jdump(self, api_backtest("equity"))
            if path == "/api/backtest/per-date":
                return jdump(self, api_backtest("per_date"))
            if path == "/api/picks":
                d = q.get("date", [""])[0]
                if not DATE_RE.match(d):
                    return jdump(self, {"error": "date 格式须为 YYYY-MM-DD"}, 400)
                return jdump(self, api_picks(d))
            if path == "/api/forward":
                d = q.get("date", [""])[0]
                if not DATE_RE.match(d):
                    return jdump(self, {"error": "date 格式须为 YYYY-MM-DD"}, 400)
                return jdump(self, forward_for_date(d))
            if path == "/api/tracking":
                try:
                    rows = max(5, min(int(q.get("rows", ["30"])[0]), 500))
                except ValueError:
                    rows = 30
                return jdump(self, api_tracking(rows))
            m = re.match(r"^/api/job/([a-f0-9]{8})$", path)
            if m:
                job = JOBS.get(m.group(1))
                if not job:
                    return jdump(self, {"error": "job 不存在"}, 404)
                return jdump(self, job)
            return jdump(self, {"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            return jdump(self, {"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self):
        from urllib.parse import urlparse
        path = urlparse(self.path).path
        try:
            if path == "/api/job/scan":
                body = self._read_body()
                as_of = str(body.get("as_of") or "")[:10]
                if not DATE_RE.match(as_of):
                    return jdump(self, {"error": "as_of 格式须为 YYYY-MM-DD"}, 400)
                job_id, err = submit_scan(as_of)
                if err:
                    return jdump(self, {"error": err}, 409)
                return jdump(self, {"job_id": job_id}, 201)
            if path == "/api/job/backtest":
                job_id, err = submit_backtest()
                if err:
                    return jdump(self, {"error": err}, 409)
                return jdump(self, {"job_id": job_id}, 201)
            return jdump(self, {"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            return jdump(self, {"error": f"{type(exc).__name__}: {exc}"}, 500)


# ---------------------------------------------------------------- api impls

def api_status() -> dict:
    hist = read_breadth()
    last = hist.iloc[-1]
    rs = regime_status()
    last_scan = None
    if TRACKING.exists():
        t = pd.read_csv(TRACKING)
        if not t.empty:
            r = t.iloc[-1].to_dict()
            last_scan = {k: clean(v) for k, v in r.items()}
    return {
        "as_of": str(last["cal_date"].date()),
        "breadth": float(last["breadth"]),
        "breadth_pass": bool(last["breadth"] >= THR),
        **rs,
        "v3_gate": bool(last["breadth"] >= THR) and rs["regime_pass"],
        "last_scan": last_scan,
        "data_last_date": data_last_date(),
    }


def api_breadth_history(days: int | None = None) -> dict:
    hist = read_breadth()
    if days is not None:
        hist = hist.tail(days)
    return {
        "dates": [str(d.date()) for d in hist["cal_date"]],
        "breadth": hist["breadth"].tolist(),
        "gate": (hist["breadth"] >= THR).tolist(),
    }


def live_pick_counts() -> dict[str, int]:
    """Selected-row counts from outputs/daily_<date>_selected.csv scan files."""
    counts: dict[str, int] = {}
    for f in OUT.glob("daily_*_selected.csv"):
        date = f.stem.replace("daily_", "").replace("_selected", "")
        try:
            df = pd.read_csv(f, usecols=["selected"])
            n = int(df["selected"].astype(str).str.lower()
                    .isin(("true", "1")).sum())
        except Exception:
            continue
        counts[date] = n
    return counts


def breadth_only_counts() -> dict[str, int]:
    """Top10-per-date counts from the cached no-regime signal scan
    (long_period_signals.csv, breadth >= 0.9%) — the yellow-tier reference."""
    try:
        sig = pd.read_csv(SIGNALS_CSV, parse_dates=["date"])
        hist = read_breadth().set_index("cal_date")["breadth"]
        sig = sig[sig["date"].map(hist) >= THR]
        return (sig.sort_values("base_total_score", ascending=False)
                .groupby("date").head(10)
                .groupby("date").size()
                .map(int).to_dict())
    except Exception:
        return {}


def official_trigger_dates() -> set[str]:
    try:
        data = json.loads(DATA_JSON.read_text(encoding="utf-8"))
        return {c["date"] for c in data["calendar"] if c.get("v3_trigger")}
    except Exception:
        return set()


def api_backtest(part: str) -> dict:
    if not DATA_JSON.exists():
        return {"error": "dashboard_data.json 不存在，请先跑 scripts/build_dashboard_data.py"}
    data = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    if part == "summary":
        live = live_pick_counts()
        ref = breadth_only_counts()
        for c in data["calendar"]:
            c["live_count"] = live.get(c["date"], 0)
            c["ref_count"] = ref.get(pd.Timestamp(c["date"]), 0)
        return {k: data[k] for k in
                ("generated_at", "data_last_date", "sessions", "summary", "eras",
                 "clusters", "calendar")}
    if part == "equity":
        return data["equity"]
    return {"per_date": data["per_date"]}


def api_picks(date: str) -> dict:
    rows, source = load_picks_for_date(date)
    hist = read_breadth()
    day = hist[hist["cal_date"] == pd.Timestamp(date)]
    gate = None
    if not day.empty:
        from qfree.data.index_cache import regime_flags
        rf = regime_flags(load_index(str(ROOT / "data")), 20)
        gate = {
            "breadth": float(day["breadth"].iloc[0]),
            "breadth_pass": bool(day["breadth"].iloc[0] >= THR),
            "regime_pass": bool(rf.get(pd.Timestamp(date), False)),
        }
    return {"date": date, "source": source, "gate": gate, "picks": rows,
            "v3_trigger": date in official_trigger_dates()}


def api_tracking(rows: int = 30) -> dict:
    if not TRACKING.exists():
        return {"rows": []}
    t = pd.read_csv(TRACKING)
    t = t.tail(rows).iloc[::-1]
    cols = ["as_of", "gate_pass", "selected_count", "startup_breadth",
            "regime_pass", "elapsed_seconds", "config"]
    cols = [c for c in cols if c in t.columns]
    return {"rows": clean(t[cols].to_dict("records"))}


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    if not VENV_PY.exists():
        print(f"error: {VENV_PY} 不存在", file=sys.stderr)
        return 1
    LOGS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if not INDEX_CSV.exists():
        print(f"error: 指数缓存 {INDEX_CSV} 缺失（regime 门不可用），请先同步数据",
              file=sys.stderr)
        return 1

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"QFree dashboard: http://{args.host}:{args.port}  (Ctrl-C 停止)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
