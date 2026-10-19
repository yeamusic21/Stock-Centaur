"""Run the data pipeline (pull prices, then rebuild the trend table) in a background thread."""
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELING_DIR = REPO_ROOT / "src" / "modeling"

STEPS = [
    ("Downloading prices", [MODELING_DIR / "data_gathering" / "01_pull_stock_data.py", "--force"]),
    ("Building rankings", [MODELING_DIR / "02_build_trend_table.py"]),
]
PROGRESS_RE = re.compile(r"Pulling tickers \d+-(\d+) of (\d+)")

_lock = threading.Lock()
_status = {"state": "idle"}


def status():
    with _lock:
        return dict(_status)


def _update(**fields):
    with _lock:
        _status.update(fields)


def _run_step(index, label, args):
    _update(step=label, step_index=index, progress=None, message="Starting…")
    proc = subprocess.Popen(
        [sys.executable, "-u", *map(str, args)],
        cwd=Path(args[0]).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    last_line = ""
    for line in proc.stdout:
        line = line.strip()
        if not line or "Failed download" in line or line.startswith("["):  # skip yfinance per-ticker noise
            continue
        last_line = line
        match = PROGRESS_RE.search(line)
        if match:
            _update(progress=int(match.group(1)) / int(match.group(2)))
        _update(message=line)
    if proc.wait() != 0:
        raise RuntimeError(last_line or f"{label} failed")


def _run():
    try:
        for i, (label, args) in enumerate(STEPS):
            _run_step(i, label, args)
        _update(state="done", message="Data refreshed.", finished_at=datetime.now().isoformat(timespec="seconds"))
    except Exception as err:
        _update(state="error", message=str(err), finished_at=datetime.now().isoformat(timespec="seconds"))


def start():
    """Start a refresh unless one is already running. Returns True if a new refresh started."""
    with _lock:
        if _status.get("state") == "running":
            return False
        _status.clear()
        _status.update(state="running", steps=len(STEPS), started_at=datetime.now().isoformat(timespec="seconds"))
    threading.Thread(target=_run, daemon=True).start()
    return True
