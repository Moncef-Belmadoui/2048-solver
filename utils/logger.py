"""
logger.py
---------
Episode statistics logger.

Writes data to:
  logs/<run_name>/episodes.csv   – row-per-episode CSV
  logs/<run_name>/summary.json   – aggregate statistics

Also keeps an in-memory deque for live moving-average access.
"""

from __future__ import annotations

import csv
import json
import os
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional


class EpisodeLogger:
    """
    Logs per-episode statistics to disk and memory.

    Parameters
    ----------
    log_dir :   Root directory for all runs.
    run_name :  Subdirectory name for this run.
    window :    Window size for moving averages.
    """

    def __init__(
        self,
        log_dir:  str = "logs",
        run_name: str = "run",
        window:   int = 100,
    ):
        self.run_name = run_name
        self.window   = window

        # Make output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(log_dir, f"{run_name}_{timestamp}")
        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_path     = os.path.join(self.out_dir, "episodes.csv")
        self.summary_path = os.path.join(self.out_dir, "summary.json")

        self._records: List[Dict[str, Any]] = []
        self._fieldnames: Optional[List[str]] = None
        self._csv_file = None
        self._csv_writer = None

        # Moving average buffers (keyed by stat name)
        self._windows: Dict[str, Deque[float]] = {}

    # ------------------------------------------------------------------

    def log(self, stats: Dict[str, Any]):
        """
        Record one episode's statistics dict.

        Automatically initialises the CSV on the first call.
        """
        self._records.append(stats)

        # Init CSV writer on first log
        if self._fieldnames is None:
            self._fieldnames = list(stats.keys()) + ["ma_reward", "ma_score"]
            self._csv_file   = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            self._csv_writer.writeheader()

        # Update moving average buffers
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                if k not in self._windows:
                    self._windows[k] = deque(maxlen=self.window)
                self._windows[k].append(float(v))

        # Augment with moving averages
        row = dict(stats)
        row["ma_reward"] = self.moving_average("reward")
        row["ma_score"]  = self.moving_average("score")

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def moving_average(self, key: str) -> float:
        """Return the current moving average for *key* (0.0 if unavailable)."""
        buf = self._windows.get(key)
        if buf and len(buf) > 0:
            return sum(buf) / len(buf)
        return 0.0

    def latest(self, key: str) -> Optional[float]:
        """Return the most recent value for *key*."""
        buf = self._windows.get(key)
        return buf[-1] if buf else None

    def save(self):
        """
        Flush CSV and write the summary JSON.
        """
        if self._csv_file:
            self._csv_file.close()

        if not self._records:
            return

        # Build summary
        numeric_keys = [
            k for k, v in self._records[0].items()
            if isinstance(v, (int, float))
        ]
        summary: Dict[str, Any] = {
            "run_name":     self.run_name,
            "total_episodes": len(self._records),
        }
        for k in numeric_keys:
            vals = [r[k] for r in self._records if isinstance(r.get(k), (int, float))]
            if vals:
                summary[k] = {
                    "mean": float(sum(vals) / len(vals)),
                    "min":  float(min(vals)),
                    "max":  float(max(vals)),
                    "last": float(vals[-1]),
                }

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Logs saved → {self.out_dir}")

    # Convenience
    def __len__(self) -> int:
        return len(self._records)

    def get_records(self) -> List[Dict[str, Any]]:
        return list(self._records)
