from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO


class SchedulerMetricSink:
    def __init__(self, path: str, *, enabled: bool, interval: int) -> None:
        self._file: TextIO | None = None
        self._counters: dict[str, int] = {}
        self.interval = max(interval, 1)
        if not enabled or not path:
            return

        file_path = Path(path).expanduser()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = file_path.open("a", buffering=1, encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def emit(self, event: str, **fields: Any) -> None:
        if self._file is None:
            return
        record = {"ts_ns": time.time_ns(), "event": event, **fields}
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def emit_sampled(self, event: str, *, key: str | None = None, **fields: Any) -> None:
        if self._file is None:
            return
        key = event if key is None else key
        count = self._counters.get(key, 0) + 1
        self._counters[key] = count
        if count != 1 and count % self.interval != 0:
            return
        self.emit(event, sample_count=count, **fields)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
