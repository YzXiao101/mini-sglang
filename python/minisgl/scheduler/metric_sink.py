from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO


class SchedulerMetricSink:
    def __init__(self, path: str | None, enabled: bool):
        self._file: TextIO | None = None
        if not enabled or not path:
            return
        file_path = Path(path).expanduser()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = file_path.open("a", buffering=1, encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        if self._file is None:
            return
        record = {"ts_ns": time.time_ns(), "event": event, **fields}
        self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
