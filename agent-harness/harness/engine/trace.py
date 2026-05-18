"""Trace writer with stable record schema."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness.errors import ERROR_CODE_MAP
from harness.state import TraceStep

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TraceHeader:
    """Run-level trace metadata."""

    record_type: str = "run"
    run_id: str = ""
    timestamp: str = ""
    query: str = ""
    conversation_id: str = ""
    llm_model: str = ""
    llm_provider: str = ""
    total_steps: int = 0
    total_latency_ms: float = 0.0
    final_status: str = "success"


class TraceRecorder:
    """Writes JSONL trace records with graceful IO fallback."""

    def __init__(self, trace_dir: Path | None = None) -> None:
        base_dir = trace_dir or (Path.home() / ".zzk" / "traces")
        self._trace_dir = base_dir
        self._run_id = f"run-{uuid.uuid4().hex[:12]}"
        self._path = self._trace_dir / f"{self._run_id}.jsonl"
        self._steps: int = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    def start_run(
        self,
        query: str,
        llm_provider: str,
        llm_model: str,
        conversation_id: str = "",
    ) -> None:
        header = TraceHeader(
            run_id=self._run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            conversation_id=conversation_id,
            llm_provider=llm_provider,
            llm_model=llm_model,
            total_steps=0,
            final_status="running",
        )
        self._append_record(asdict(header))

    def append_step(self, step: TraceStep) -> None:
        self._steps += 1
        self._append_record(asdict(step))

    def finish_run(self, final_status: str) -> None:
        summary = {
            "record_type": "run_summary",
            "run_id": self._run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_steps": self._steps,
            "final_status": final_status,
        }
        self._append_record(summary)

    def _append_record(self, record: dict[str, object]) -> None:
        try:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        except OSError as exc:
            LOGGER.warning("trace write failed: %s", exc)
