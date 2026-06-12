import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AgentActivityLogger:
    def __init__(self, log_path: str = "outputs/agent_activity_log.json"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = self._load_entries()

    def _load_entries(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        try:
            data = json.loads(self.log_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def log_event(
        self,
        agent: str,
        stage: str,
        status: str = "success",
        details: str | None = None,
        iteration: int | None = None,
        input_paths: dict | None = None,
        output_paths: dict | None = None,
        metadata: dict | None = None,
    ) -> dict:
        event = {
            "timestamp": self._timestamp(),
            "agent": agent,
            "stage": stage,
            "status": status,
            "iteration": iteration,
            "details": details or "",
            "input_paths": input_paths or {},
            "output_paths": output_paths or {},
            "metadata": metadata or {},
        }
        self.entries.append(event)
        self.save()
        return event

    def log_commit_summary(
        self,
        agent: str,
        iteration: int | None,
        summary: str,
        changes: dict | None = None,
        output_paths: dict | None = None,
    ) -> dict:
        return self.log_event(
            agent=agent,
            stage="commit_summary",
            status="success",
            details=summary,
            iteration=iteration,
            output_paths=output_paths,
            metadata={"commit_summary": summary, "changes": changes or {}},
        )

    def save(self) -> None:
        self.log_path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")
