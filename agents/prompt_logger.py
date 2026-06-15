import json
from datetime import datetime, timezone
from pathlib import Path


class PromptResponseLogger:
    LOG_PATH = "outputs/prompt_response_log.json"

    def log(
        self,
        agent: str,
        call_type: str,
        system_prompt: str,
        human_prompt: str,
        raw_response: str,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "agent": agent,
            "call_type": call_type,
            "system_prompt": system_prompt,
            "human_prompt": human_prompt,
            "raw_response": raw_response,
        }
        log_path = Path(self.LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entries = self.load()
        entries.append(entry)
        log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def clear(self) -> None:
        log_path = Path(self.LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("[]", encoding="utf-8")

    @staticmethod
    def load() -> list[dict]:
        log_path = Path(PromptResponseLogger.LOG_PATH)
        if not log_path.exists():
            return []
        try:
            return json.loads(log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
