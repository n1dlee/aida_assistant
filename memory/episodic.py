"""
memory/episodic.py
Episodic memory: logs timestamped events to a JSON file.
Useful for "what did I ask yesterday?" queries.
"""
import json
import os
import logging
from datetime import datetime
from typing import List, Dict

log = logging.getLogger("aida.memory.episodic")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "episodes.json")


class EpisodicMemory:
    def __init__(self):
        self._episodes: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(self._episodes[-500:], f, ensure_ascii=False, indent=2)

    def log_event(self, event_type: str, content: str, metadata: Dict = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "content": content,
            "metadata": metadata or {},
        }
        self._episodes.append(entry)
        self._save()

    def recent(self, n: int = 10) -> List[Dict]:
        return self._episodes[-n:]

    def search_by_date(self, date_str: str) -> List[Dict]:
        return [e for e in self._episodes if e["timestamp"].startswith(date_str)]
