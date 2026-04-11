"""
tools/calendar_tool.py
Simple local calendar tool — reads/writes events to a JSON file.
For full Google Calendar integration, swap _load/_save with gcal API calls.
"""
import json
import os
import logging
from datetime import datetime
from typing import List, Dict
from tools.base_tool import BaseTool

log = logging.getLogger("aida.tools.calendar")
CALENDAR_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "calendar.json")


class CalendarTool(BaseTool):
    @property
    def name(self) -> str:
        return "calendar"

    @property
    def description(self) -> str:
        return "Add or list calendar events stored locally."

    def _load(self) -> List[Dict]:
        if os.path.exists(CALENDAR_PATH):
            with open(CALENDAR_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self, events: List[Dict]):
        os.makedirs(os.path.dirname(CALENDAR_PATH), exist_ok=True)
        with open(CALENDAR_PATH, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

    async def run(self, query: str, **kwargs) -> str:
        lower = query.lower()
        events = self._load()

        if any(w in lower for w in ["list", "show", "what", "today", "покажи", "список"]):
            today = datetime.now().strftime("%Y-%m-%d")
            today_events = [e for e in events if e.get("date", "").startswith(today)]
            if not today_events:
                return "No events today."
            return "Today's events:\n" + "\n".join(
                f"• {e['time']} — {e['title']}" for e in today_events
            )

        # Basic add: "add meeting at 15:00"
        if any(w in lower for w in ["add", "create", "schedule", "добавь", "создай"]):
            event = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": "??:??",
                "title": query,
                "created": datetime.now().isoformat(),
            }
            events.append(event)
            self._save(events)
            return f"Event added: '{query}'. (Edit data/calendar.json for exact time.)"

        return "Calendar: say 'list events' or 'add [event name]'."
