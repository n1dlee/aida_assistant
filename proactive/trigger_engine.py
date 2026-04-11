"""
proactive/trigger_engine.py
Watches for conditions and fires actions proactively.
Example triggers: time-based reminders, anomaly detection, daily briefing.
"""
import asyncio
import logging
from datetime import datetime
from typing import Callable, Awaitable

log = logging.getLogger("aida.proactive.triggers")


class TriggerEngine:
    def __init__(self, notify_callback: Callable[[str], Awaitable] = None):
        """
        notify_callback: async function that receives a message string.
        Typically wired to orchestrator.process() or speaker.speak().
        """
        self._notify = notify_callback or self._default_notify
        self._running = False

    async def _default_notify(self, msg: str):
        print(f"\n[AIDA Proactive] {msg}")

    async def start(self):
        self._running = True
        await asyncio.gather(
            self._daily_briefing(),
            self._heartbeat(),
        )

    async def _daily_briefing(self):
        """Sends a morning briefing at 9:00 AM."""
        while self._running:
            now = datetime.now()
            if now.hour == 9 and now.minute == 0:
                await self._notify(
                    f"Good morning! It's {now.strftime('%A, %d %B')}. "
                    "Say 'what's my schedule today' to get your agenda."
                )
                await asyncio.sleep(60)  # Avoid re-triggering within same minute
            await asyncio.sleep(30)

    async def _heartbeat(self):
        """Logs a heartbeat every 5 minutes (useful for debugging)."""
        while self._running:
            await asyncio.sleep(300)
            log.debug("Proactive engine heartbeat — %s", datetime.now().strftime("%H:%M"))

    def stop(self):
        self._running = False
