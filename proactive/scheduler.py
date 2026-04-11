"""
proactive/scheduler.py
Runs background tasks on a schedule using APScheduler.
Falls back to asyncio sleep loops if APScheduler is not installed.
"""
import asyncio
import logging
from typing import Callable, Awaitable

log = logging.getLogger("aida.proactive.scheduler")


class Scheduler:
    def __init__(self):
        self._jobs = []
        self._scheduler = None
        self._init()

    def _init(self):
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            self._scheduler = AsyncIOScheduler()
            log.info("APScheduler loaded.")
        except ImportError:
            log.warning("APScheduler not installed — using asyncio fallback.")

    def add_interval_job(self, func: Callable[[], Awaitable], seconds: int, job_id: str = None):
        """Run async func every N seconds."""
        if self._scheduler:
            self._scheduler.add_job(func, "interval", seconds=seconds, id=job_id)
            log.info("Scheduled job '%s' every %ds.", job_id or func.__name__, seconds)
        else:
            self._jobs.append((func, seconds))

    def add_cron_job(self, func: Callable[[], Awaitable], hour: int, minute: int, job_id: str = None):
        """Run async func daily at HH:MM."""
        if self._scheduler:
            self._scheduler.add_job(func, "cron", hour=hour, minute=minute, id=job_id)
            log.info("Scheduled cron job '%s' at %02d:%02d.", job_id or func.__name__, hour, minute)
        else:
            log.warning("Cron jobs require APScheduler. Install it: pip install apscheduler")

    def start(self):
        if self._scheduler:
            self._scheduler.start()
        else:
            # Start simple asyncio loops as fallback
            for func, interval in self._jobs:
                asyncio.ensure_future(self._loop(func, interval))

    def stop(self):
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    @staticmethod
    async def _loop(func: Callable, interval: int):
        while True:
            try:
                await func()
            except Exception as e:
                log.error("Scheduled job error: %s", e)
            await asyncio.sleep(interval)
