"""
AIDA — AI Desktop Assistant
Entry point
"""
import asyncio
import logging
from core.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aida")


async def main():
    log.info("Starting AIDA...")
    orchestrator = Orchestrator()
    await orchestrator.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("AIDA stopped.")
