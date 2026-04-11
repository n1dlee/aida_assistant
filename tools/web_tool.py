"""
tools/web_tool.py
Web search using DuckDuckGo (no API key needed).
Falls back to a plain URL suggestion if duckduckgo-search is not installed.
"""
import logging
from tools.base_tool import BaseTool

log = logging.getLogger("aida.tools.web")


class WebTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo."

    async def run(self, query: str, **kwargs) -> str:
        import asyncio

        def _search():
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                if not results:
                    return "No results found."
                lines = []
                for r in results:
                    lines.append(f"• {r['title']}\n  {r['body']}\n  {r['href']}")
                return "\n\n".join(lines)
            except ImportError:
                return (
                    f"Web search not available (install duckduckgo-search).\n"
                    f"Manual search: https://duckduckgo.com/?q={query.replace(' ', '+')}"
                )
            except Exception as e:
                log.error("Web search error: %s", e)
                return f"Search failed: {e}"

        return await asyncio.get_running_loop().run_in_executor(None, _search)
