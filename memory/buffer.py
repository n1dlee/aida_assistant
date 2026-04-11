"""
memory/buffer.py
Short-term conversation memory. Keeps last N turns in RAM.
"""
from collections import deque
from typing import List, Dict


class ConversationBuffer:
    def __init__(self, max_turns: int = 20):
        self._history: deque = deque(maxlen=max_turns * 2)  # user+assistant pairs

    def add(self, role: str, content: str):
        self._history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def clear(self):
        self._history.clear()

    def __len__(self):
        return len(self._history)
