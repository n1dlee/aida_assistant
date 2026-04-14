"""
memory/vector_store.py
Long-term semantic memory using ChromaDB (local, no server needed).
Falls back to simple keyword search if ChromaDB is not installed or
if the embedding model cannot be loaded/downloaded.

Role filtering: pass role="user" or role="assistant" to search() to
restrict results to one side of the conversation, preventing assistant
responses from drowning out user-side memories.
"""
import logging
import os
from typing import List, Dict, Any, Optional

log = logging.getLogger("aida.memory.vector")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")


class VectorStore:
    def __init__(self):
        self._collection = None
        self._fallback: List[Dict[str, str]] = []   # {"text": ..., "role": ...}
        self._init()

    def _init(self):
        try:
            import chromadb
            import uuid
            client = chromadb.PersistentClient(path=DB_PATH)
            collection = client.get_or_create_collection(
                name="aida_memory",
                metadata={"hnsw:space": "cosine"},
            )
            # Probe the embedding function with a dummy add + delete.
            probe_id = "_probe_" + str(uuid.uuid4())
            collection.add(documents=["probe"], metadatas=[{"_": "probe"}], ids=[probe_id])
            collection.delete(ids=[probe_id])
            self._collection = collection
            log.info("ChromaDB loaded (%d entries).", self._collection.count())
        except ImportError:
            log.warning("ChromaDB not installed — using fallback keyword memory.")
        except Exception as e:
            log.warning(
                "ChromaDB unavailable (%s) — using fallback keyword memory. "
                "Tip: ChromaDB's default embedder downloads a model on first run; "
                "ensure internet access or install sentence-transformers manually.",
                e,
            )
            self._collection = None

    def add(self, text: str, metadata: Dict[str, Any] = None):
        if not text.strip():
            return
        meta = metadata if metadata else {}
        if self._collection is not None:
            try:
                import uuid
                self._collection.add(
                    documents=[text],
                    metadatas=[meta],
                    ids=[str(uuid.uuid4())],
                )
            except Exception as e:
                log.warning("ChromaDB add failed (%s) — switching to keyword fallback.", e)
                self._collection = None
                self._fallback.append({"text": text, **meta})
        else:
            self._fallback.append({"text": text, **meta})

    def search(self, query: str, top_k: int = 3, role: Optional[str] = None) -> List[str]:
        """
        Search semantic memory.
        role: optional "user" or "assistant" — filters results to one speaker.
        """
        if self._collection is not None:
            count = self._collection.count()
            if count == 0:
                return []
            try:
                where = {"role": role} if role else None
                kwargs: Dict[str, Any] = dict(
                    query_texts=[query],
                    n_results=min(top_k, count),
                )
                if where:
                    kwargs["where"] = where
                results = self._collection.query(**kwargs)
                return results["documents"][0] if results["documents"] else []
            except Exception as e:
                log.warning("ChromaDB search failed: %s", e)
                return []

        # Fallback: simple keyword match (respects role filter)
        query_words = set(query.lower().split())
        pool = self._fallback[-200:]
        if role:
            pool = [d for d in pool if d.get("role") == role]
        scored = []
        for doc in pool:
            text = doc.get("text", "")
            overlap = len(query_words & set(text.lower().split()))
            if overlap:
                scored.append((overlap, text))
        scored.sort(reverse=True)
        return [t for _, t in scored[:top_k]]

    def count(self) -> int:
        if self._collection:
            return self._collection.count()
        return len(self._fallback)
