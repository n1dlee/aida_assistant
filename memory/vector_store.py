"""
memory/vector_store.py
Long-term semantic memory using ChromaDB (local, no server needed).
Falls back to simple keyword search if ChromaDB is not installed or
if the embedding model cannot be loaded/downloaded.
"""
import logging
import os
from typing import List, Dict, Any

log = logging.getLogger("aida.memory.vector")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")


class VectorStore:
    def __init__(self):
        self._collection = None
        self._fallback: List[str] = []
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
            # ChromaDB's default embedder (ONNXMiniLM) downloads a model on
            # first use. If that download fails (no internet, proxy, etc.)
            # we detect it HERE and fall back — otherwise every subsequent
            # add() would silently do nothing (caught at debug level).
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
        if self._collection is not None:
            try:
                import uuid
                self._collection.add(
                    documents=[text],
                    metadatas=[metadata if metadata else {"_": "1"}],
                    ids=[str(uuid.uuid4())],
                )
            except Exception as e:
                # Demote to fallback so memory is NEVER silently lost
                log.warning("ChromaDB add failed (%s) — switching to keyword fallback.", e)
                self._collection = None
                self._fallback.append(text)
        else:
            self._fallback.append(text)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if self._collection is not None:
            count = self._collection.count()
            if count == 0:
                return []
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(top_k, count),
                )
                return results["documents"][0] if results["documents"] else []
            except Exception as e:
                log.warning("ChromaDB search failed: %s", e)
                return []
        # Fallback: simple keyword match
        query_words = set(query.lower().split())
        scored = []
        for doc in self._fallback[-100:]:
            overlap = len(query_words & set(doc.lower().split()))
            if overlap:
                scored.append((overlap, doc))
        scored.sort(reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def count(self) -> int:
        if self._collection:
            return self._collection.count()
        return len(self._fallback)
