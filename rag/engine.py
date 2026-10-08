import re
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "kb"
CHROMA_DIR = BASE_DIR / "index" / "chroma"
COLLECTION_NAME = "kb_chunks"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
RRF_K = 60
DENSE_CANDIDATES = 10
SPARSE_CANDIDATES = 10
FUSION_CANDIDATES = 12
DEFAULT_K = 4
CE_RELEVANCE_GATE = 0.0

STOPWORDS = frozenset(
    """
    a an the and or but if then else is are was were be been being am do does
    did doing to of in on at by for with about as into from up down out off over
    under again please can could would should will shall may might must i me my
    mine we our ours you your yours he him his she her hers it its they them
    their theirs this that these those what which who whom whose how when where
    why there here than too very just also any some all most more other own same
    so only get got
    """.split()
)

_SPLIT_MARKER = re.compile(
    r"\?|;|\s+and also\s+|\s+as well as\s+|\s+and then\s+", re.IGNORECASE
)
_LEADING_CONNECTOR = re.compile(r"^(?:also|and)\s*,?\s+", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_name: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_name: str
    chunk_index: int
    text: str
    rrf_score: float
    ce_score: float
    dense_rank: int | None
    sparse_rank: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS
    ]


def decompose_query(query: str) -> list[str]:
    parts = []
    for part in _SPLIT_MARKER.split(query):
        part = _LEADING_CONNECTOR.sub("", part.strip())
        if len(part.split()) >= 2:
            parts.append(part)
    if not parts:
        return [query]
    seen: set[str] = set()
    unique = []
    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            unique.append(part)
    return unique


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n"],
    )
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        for index, piece in enumerate(splitter.split_text(path.read_text(encoding="utf-8"))):
            chunks.append(
                Chunk(
                    chunk_id=f"{path.name}#{index}",
                    doc_name=path.name,
                    chunk_index=index,
                    text=piece,
                )
            )
    return chunks


class HybridRAG:
    def __init__(self, kb_dir: Path = KB_DIR, chroma_dir: Path = CHROMA_DIR):
        self.chunks = load_chunks(kb_dir)
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self._collection = self._ensure_collection(chroma_dir)
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        self._reranker = None

    def _ensure_collection(self, chroma_dir: Path):
        client = chromadb.PersistentClient(path=str(chroma_dir))
        existing = {c.name for c in client.list_collections()}
        if COLLECTION_NAME in existing:
            collection = client.get_collection(COLLECTION_NAME)
            if collection.count() == len(self.chunks):
                return collection
            client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        collection.add(
            ids=[c.chunk_id for c in self.chunks],
            documents=[c.text for c in self.chunks],
            metadatas=[
                {"doc_name": c.doc_name, "chunk_index": c.chunk_index}
                for c in self.chunks
            ],
        )
        return collection

    @property
    def cross_encoder(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        return self._reranker

    def dense_search(self, query: str, n: int = DENSE_CANDIDATES) -> list[str]:
        limit = max(1, min(n, len(self.chunks)))
        result = self._collection.query(
            query_texts=[query], n_results=limit, include=["distances"]
        )
        return result["ids"][0]

    def sparse_search(self, query: str, n: int = SPARSE_CANDIDATES) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        hits = [
            self.chunks[i].chunk_id
            for i in order
            if scores[i] > 0
        ]
        return hits[:n]

    def rerank(self, query: str, chunk_ids: list[str]) -> list[float]:
        if not chunk_ids:
            return []
        texts = [self.by_id[cid].text for cid in chunk_ids]
        scores = self.cross_encoder.predict([(query, text) for text in texts])
        return [float(score) for score in scores]

    def retrieve(self, query: str, k: int = DEFAULT_K, rerank: bool = True) -> list[RetrievedChunk]:
        sub_queries = decompose_query(query)
        dense_rank: dict[str, int] = {}
        sparse_rank: dict[str, int] = {}
        per_sub: list[list[tuple[str, float, float]]] = []

        for sub in sub_queries:
            rrf: dict[str, float] = {}
            for rank, cid in enumerate(self.dense_search(sub), 1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank)
                dense_rank[cid] = min(dense_rank.get(cid, rank), rank)
            for rank, cid in enumerate(self.sparse_search(sub), 1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank)
                sparse_rank[cid] = min(sparse_rank.get(cid, rank), rank)

            fused = sorted(rrf, key=lambda cid: -rrf[cid])[:FUSION_CANDIDATES]
            if not rerank:
                per_sub.append([(cid, 0.0, rrf[cid]) for cid in fused])
                continue
            scores = self.rerank(sub, fused)
            score_by_cid = dict(zip(fused, scores))
            if scores and max(scores) >= CE_RELEVANCE_GATE:
                order = sorted(fused, key=lambda cid: -score_by_cid[cid])
            else:
                order = fused
            per_sub.append(
                [(cid, score_by_cid[cid], rrf[cid]) for cid in order]
            )

        merged: list[tuple[str, float, float]] = []
        seen: set[str] = set()
        for depth in range(max(len(group) for group in per_sub)):
            for group in per_sub:
                if depth >= len(group):
                    continue
                cid = group[depth][0]
                if cid not in seen:
                    seen.add(cid)
                    merged.append(group[depth])

        return [
            RetrievedChunk(
                chunk_id=cid,
                doc_name=self.by_id[cid].doc_name,
                chunk_index=self.by_id[cid].chunk_index,
                text=self.by_id[cid].text,
                rrf_score=rrf_score,
                ce_score=score,
                dense_rank=dense_rank.get(cid),
                sparse_rank=sparse_rank.get(cid),
            )
            for cid, score, rrf_score in merged[:k]
        ]


_engine: HybridRAG | None = None


def get_engine() -> HybridRAG:
    global _engine
    if _engine is None:
        _engine = HybridRAG()
    return _engine
