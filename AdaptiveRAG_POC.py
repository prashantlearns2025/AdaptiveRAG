import streamlit as st

# This MUST be the first Streamlit command
st.set_page_config(
    page_title="🤖 Adaptive RAG Agent", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from typing import Any, Dict, List, Literal, Optional, Tuple
from langchain.schema import Document
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

import hashlib
import os
import pickle
import re
import shutil
from pathlib import Path
from dotenv import load_dotenv
import io
import sys
from contextlib import redirect_stdout
import time

load_dotenv(override=True)
groq_api_key=os.getenv("GROQ_API_KEY")

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = os.getenv("LANGSMITH_ENDPOINT") 
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
llm= ChatOpenAI(model="gpt-4.1-nano") #ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key) 

# Custom print capture class
class PrintCapture:
    def __init__(self):
        self.messages = []
    
    def write(self, message):
        if message.strip():
            self.messages.append(message.strip())
            # Update Streamlit interface in real-time
            if hasattr(st, 'session_state') and 'step_container' in st.session_state:
                with st.session_state.step_container:
                    st.write(f"🔄 {message.strip()}")
    
    def flush(self):
        pass

# %%
class State(TypedDict, total=False):
    query: str
    query_redirect: str
    retrieved_docs: List[Document]
    retrieved_docs_with_scores: List[Tuple[Document, float]]
    dense_retrieval_with_scores: List[Tuple[Document, float]]
    bm25_retrieval_with_scores: List[Tuple[Document, float]]
    retrieval_metrics: Dict[str, int]
    generation: str

# %%
class QueryCategorizer(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    query_category: Literal["Vector Store", "Web Search"]

structured_llm_categorizer = llm.with_structured_output(QueryCategorizer)

# %%
# Query Analysis:
def QueryAnalyser(state: State):
    print("---Query analyser checks the query---")
    # Prompt
    system = """You are an intelligent routing expert that determines the best source to answer user questions.

                **Your Knowledge Base** contains three comprehensive technical articles by Lilian Weng:
                1. **LLM-Powered Autonomous Agents** - Covers agent architecture (planning, memory, tool use), frameworks like ReAct and Reflexion, task decomposition methods (Chain-of-Thought, Tree of Thoughts), memory systems (MIPS algorithms), and case studies like AutoGPT, BabyAGI, and Generative Agents.

                2. **Prompt Engineering** - Covers zero-shot and few-shot learning, instruction prompting, Chain-of-Thought (CoT) techniques, self-consistency sampling, automatic prompt design (APE, AutoPrompt), and augmented language models with retrieval, programming, and external APIs (PAL, Toolformer, TALM).

                3. **Adversarial Attacks on LLMs** - Covers threat models, token manipulation attacks (TextFooler, BERT-Attack), gradient-based attacks (GBDA, HotFlip, UAT), jailbreak prompting techniques, red-teaming approaches (human and model-based), and defense mechanisms.

                **Routing Decision:**
                - **Use VECTORSTORE** if the question asks about:
                • Agent architectures, components, or frameworks (planning, memory, tool use, ReAct, Reflexion, MRKL)
                • Prompt engineering techniques (CoT, few-shot, zero-shot, instruction prompting, Tree of Thoughts)
                • Memory systems and retrieval (MIPS, vector stores, FAISS, HNSW, embeddings)
                • Task decomposition and reasoning chains
                • Adversarial attacks on LLMs (UAT, jailbreaking, red-teaming, GBDA, HotFlip)
                • Tool use and augmented LLMs (Toolformer, HuggingGPT, API-Bank)
                • Specific papers, methods, or researchers mentioned in these topics
                • Defense mechanisms against adversarial attacks

                - **Use WEB-SEARCH** for:
                • Current events, news, or information after January 2025
                • Topics outside the three articles above
                • Real-time data (stock prices, weather, recent releases)
                • Questions requiring up-to-date information
                • General knowledge not covered in the knowledge base

                **Examples:**
                - "What is ReAct framework?" → VECTORSTORE
                - "How does Chain-of-Thought prompting work?" → VECTORSTORE
                - "What are Universal Adversarial Triggers?" → VECTORSTORE
                - "What is the latest version of GPT?" → WEB-SEARCH
                - "Current AI news today" → WEB-SEARCH

                Analyze the user's question carefully and route to the appropriate source."""
    analyse_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{query}"),
        ]
    )

    analyse_chain = analyse_prompt | structured_llm_categorizer
    analyse = analyse_chain.invoke({"query":state["query"]})
    return{"query_redirect": analyse.query_category}

# %%
def router(state: State):
    if state["query_redirect"] == "Vector Store":
        print("---Decides to go to Vector store---")
        return "Vector Store"
    elif state["query_redirect"] == "Web Search":
        print("---Decides to go for Web search---")
        return "Web Search"

# %%
# Retrieve more chunks than we need; cross-encoder re-ranker scores and keeps the best.
RETRIEVE_K = 10
RERANK_TOP_N = 4
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_INDEX_DIR = Path(__file__).resolve().parent / "data" / "bm25_index"
BM25_INDEX_FILE = BM25_INDEX_DIR / "bm25.pkl"
RRF_K = 60

VECTORSTORE_DIR = Path(__file__).resolve().parent / "data" / "faiss_index"
URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def _faiss_index_files_exist(index_dir: Path) -> bool:
    """True only when LangChain FAISS save_local artifacts are present."""
    return (index_dir / "index.faiss").is_file() and (index_dir / "index.pkl").is_file()


def _remove_incomplete_faiss_dir(index_dir: Path) -> None:
    if index_dir.exists():
        shutil.rmtree(index_dir)


def _get_embeddings():
    return OpenAIEmbeddings()


def _build_vectorstore():
    print("---Fetching documents, chunking, and embedding---")
    docs = [WebBaseLoader(url).load() for url in URLS]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    for idx, doc in enumerate(doc_splits):
        metadata = dict(doc.metadata or {})
        metadata["chunk_id"] = f"chunk_{idx}"
        if "source_url" not in metadata and "source" in metadata:
            metadata["source_url"] = metadata["source"]
        doc.metadata = metadata

    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embeddings)

    VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"---FAISS index saved to {VECTORSTORE_DIR}---")

    _build_bm25_index(doc_splits)
    return vectorstore


def _load_vectorstore():
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def _bm25_index_files_exist(index_dir: Path) -> bool:
    return (index_dir / "bm25.pkl").is_file()


def _remove_incomplete_bm25_dir(index_dir: Path) -> None:
    if index_dir.exists():
        shutil.rmtree(index_dir)


def bm25_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _save_bm25_index(documents: List[Document], tokenized_corpus: List[List[str]]) -> None:
    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    docs_data = [
        {"page_content": doc.page_content, "metadata": dict(doc.metadata or {})}
        for doc in documents
    ]
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump({"docs": docs_data, "corpus": tokenized_corpus}, f)


def _load_bm25_index():
    if BM25Okapi is None:
        print("---rank_bm25 package not installed; BM25 retrieval disabled---")
        return None
    if not _bm25_index_files_exist(BM25_INDEX_DIR):
        return None
    try:
        with open(BM25_INDEX_FILE, "rb") as f:
            data = pickle.load(f)
        docs = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in data["docs"]
        ]
        bm25 = BM25Okapi(data["corpus"])
        return {"bm25": bm25, "docs": docs}
    except Exception as exc:
        print(f"---BM25 load failed ({exc}); clearing BM25 index---")
        _remove_incomplete_bm25_dir(BM25_INDEX_DIR)
        return None


def _build_bm25_index(documents: List[Document]):
    if BM25Okapi is None:
        raise ImportError(
            "rank_bm25 is required for BM25 index building. Install it with 'pip install rank_bm25'."
        )
    print("---Building BM25 index---")
    tokenized_corpus = [bm25_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    _save_bm25_index(documents, tokenized_corpus)
    return {"bm25": bm25, "docs": documents}


@st.cache_resource
def get_bm25_index():
    if _bm25_index_files_exist(BM25_INDEX_DIR):
        existing = _load_bm25_index()
        if existing is not None:
            return existing
    if VECTORSTORE_DIR.exists():
        print("---BM25 index missing; rebuilding FAISS and BM25 indexes---")
        _build_vectorstore()
        return _load_bm25_index()
    print("---No BM25 index found; building FAISS and BM25 indexes---")
    _build_vectorstore()
    return _load_bm25_index()


@st.cache_resource
def get_cross_encoder():
    return HuggingFaceCrossEncoder(model_name=RERANK_MODEL)


@st.cache_resource
def get_retriever():
    if _faiss_index_files_exist(VECTORSTORE_DIR):
        try:
            print("---Loading existing FAISS index from disk---")
            vectorstore = _load_vectorstore()
        except Exception as exc:
            print(f"---FAISS load failed ({exc}); rebuilding index---")
            _remove_incomplete_faiss_dir(VECTORSTORE_DIR)
            vectorstore = _build_vectorstore()
    else:
        if VECTORSTORE_DIR.exists():
            print("---Incomplete FAISS index folder; rebuilding---")
            _remove_incomplete_faiss_dir(VECTORSTORE_DIR)
        else:
            print("---No index found; building and saving FAISS index---")
        vectorstore = _build_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_K})


def rebuild_vectorstore():
    get_retriever.clear()
    get_bm25_index.clear()
    _remove_incomplete_faiss_dir(VECTORSTORE_DIR)
    _remove_incomplete_bm25_dir(BM25_INDEX_DIR)
    _build_vectorstore()
    get_retriever()
    get_bm25_index()


# Semantic cache (separate FAISS index: query embedding -> cached answer)
SEMANTIC_CACHE_DIR = Path(__file__).resolve().parent / "data" / "semantic_cache"
CACHE_SIMILARITY_THRESHOLD = 0.82



def _load_semantic_cache():
    if not _faiss_index_files_exist(SEMANTIC_CACHE_DIR):
        if SEMANTIC_CACHE_DIR.exists():
            _remove_incomplete_faiss_dir(SEMANTIC_CACHE_DIR)
        return None
    try:
        return FAISS.load_local(
            str(SEMANTIC_CACHE_DIR),
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        print(f"---Semantic cache load failed ({exc}); clearing cache---")
        _remove_incomplete_faiss_dir(SEMANTIC_CACHE_DIR)
        get_semantic_cache.clear()
        return None


@st.cache_resource
def get_semantic_cache():
    return _load_semantic_cache()


def cache_eval_metadata(
    *,
    cache_hit: bool,
    cache_similarity_score: Optional[float],
    cached_query: Optional[str] = None,
) -> dict:
    """Metadata fields for LangSmith evaluation and run filtering."""
    meta = {
        "cache_hit": cache_hit,
        "cache_similarity_score": cache_similarity_score,
    }
    if cached_query is not None:
        meta["cached_query"] = cached_query
    return meta


def _get_cache_eval_holder(config: Optional[RunnableConfig]) -> dict:
    if config:
        holder = (config.get("configurable") or {}).get("cache_eval_holder")
        if isinstance(holder, dict):
            return holder
    return {}


def _write_cache_eval_holder(config: RunnableConfig, outcome: dict) -> dict:
    """Persist cache eval in RunnableConfig (shared across LangGraph nodes)."""
    meta = cache_eval_metadata(
        cache_hit=bool(outcome.get("cache_hit", False)),
        cache_similarity_score=outcome.get("cache_similarity_score"),
        cached_query=outcome.get("cached_query"),
    )
    holder = _get_cache_eval_holder(config)
    if holder is not None:
        holder.update(meta)
    return meta


def update_run_metrics(config: Optional[RunnableConfig], metrics: dict) -> None:
    holder = _get_cache_eval_holder(config)
    if holder is not None:
        holder.update(metrics)


def patch_langsmith_run_metadata(run_id: str, cache_meta: dict) -> bool:
    """Merge cache eval into an existing LangSmith run (same Metadata panel as input_query)."""
    if not run_id or not cache_meta:
        return False

    from langsmith import Client

    client = Client()
    last_error: Optional[Exception] = None

    for attempt in range(6):
        try:
            run = client.read_run(run_id)
            merged_extra = dict(run.extra or {})
            merged_meta = dict(merged_extra.get("metadata") or {})
            merged_meta.update(cache_meta)
            merged_extra["metadata"] = merged_meta
            client.update_run(run_id, extra=merged_extra)
            client.flush()
            return True
        except Exception as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))

    print(f"---LangSmith cache metadata update failed: {last_error}---")
    return False


class CacheEvalLangSmithHandler(BaseCallbackHandler):
    """Patches root LangGraph run metadata when the graph finishes."""

    def __init__(self, cache_eval_holder: dict):
        super().__init__()
        self.cache_eval_holder = cache_eval_holder
        self.root_run_id: Optional[str] = None
        self._metadata_patched = False

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name", "")
        if name == "LangGraph":
            self.root_run_id = str(run_id)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if self._metadata_patched or not self.cache_eval_holder:
            return
        rid = str(run_id)
        if self.root_run_id and rid == self.root_run_id:
            self._metadata_patched = patch_langsmith_run_metadata(
                rid, dict(self.cache_eval_holder)
            )

    def patch_root_run_metadata(self) -> None:
        """Fallback patch after invoke (tracer may finish slightly later)."""
        if self._metadata_patched or not self.root_run_id or not self.cache_eval_holder:
            return
        meta = dict(self.cache_eval_holder)
        self._metadata_patched = patch_langsmith_run_metadata(
            self.root_run_id, meta
        )
        if self._metadata_patched:
            print(
                f"---LangSmith metadata saved: cache_hit={meta.get('cache_hit')}, "
                f"cache_similarity_score={meta.get('cache_similarity_score')}---"
            )


def lookup_semantic_cache(query: str) -> dict:
    vectorstore = get_semantic_cache()
    if vectorstore is None:
        return cache_eval_metadata(cache_hit=False, cache_similarity_score=None)

    results = vectorstore.similarity_search_with_relevance_scores(query, k=1)
    if not results:
        return cache_eval_metadata(cache_hit=False, cache_similarity_score=None)

    doc, score = results[0]
    score = float(score)
    if score < CACHE_SIMILARITY_THRESHOLD:
        print(f"---Semantic cache MISS (best relevance={score:.3f})---")
        return cache_eval_metadata(cache_hit=False, cache_similarity_score=score)

    print(f"---Semantic cache HIT (relevance={score:.3f})---")
    return {
        **cache_eval_metadata(
            cache_hit=True,
            cache_similarity_score=score,
            cached_query=doc.metadata.get("query", ""),
        ),
        "generation": doc.page_content,
        "query_redirect": doc.metadata.get("query_redirect", ""),
    }


def add_to_semantic_cache(query: str, generation: str, query_redirect: str):
    doc = Document(
        page_content=generation,
        metadata={"query": query, "query_redirect": query_redirect},
    )
    embeddings = _get_embeddings()
    existing = _load_semantic_cache()

    if existing is None:
        vectorstore = FAISS.from_documents([doc], embedding=embeddings)
    else:
        vectorstore = existing
        vectorstore.add_documents([doc])

    SEMANTIC_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(SEMANTIC_CACHE_DIR))
    get_semantic_cache.clear()
    print("---Answer stored in semantic cache---")


def clear_semantic_cache():
    get_semantic_cache.clear()
    _remove_incomplete_faiss_dir(SEMANTIC_CACHE_DIR)


def _attach_metadata_to_active_trace(cache_meta: dict) -> None:
    """Best-effort: attach to in-flight LangSmith runs before they are persisted."""
    try:
        from langsmith import run_helpers

        run_tree = run_helpers.get_current_run_tree()
        while run_tree is not None:
            if hasattr(run_tree, "add_metadata"):
                run_tree.add_metadata(cache_meta)
            run_tree = getattr(run_tree, "parent_run", None)
    except Exception:
        pass


def semantic_cache_lookup(state: State, config: RunnableConfig):
    print("---Checking semantic cache---")
    outcome = lookup_semantic_cache(state["query"])
    meta = _write_cache_eval_holder(config, outcome)
    _attach_metadata_to_active_trace(meta)
    if outcome["cache_hit"]:
        return {
            "generation": outcome["generation"],
            "query_redirect": outcome["query_redirect"],
        }
    return {}


def semantic_cache_route(state: State, config: RunnableConfig):
    holder = _get_cache_eval_holder(config)
    # Route from holder (reliable across nodes); generation set only on cache hit.
    if holder.get("cache_hit") or state.get("generation"):
        return "cache_hit"
    return "cache_miss"


def semantic_cache_store(state: State):
    generation = state.get("generation")
    query = state.get("query")
    route = state.get("query_redirect", "")
    if generation and query and route == "Vector Store":
        add_to_semantic_cache(query, generation, route)
    elif generation and query:
        print("---Skipping semantic cache store for Web Search route---")
    return {}


# %%
# Vector Store Retrieval:
def get_doc_id(doc: Document) -> str:
    metadata = dict(doc.metadata or {})
    if metadata.get("chunk_id"):
        return str(metadata["chunk_id"])
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def bm25_retrieve(query: str, k: int):
    if BM25Okapi is None:
        print("---rank_bm25 not installed; skipping BM25 retrieval---")
        return []

    bm25_index = get_bm25_index()
    if bm25_index is None:
        return []

    tokenized_query = bm25_tokenize(query)
    bm25 = bm25_index["bm25"]
    docs = bm25_index["docs"]
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for idx in ranked_indices:
        doc = docs[idx]
        score = float(scores[idx])
        meta = dict(doc.metadata or {})
        meta["bm25_score"] = score
        results.append((Document(page_content=doc.page_content, metadata=meta), score))
    return results


def rrf_fusion(
    dense_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    top_n: int = RETRIEVE_K,
    rrf_k: int = RRF_K,
):
    fused = {}

    for rank, result in enumerate(dense_results, start=1):
        doc, score = result
        item = fused.setdefault(
            get_doc_id(doc),
            {
                "doc": doc,
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": None,
                "bm25_score": None,
                "sources": set(),
            },
        )
        item["doc"] = doc
        item["dense_rank"] = rank
        item["dense_score"] = float(score)
        item["sources"].add("dense")

    for rank, result in enumerate(bm25_results, start=1):
        doc, score = result
        item = fused.setdefault(
            get_doc_id(doc),
            {
                "doc": doc,
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": None,
                "bm25_score": None,
                "sources": set(),
            },
        )
        item["doc"] = doc
        item["bm25_rank"] = rank
        item["bm25_score"] = float(score)
        item["sources"].add("bm25")

    fused_list = []
    for item in fused.values():
        score = 0.0
        if item["dense_rank"] is not None:
            score += 1.0 / (rrf_k + item["dense_rank"])
        if item["bm25_rank"] is not None:
            score += 1.0 / (rrf_k + item["bm25_rank"])
        item["rrf_score"] = score
        fused_list.append(item)

    fused_list.sort(key=lambda item: item["rrf_score"], reverse=True)
    fused_docs = []
    for item in fused_list[:top_n]:
        doc = item["doc"]
        meta = dict(doc.metadata or {})
        meta["retrieval_sources"] = sorted(list(item["sources"]))
        if item["dense_score"] is not None:
            meta["retrieval_dense_score"] = item["dense_score"]
            meta["dense_rank"] = item["dense_rank"]
        if item["bm25_score"] is not None:
            meta["retrieval_bm25_score"] = item["bm25_score"]
            meta["bm25_rank"] = item["bm25_rank"]
        meta["retrieval_rrf_score"] = float(item["rrf_score"])
        fused_docs.append(Document(page_content=doc.page_content, metadata=meta))

    bm25_hits = sum(1 for item in fused_list[:top_n] if item["dense_rank"] is None and item["bm25_rank"] is not None)
    dense_hits = sum(1 for item in fused_list[:top_n] if item["bm25_rank"] is None and item["dense_rank"] is not None)
    overlap_hits = sum(1 for item in fused_list[:top_n] if item["dense_rank"] is not None and item["bm25_rank"] is not None)

    metrics = {
        "bm25_hits": bm25_hits,
        "dense_hits": dense_hits,
        "overlap_hits": overlap_hits,
    }
    return fused_docs, metrics


def retrieval(state: State, config: Optional[RunnableConfig] = None):
    print("---Hybrid Dense + BM25 retrieval begins---")
    query = state["query"]

    vectorstore = _load_vectorstore()
    dense_results = vectorstore.similarity_search_with_relevance_scores(query, k=RETRIEVE_K)
    bm25_results = bm25_retrieve(query, k=RETRIEVE_K)

    fused_docs, metrics = rrf_fusion(dense_results, bm25_results, top_n=RETRIEVE_K)
    update_run_metrics(config, metrics)

    retrieved_docs_with_scores = [
        (
            doc,
            float(doc.metadata.get("retrieval_rrf_score", 0.0)),
        )
        for doc in fused_docs
    ]

    return {
        "retrieved_docs": fused_docs,
        "retrieved_docs_with_scores": retrieved_docs_with_scores,
        "dense_retrieval_with_scores": dense_results,
        "bm25_retrieval_with_scores": bm25_results,
        "retrieval_metrics": metrics,
    }


def should_skip_reranking(docs_with_scores: List[Tuple[Document, float]]) -> Tuple[bool, str]:
    """
    Determine if reranking should be skipped based on AND conditions:
    Skip reranking ONLY if ALL conditions are true:
    1. Document count <= 10 (threshold for expensive cross-encoder)
    2. Top score > 0.8 (very confident retrieval)
    3. Any score difference > 0.1 (clear ranking already established)
    
    If ANY condition is NOT TRUE → perform reranking
    
    Returns: (skip_reranking: bool, reason: str)
    """
    if not docs_with_scores:
        return True, "No documents to rerank"
    
    # Extract scores for condition checks
    scores = [score for _, score in docs_with_scores]
    top_score = scores[0]
    
    condition1 = len(docs_with_scores) <= 10
    condition2 = top_score > 0.8
    condition3 = any(scores[i] - scores[i + 1] > 0.1 for i in range(len(scores) - 1))
    
    print(f"---Condition 1 (doc_count <= 10): {condition1} ({len(docs_with_scores)} docs)---")
    print(f"---Condition 2 (top_score > 0.8): {condition2} (score: {top_score:.4f})---")
    print(f"---Condition 3 (score_diff > 0.1): {condition3}---")
    
    # Skip reranking only if ALL conditions are true (AND logic)
    if condition1 and condition2 and condition3:
        reason = "All conditions met: low doc count AND high confidence AND clear ranking"
        print(f"---Skipping rerank: {reason}---")
        return True, reason
    
    # Perform reranking if ANY condition is NOT true
    print("---Performing rerank: At least one condition is not met---")
    return False, "Applying rerank"


def rerank_after_retrieval(state: State):
    """Re-rank retrieved chunks by cross-encoder relevance scores (higher = more relevant).
    
    Conditionally skips reranking if all these are true:
    - Retrieved docs <= 10
    - Top retrieval score > 0.8
    - Score difference between any 2 docs > 0.1
    """
    docs_with_scores = state.get("retrieved_docs_with_scores", [])
    docs = state["retrieved_docs"]
    
    if not docs:
        return {"retrieved_docs": docs}
    
    # Check if we should skip reranking
    skip_rerank, reason = should_skip_reranking(docs_with_scores)
    
    if skip_rerank:
        print(f"---Reranking skipped: {reason}---")
        # Attach fused retrieval scores to metadata for reference
        docs_with_metadata = []
        for doc, score in docs_with_scores:
            meta = dict(doc.metadata)
            meta["retrieval_rrf_score"] = float(score)
            docs_with_metadata.append(
                Document(page_content=doc.page_content, metadata=meta)
            )
        return {"retrieved_docs": docs_with_metadata}
    
    # Apply reranking if conditions not met
    print("---Re-ranking retrieved chunks by relevance score---")
    query = state["query"]
    model = get_cross_encoder()
    scores = model.score([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)[
        : RERANK_TOP_N
    ]

    reranked_docs = []
    for doc, score in ranked:
        meta = dict(doc.metadata)
        meta["reranker_score"] = float(score)
        reranked_docs.append(
            Document(page_content=doc.page_content, metadata=meta)
        )
        print(f"---Re-rank relevance_score={score:.4f}---")

    return {"retrieved_docs": reranked_docs}

# %%
class grade_format(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    grade_check: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(grade_format)

# %%
# Grader - grades retrieved docs
def grader(state: State):
    print("---Docs are being graded for relevancy---")
    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user's query, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User query: {query}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    query  = state["query"]
    documents = state["retrieved_docs"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"query": query, "document": d.page_content}
        )
        grade = score.grade_check
        
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            print("---Relevant doc Selected---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            print("---Irrelevant doc Rejected---")
            continue
    return {"retrieved_docs": filtered_docs}

# %%
def decider(state:State):
    if not state["retrieved_docs"]:
        print("---All docs irrelevant, proceed to Transform the User Query---")
        return "transform query"
    else:
        print("---Proceeding to generate---")
        return "generate"

# %%
# Query transformer
def transform_query(state: State):
    print("---Transforming the user query---")
    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. Your response should just contain only one transformed user question and nothing else, no extra content, not even a label such as transformed query:"""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human","Here is the initial query: \n\n {query} \n Formulate an improved question/ query.")
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    query = state["query"]
    updated_query = question_rewriter.invoke({"query": query})
    print(f"Transformed user query: {updated_query}")
    return{"query": updated_query}

# %%
### Generate
def generate(state: State):
    print("---Generating---")
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    query = state["query"]
    docs = state["retrieved_docs"]

    generation = rag_chain.invoke({"context": docs, "question": query})
    return{"generation": generation}

# %%
# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    model_config = {"arbitrary_types_allowed": True}
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)

# %%
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    model_config = {"arbitrary_types_allowed": True}
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

answer_grader_structure = llm.with_structured_output(GradeAnswer)

# LLM with function call
# Prompt
system = """You are a strict binary grader whose ONLY job is to decide whether an LLM's response directly answers the user's question.
Return exactly and only the single token 'yes' or 'no' (lowercase) with no additional text, punctuation, explanation, or whitespace.

Rules:
- Return 'yes' when the generation directly and explicitly answers the question (this includes short factual replies, dates, numbers, single-word answers, or an explicit sentence that resolves the user's question).
- Return 'no' when the generation does not answer the question, says it cannot answer, is unrelated, only restates the question, or is ambiguous/hedging without a clear answer.

Behavioral notes:
- If the answer is embedded inside a longer sentence (e.g. "The answer is 42."), still return 'yes'.
- If the model replies with uncertainty like "I don't know" or "I cannot answer that", return 'no'.
- Be tolerant of paraphrases — focus on whether the user's intent was resolved, not on exact wording.

Examples (for clarification only — DO NOT output these examples):
Q: "What is 2+2?" A: "2+2 equals 4." -> yes
Q: "When was X released?" A: "I don't know." -> no
Q: "Summarize the steps to reset a password." A: "Step 1: ... Step 2: ..." -> yes

Output rule: only 'yes' or 'no'.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question:\n\n{question}\n\nLLM generation:\n\n{generation}"),
    ]
)

answer_grader = answer_prompt | answer_grader_structure

# %%
### Hallucination Grader
def hallucination_check(state: State):
    print("---Checking for hallucination in generated response---")
    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )
    query  = state["query"]
    documents = state["retrieved_docs"]
    generation = state["generation"]

    hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---No Hallucination, proceed to check if Response answers the Question---")
        # Check question-answering
        score = answer_grader.invoke({"question": query, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---Yes! Response answered the Question---")
            return "question answered"
        else:
            print("---No! Response didn't answer the Question, transforming the user Query---")
            return "question not answered"
    else:
        print("---Response is Hallucinated, Regenerating the response---")
        return "not supported"

# %%
# Web Search:
def web_search(state: State):
    print("---Web search begins---")
    web_search_tool = TavilySearchResults(k=3)
    query = state["query"]

    search_result = web_search_tool.invoke(query)
    web_results = "\n".join([indiv_result["content"] for indiv_result in search_result])
    web_results = Document(page_content=web_results)
    
    return{"retrieved_docs": [web_results]}

# %%
# Build the graph
@st.cache_resource
def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("SemanticCacheLookup", semantic_cache_lookup)
    graph_builder.add_node("CacheStore", semantic_cache_store)
    graph_builder.add_node("QueryAnalyser",QueryAnalyser)
    graph_builder.add_node("Retrieve",retrieval)
    graph_builder.add_node("Rerank", rerank_after_retrieval)
    graph_builder.add_node("RtvDocsGrader",grader)
    graph_builder.add_node("QueryTransformer",transform_query)
    graph_builder.add_node("Generator",generate)
    graph_builder.add_node("Web Search",web_search)

    graph_builder.add_edge(START, "SemanticCacheLookup")
    graph_builder.add_conditional_edges(
        "SemanticCacheLookup",
        semantic_cache_route,
        {"cache_hit": END, "cache_miss": "QueryAnalyser"},
    )
    graph_builder.add_conditional_edges("QueryAnalyser", router, {"Vector Store": "Retrieve","Web Search": "Web Search"})
    graph_builder.add_edge("Retrieve", "Rerank")
    graph_builder.add_edge("Rerank", "RtvDocsGrader")
    graph_builder.add_conditional_edges("RtvDocsGrader", decider, {"transform query": "QueryTransformer","generate": "Generator"})
    graph_builder.add_conditional_edges(
        "Generator",
        hallucination_check,
        {
            "not supported": "Generator",
            "question not answered": "QueryTransformer",
            "question answered": "CacheStore",
        },
    )
    graph_builder.add_edge("CacheStore", END)
    graph_builder.add_edge("QueryTransformer","Retrieve")
    graph_builder.add_edge("Web Search","Generator")

    return graph_builder.compile()

graph = build_graph()


# Langsmith Metadata Configuration to be passed during graph invocation & traced at Langsmith ui
user_id = "user_12345"

BASE_RUN_METADATA = {
    "user_id": user_id,
    "user_type": "premium",
    "session_id": "abc123",
}


def build_invoke_config(query: str) -> Tuple[dict, CacheEvalLangSmithHandler, dict]:
    """RunnableConfig + callback handler + shared cache eval holder for LangSmith."""
    cache_eval_holder: dict = {}
    handler = CacheEvalLangSmithHandler(cache_eval_holder)
    invoke_config = {
        "configurable": {
            "thread_id": f"session_{user_id}",
            "cache_eval_holder": cache_eval_holder,
        },
        "metadata": {
            **BASE_RUN_METADATA,
            "input_query": query,
        },
        "callbacks": [handler],
    }
    return invoke_config, handler, cache_eval_holder

# %%
# Streamlit Frontend
def main():
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .step-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .result-container {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Adaptive RAG Agent</h1>
        <p>Advanced Retrieval-Augmented Generation with Adaptive Routing & Quality Control</p>
    </div>
    """, unsafe_allow_html=True)

    st.image("adaptive_rag_arch.png", use_container_width=True)
    
    # Sidebar with agent architecture
    with st.sidebar:
        st.header("🏗️ Agent Architecture")
        
        # Agent workflow diagram
        st.markdown("""
        ### 🔄 Workflow Overview:
        1. **Semantic cache** - Returns similar past answers (FAISS)
        2. **Query Analysis** - Categorizes query type
        3. **Routing** - Vector Store vs Web Search
        4. **Retrieval** - Fetches candidate documents
        5. **Re-ranking** - Cross-encoder scores and orders chunks by relevance
        6. **Grading** - Validates document relevance
        7. **Generation** - Creates response
        8. **Quality Check** - Hallucination & relevance check
        9. **Query Transform** - Optimizes if needed
        """)

        st.divider()
        st.subheader("⚡ Semantic Cache")
        if _faiss_index_files_exist(SEMANTIC_CACHE_DIR):
            st.success("Semantic cache index on disk")
        else:
            st.info("Cache empty — fills after successful Vector Store answers")
        if st.button("Clear semantic cache", use_container_width=True):
            clear_semantic_cache()
            build_graph.clear()
            st.success("Semantic cache cleared.")
            st.rerun()

        st.divider()
        st.subheader("📚 Knowledge Base")
        if _faiss_index_files_exist(VECTORSTORE_DIR):
            st.success("FAISS index on disk")
        else:
            st.info("No index on disk yet — will build on first vector retrieval")

        if st.button("Rebuild knowledge base", use_container_width=True):
            with st.spinner("Rebuilding index (web fetch, chunking, embedding)..."):
                rebuild_vectorstore()
            st.success("Knowledge base rebuilt successfully.")
            st.rerun()

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("💬 Query Input")
        
        # Query input
        user_query = st.text_area(
            "Enter your question:",
            placeholder="e.g., What is hallucination in language models?",
            height=100,
            help="Ask questions about agents, prompt engineering, or adversarial attacks for vector store retrieval, or any other topic for web search."
        )
        
        # Submit button
        if st.button("🚀 Process Query", type="primary", use_container_width=True):
            if user_query.strip():
                process_query(user_query.strip())
            else:
                st.warning("⚠️ Please enter a query first!")

        st.image("adaptive_rag_graph.png", use_container_width=True)
        
       
    with col2:
        st.header("⚙️ Agent Execution Steps")
        
        # Container for steps
        if 'step_messages' not in st.session_state:
            st.session_state.step_messages = []
        
        step_placeholder = st.empty()
        
        # Store reference for real-time updates
        st.session_state.step_container = step_placeholder
        
        if st.session_state.step_messages:
            with step_placeholder.container():
                for msg in st.session_state.step_messages:
                    st.markdown(f'<div class="step-container">🔄 {msg}</div>', unsafe_allow_html=True)

def process_query(query):
    """Process the user query and show steps in real-time"""
    
    # Clear previous messages
    st.session_state.step_messages = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Capture print statements
    print_capture = PrintCapture()
    original_stdout = sys.stdout
    
    try:
        # Redirect stdout to capture print statements
        sys.stdout = print_capture
        
        status_text.text("🚀 Starting agent processing...")
        progress_bar.progress(10)
        
        # Process the query
        invoke_config, ls_handler, cache_holder = build_invoke_config(query)
        result = graph.invoke({"query": query}, config=invoke_config)
        ls_handler.patch_root_run_metadata()
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Store the captured messages
        st.session_state.step_messages = print_capture.messages
        
        # Display final result
        st.markdown("---")
        st.header("📋 Final Response")
        
        if result and "generation" in result:
            st.markdown(f'''
            <div class="result-container">
                <h4>🎯 Answer:</h4>
                <p>{result["generation"]}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Show additional details
            with st.expander("🔍 Process Details"):
                cache_ctx = cache_holder or cache_eval_metadata(
                    cache_hit=False, cache_similarity_score=None
                )
                st.json({
                    "Original Query": result.get("query", "N/A"),
                    "Routing Decision": result.get("query_redirect", "N/A"),
                    "Documents Retrieved": len(result.get("retrieved_docs", [])) if result.get("retrieved_docs") else 0,
                    "LangSmith metadata (cache)": cache_ctx,
                })
        else:
            st.error("❌ No response generated. Please try again.")
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        
        # Update the steps display
        step_placeholder = st.session_state.get('step_container')
        if step_placeholder and st.session_state.step_messages:
            with step_placeholder.container():
                for msg in st.session_state.step_messages:
                    st.markdown(f'<div class="step-container">🔄 {msg}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()