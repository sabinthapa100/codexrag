"""
CodeXRAG Web GUI
================
A Streamlit-based web interface for CodeXRAG with native LaTeX support.

Key Features:
- Chat Interface (like ChatGPT)
- Code & Latex Rendering
- Safety Audit View
- Source Citation Viewer

Author: Sabin Thapa (sthapa3@kent.edu)
"""


import streamlit as st
import time
from pathlib import Path
from codexrag.indexer import IndexStore
from codexrag.retriever import HybridRetriever
from codexrag.agents.orchestrator import Orchestrator
from codexrag.agents.safety_agent import SafetyAgent
from codexrag.llm_ollama import make_ollama
from codexrag.config import load_config
from codexrag.utils_warnings import suppress_common_warnings

# Suppress harmless warnings
suppress_common_warnings()

# Page Configuration
st.set_page_config(
    page_title="CodeXRAG: Scientific Assistant",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for "Scientific" Look
st.markdown("""
<style>
    .stChatMessage {
        font-family: 'Source Code Pro', monospace;
    }
    .citation-box {
        border-left: 3px solid #00d462;
        padding-left: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin-top: 5px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    """Load the RAG system (cached for performance)."""
    # Heuristic: Find where the .codexrag index actually lives
    cwd = Path(".").resolve()
    # Check parent FIRST (common for nested usage), then current
    candidates = [cwd.parent, cwd]
    
    repo = cwd
    for cand in candidates:
        # Check for actual index directory (avoid empty log folders)
        if (cand / ".codexrag" / "index").exists():
            repo = cand
            break

    # Try multiple config locations for flexibility
    config_candidates = [
        repo / "rag4mycodex" / "config.yaml",    # Priority: Project specific
        repo / "config.yaml",                    # Fallback: Dist default
        cwd / "config.yaml",                     # Local fallback
    ]
    
    config_path = None
    for candidate in config_candidates:
        if candidate.exists():
            config_path = candidate
            break
    
    if config_path is None:
        st.error("❌ config.yaml not found!")
        st.info("Expected locations: " + ", ".join(str(c) for c in config_candidates))
        return None, None
        
    cfg = load_config(str(config_path))
    
    # Check index
    index_path = repo / cfg.index_dir
    if not index_path.exists():
        return None, None
        
    store = IndexStore(index_dir=index_path, embedding_model=cfg.embedding_model)
    store.load()
    
    base_retr = HybridRetriever(store, reranker_model=cfg.reranker_model)
    
    # Try to load Graph
    try:
        from codexrag.graph import CodeGraph, GraphEnhancedRetriever
        graph_path = (repo / cfg.index_dir / "graph.json")
        if graph_path.exists():
            graph = CodeGraph()
            graph.load(graph_path)
            retr = GraphEnhancedRetriever(base_retr, graph)
        else:
            retr = base_retr
    except ImportError:
        retr = base_retr
    llm = make_ollama(model=cfg.ollama_model, base_url=cfg.ollama_base_url)
    safety = SafetyAgent(repo / ".codexrag")
    
    orch = Orchestrator(llm, retr, safety_agent=safety)
    return orch, cfg

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.shields.io/badge/CodeXRAG-v2.0-blue?style=for-the-badge", width="stretch")
    st.markdown("### ⚛️ Verified Research Assistant")
    st.markdown("---")
    
    st.subheader("System Status")
    orch, cfg = load_rag_system()
    
    if orch:
        st.success("✅ RAG Index Loaded")
        # Handle wrapped retriever (GraphEnhancedRetriever)
        if hasattr(orch.retriever, "base"):
            chunk_count = len(orch.retriever.base.store.chunks)
        else:
            chunk_count = len(orch.retriever.store.chunks)
        st.info(f"📚 Knowledge Base: {chunk_count} chunks")
        st.info(f"🧠 Model: `{cfg.ollama_model}`")
    else:
        st.error("❌ Index Not Found")
        st.warning("Run `codexrag index` first!")
        st.stop()
        
    st.markdown("---")
    if st.button("🛡️ View Safety Audit"):
        report = orch.safety.generate_safety_report()
        st.text_area("Audit Log", report, height=300)

# --- Main Interface ---

st.title("Code Research Assistant")
st.caption("Ask about code, physics equations, or experimental data. I use **Hessian Uncertainty**.")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("How do we calculate RpA?"):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Assistant Message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyzing codebase..."):
            # Get response from Orchestrator
            response = orch.query(
                prompt,
                cfg.top_k_bm25,
                cfg.top_k_vector,
                cfg.top_k_rerank,
                cfg.max_context_chunks
            )
            
            # Stream the response (simulate typing)
            for chunk in response.split(" "):
                full_response += chunk + " "
                time.sleep(0.01)
                message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
