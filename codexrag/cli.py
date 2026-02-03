from __future__ import annotations
import argparse
import sys
from pathlib import Path
from codexrag.config import load_config
from codexrag.indexer import build_index
from codexrag.index_store import IndexStore
from codexrag.retriever import HybridRetriever
from codexrag.agents.orchestrator import Orchestrator
from codexrag.agents.safety_agent import SafetyAgent
from codexrag.llm_ollama import make_ollama

# Optional: Try importing rich for pretty output
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class Console:
        def print(self, *args, **kwargs): print(*args)
    console = Console()

def _print_markdown(text: str):
    if HAS_RICH:
        console.print(Markdown(text))
    else:
        print(text)

def cmd_index(args):
    """Run the indexing process."""
    if HAS_RICH:
        console.print(Panel("[bold blue]CodeXRAG Indexer[/bold blue]", subtitle="Building Knowledge Base"))
        with console.status("[bold green]Scanning and indexing...[/bold green]"):
            cfg = load_config(args.config)
            repo = Path(args.repo).resolve()
            
            # Setup Safety Agent to log indexing
            safety = SafetyAgent(repo / ".codexrag")
            
            store = build_index(cfg, repo)
            safety.log_index(str(repo), len(store.chunks), 0) # 0 chunks for now as build_index returns store
            
        console.print(f"[bold green]✅ Success![/bold green] Indexed repo at: {repo}")
        console.print(f"Total Chunks: {len(store.chunks)}")
    else:
        cfg = load_config(args.config)
        repo = Path(args.repo).resolve()
        build_index(cfg, repo)
        print(f"[OK] Indexed repo at {repo}")

def _load_system(args) -> Orchestrator:
    """Initialize the full RAG system."""
    cfg = load_config(args.config)
    repo = Path(args.repo).resolve()
    
    # 1. Load Index
    store = IndexStore(index_dir=(repo / cfg.index_dir).resolve(), embedding_model=cfg.embedding_model)
    store.load()
    
    if not store.chunks:
        if HAS_RICH:
            console.print("[bold red]Error:[/bold red] Index is empty.")
            console.print("Run: [yellow]codexrag index --repo .[/yellow]")
        else:
            print("Index is empty. Run index command first.")
        sys.exit(1)
        
    # 2. Setup Retriever
    retr = HybridRetriever(store, reranker_model=cfg.reranker_model)
    
    # 3. Setup LLM
    llm = make_ollama(model=cfg.ollama_model, base_url=cfg.ollama_base_url)
    
    # 4. Setup Safety Agent
    safety = SafetyAgent(repo / ".codexrag")
    
    # 5. Create Orchestrator
    orch = Orchestrator(llm, retr, safety_agent=safety)
    return orch, cfg

def cmd_ask(args):
    """Run a single query."""
    orchestrator, cfg = _load_system(args)
    
    if HAS_RICH:
        console.print(Panel(f"[bold]Question:[/bold] {args.question}", title="CodeXRAG Query"))
        with console.status("[bold blue]Thinking (Agents working)...[/bold blue]"):
            response = orchestrator.query(
                args.question,
                cfg.top_k_bm25,
                cfg.top_k_vector,
                cfg.top_k_rerank,
                cfg.max_context_chunks
            )
        console.print(Markdown(response))
    else:
        response = orchestrator.query(
            args.question,
            cfg.top_k_bm25,
            cfg.top_k_vector,
            cfg.top_k_rerank,
            cfg.max_context_chunks
        )
        print(response)

def cmd_chat(args):
    """Run interactive chat mode."""
    orchestrator, cfg = _load_system(args)
    
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]CodeXRAG Interactive Chat[/bold cyan]\n"
            "Type 'exit' to quit.",
            border_style="blue"
        ))
    else:
        print("CodeXRAG Chat. Type 'exit' to quit.\n")
        
    while True:
        try:
            if HAS_RICH:
                q = console.input("[bold green]>>> [/bold green]")
            else:
                q = input(">>> ")
                
            q = q.strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
                
            if HAS_RICH:
                with console.status("[bold blue]Agents analyzing...[/bold blue]"):
                    ans = orchestrator.query(
                        q,
                        cfg.top_k_bm25,
                        cfg.top_k_vector,
                        cfg.top_k_rerank,
                        cfg.max_context_chunks
                    )
                console.print(Markdown(ans))
                console.print("---")
            else:
                ans = orchestrator.query(
                    q,
                    cfg.top_k_bm25,
                    cfg.top_k_vector,
                    cfg.top_k_rerank,
                    cfg.max_context_chunks
                )
                print("\n" + ans + "\n")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            if HAS_RICH:
                console.print(f"[bold red]Error:[/bold red] {e}")
            else:
                print(f"Error: {e}")

def cmd_audit(args):
    """View safety audit log."""
    repo = Path(args.repo).resolve()
    safety = SafetyAgent(repo / ".codexrag")
    report = safety.generate_safety_report()
    print(report)

def main():
    p = argparse.ArgumentParser(prog="codexrag", description="State-of-the-Art Multi-Agent RAG.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Index Command
    p_index = sub.add_parser("index", help="Index a repo.")
    p_index.add_argument("--repo", default=".")
    p_index.add_argument("--config", default="config.yaml")
    p_index.set_defaults(func=cmd_index)

    # Ask Command
    p_ask = sub.add_parser("ask", help="Ask a single question.")
    p_ask.add_argument("--repo", default=".")
    p_ask.add_argument("--config", default="config.yaml")
    p_ask.add_argument("--question", required=True)
    p_ask.set_defaults(func=cmd_ask)

    # Chat Command
    p_chat = sub.add_parser("chat", help="Interactive chat.")
    p_chat.add_argument("--repo", default=".")
    p_chat.add_argument("--config", default="config.yaml")
    p_chat.set_defaults(func=cmd_chat)
    
    # Audit Command
    p_audit = sub.add_parser("audit", help="View safety audit log.")
    p_audit.add_argument("--repo", default=".")
    p_audit.set_defaults(func=cmd_audit)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
