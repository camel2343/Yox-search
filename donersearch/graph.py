import sqlite3
import time
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from . import db as dbmod
from . import crawler

def calculate_pagerank(
    db_path: str,
    damping: float = 0.85,
    epsilon: float = 1.0e-5,
    max_iterations: int = 100
) -> int:
    """Calculates PageRank score for all documents in the database.
    
    Updates the 'pagerank' column in the documents table.
    Returns the number of documents updated.
    """
    conn = dbmod.open_db(db_path)
    dbmod.ensure_schema(conn)
    try:
        # 1. Load the graph
        links = dbmod.get_all_links(conn)
        
        # Build adjacency list: source -> [targets]
        # and inverse: target -> [sources]
        outgoing: Dict[int, List[int]] = {}
        incoming: Dict[int, List[int]] = {}
        
        all_nodes = set()
        
        for src, dst in links:
            outgoing.setdefault(src, []).append(dst)
            incoming.setdefault(dst, []).append(src)
            all_nodes.add(src)
            all_nodes.add(dst)
            
        N = len(all_nodes)
        if N == 0:
            print("[pagerank] No nodes in graph.")
            return 0
            
        # Initialize PR
        initial_pr = 1.0 / N
        pagerank = {node: initial_pr for node in all_nodes}
        
        # Power iteration
        for i in range(max_iterations):
            new_pagerank = {}
            diff = 0.0
            
            # Sink mass: nodes with no outgoing links distributed to everyone
            sink_pr = 0.0
            for node in all_nodes:
                if node not in outgoing:
                    sink_pr += pagerank[node]
            
            for node in all_nodes:
                # Sum of PR from incoming nodes
                incoming_sum = 0.0
                if node in incoming:
                    for src in incoming[node]:
                        # src distributes its PR equally among its targets
                        num_out = len(outgoing[src])
                        incoming_sum += pagerank[src] / num_out
                
                # Formula: (1-d)/N + d * (incoming_sum + sink_mass/N)
                pr = (1.0 - damping) / N + damping * (incoming_sum + sink_pr / N)
                new_pagerank[node] = pr
                diff += abs(pr - pagerank[node])
            
            pagerank = new_pagerank
            print(f"[pagerank] Iteration {i+1}: diff={diff:.6f}")
            if diff < epsilon:
                break
        
        # Normalize? Standard PR sums to 1.
        # But we might want scores roughly like "1.0 is average" for easier boosting math.
        # Standard PR: avg score is 1/N. 
        # Let's scale so that Average PR = 1.0. 
        # This makes it interchangeable with our default database value of 1.0.
        scale = N
        scaled_pr = {k: v * scale for k, v in pagerank.items()}
        
        print(f"[pagerank] Saving scores for {N} nodes...")
        dbmod.update_pagerank_scores(conn, scaled_pr)
        return N
        
    finally:
        conn.close()


def rebuild_link_graph(db_path: str, threads: int = 4) -> None:
    """Iterates over existing documents, re-fetches HTML, extracts links, and populates 'links' table.
    
    This is a 'repair' function to populate the graph from an existing index 
    that didn't save links initially.
    """
    conn = dbmod.open_db(db_path)
    dbmod.ensure_schema(conn)
    
    # Get all URLs
    cur = conn.execute("SELECT id, url FROM documents")
    rows = cur.fetchall()
    conn.close()
    
    total = len(rows)
    print(f"[graph] Found {total} documents to scan for links.")
    
    # We can reuse crawler's fetch logic but simpler: we just want links.
    # We can use a thread pool.
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Split into chunks for easier progress tracking
    chunk_size = 50
    chunks = [rows[i:i + chunk_size] for i in range(0, total, chunk_size)]
    
    processed = 0
    
    # We open a NEW connection per thread or share one properly? 
    # SQLite is thread-safe in serialized mode, but best practice is one conn per thread.
    # Actually, simpler: fetch in threads, write in main thread.
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for chunk in chunks:
            futures = []
            for doc_id, url in chunk:
                futures.append(executor.submit(_fetch_and_extract_links, url))
                
            results = []
            for f in as_completed(futures):
                try:
                    url, links = f.result()
                    results.append((url, links))
                except Exception as e:
                    # Ignore fetch errors (404s etc are expected on old indexes)
                    pass
            
            # Batch write
            write_conn = dbmod.open_db(db_path)
            for url, links in results:
                # We need the doc_id. We have it in 'chunk' but results are out of order.
                # Actually _fetch_and_extract_links returns url. We can lookup id again or pass it through.
                # Let's pass doc_id through.
                pass
            
            # Map url -> doc_id
            # Optimization: just reuse the ids we had in 'chunk'
            # But the results are (url, links).
            # Let's refactor _fetch_and_extract_links to accept and return doc_id.
            
            # Retry with proper mapping
            pass
            write_conn.close()
            processed += len(chunk)
            print(f"[graph] Processed {processed}/{total}...")


def _fetch_and_extract_links_wrapper(args):
    doc_id, url = args
    return _fetch_links_worker(doc_id, url)

def _fetch_links_worker(doc_id: int, url: str) -> Tuple[int, List[str]]:
    # Use crawler's fetch mechanisms
    # We just need raw html -> extract_text_and_links
    
    from .crawler import _fetch_html, _read_text, extract_text_and_links, USER_AGENT
    
    # Short timeout, we want speed
    raw = _fetch_html(url, timeout=5.0, user_agent=USER_AGENT)
    if not raw:
        return doc_id, []
        
    html = _read_text(raw)
    if not html:
        return doc_id, []
        
    try:
        _, _, links, _, _ = extract_text_and_links(html, url)
        return doc_id, links
    except Exception:
        return doc_id, []


def rebuild_link_graph_threaded(db_path: str, workers: int = 8) -> None:
    conn = dbmod.open_db(db_path)
    dbmod.ensure_schema(conn)
    cur = conn.execute("SELECT id, url FROM documents")
    rows = cur.fetchall()
    conn.close() # Close mainly read connection
    
    total = len(rows)
    print(f"[graph] Starting link recovery for {total} docs with {workers} threads...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # We'll accumulate results and write in batches to avoid locking DB too much
    batch_size = 50
    batch_results = []
    
    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Submit all tasks? If total is huge (100k), this consumes memory.
        # Better: submit in batches.
        
        # But for simplicity in this script, let's just iterative chunk submit
        
        for i in range(0, total, batch_size):
            chunk = rows[i:i+batch_size]
            futures = {ex.submit(_fetch_links_wrapper, (r[0], r[1])): r[0] for r in chunk}
            
            results_map = {}
            for fut in as_completed(futures):
                try:
                    did, links = fut.result()
                    results_map[did] = links
                except Exception:
                    pass
            
            # Write batch
            w_conn = dbmod.open_db(db_path)
            for doc_id in results_map:
                links = results_map[doc_id]
                if links:
                    dbmod.save_links(w_conn, doc_id, links)
            w_conn.commit()
            w_conn.close()
            
            print(f"[graph] Progress: {min(i+batch_size, total)}/{total}")

