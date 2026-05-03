import sys
import argparse
import os
from pathlib import Path

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from donersearch import web, graph

def main():
    parser = argparse.ArgumentParser(description="DonerSearch Management")
    subparsers = parser.add_subparsers(dest="command")

    # Serve
    p_serve = subparsers.add_parser("serve", help="Run web server")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--db", default="doner.db")

    # Graph
    p_graph = subparsers.add_parser("rebuild-graph", help="Rebuild link graph from existing docs (re-fetch html)")
    p_graph.add_argument("--db", default="doner.db")
    p_graph.add_argument("--workers", type=int, default=8)

    # PageRank
    p_rank = subparsers.add_parser("calc-pagerank", help="Calculate PageRank scores")
    p_rank.add_argument("--db", default="doner.db")
    p_rank.add_argument("--iterations", type=int, default=20)

    args = parser.parse_args()

    if args.command == "serve":
        web.serve(args.db, host=args.host, port=args.port)
    elif args.command == "rebuild-graph":
        print(f"Rebuilding link graph using {args.workers} workers... This may take a while as it re-fetches content.")
        graph.rebuild_link_graph_threaded(args.db, max_workers=args.workers)
        print("Link graph rebuild complete.")
        # Auto-calc after rebuild
        print("Calculating PageRank...")
        graph.calculate_pagerank(args.db)
        print("Done.")
    elif args.command == "calc-pagerank":
        print(f"Calculating PageRank ({args.iterations} iterations)...")
        graph.calculate_pagerank(args.db, max_iterations=args.iterations)
        print("Done.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
