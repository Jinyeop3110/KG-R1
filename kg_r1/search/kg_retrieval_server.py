"""
Entry point for the Knowledge Graph Retrieval Server.

This is the main entry point that initializes and runs the modular KG retrieval server.

Environment Variables:
    KG_RELATION_FORMAT: Relation output format (flat, full_indent, mixed, compact)
                       Default: full_indent (provides hierarchical structure)
    KG_BASE_DATA_PATH: Base path to RoG processed data directory
    KG_SUPPORTED_KGS: Comma-separated list of supported KGs (webqsp,cwq)
    KG_ENABLED_ACTIONS: Comma-separated list of enabled actions
    KG_USE_ENTITIES_TEXT: Use entities_text.txt instead of entities.txt
    KG_THREAD_WORKERS: Number of ThreadPoolExecutor workers
"""

import argparse
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the KG Retrieval Server."""
    parser = argparse.ArgumentParser(description="Knowledge Graph Retrieval Server for RoG Subgraphs")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--thread_workers", type=int, default=8, help="Number of ThreadPoolExecutor workers for concurrent request processing")
    parser.add_argument("--base_data_path", type=str, required=True, 
                       help="Base path to the RoG processed data directory (e.g., ./data_kg/) which contains webqsp/subgraphs and cwq/subgraphs.")
    parser.add_argument("--kgs", type=str, nargs="+", default=["webqsp", "cwq"],
                       help="Knowledge graphs to support (default: webqsp cwq)")
    parser.add_argument("--actions", type=str, nargs="+", 
                       help="Actions to enable (default: all available). Options: get_relations (deprecated), get_head_relations, get_tail_relations, get_head_entities, get_tail_entities")
    parser.add_argument("--use_entities_text", action="store_true",
                       help="Use entities_text.txt instead of entities.txt for entity loading")
    parser.add_argument("--relation_format", type=str, choices=["flat", "full_indent", "mixed", "compact"], 
                       default="full_indent",
                       help="Relation output format (default: full_indent for hierarchical structure)")
    
    args = parser.parse_args()
    
    if not args.base_data_path or not os.path.isdir(args.base_data_path):
        logger.error(f"Error: --base_data_path '{args.base_data_path}' is not a valid directory.")
        return
    
    # Validate KGs exist in the data path
    for kg in args.kgs:
        kg_path = os.path.join(args.base_data_path, kg)
        if not os.path.isdir(kg_path):
            logger.warning(f"Warning: KG directory not found: {kg_path}")
    
    # Set relation format environment variable
    os.environ['KG_RELATION_FORMAT'] = args.relation_format
    logger.info(f"Relation format set to: {args.relation_format}")
    
    # Import and run the server
    import sys
    # Add the current directory to Python path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from .server import run_server
    
    try:
        run_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            thread_workers_param=args.thread_workers,
            base_data_path=args.base_data_path,
            supported_kgs=args.kgs,
            enabled_actions=args.actions,
            use_entities_text=args.use_entities_text
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


if __name__ == "__main__":
    main()
