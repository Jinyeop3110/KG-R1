"""
KG Search module for modular knowledge graph retrieval.
"""

from .actions import ActionType, ActionHandler, ACTION_REGISTRY, SearchRequest

try:
    from .server import KnowledgeGraphRetriever, app
    __all__ = [
        "ActionType", 
        "ActionHandler", 
        "ACTION_REGISTRY",
        "SearchRequest",
        "KnowledgeGraphRetriever",
        "app"
    ]
except ImportError:
    # FastAPI not available
    __all__ = [
        "ActionType", 
        "ActionHandler", 
        "ACTION_REGISTRY",
        "SearchRequest"
    ]
