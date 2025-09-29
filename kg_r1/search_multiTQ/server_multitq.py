"""
FastAPI server for MultiTQ Knowledge Graph Retrieval.

Handles temporal KGQA with split-wise knowledge graphs.
"""

import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Optional, Dict, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from .actions_multitq import (
    ActionType, SearchRequest, ACTION_REGISTRY_MULTITQ,
    kg_retrieval_completion_response
)
from .knowledge_graph_multitq import KnowledgeGraphMultiTQ

logger = logging.getLogger(__name__)


class KnowledgeGraphRetrieverMultiTQ:
    """
    Knowledge Graph retriever for MultiTQ temporal KGQA dataset.
    Uses split-wise temporal KGs instead of per-sample subgraphs.
    """
    
    def __init__(self, data_path: str, splits: List[str] = None):
        """
        Initialize MultiTQ KG retriever.
        
        Args:
            data_path: Path to MultiTQ data directory
            splits: List of splits to support (default: ["train", "valid", "test"])
        """
        self.data_path = data_path
        self.splits = splits or ["train", "valid", "test"]
        self.knowledge_graphs = {}
        self.action_handlers = ACTION_REGISTRY_MULTITQ
        
        # Initialize knowledge graphs for each split
        for split in self.splits:
            try:
                self.knowledge_graphs[split] = KnowledgeGraphMultiTQ(data_path, split)
                logger.info(f"Initialized MultiTQ KG for split: {split}")
            except Exception as e:
                logger.error(f"Failed to initialize KG for split {split}: {e}")
        
        logger.info(f"MultiTQ Retriever initialized with data_path: {data_path}")
        logger.info(f"Available splits: {list(self.knowledge_graphs.keys())}")
        logger.info(f"Available actions: {list(self.action_handlers.keys())}")
    
    def get_kg_for_sample(self, sample_id: str) -> Optional[KnowledgeGraphMultiTQ]:
        """
        Determine which KG to use based on sample ID.
        MultiTQ sample IDs now contain the split prefix:
        - train_xxxxxxx = train
        - dev_xxxxxxx = dev/valid
        - test_xxxxxxx = test
        """
        if not sample_id:
            return self.knowledge_graphs.get("train")  # Default
            
        # Convert to string and check prefix
        sample_id_str = str(sample_id)
        
        if sample_id_str.startswith('train_'):
            return self.knowledge_graphs.get("train")
        elif sample_id_str.startswith('dev_'):
            # dev/valid split
            if "valid" in self.knowledge_graphs:
                return self.knowledge_graphs.get("valid")
            elif "dev" in self.knowledge_graphs:
                return self.knowledge_graphs.get("dev")
            else:
                logger.warning(f"No dev/valid split found for sample {sample_id}, using train")
                return self.knowledge_graphs.get("train")
        elif sample_id_str.startswith('test_'):
            return self.knowledge_graphs.get("test")
        else:
            logger.warning(f"Unknown sample ID pattern: {sample_id}, using train split")
            return self.knowledge_graphs.get("train")
    
    def execute_action(self, request: SearchRequest, split: str = None) -> Any:
        """Execute an action on the MultiTQ KG."""
        
        # Determine which split to use
        if split and split in self.knowledge_graphs:
            kg = self.knowledge_graphs[split]
        elif request.sample_id:
            # Try to infer split from sample_id if possible
            kg = self.get_kg_for_sample(request.sample_id)
            if not kg:
                kg = self.knowledge_graphs.get("train")  # Default
        else:
            kg = self.knowledge_graphs.get("train")  # Default
        
        if not kg:
            error_content = f"No KG available for split: {split}"
            return kg_retrieval_completion_response(
                error_content, "server_error",
                is_error=True, error_type="KG_SERVER_ERROR"
            )
        
        # Check if action is available
        if request.action_type not in self.action_handlers:
            available_actions = [action.value for action in self.action_handlers.keys()]
            error_content = f'Action "{request.action_type}" not available (use: {", ".join(available_actions)})'
            return kg_retrieval_completion_response(
                error_content, "server_error",
                is_error=True, error_type="KG_SERVER_ERROR"
            )
        
        # Create handler and execute
        handler_class = self.action_handlers[request.action_type]
        handler = handler_class(kg)
        
        # Prepare kwargs based on action type
        kwargs = {}
        if request.entity_id:
            kwargs['entity_id'] = request.entity_id
        if request.relation:
            kwargs['relation'] = request.relation
        if request.concept:
            kwargs['concept'] = request.concept
        if request.timestamp:
            kwargs['timestamp'] = request.timestamp
        
        return handler.execute(
            sample_id=request.sample_id,
            **kwargs
        )


# Create FastAPI app
app = FastAPI(title="MultiTQ KG Retrieval Server", version="1.0.0")

# Global retriever instance
retriever = None
executor = ThreadPoolExecutor(max_workers=4)


@app.on_event("startup")
async def startup_event():
    """Initialize retriever on startup."""
    global retriever
    import os
    
    # Get data path from environment or use default
    data_path = os.environ.get(
        "MULTITQ_DATA_PATH",
        "/nobackup/users/yeopjin/workspace/RL_KG/data_multitq_kg/MultiTQ"
    )
    
    retriever = KnowledgeGraphRetrieverMultiTQ(data_path)
    logger.info("MultiTQ KG Retrieval Server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    executor.shutdown(wait=True)
    logger.info("MultiTQ KG Retrieval Server shutdown")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "MultiTQ KG Retrieval Server",
        "status": "running",
        "available_splits": list(retriever.knowledge_graphs.keys()) if retriever else [],
        "available_actions": [action.value for action in ActionType]
    }


@app.post("/retrieve")
async def retrieve(request: Union[SearchRequest, List[Dict], Dict]):
    """Main retrieval endpoint for MultiTQ - handles both single and batch requests."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    # Handle both single request and batch request formats
    requests = []
    is_batch = False
    
    if isinstance(request, list):
        # Batch request from client
        is_batch = True
        for req_dict in request:
            try:
                requests.append(SearchRequest(**req_dict))
            except Exception as e:
                logger.error(f"Error parsing batch request item: {e}, request: {req_dict}")
                # Return error response for this item
                requests.append(None)
    elif isinstance(request, dict):
        # Single request as dict
        try:
            requests = [SearchRequest(**request)]
        except Exception as e:
            logger.error(f"Error parsing single request dict: {e}")
            raise HTTPException(status_code=422, detail=f"Invalid request format: {str(e)}")
    elif isinstance(request, SearchRequest):
        # Already parsed by FastAPI
        requests = [request]
    else:
        raise HTTPException(status_code=422, detail="Invalid request format")
    
    results = []
    for i, req in enumerate(requests):
        if req is None:
            # Error parsing this request
            results.append(kg_retrieval_completion_response(
                "Invalid request format",
                "server_error",
                is_error=True,
                error_type="KG_FORMAT_ERROR"
            ))
            continue
            
        try:
            # Determine split from dataset_name or sample_id
            split = None
            if req.dataset_name in ["train", "valid", "test", "dev"]:
                split = req.dataset_name
            elif req.dataset_name == "multitq" and req.sample_id:
                # Infer split from sample_id prefix
                sample_id_str = str(req.sample_id)
                if sample_id_str.startswith('train_'):
                    split = "train"
                elif sample_id_str.startswith('dev_'):
                    split = "valid"  # or "dev" depending on what's loaded
                elif sample_id_str.startswith('test_'):
                    split = "test"
                else:
                    split = "train"  # Default
            else:
                split = "train"  # Default
            
            # Run retrieval in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                retriever.execute_action,
                req,
                split
            )
            results.append(result)
        
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            results.append(kg_retrieval_completion_response(
                f"Internal server error: {str(e)}",
                "server_error",
                is_error=True,
                error_type="KG_SERVER_ERROR"
            ))
    
    # Return list for batch, single result for single request
    if is_batch:
        return results
    elif len(results) == 1:
        return results[0]
    else:
        return results


@app.get("/actions")
async def get_supported_actions():
    """Get list of supported action types."""
    return {
        "actions": [action.value for action in ActionType]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    return {
        "status": "healthy",
        "splits_loaded": list(retriever.knowledge_graphs.keys()),
        "timestamp": time.time()
    }


def run_server(host: str = "127.0.0.1", port: int = 8001):
    """Run the MultiTQ KG server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    run_server()