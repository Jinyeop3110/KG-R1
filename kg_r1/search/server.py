"""
FastAPI server for Knowledge Graph Retrieval.

This module contains the FastAPI application and the KnowledgeGraphRetriever class.
"""

import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Optional
from fastapi import FastAPI, HTTPException
import uvicorn

from .actions import ActionType, SearchRequest, ACTION_REGISTRY, get_filtered_action_registry, kg_retrieval_completion_response
from .knowledge_graph import KnowledgeGraph
from .error_types import KGErrorType

logger = logging.getLogger(__name__)


class KnowledgeGraphRetriever:
    """
    Knowledge Graph retriever for WebQuestionsSP and ComplexWebQuestions using RoG per-sample subgraphs.
    """
    
    def __init__(self, base_data_path: str, supported_kgs: List[str] = None, enabled_actions: List[str] = None, use_entities_text: bool = False):
        self.base_data_path = base_data_path
        self.supported_kgs = supported_kgs or ["webqsp", "CWQ"]
        self.enabled_actions = enabled_actions
        self.use_entities_text = use_entities_text
        self.action_handlers = {}
        self.knowledge_graphs = {}
        
        # Initialize knowledge graphs
        for kg_name in self.supported_kgs:
            kg_path = f"{base_data_path}/{kg_name}"
            self.knowledge_graphs[kg_name] = KnowledgeGraph(kg_name, kg_path, use_entities_text=use_entities_text)
            logger.info(f"Initialized KG: {kg_name} with {len(self.knowledge_graphs[kg_name].subgraphs)} subgraphs (entities_text: {use_entities_text})")
        
        # Get filtered action registry based on enabled actions
        action_registry = get_filtered_action_registry(enabled_actions)
        
        # Initialize action handlers with knowledge graphs
        for action_type, handler_class in action_registry.items():
            # Each action handler will receive the appropriate KG when executing
            self.action_handlers[action_type] = handler_class
        
        logger.info(f"Retriever initialized with base_data_path: {self.base_data_path}")
        logger.info(f"Supported KGs: {self.supported_kgs}")
        logger.info(f"Available actions: {list(self.action_handlers.keys())}")

    def is_kg_supported(self, kg_name: str) -> bool:
        """Check if a KG is supported."""
        # Handle case-insensitive matching for cwq/CWQ
        normalized_kg_name = "CWQ" if kg_name.lower() == "cwq" else kg_name
        return normalized_kg_name in self.supported_kgs

    def execute_action(self, request: SearchRequest, dataset_name: str) -> Any:
        """Execute an action based on the request."""
        # Check if KG is supported
        if not self.is_kg_supported(dataset_name):
            from .error_types import KGErrorType
            error_content = f'Dataset "{dataset_name}" not supported (available: {", ".join(self.supported_kgs)})'
            return kg_retrieval_completion_response(
                error_content, "server_error", 
                is_error=True, error_type=KGErrorType.SERVER_ERROR
            )
        
        # Check if action is available
        if request.action_type not in self.action_handlers:
            from .error_types import KGErrorType
            available_actions = [action.value for action in self.action_handlers.keys()]
            error_content = f'Action "{request.action_type}" not available (use: {", ".join(available_actions)})'
            return kg_retrieval_completion_response(
                error_content, "server_error", 
                is_error=True, error_type=KGErrorType.SERVER_ERROR
            )
        
        handler_class = self.action_handlers[request.action_type]
        # Normalize dataset name for KG lookup (handle cwq -> CWQ)
        normalized_dataset_name = "CWQ" if dataset_name.lower() == "cwq" else dataset_name
        kg = self.knowledge_graphs[normalized_dataset_name]
        handler = handler_class(kg)
        
        # Prepare kwargs based on action type
        kwargs = {}
        if request.entity_id:
            kwargs['entity_id'] = request.entity_id
        if request.relation:
            kwargs['relation'] = request.relation
        if request.concept:
            kwargs['concept'] = request.concept
            
        return handler.execute(
            sample_id=request.sample_id,
            **kwargs
        )


# Global retriever instance
retriever = None

# Global configuration
thread_workers = 8  # Default ThreadPoolExecutor workers

# FastAPI app
app = FastAPI(title="Knowledge Graph Retrieval Server", version="2.0.0")


def initialize_from_env():
    """Initialize server from environment variables if available."""
    global retriever, thread_workers
    import os
    
    if retriever is None and 'KG_BASE_DATA_PATH' in os.environ:
        base_data_path = os.environ.get('KG_BASE_DATA_PATH')
        supported_kgs = os.environ.get('KG_SUPPORTED_KGS', 'webqsp,cwq').split(',')
        enabled_actions = os.environ.get('KG_ENABLED_ACTIONS', '').split(',') if os.environ.get('KG_ENABLED_ACTIONS') else None
        use_entities_text = os.environ.get('KG_USE_ENTITIES_TEXT', 'False').lower() == 'true'
        thread_workers = int(os.environ.get('KG_THREAD_WORKERS', '8'))
        
        logger.info(f"Initializing server from environment variables...")
        initialize_server(base_data_path, supported_kgs, enabled_actions, use_entities_text)


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    initialize_from_env()


def get_retriever() -> KnowledgeGraphRetriever:
    """Get the global retriever instance."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever service not available.")
    return retriever


def _validate_request_fields(request: SearchRequest) -> Optional[str]:
    """Validate required fields for a request based on action type.
    
    Returns:
        Error message if validation fails, None if validation passes.
    """
    # Define required fields for each action type
    required_fields = {
        ActionType.GET_RELATIONS: ["sample_id", "dataset_name", "entity_id"],  # Deprecated
        ActionType.GET_HEAD_RELATIONS: ["sample_id", "dataset_name", "entity_id"],
        ActionType.GET_TAIL_RELATIONS: ["sample_id", "dataset_name", "entity_id"],
        ActionType.GET_HEAD_ENTITIES: ["sample_id", "dataset_name", "entity_id", "relation"],
        ActionType.GET_TAIL_ENTITIES: ["sample_id", "dataset_name", "entity_id", "relation"],
        ActionType.GET_CONDITIONAL_RELATIONS: ["sample_id", "dataset_name", "entity_id", "concept"]
    }
    
    # Get required fields for this action type
    fields = required_fields.get(request.action_type, [])
    
    # Check each required field
    missing_fields = []
    for field in fields:
        value = getattr(request, field, None)
        if not value:
            # Create human-readable field descriptions
            field_descriptions = {
                "entity_id": "entity_id (tail entity)" if request.action_type == ActionType.GET_HEAD_ENTITIES 
                           else "entity_id (head entity)" if request.action_type == ActionType.GET_TAIL_ENTITIES 
                           else "entity_id",
                "relation": "relation_name",
                "concept": "concept (e.g., 'people', 'location', 'organization')"
            }
            field_name = field_descriptions.get(field, field)
            missing_fields.append(field_name)
    
    if missing_fields:
        action_name = request.action_type.value
        return f"Missing required fields for {action_name}: {', '.join(missing_fields)}."
    
    return None


@app.post("/retrieve")
async def retrieve(requests_list: List[SearchRequest]):
    """Main retrieval endpoint that handles a list of search requests."""
    current_retriever = get_retriever()
    
    # Debug: Log incoming requests
    print(f"[KG_SERVER_DEBUG] Received {len(requests_list)} requests")
    for i, req in enumerate(requests_list[:3]):  # Show first 3 requests
        print(f"[KG_SERVER_DEBUG] Request {i}: action_type={req.action_type}, dataset_name={req.dataset_name}, sample_id={req.sample_id}, entity_id={req.entity_id}, relation={req.relation}")
    
    # Check if retriever has any enabled actions
    if not current_retriever.action_handlers:
        # Get available actions dynamically
        from actions import ACTION_REGISTRY
        from .error_types import KGErrorType
        all_available_actions = [action.value for action in ACTION_REGISTRY.keys()]
        error_content = f"No actions enabled on server (available actions: {', '.join(all_available_actions)})"
        error_response = kg_retrieval_completion_response(
            error_content, 
            "system_error", 
            is_error=True,
            error_type=KGErrorType.SERVER_ERROR
        )
        return [error_response]
    
    def process_single_request(request_item):
        """Process a single request (for concurrent execution)."""
        start_time = time.time()
        
        # Validate required fields
        validation_error = _validate_request_fields(request_item)
        if validation_error:
            from .error_types import KGErrorType
            print(f"[KG_SERVER_DEBUG] Validation failed for request: {validation_error}")
            error_content = validation_error
            response = kg_retrieval_completion_response(
                error_content, request_item.action_type, 
                is_error=True, error_type=KGErrorType.FORMAT_ERROR
            )
            return response
        
        # Execute the action - this now returns an OpenAI-style response
        action_results = current_retriever.execute_action(request_item, request_item.dataset_name)
        
        # Add timing information to the response
        end_time = time.time()
        query_time = end_time - start_time
        action_results["query_time"] = query_time
        
        return action_results

    # Process requests concurrently for better latency
    with ThreadPoolExecutor(max_workers=min(len(requests_list), thread_workers)) as executor:
        all_responses = list(executor.map(process_single_request, requests_list))
        
    return all_responses


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "kg_retrieval"}


@app.get("/actions")
def get_supported_actions():
    """Get list of supported action types."""
    current_retriever = get_retriever()
    available_actions = list(current_retriever.action_handlers.keys())
    
    action_details = {}
    for action_type in available_actions:
        if action_type == ActionType.GET_RELATIONS:
            action_details[action_type.value] = {
                "description": "Get all relations for a given entity (DEPRECATED - use get_head_relations or get_tail_relations)",
                "required_fields": ["sample_id", "dataset_name", "entity_id"],
                "deprecated": True
            }
        elif action_type == ActionType.GET_HEAD_RELATIONS:
            action_details[action_type.value] = {
                "description": "Get relations where the entity is the tail (object)",
                "required_fields": ["sample_id", "dataset_name", "entity_id"]
            }
        elif action_type == ActionType.GET_TAIL_RELATIONS:
            action_details[action_type.value] = {
                "description": "Get relations where the entity is the head (subject)",
                "required_fields": ["sample_id", "dataset_name", "entity_id"]
            }
        elif action_type == ActionType.GET_HEAD_ENTITIES:
            action_details[action_type.value] = {
                "description": "Get head entities for a given tail entity and relation",
                "required_fields": ["sample_id", "dataset_name", "entity_id", "relation"]
            }
        elif action_type == ActionType.GET_TAIL_ENTITIES:
            action_details[action_type.value] = {
                "description": "Get tail entities for a given head entity and relation",
                "required_fields": ["sample_id", "dataset_name", "entity_id", "relation"]
            }
        elif action_type == ActionType.GET_CONDITIONAL_RELATIONS:
            action_details[action_type.value] = {
                "description": "Get relations for a given entity filtered by concept (e.g., 'people', 'location', 'organization')",
                "required_fields": ["sample_id", "dataset_name", "entity_id", "concept"]
            }
    
    return {
        "actions": [action.value for action in available_actions],  # Simple list for LLM client
        "supported_actions": [action.value for action in available_actions],
        "supported_kgs": current_retriever.supported_kgs,
        "action_details": action_details
    }


def initialize_server(base_data_path: str, supported_kgs: List[str] = None, enabled_actions: List[str] = None, use_entities_text: bool = False):
    """Initialize the global retriever instance."""
    global retriever
    retriever = KnowledgeGraphRetriever(
        base_data_path=base_data_path,
        supported_kgs=supported_kgs,
        enabled_actions=enabled_actions,
        use_entities_text=use_entities_text
    )
    logger.info("KG Retrieval Server initialized successfully")


def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1, thread_workers_param: int = 8, base_data_path: str = None, 
               supported_kgs: List[str] = None, enabled_actions: List[str] = None, use_entities_text: bool = False):
    """Run the FastAPI server."""
    global thread_workers
    thread_workers = thread_workers_param
    
    if not base_data_path:
        raise ValueError("base_data_path is required")
    
    # Store configuration in global variables for multi-worker access
    import os
    os.environ['KG_BASE_DATA_PATH'] = base_data_path
    os.environ['KG_SUPPORTED_KGS'] = ','.join(supported_kgs or ['webqsp', 'cwq'])
    os.environ['KG_ENABLED_ACTIONS'] = ','.join(enabled_actions or ['get_head_relations', 'get_tail_relations', 'get_head_entities', 'get_tail_entities', 'get_conditional_relations'])
    os.environ['KG_USE_ENTITIES_TEXT'] = str(use_entities_text)
    os.environ['KG_THREAD_WORKERS'] = str(thread_workers_param)
    
    initialize_server(base_data_path, supported_kgs, enabled_actions, use_entities_text)
    
    logger.info(f"Starting Knowledge Graph Retrieval Server on {host}:{port}")
    logger.info(f"Using {workers} worker process(es)")
    logger.info(f"Using {thread_workers_param} ThreadPoolExecutor workers for concurrent processing")
    logger.info(f"Using RoG base data path: {base_data_path}")
    logger.info(f"Supported KGs: {supported_kgs or ['webqsp', 'cwq']}")
    logger.info(f"Enabled actions: {enabled_actions or 'all'}")
    logger.info(f"Use entities text: {use_entities_text}")
    
    if workers > 1:
        logger.warning("Multiple workers requested but this will use single worker due to uvicorn limitations with app object")
        logger.info("For true multi-worker support, use: uvicorn kg_r1.search.server:app --workers N")
    
    # Use single worker for now - the ThreadPoolExecutor provides concurrency within each worker
    uvicorn.run(app, host=host, port=port, workers=1)
