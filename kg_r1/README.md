# KG-R1: Knowledge Graph-based Search-R1 Framework

## Overview

KG-R1 is a knowledge graph-based retrieval system that extends the Search-R1 framework for question answering tasks. Instead of traditional document retrieval, it uses pre-computed knowledge subgraphs from datasets like WebQuestionsSP (WebQSP) and ComplexWebQuestions (CWQ) to provide structured knowledge for language model reasoning.

## Architecture

The system consists of a clean, modular architecture with two main components:

### 1. Actions Module (`kg_r1/search/actions.py`)
Contains all action definitions and their implementation logic:
- **ActionType Enum**: Defines available action types (GET_RELATIONS, GET_HEAD_ENTITIES, GET_TAIL_ENTITIES)
- **SearchRequest**: Pydantic model for incoming requests
- **ActionHandler Base Class**: Abstract base class that provides common functionality like subgraph loading
- **Concrete Action Classes**: Each action implements its own retrieval algorithm:
  - **GetRelationsAction**: Retrieves all relations for a given entity
  - **GetHeadEntitiesAction**: Finds head entities given a tail entity and relation  
  - **GetTailEntitiesAction**: Finds tail entities given a head entity and relation
- **ACTION_REGISTRY**: Registry mapping action types to their handler classes

### 2. Server Module (`kg_r1/search/server.py`)
Contains the FastAPI server and coordination logic:
- **KnowledgeGraphRetriever**: Initializes action handlers and executes requests
- **FastAPI Application**: Defines endpoints (`/retrieve`, `/health`, `/actions`)
- **Request Validation**: Ensures required fields are present for each action type
- **Response Formatting**: Standardized response format with timing and result counts

### 3. Entry Point (`kg_r1/search/kg_retrieval_server.py`)
Simple command-line entry point that:
- Parses command-line arguments
- Validates the data path
- Initializes and runs the server

**Key Benefits:**
- **Self-contained Actions**: Each action contains its own algorithm implementation
- **Easy Extension**: Add new actions by creating a new ActionHandler subclass
- **No Central Algorithms**: The server doesn't need to know about specific retrieval logic
- **Clean Separation**: Actions handle logic, server handles HTTP and coordination

## Data Structure

The system expects data in the following structure:
```
data_kg/
├── webqsp/
│   └── subgraphs/
│       ├── sample_id_1.json
│       ├── sample_id_2.json
│       └── ...
└── cwq/
    └── subgraphs/
        ├── sample_id_1.json
        ├── sample_id_2.json
        └── ...
```

Each subgraph JSON file contains:
```json
{
  "triples": [
    ["head_entity_id", "relation", "tail_entity_id"],
    ["head_entity_id", "relation", "tail_entity_id"],
    ...
  ]
}
```

## API Endpoints

### Knowledge Graph Retrieval Server

**Base URL**: `http://127.0.0.1:8001` (default)

#### POST `/retrieve`
Main endpoint for KG operations. Accepts a list of `SearchRequest` objects.

**Request Format**:
```json
[
  {
    "action_type": "get_relations",
    "dataset_name": "webqsp",
    "sample_id": "sample_123",
    "entity_id": "m.0123456"
  }
]
```

**Response Format**:
```json
[
  {
    "results": [
      {
        "relations": ["relation1", "relation2", "relation3"]
      }
    ],
    "query_time": 0.001,
    "total_results": 3
  }
]
```

#### GET `/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "service": "kg_retrieval",
  "version": "2.0.0",
  "uptime": 1234.56
}
```

#### GET `/actions`
List all available actions and their requirements.

**Response**:
```json
{
  "available_actions": ["get_relations", "get_head_entities", "get_tail_entities"],
  "action_details": {
    "get_relations": {
      "required_fields": ["sample_id", "dataset_name", "entity_id"],
      "description": "Action to get all relations for a given entity"
    }
  }
}
```

#### GET `/cache/info`
Get information about the subgraph cache.

**Response**:
```json
{
  "cache_enabled": true,
  "cache_size": 45,
  "max_cache_size": 1000,
  "cached_items": ["webqsp:sample_1", "webqsp:sample_2"]
}
```

#### POST `/cache/clear`
Clear the subgraph cache.

**Response**:
```json
{
  "message": "Cache cleared successfully"
}
```

## Configuration

### KG Retrieval Server Configuration
- `--host`: Server host address (default: "0.0.0.0")
- `--port`: Server port (default: 8000)
- `--base_data_path`: Path to the knowledge graph data directory (required)

### Search Configuration in Training
```yaml
actor_rollout_ref:
  rollout:
    search:
      enable: true
      enable_during_training: true
      enable_during_validation: true
      search_url: "http://127.0.0.1:8001/retrieve"
      max_turns: 2
      topk: 3
```

### Generation Configuration
```python
GenerationConfig(
    max_turns=10,           # Maximum interaction turns
    max_start_length=1024,  # Maximum initial prompt length
    max_prompt_length=4096, # Maximum total prompt length
    max_response_length=500,# Maximum response length
    max_obs_length=512,     # Maximum observation length
    num_gpus=4,             # Number of GPUs
    no_think_rl=False,      # Disable thinking process
    search_url="http://127.0.0.1:8001/retrieve",
    topk=3                  # Top-k results (legacy parameter)
)
```

## Usage

### 1. Start the KG Retrieval Server

```bash
# Using the launch script
./kg_retrieval_launch.sh

# Or directly
python kg_r1/search/kg_retrieval_server.py \
    --base_data_path data_kg \
    --port 8001
```

### 2. Training with KG Integration

The system integrates with the VERL training framework:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m verl.trainer.main_ppo \
    # ... other training parameters ...
    actor_rollout_ref.rollout.search.enable=true \
    actor_rollout_ref.rollout.search.search_url=http://127.0.0.1:8001/retrieve \
    actor_rollout_ref.rollout.search.max_turns=2 \
    actor_rollout_ref.rollout.search.topk=3
```

### 3. Generation Process

The language model generates responses using special tags:
- `<think>`: Internal reasoning process
- `<search>`: Knowledge graph search operations
- `<information>`: Retrieved information from KG
- `<answer>`: Final answer

Example generation flow:
```
<think>I need to find information about Barack Obama's birth place.</think>
<search>{"action_type": "get_relations", "sample_id": "123", "dataset_name": "webqsp", "entity_id": "m.02mjmr"}</search>
<information>Relations: ["place_of_birth", "profession", "political_party"]</information>
<search>{"action_type": "get_tail_entities", "sample_id": "123", "dataset_name": "webqsp", "entity_id": "m.02mjmr", "relation": "place_of_birth"}</search>
<information>Tail entities: ["m.02hrh0_"]</information>
<answer>Barack Obama was born in Honolulu, Hawaii.</answer>
```

## Integration with VERL Framework

The KG-R1 system integrates with the VERL (Versatile Efficient Reinforcement Learning) framework through:

1. **Custom Ray Trainer** (`verl/trainer/ppo/ray_trainer_kg.py`): Extended PPO trainer with KG search capabilities
2. **Generation Manager Integration**: Automatic initialization when search is enabled
3. **Multi-turn Support**: Handles iterative reasoning and search operations
4. **Tensor Management**: Efficient handling of variable-length sequences

## Performance Considerations

### Memory Management
- Uses FSDP (Fully Sharded Data Parallel) for large model training
- Implements gradient checkpointing and activation offloading
- GPU memory utilization configuration for VLLM engine

### Batch Processing
- Dynamic batch sizing support
- GPU-aware padding for multi-GPU setups
- Efficient tensor concatenation and masking

### Search Optimization
- Pre-computed subgraphs for fast retrieval
- In-memory JSON loading for low latency
- Batch request processing

## Error Handling

Common issues and solutions:

1. **Data Type Errors**: Ensure `input_ids` remain as integer tensors (torch.long)
2. **Memory Issues**: Adjust `max_obs_length` if observations are too long
3. **Server Connection**: Verify KG retrieval server is running and accessible
4. **Subgraph Missing**: Check that sample IDs exist in the subgraph directory

## Monitoring and Logging

The system provides comprehensive logging through:
- WandB integration for experiment tracking
- Validation generation logging
- Performance metrics (query time, response quality)
- Error tracking and debugging information

## Extending the System

To add new KG operations, follow these steps:

### 1. Define a New Action Type
Add the new action to the `ActionType` enum in `actions.py`:
```python
class ActionType(str, Enum):
    GET_RELATIONS = "get_relations"
    GET_HEAD_ENTITIES = "get_head_entities"
    GET_TAIL_ENTITIES = "get_tail_entities"
    # Add your new action
    GET_ENTITY_TYPES = "get_entity_types"
```

### 2. Create an Action Handler Class
Implement a new action handler that inherits from `ActionHandler`:
```python
class GetEntityTypesAction(ActionHandler):
    """Handler for getting entity types."""
    
    def execute(self, sample_id: str, dataset_name: str, entity_id: str, **kwargs):
        """Get entity types for a given entity."""
        subgraph_data = self.load_subgraph(sample_id, dataset_name)
        if not subgraph_data or "triples" not in subgraph_data:
            return {"error": f"Subgraph for {sample_id} could not be loaded"}

        # Your custom logic here
        entity_types = []
        triples = subgraph_data["triples"]
        for head, rel, tail in triples:
            if head == entity_id and rel == "type":
                entity_types.append(tail)
        
        return {"entity_types": sorted(list(set(entity_types)))}
```

### 3. Register the New Action
Add your action to the `ACTION_REGISTRY`:
```python
ACTION_REGISTRY = {
    ActionType.GET_RELATIONS: GetRelationsAction,
    ActionType.GET_HEAD_ENTITIES: GetHeadEntitiesAction,
    ActionType.GET_TAIL_ENTITIES: GetTailEntitiesAction,
    ActionType.GET_ENTITY_TYPES: GetEntityTypesAction,  # Add this line
}
```

### 4. Update Request Validation (Optional)
If needed, add validation logic in `server.py` for your new action's required fields.

That's it! The server will automatically discover and support your new action without any changes to the server code.

## Dependencies

Core dependencies:
- PyTorch (with CUDA support)
- Transformers (Hugging Face)
- FastAPI and Uvicorn
- Ray (for distributed training)
- VERL framework
- WandB (for logging)
- Pydantic (for data validation)

## File Structure

```
kg_r1/
├── __init__.py
├── README.md                  # This documentation
├── llm_agent/
│   ├── __init__.py
│   ├── generation.py          # Multi-turn generation manager
│   └── tensor_helper.py       # Tensor utility functions
└── search/
    ├── __init__.py
    ├── actions.py             # Action definitions and handlers  
    ├── server.py              # FastAPI server and KnowledgeGraphRetriever
    └── kg_retrieval_server.py # Command-line entry point
```

**Key Files:**
- `actions.py` - All action logic is self-contained here
- `server.py` - HTTP server and request coordination  
- `kg_retrieval_server.py` - Simple entry point for command-line usage

## Adding New Actions

The modular architecture makes it easy to add new actions:

### 1. Define the Action Type
```python
# In actions.py
class ActionType(str, Enum):
    GET_RELATIONS = "get_relations"
    GET_HEAD_ENTITIES = "get_head_entities" 
    GET_TAIL_ENTITIES = "get_tail_entities"
    YOUR_NEW_ACTION = "your_new_action"  # Add here
```

### 2. Create the Action Class
```python
# In actions.py or extensions.py
class YourNewAction(BaseAction):
    def get_required_fields(self) -> List[str]:
        return ["sample_id", "dataset_name", "your_param"]
    
    def validate_request(self, request_data: Dict[str, Any]) -> bool:
        required_fields = self.get_required_fields()
        return all(request_data.get(field) is not None for field in required_fields)
    
    def execute(self, retriever, request_data: Dict[str, Any]) -> Union[Dict[str, Any], Dict[str, str]]:
        # Your implementation here
        return {"result": "your_result"}
```

### 3. Register the Action
```python
# In your initialization code
action_registry.register(ActionType.YOUR_NEW_ACTION, YourNewAction())
```

### 4. Update Request Model (if needed)
```python
# In models.py
class SearchRequest(BaseModel):
    # ... existing fields ...
    your_param: Optional[str] = None  # Add new fields as needed
```

The new action becomes immediately available through the `/retrieve` endpoint!

## Contributing

When modifying the system:
1. Ensure backward compatibility with existing data formats
2. Add appropriate error handling and logging
3. Update configuration documentation
4. Test with both WebQSP and CWQ datasets
5. Verify multi-GPU compatibility

## License

This project follows the same license as the parent VERL framework.
