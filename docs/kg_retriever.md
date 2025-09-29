\
# Knowledge Graph (KG) Retriever Server

This document outlines how to launch and use the Knowledge Graph (KG) Retriever Server. This server is designed to work with pre-computed per-sample subgraphs for datasets like WebQuestionsSP (WebQSP) and ComplexWebQuestions (CWQ), specifically using the RoG (Reasoning over Graphs) format.

It replaces traditional document retrieval with KG-based operations, allowing for targeted information extraction from these subgraphs.

## Overview

The KG Retriever Server provides an API to perform specific actions on knowledge subgraphs associated with individual data samples (e.g., questions). Instead of retrieving a list of documents, it allows querying for relations connected to an entity, or entities connected by a specific relation within a given sample's subgraph.

**Key Features:**

*   **Per-Sample Subgraph Operations:** All actions are performed on the context of a specific sample's pre-loaded knowledge subgraph.
*   **Supported Actions:**
    *   `GET_RELATIONS`: Retrieves all unique relations connected to a specified entity within the sample's subgraph.
    *   `GET_HEAD_ENTITIES`: Given a tail entity and a relation, retrieves all head entities connected by that relation within the sample's subgraph.
    *   `GET_TAIL_ENTITIES`: Given a head entity and a relation, retrieves all tail entities connected by that relation within the sample's subgraph.
*   **Extensibility:** Additional actions can be supported by editing the `ActionType` enum and `SearchRequest` model in `kg_r1/search/kg_retrieval_server.py`, implementing the new action logic within the `KnowledgeGraphRetriever` class, and updating the request handling in the `/retrieve` endpoint. Corresponding tests should be added to `scripts/test_kg_actions.py`.
*   **Batch Processing:** The server can handle a list of requests, allowing multiple actions to be performed in a single API call.
*   **RoG Format:** Designed to work with subgraphs from `rmanluo/RoG-webqsp` and `rmanluo/RoG-cwq` Hugging Face datasets.

## Setup and Launching

### 1. Prerequisites

*   Ensure you have a Conda environment with the necessary packages. If not, create one:
    ```bash
    conda create -n kg_retriever python=3.9  # Or your preferred Python version
    conda activate kg_retriever
    pip install fastapi uvicorn pydantic requests
    ```
*   Download the RoG-formatted data, including the per-sample subgraphs. The `scripts/download_kg.py` script can be used for this:
    ```bash
    # Example:
    python scripts/download_kg.py --save_path ./data_kg_rog 
    ```
    This will download the data and organize it into `webqsp/subgraphs/` and `cwq/subgraphs/` under the specified `save_path`. Each subgraph is stored as a JSON file named after its sample ID (e.g., `WebQTrn-0.json`).

### 2. Launching the Server

The server script is `kg_r1/search/kg_retrieval_server.py`.

```bash
conda activate kg_retriever # Or your relevant conda environment name

# The --base_data_path should point to the directory containing 
# the 'webqsp' and 'cwq' folders with their respective 'subgraphs' subdirectories.
python kg_r1/search/kg_retrieval_server.py --base_data_path /path/to/your/data_kg_rog/
```

**Server Arguments:**

*   `--host`: Host address to bind the server to (default: `0.0.0.0`).
*   `--port`: Port number to run the server on (default: `8000`).
*   `--base_data_path` (Required): The absolute or relative path to the directory where the RoG dataset (e.g., `data_kg_rog`) is stored. This directory must contain subfolders for each dataset (e.g., `webqsp`, `cwq`), which in turn contain a `subgraphs` folder with the individual JSON subgraph files.

Example: If your subgraphs are in `/nobackup/users/yeopjin/workspace/RL_KG/data_kg_rog/webqsp/subgraphs/`, then `--base_data_path /nobackup/users/yeopjin/workspace/RL_KG/data_kg_rog`.

Upon successful launch, you should see log messages indicating the server has started, e.g.:
```
INFO:__main__:Retriever initialized with base_data_path: ./data_kg_rog
INFO:__main__:Starting Knowledge Graph Retrieval Server on 0.0.0.0:8000
INFO:__main__:Using RoG base data path: ./data_kg_rog
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## API Endpoint: `/retrieve`

The server exposes a single POST endpoint `/retrieve` that accepts a list of `SearchRequest` objects.

*   **URL:** `http://<host>:<port>/retrieve`
*   **Method:** `POST`
*   **Request Body:** A JSON list of `SearchRequest` objects.

### `SearchRequest` Model

Each `SearchRequest` object in the list should conform to the following structure:

```json
{
    "action_type": "get_relations", // or "get_head_entities", "get_tail_entities"
    "dataset_name": "webqsp",       // or "cwq"
    "sample_id": "WebQTrn-0",       // The ID of the sample (e.g., question ID)
    "entity_id": "Justin Bieber",   // Required for all actions. Represents the primary entity for the action.
                                    // For get_head_entities, this is the TAIL entity.
                                    // For get_tail_entities, this is the HEAD entity.
    "relation": "music.artist.album" // Required for get_head_entities and get_tail_entities
}
```

**Fields:**

*   `action_type` (required, string): The type of KG action to perform. Must be one of:
    *   `"get_relations"`
    *   `"get_head_entities"`
    *   `"get_tail_entities"`
*   `dataset_name` (required, string): The name of the dataset (e.g., `"webqsp"`, `"cwq"`). This is used to locate the correct subgraph directory.
*   `sample_id` (required, string): The unique identifier for the data sample (e.g., `WebQTrn-0`, `CWQTest-0`). The server will load `{base_data_path}/{dataset_name}/subgraphs/{sample_id}.json`.
*   `entity_id` (required, string): The identifier of the entity of interest within the subgraph.
    *   For `get_relations`: The entity whose relations are to be fetched.
    *   For `get_head_entities`: This is treated as the *tail entity* of the triples to search.
    *   For `get_tail_entities`: This is treated as the *head entity* of the triples to search.
*   `relation` (optional, string): The specific relation to consider.
    *   Required for `get_head_entities` and `get_tail_entities`.
    *   Not used by `get_relations`.

### Response Format

The `/retrieve` endpoint returns a JSON list, where each item in the list corresponds to the result of a `SearchRequest` from the input list, maintaining the order.

Each response item has the following structure:

```json
{
    "results": [ 
        // For GET_RELATIONS:
        { "relations": ["relation1", "relation2", ...] }
        // For GET_HEAD_ENTITIES:
        { "head_entities": ["entity1", "entity2", ...] }
        // For GET_TAIL_ENTITIES:
        { "tail_entities": ["entity1", "entity2", ...] }
        // If an error occurred loading the subgraph for this request:
        { "error": "Subgraph for ... could not be loaded or is empty." }
    ],
    "query_time": 0.005, // Time taken in seconds for this specific request
    "total_results": 15  // Number of items in the primary list (e.g., number of relations)
}
```

**Error Handling in Batch Requests:**

If a specific `SearchRequest` within a batch is invalid (e.g., missing required fields), the corresponding item in the response list will contain an "error" field:

```json
{
    "error": "For GET_RELATIONS, sample_id, dataset_name, and entity_id are required in request: {...request_details...}",
    "query_time": 0.0001,
    "total_results": 0
}
```
The server will continue to process other valid requests in the batch.

## Example Usage (Python `requests`)

```python
import requests
import json

SERVER_URL = "http://localhost:8000/retrieve"

payloads = [
    { # Request 1: Get relations for Justin Bieber in WebQTrn-0
        "action_type": "get_relations",
        "dataset_name": "webqsp",
        "sample_id": "WebQTrn-0",
        "entity_id": "Justin Bieber"
    },
    { # Request 2: Get tail entities (albums) for Justin Bieber in WebQTrn-0
        "action_type": "get_tail_entities",
        "dataset_name": "webqsp",
        "sample_id": "WebQTrn-0",
        "entity_id": "Justin Bieber",
        "relation": "music.artist.album"
    },
    { # Request 3: Invalid request - missing entity_id for get_relations
        "action_type": "get_relations",
        "dataset_name": "webqsp",
        "sample_id": "WebQTrn-0"
        # "entity_id": "Justin Bieber" # Missing
    }
]

try:
    response = requests.post(SERVER_URL, json=payloads)
    response.raise_for_status() 
    print("--- Response (Status:", response.status_code, ") ---")
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.RequestException as e:
    print("--- Error ---")
    print("Request failed:", e)
    if hasattr(e, 'response') and e.response is not None:
        try:
            print("Server response:", e.response.json())
        except json.JSONDecodeError:
            print("Server response (not JSON):", e.response.text)

```

This example demonstrates sending a batch of requests, including one intentionally invalid request, to the server. The output will show the results for the valid requests and an error message for the invalid one.

## Testing

A test script `scripts/test_kg_actions.py` is available to verify the server's functionality. It sends various valid and invalid requests to the server and prints the responses.

```bash
conda activate kg_retriever # Or your environment
python scripts/test_kg_actions.py
```
Ensure the KG Retriever server is running before executing the test script.
