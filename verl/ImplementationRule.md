# KG-Augmented Generation RL Framework Implementation Plan

## Overview
This document outlines the implementation plan for integrating Knowledge Graph (KG) augmented generation into the VERL PPO training framework. The system enables LLMs to interact with external knowledge graphs during training through structured queries and reward optimization.

## ✅ IMPLEMENTATION STATUS

### Completed Components
- ✅ **KG-Specific Ray Trainer** (`ray_trainer_kg.py`) - Extended PPO trainer with KG interaction capabilities
- ✅ **KG Reward System** (`kg_format.py`) - Specialized reward manager for KG-augmented generation
- ✅ **KG Data Pipeline** - Data processing for WebQSP and CWQ datasets
- ✅ **Main Entry Integration** (`main_ppo.py`) - Mode detection and trainer routing
- ✅ **KG Scoring Functions** (`qa_em_format_kg.py`) - KG-aware scoring with proper tag validation
- ✅ **Training Scripts** - Both standard (`train_ppo.sh`) and KG (`train_ppo_kg.sh`) training scripts

### Integration Points
- ✅ Trainer mode selection based on `trainer.mode` configuration
- ✅ KG-specific search configuration with `<kg-query>` tags
- ✅ Reward manager registration and loading
- ✅ Dataset compatibility (WebQSP, CWQ, NQ formats)

## Architecture Overview

### Core Components (Current Status)
1. ✅ **KG-Specific Ray Trainer** (`ray_trainer_kg.py`) - Extended PPO trainer with KG interaction capabilities
2. ✅ **KG Generation Manager** - Integrated with existing `LLMGenerationManager` from `kg_r1/llm_agent/generation.py`
3. ✅ **KG Reward System** (`kg_format.py`) - Specialized reward manager with KG-aware scoring
4. ✅ **KG Data Pipeline** - Data processing for KG-enhanced training (WebQSP, CWQ)
5. ✅ **KG Scoring Functions** (`qa_em_format_kg.py`) - KG-specific exact match and format validation

## Implementation Structure (Current)

```
verl/
├── trainer/
│   ├── main_ppo.py (✅ modified)                 # Entry point with KG mode detection
│   └── ppo/
│       ├── ray_trainer_kg.py (✅ implemented)    # KG-specific PPO trainer
│       └── reward.py (✅ modified)               # Updated reward manager loading
├── workers/
│   └── reward_manager/
│       ├── kg_format.py (✅ implemented)         # KG-specific reward manager
│       └── __init__.py (✅ updated)              # Registered KG format manager
├── utils/
│   └── reward_score/
│       ├── qa_em_format_kg.py (✅ implemented)   # KG-aware scoring functions
│       └── kg_webqsp.py (✅ existing)            # WebQSP-specific scoring
└── scripts/
    └── data_process_kg/
        ├── webqsp.py (✅ implemented)            # WebQSP data processing
        └── cwq.py (✅ implemented)               # CWQ data processing
```

## Execution Flow (Current Implementation)

### Training Script: `train_ppo_kg.sh`
- ✅ Entry point for KG-augmented training
- ✅ Sets KG-specific environment variables
- ✅ Calls `main_ppo.py` with `trainer.mode=kg-search`
- ✅ Configures KG server URL and search parameters

### Main Entry: `main_ppo.py`
- ✅ Detects training mode from config (`trainer.mode`)
- ✅ Routes to appropriate trainer:
  - Standard mode (`search`): `RayPPOTrainer` from `ray_trainer.py`
  - KG mode (`kg-search`): `RayPPOTrainer` from `ray_trainer_kg.py`
- ✅ Loads appropriate reward manager based on `reward_model.reward_manager`

### KG Data Flow
- ✅ WebQSP/CWQ datasets processed with proper VERL structure
- ✅ Ground truth includes both `target_text` and `target_kb_id`
- ✅ Data compatible with existing VERL RLHFDataset loader

## Detailed Implementation Status

### 1. ✅ Modified Main Entry (`main_ppo.py`)

**Changes Completed:**
- ✅ Added mode detection logic based on `config.trainer.mode`
- ✅ Import and instantiate KG-specific trainer when `mode=kg-search`
- ✅ Route to appropriate `RayPPOTrainer` based on mode
- ✅ Logging for trainer selection

**Implementation:**
```python
# Added to main_ppo.py around line 135
training_mode = getattr(config.trainer, 'mode', 'search')

if training_mode == 'kg-search':
    from verl.trainer.ppo.ray_trainer_kg import RayPPOTrainer as RayPPOTrainerKG
    trainer_cls = RayPPOTrainerKG
    logger.info("Using KG-specific PPO trainer")
else:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    trainer_cls = RayPPOTrainer
    logger.info("Using standard PPO trainer")
```

### 2. ✅ KG-Specific Ray Trainer (`ray_trainer_kg.py`)

**Purpose:** Extends the standard PPO trainer with KG interaction capabilities

**Key Features Implemented:**
- ✅ Inherits from `RayPPOTrainer` 
- ✅ Integrates `LLMGenerationManager` for KG interactions
- ✅ Manages KG server connections and configuration
- ✅ Handles multi-turn conversation flows with `<kg-query>` tags
- ✅ Custom reward computation integration
- ✅ Preference for `kg_config` and `generation_config` over `search_config`

**Core Methods:**
```python
class RayPPOTrainer(RayPPOTrainer):  # Note: Same class name but from different file
    def __init__(self, config):
        super().__init__(config)
        # KG-specific initialization
        self.use_search_generation = self._check_kg_enabled(config)
        if self.use_search_generation:
            self.generation_manager = LLMGenerationManager(...)
    
    def _check_kg_enabled(self, config):
        """Check if KG generation is enabled via kg_config or search_config"""
        
    def _setup_kg_generation(self, config):
        """Setup KG generation with preferred kg_config"""
```

### 3. ✅ KG Reward System (`kg_format.py`)

**Purpose:** Specialized reward manager for KG-augmented generation

**Implementation Status:**
- ✅ Follows same pattern as existing `format.py` reward manager
- ✅ Uses `qa_em_format_kg.compute_score_em_kg` for scoring
- ✅ Supports both KG datasets (WebQSP, CWQ) and standard datasets (NQ)
- ✅ Registered in `verl.workers.reward_manager.__init__.py`
- ✅ Integrated with reward loading system in `verl.trainer.ppo.reward.py`

**Reward Components:**
1. ✅ **Structure Reward**: Proper use of `<think>`, `<kg-query>`, `<answer>`, `<information>` tags
2. ✅ **Retrieval Reward**: Successful KG queries and relevant results
3. ✅ **Format Reward**: Correct sequence validation and tag balance
4. ✅ **Final Answer Reward**: Correctness of final answer against target_text and target_kb_id
5. ✅ **Exact Match Scoring**: Enhanced with KG-aware answer extraction

**Implementation:**
```python
class KGFormatRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", 
                 structure_format_score=0., final_format_score=0., retrieval_score=0., format_score=0.):
        # Initialize with format-specific scoring parameters
        
    def __call__(self, data: DataProto, return_dict=False):
        # Use _select_kg_rm_score_fn to choose appropriate scoring function
        # Support both custom compute_score and built-in KG scoring
```

### 4. ✅ KG Scoring Functions (`qa_em_format_kg.py`)

**Purpose:** KG-aware scoring with proper tag validation and ground truth handling

**Key Features Implemented:**
- ✅ **Tag Pattern Support**: Uses `<kg-query>` instead of `<search>` for KG datasets
- ✅ **Ground Truth Handling**: Supports both `target_text` and `target_kb_id` as valid answers
- ✅ **Sequence Validation**: Follows same pattern as `qa_em_format.py` with state machine validation
- ✅ **Backward Compatibility**: Also supports legacy `<search>` tags
- ✅ **Enhanced Answer Extraction**: Uses existing `extract_answer` from `kg_webqsp.py`

**Core Functions:**
```python
def is_valid_kg_sequence(text: str) -> tuple[bool, str]:
    """Validates KG sequence with proper state machine like qa_em_format.py"""
    # Checks for: <think> -> <kg-query> -> <information> -> <answer> flow
    
def compute_score_em_kg(solution_str, ground_truth, method='kg_aware', ...):
    """KG-aware exact match scoring with enhanced ground truth handling"""
    # Combines target_text and target_kb_id as potential correct answers
    # Uses kg_webqsp.compute_score for KG-aware scoring when available
```

### 5. ✅ KG Data Pipeline

**Purpose:** Data processing pipeline for KG-enhanced training data

**Implementation Status:**
- ✅ **WebQSP Processing** (`scripts/data_process_kg/webqsp.py`): Converts WebQSP JSONL to VERL-compatible parquet
- ✅ **CWQ Processing** (`scripts/data_process_kg/cwq.py`): Converts ComplexWebQuestions to VERL format
- ✅ **Data Structure**: Compatible with existing VERL RLHFDataset loader
- ✅ **Ground Truth Format**: Includes both `target_text` and `target_kb_id` fields
- ✅ **Validation**: Output validated for compatibility with reward managers

**Data Format:**
```json
{
    "prompt": "What is the capital of France?",
    "sample_id": "webqsp_train_0001",
    "data_source": "webqsp",
    "ground_truth": {
        "target_text": ["Paris"],
        "target_kb_id": ["m.05qtj"]
    },
    "extra_info": {
        "sample_id": "webqsp_train_0001",
        "original_question": "What is the capital of France?",
        "parse_info": {...}
    }
}
```

### 6. ✅ Training Configuration

**Purpose:** KG-specific training script and configuration

**Implementation Status:**
- ✅ **Training Script** (`train_ppo_kg.sh`): Complete KG training configuration
- ✅ **Mode Setting**: `trainer.mode=kg-search` for KG trainer selection
- ✅ **Search Configuration**: Enables KG search with proper URL and parameters
- ✅ **Reward Configuration**: Uses `reward_model.reward_manager=kg_format`
- ✅ **KG Configuration**: Server URL, max turns, and generation parameters

**Key Configuration:**
```bash
trainer.mode=kg-search \
actor_rollout_ref.rollout.search.enable=true \
actor_rollout_ref.rollout.search.enable_during_training=true \
actor_rollout_ref.rollout.search.enable_during_validation=true \
actor_rollout_ref.rollout.search.search_url="http://127.0.0.1:8001/retrieve" \
kg_config.server_url="http://127.0.0.1:8001/retrieve" \
kg_config.max_turns=6 \
kg_config.enable_kg_during_training=true \
reward_model.enable=true \
reward_model.reward_manager=kg_format \
```
  max_start_length: 512
  max_prompt_length: 2048
  max_response_length: 100
  max_obs_length: 256
  
reward_config:
  structure_weight: 0.1
  retrieval_weight: 0.2
  format_weight: 0.1
  answer_weight: 0.5
  efficiency_weight: 0.1
```

## Integration Points (Current Status)

### 1. ✅ Training Script Integration

**File:** `train_ppo_kg.sh`

**Key Parameters:**
```bash
# KG-specific parameters
trainer.mode=kg-search \
actor_rollout_ref.rollout.search.enable=true \
actor_rollout_ref.rollout.search.enable_during_training=true \
actor_rollout_ref.rollout.search.enable_during_validation=true \
actor_rollout_ref.rollout.search.search_url="http://127.0.0.1:8001/retrieve" \
actor_rollout_ref.rollout.search.max_turns=6 \
kg_config.server_url="http://127.0.0.1:8001/retrieve" \
kg_config.max_turns=6 \
kg_config.enable_kg_during_training=true \
generation_config.max_start_length=512 \
generation_config.max_prompt_length=2048 \
reward_model.enable=true \
reward_model.reward_manager=kg_format \
reward_config.structure_weight=0.1 \
reward_config.retrieval_weight=0.2 \
```

### 2. ✅ Worker Integration

**Rollout Workers:**
- ✅ Modified to support KG interaction loops via `LLMGenerationManager`
- ✅ Handle multi-turn generation with `<kg-query>` tags
- ✅ Process KG server responses through existing infrastructure

**Critic Workers:**
- ✅ Evaluate KG-augmented sequences using `kg_format` reward manager
- ✅ Compute value estimates for multi-turn interactions

### 3. ✅ Data Flow Integration

**Training Loop:**
1. ✅ Sample batch from KG dataset (WebQSP/CWQ format)
2. ✅ Run KG-augmented rollout with generation manager
3. ✅ Query KG server for relevant information via search mechanism
4. ✅ Generate responses with KG context using multi-turn capability
5. ✅ Compute KG-specific rewards using `kg_format` reward manager
6. ✅ Update policy and critic networks using standard PPO mechanism

## Current Testing Results

### ✅ Validated Components
- ✅ KG format reward manager imports and instantiates correctly
- ✅ QA EM format KG module handles ground truth parsing
- ✅ Trainer selection logic works based on configuration mode
- ✅ WebQSP and CWQ data processing produces valid VERL format
- ✅ Data structure compatibility with existing RLHFDataset loader

## Implementation Timeline (Completed)

### ✅ Phase 1: Core Components (Week 1-2)
1. ✅ Created `ray_trainer_kg.py` with KG integration and LLMGenerationManager
2. ✅ Implemented `kg_format.py` reward manager for comprehensive KG scoring
3. ✅ Developed `qa_em_format_kg.py` with KG-aware scoring functions
4. ✅ Set up configuration in training scripts

### ✅ Phase 2: Data Pipeline (Week 2-3)
1. ✅ Implemented WebQSP and CWQ data processing for KG data handling
2. ✅ Created VERL-compatible data format with target_text and target_kb_id
3. ✅ Developed data validation and preprocessing
4. ✅ Integration with existing RLHFDataset loaders

### ✅ Phase 3: Training Integration (Week 3-4)
1. ✅ Modified `main_ppo.py` for mode routing (kg-search vs search)
2. ✅ Integrated with Ray-based distributed training framework
3. ✅ Implemented KG search configuration and server connectivity
4. ✅ Added reward manager registration and loading

### 🔄 Phase 4: Testing and Validation (Week 4-5)
1. ✅ Basic component testing and import validation
2. 🔄 End-to-end training pipeline testing
3. ⏳ Performance benchmarking
4. ⏳ Documentation updates

## File Naming Convention (Current)

All KG-specific files follow the pattern: `{base_name}_kg.py` or use descriptive KG-related names

**Implemented Files:**
- ✅ `ray_trainer_kg.py` - KG-specific PPO trainer
- ✅ `kg_format.py` - KG reward manager
- ✅ `qa_em_format_kg.py` - KG-aware scoring functions
- ✅ `webqsp.py` - WebQSP data processing
- ✅ `cwq.py` - CWQ data processing
- ✅ `train_ppo_kg.sh` - KG training script

## Dependencies (Current)

### External Dependencies
- ✅ `requests`: For KG server communication
- ✅ `torch`: For tensor operations  
- ✅ `transformers`: For model and tokenizer handling
- ✅ `ray`: For distributed training
- ✅ `pandas`: For data processing
- ✅ `pyarrow`: For parquet file handling

### Internal Dependencies
- ✅ Existing VERL PPO framework
- ✅ LLMGenerationManager from `kg_r1/llm_agent/generation.py`
- ✅ KG scoring functions from `verl/utils/reward_score/kg_webqsp.py`
- ✅ Existing reward manager infrastructure

## Success Criteria (Current Status)

1. ✅ **Functional Integration**: KG-augmented training framework integrated with existing PPO system
2. ⏳ **Performance**: Training pipeline ready for performance validation
3. ✅ **Scalability**: Supports distributed training across multiple GPUs/nodes
4. ⏳ **Accuracy**: Ready for evaluation on KG-dependent tasks  
5. ✅ **Maintainability**: Clean integration with minimal changes to existing code

## Next Steps

### Immediate Actions
1. **End-to-end Testing**: Run complete training pipeline with KG server
2. **Performance Validation**: Benchmark training speed and resource usage
3. **Accuracy Evaluation**: Test on WebQSP and CWQ validation sets
4. **Error Handling**: Validate robustness with server failures and edge cases

### Future Enhancements
1. **Advanced KG Reasoning**: Enhanced multi-hop reasoning capabilities
2. **Dynamic KG Selection**: Adaptive KG source selection during training
3. **Caching Optimization**: KG query caching for improved performance
4. **Advanced Reward Functions**: More sophisticated KG interaction scoring

This implementation plan provides a comprehensive roadmap for integrating KG-augmented generation into the VERL PPO training framework while maintaining compatibility with existing systems and ensuring scalability for distributed training environments.
