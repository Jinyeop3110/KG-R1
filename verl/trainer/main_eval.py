# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Main evaluation entry point supporting multiple evaluation modes:
1. Offline evaluation: Performance evaluation of pre-generated sequences
2. KG evaluation: Live generation with KG search and Pass@K metrics

"""

import os
from collections import defaultdict
from typing import Dict, Any, List

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


def offline_evaluation(config):
    """
    Original offline evaluation mode for pre-generated responses.
    """
    local_path = copy_to_local(config.data.path, use_shm=config.data.get('use_shm', False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    compute_score = get_custom_reward_fn(config)

    # Create remote tasks
    remote_tasks = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)

    print(metric_dict)
    return metric_dict


def kg_evaluation(config):
    """
    KG evaluation mode with live generation and Pass@K metrics.
    """
    print("Starting KG evaluation mode")
    
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = EvaluationTaskRunner.remote()
    return ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class EvaluationTaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        from verl.trainer.ppo.reward import load_reward_manager
        reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        
        # Get evaluation parameters
        n_rollout_eval = config.get('n_rollout_eval', 8)
        k_values_raw = config.get('k_values', [1, 3, 5, 8])
        eval_samples = config.get('eval_samples', 0)  # 0 means evaluate all samples
        save_detailed_results = config.get('save_detailed_results', False)  # Save prompts + responses
        
        # Parse k_values if it's a string (from command line)
        if isinstance(k_values_raw, str):
            import ast
            try:
                k_values = ast.literal_eval(k_values_raw)
                if not isinstance(k_values, list):
                    k_values = [k_values]
            except (ValueError, SyntaxError):
                # Fallback: try splitting by comma
                k_values = [int(k.strip()) for k in k_values_raw.strip('[]').split(',')]
        else:
            k_values = k_values_raw
        
        print(f"Configuration:")
        print(f"  n_rollout_eval: {n_rollout_eval}")
        print(f"  k_values: {k_values}")
        print(f"  eval_samples: {eval_samples} (0 = all samples)")
        print(f"  save_detailed_results: {save_detailed_results}")
        
        # Create KG evaluator
        from verl.trainer.ppo.ray_evaluator_kg import RayKGEvaluator
        
        evaluator = RayKGEvaluator(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
            n_rollout_eval=n_rollout_eval,
            k_values=k_values,
            eval_samples=eval_samples,
            save_detailed_results=save_detailed_results,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        
        # Initialize workers first (creates worker groups)
        evaluator.init_workers()
        
        # Set global_steps = 0 and load checkpoint (like in training fit method)
        evaluator.global_steps = 0  
        if config.trainer.get('resume_mode') == 'resume_path':
            print(f"Loading checkpoint after worker initialization")
            evaluator._load_checkpoint()
        
        try:
            metrics = evaluator.evaluate_dataset("test")
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Clean shutdown of Ray workers to prevent SIGTERM
            try:
                print("Cleaning up Ray workers...")
                if hasattr(evaluator, 'actor_rollout_wg'):
                    evaluator.actor_rollout_wg.shutdown()
                if hasattr(evaluator, 'critic_wg'):
                    evaluator.critic_wg.shutdown()
            except Exception as cleanup_e:
                print(f"Cleanup warning: {cleanup_e}")
        
        # Save results
        output_dir = config.trainer.get('default_local_dir', 'evaluation_results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine dataset name from the data files for naming
        data_files = config.data.get('val_files', 'test')
        if 'cwq' in str(data_files).lower():
            dataset_name = 'cwq'
        elif 'webqsp' in str(data_files).lower():
            dataset_name = 'webqsp'
        else:
            dataset_name = 'test'
        
        results_file = os.path.join(output_dir, f'{dataset_name}_passatk_results.json')
        import json
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        
        print(f"\nResults saved to: {results_file}")
        
        return metrics


def kg_llm_judge_evaluation(config):
    """
    KG evaluation mode with LLM judge scoring.
    
    This mode combines:
    - KG search and multi-turn reasoning from kg_evaluation
    - LLM judge evaluation from vanilla_evaluation
    - Designed for MultiTQ and other temporal KG datasets
    """
    print("Starting KG-LLM-Judge evaluation mode")
    
    if not ray.is_initialized():
        ray_init_kwargs = {
            "runtime_env": {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
        }
        if hasattr(config, 'ray_init'):
            if hasattr(config.ray_init, 'num_cpus') and config.ray_init.num_cpus is not None:
                ray_init_kwargs["num_cpus"] = config.ray_init.num_cpus
            if hasattr(config.ray_init, 'timeline_json_file') and config.ray_init.timeline_json_file is not None:
                ray_init_kwargs["timeline_json_file"] = config.ray_init.timeline_json_file
        
        ray.init(**ray_init_kwargs)
    
    try:
        # Use KG-LLM-Judge hybrid evaluator
        from verl.trainer.ppo.ray_evaluator_kg_llm_judge import create_kg_llm_judge_evaluator
        
        # Import the same helper functions as kg_evaluation
        from transformers import AutoTokenizer
        
        # Create tokenizer (same as kg_evaluation)
        tokenizer = AutoTokenizer.from_pretrained(
            config.actor_rollout_ref.model.path,
            trust_remote_code=config.actor_rollout_ref.model.get('trust_remote_code', True)
        )
        
        # Use same worker setup as kg_evaluation
        processor = None  # Not needed for text-only
        
        # Get evaluation parameters
        n_rollout_eval = config.get('n_rollout_eval', 4)
        k_values = config.get('k_values', [1, 2, 3, 4])
        eval_samples = config.get('eval_samples', 0)
        save_detailed_results = config.get('save_detailed_results', False)
        
        print(f"[KG-LLM-JUDGE] Configuration:")
        print(f"  - N rollout eval: {n_rollout_eval}")
        print(f"  - K values: {k_values}")
        print(f"  - Eval samples: {eval_samples}")
        print(f"  - Save detailed results: {save_detailed_results}")
        
        # Create KG-LLM-Judge evaluator
        evaluator = create_kg_llm_judge_evaluator(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            device_name=config.trainer.device,
            n_rollout_eval=n_rollout_eval,
            k_values=k_values,
            eval_samples=eval_samples,
            save_detailed_results=save_detailed_results
        )
        
        # Initialize workers first (creates worker groups) - same as kg_evaluation
        evaluator.init_workers()
        
        # Set global_steps = 0 (same as kg_evaluation)
        evaluator.global_steps = 0
        
        # Run evaluation using the same pattern as kg_evaluation
        metrics = evaluator.evaluate_dataset("test")
        
        print(f"\nKG-LLM-Judge evaluation completed!")
        
        # Print LLM judge statistics
        if hasattr(evaluator, 'llm_judge_total_calls') and evaluator.llm_judge_total_calls > 0:
            success_rate = (evaluator.llm_judge_successful_calls / evaluator.llm_judge_total_calls) * 100
            fallback_rate = (evaluator.llm_judge_fallbacks / evaluator.llm_judge_total_calls) * 100
            print(f"\n🤖 LLM JUDGE FINAL STATISTICS:")
            print(f"Total calls: {evaluator.llm_judge_total_calls}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Fallback rate: {fallback_rate:.1f}%")
        
        # Save results to JSON file (same pattern as other evaluation modes)
        output_dir = config.trainer.get('default_local_dir', 'evaluation_results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine dataset name from data files
        data_files = config.get('data', {}).get('val_files', '')
        if isinstance(data_files, list):
            data_files = data_files[0] if data_files else ''
        
        dataset_name = 'test'  # Default
        if 'multitq' in str(data_files).lower():
            dataset_name = 'multitq'
        elif 'cwq' in str(data_files).lower():
            dataset_name = 'cwq'
        elif 'webqsp' in str(data_files).lower():
            dataset_name = 'webqsp'
        elif 'simpleqa' in str(data_files).lower():
            dataset_name = 'simpleqa'
        elif 'trex' in str(data_files).lower():
            dataset_name = 'trex'
        elif 'zero_shot_re' in str(data_files).lower():
            dataset_name = 'zero_shot_re'
        
        results_file = os.path.join(output_dir, f'{dataset_name}_passatk_results.json')
        import json
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        
        print(f"\nResults saved to: {results_file}")
        
        return metrics
        
    except Exception as e:
        print(f"KG-LLM-Judge evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean shutdown (same as kg_evaluation)
        try:
            print("Cleaning up Ray workers...")
            if hasattr(evaluator, 'actor_rollout_wg'):
                evaluator.actor_rollout_wg.shutdown()
        except Exception as cleanup_e:
            print(f"Cleanup warning: {cleanup_e}")
    
    # Save results
    output_dir = config.trainer.get('default_local_dir', 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    return metrics


def vanilla_evaluation(config):
    """
    Pure vanilla evaluation mode with VERL framework but no KG integration.
    
    This mode:
    - Uses VERL for model loading and generation
    - Applies vanilla prompt augmentation
    - No KG server, no special formatting
    - Standard NLP metrics only
    """
    print("Starting vanilla evaluation mode")
    
    if not ray.is_initialized():
        ray_init_kwargs = {
            "runtime_env": {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
        }
        if hasattr(config, 'ray_init'):
            if hasattr(config.ray_init, 'num_cpus') and config.ray_init.num_cpus is not None:
                ray_init_kwargs["num_cpus"] = config.ray_init.num_cpus
            if hasattr(config.ray_init, 'timeline_json_file') and config.ray_init.timeline_json_file is not None:
                ray_init_kwargs["timeline_json_file"] = config.ray_init.timeline_json_file
        
        ray.init(**ray_init_kwargs)
    
    try:
        # Use efficient vanilla evaluator with VERL's batched generation infrastructure
        # This follows the same setup as kg_evaluation but uses vanilla mode
        from verl.trainer.ppo.ray_evaluator_vanilla import RayVanillaEvaluator
        
        # Import the same helper functions as kg_evaluation
        from transformers import AutoTokenizer
        
        # Create tokenizer (same as kg_evaluation)
        tokenizer = AutoTokenizer.from_pretrained(
            config.actor_rollout_ref.model.path,
            trust_remote_code=config.actor_rollout_ref.model.get('trust_remote_code', True)
        )
        
        # Use same worker setup as kg_evaluation but simpler
        processor = None  # Not needed for text-only
        
        # Define worker classes (same as kg_evaluation)
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker
            
            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError("Only FSDP strategy supported for vanilla evaluation")
        
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }
        
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        
        # Create efficient vanilla evaluator
        evaluator = RayVanillaEvaluator(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            n_rollout_eval=config.get('n_rollout_eval', 4),
            k_values=config.get('k_values', [1, 2, 3, 4]),
            eval_samples=config.get('eval_samples', 0)
        )
        
        # Initialize workers first (creates worker groups) - same as kg_evaluation
        evaluator.init_workers()
        
        # Set global_steps = 0 (same as kg_evaluation)
        evaluator.global_steps = 0
        
        # Run evaluation using the same pattern as kg_evaluation
        # The dataloader is created internally by the parent class
        metrics = evaluator.evaluate_dataset("test")
        
        print(f"\nVanilla evaluation completed!")
        
    except Exception as e:
        print(f"Vanilla evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean shutdown (same as kg_evaluation)
        try:
            print("Cleaning up Ray workers...")
            if hasattr(evaluator, 'actor_rollout_wg'):
                evaluator.actor_rollout_wg.shutdown()
        except Exception as cleanup_e:
            print(f"Cleanup warning: {cleanup_e}")
    
    # Save results
    output_dir = config.trainer.get('default_local_dir', 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine dataset name
    data_files = config.data.get('val_files', 'test')
    if 'cwq' in str(data_files).lower():
        dataset_name = 'cwq'
    elif 'webqsp' in str(data_files).lower():
        dataset_name = 'webqsp'
    else:
        dataset_name = 'test'
    
    results_file = os.path.join(output_dir, f'{dataset_name}_passatk_results.json')
    import json
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    
    print(f"\nResults saved to: {results_file}")
    
    return metrics


@hydra.main(config_path="config", config_name="evaluation_k_beam", version_base=None)
def main(config):
    """
    Main evaluation entry point with mode selection.
    """
    # Check evaluation mode
    eval_mode = config.get('mode', 'offline')
    
    if eval_mode == 'kg-search':
        print("Running KG Search evaluation mode")
        return kg_evaluation(config)
    elif eval_mode == 'kg-search-llm-judge':
        print("Running KG Search with LLM Judge evaluation mode")
        return kg_llm_judge_evaluation(config)
    elif eval_mode == 'vanilla':
        print("Running Vanilla evaluation mode")
        return vanilla_evaluation(config)
    elif eval_mode == 'offline':
        print("Running offline evaluation mode")
        return offline_evaluation(config)
    else:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}. Supported modes: 'kg-search', 'kg-search-llm-judge', 'vanilla', 'offline'")


if __name__ == "__main__":
    main()
