#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='data_kg'
# BATCH SIZE OPTIMIZED VERSION - increase VLLM batch capacity for 20-30% speedup
# Fix expandable segments compatibility with memory pool
WAND_PROJECT='KG-R1-main'

# Disable VLLM usage stats to avoid permission errors
export VLLM_USAGE_SOURCE=do-not-track

rm -rf ~/.cache/torch/triton/

export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=cwq-KG-r1-grpo-qwen2.5-3b-it_f1_turn7
export RAY_LOG_DIR=".RAY_DEBUG/${EXPERIMENT_NAME}_RAY_DEBUG"
if [ "${RAY_DEBUG:-0}" -eq 1 ]; then
    mkdir -p "$RAY_LOG_DIR" # Create the directory if it doesn't exist
fi

#export RAY_DEBUG=1
#export RAY_LOG_TO_STDERR=1
#export RAY_DISABLE_IMPORT_WARNING=1

# Fix VLLM configuration issues
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
export HYDRA_FULL_ERROR=1

export MAX_LENGTH=4500

# Regex optimizations now applied directly in generation.py source code

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +trainer.mode=kg-search \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.project_name=$WAND_PROJECT \
    data.train_files=$DATA_DIR/cwq_search_augmented_initial_entities/train.parquet \
    data.val_files=$DATA_DIR/cwq_search_augmented_initial_entities/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=$MAX_LENGTH \
    data.max_response_length=256 \
    data.max_obs_length=512 \
    data.shuffle=True \
    data.trust_remote_code=true \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true\
    +algorithm.enable_multiturn_advantage=true \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=k3 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    +trainer.use_ref_model=true \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=21000 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=130000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_batched_tokens=18384 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.disable_log_stats=false \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.search.enable=true \
    actor_rollout_ref.rollout.search.enable_during_training=true \
    actor_rollout_ref.rollout.search.enable_during_validation=true \
    actor_rollout_ref.rollout.search.search_url="http://127.0.0.1:8001/retrieve" \
    actor_rollout_ref.rollout.search.max_turns=7 \
    actor_rollout_ref.rollout.search.topk=3 \
    actor_rollout_ref.rollout.search.timeout=3 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=130000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.rollout.grpo_rollout_n=16 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=20 \
    trainer.total_training_steps=400 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    +data.prompt_augmentation.enable=true \
    +data.prompt_augmentation.guideline_level=detailed_flat \
    +data.prompt_augmentation.hint_steps=500 \
    algorithm.kg_token_masking.enable=false \
    algorithm.kg_token_masking.reduction_factor=0 \
    algorithm.kg_token_masking.patterns='["<kg-query>", "</kg-query>", "<search>", "</search>", "<think>", "</think>", "get_tail_relations", "get_head_relations", "get_tail_entities", "get_head_entities", "get_conditional_relations"]' \
    algorithm.kg_token_masking.debug_logging=false \
    reward_model.enable=false \
    reward_model.reward_manager=kg_format_multiturn \
    +reward_model.reward_kwargs.turn_kg_query_validity=0.5 \
    +reward_model.reward_kwargs.turn_is_answer_score=0.5 \
    +reward_model.reward_kwargs.turn_format_score=0.5 \
    +reward_model.reward_kwargs.global_exact_match=0.5 \
    +reward_model.reward_kwargs.global_retrieval_quality=0.5 \
    +reward_model.reward_kwargs.kg_server_error_penalty=0 \
    +reward_model.reward_kwargs.kg_not_found_penalty=0 \
    +reward_model.reward_kwargs.kg_format_error_penalty=0 \
    +reward_model.reward_kwargs.kg_no_data_penalty=0 \
    +reward_model.reward_kwargs.verbose=false \
    +reward_model.reward_kwargs.debug_long_responses=true \
    +reward_model.reward_kwargs.response_length_threshold=4400 \
    +reward_model.reward_kwargs.debug_log_dir=${EXPERIMENT_NAME}_debug.log \
    +reward_model.reward_kwargs.answer_score_mode=f1 \
    reward_model.reward_kwargs.otc_scaling=false \
    2>&1 | tee $EXPERIMENT_NAME.log
