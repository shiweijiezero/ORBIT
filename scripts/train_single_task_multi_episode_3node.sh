#!/bin/bash
set -x

# 3-node, 24-GPU (8 GPUs per node) multi-episode training script for Alibaba Cloud DLC
# Usage:
#   Submit as a DLC distributed Ray job with 3 nodes (1 head + 2 workers).
#   DLC automatically sets MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK on each node.
#
#   Enable reflection (already enabled by default, override with):
#     ... +rllm.env.env_args.enable_reflection=False

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1

NNODES=3
N_GPUS_PER_NODE=8
RAY_PORT=${RAY_PORT:-6379}

# ── Ray cluster setup for DLC ───────────────────────────────────────────────
# DLC provides: MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK
# We use RANK to decide head vs worker node.
HEAD_ADDR="${MASTER_ADDR}:${RAY_PORT}"

if [ "${RANK}" == "0" ]; then
    echo "Starting Ray head node on $(hostname)"
    ray start --head --port=${RAY_PORT} \
        --num-gpus=${N_GPUS_PER_NODE} \
        --block &
    sleep 10
else
    echo "Starting Ray worker node on $(hostname), connecting to head at ${HEAD_ADDR}"
    sleep 15  # wait for head to be ready
    ray start --address=${HEAD_ADDR} \
        --num-gpus=${N_GPUS_PER_NODE} \
        --block &
    sleep 10
fi

# Wait for all nodes to join the Ray cluster
if [ "${RANK}" == "0" ]; then
    echo "Waiting for Ray cluster to have ${NNODES} nodes..."
    for i in $(seq 1 60); do
        NODE_COUNT=$(python3 -c "import ray; ray.init(address='auto'); print(len(ray.nodes()))" 2>/dev/null)
        if [ "${NODE_COUNT}" -ge "${NNODES}" ] 2>/dev/null; then
            echo "Ray cluster ready with ${NODE_COUNT} nodes."
            break
        fi
        echo "Attempt ${i}: ${NODE_COUNT:-0}/${NNODES} nodes ready, waiting..."
        sleep 10
    done
fi

# Only the head node (rank 0) runs the training script
if [ "${RANK}" != "0" ]; then
    echo "Worker node ${RANK} standing by. Waiting for training to complete..."
    wait
    exit 0
fi

# ── Training config ─────────────────────────────────────────────────────────
ENV_ID=game:Minesweeper-v0-only-reveal
TOTAL_STEP_CAP=21
MAX_TURNS_PER_EPISODE=7
MODEL_PATH=Qwen/Qwen3-1.7B

# Extract model name (last part after /)
MODEL_NAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')
# Extract env name (part after :, convert to lowercase with hyphens)
ENV_NAME=$(echo "$ENV_ID" | cut -d: -f2 | tr '[:upper:]' '[:lower:]' | tr '_' '-')
# Construct experiment name
EXPERIMENT_NAME="gem-${ENV_NAME}-multi-episode-env-${MODEL_NAME}-3node"

# ── Launch training (head node only) ────────────────────────────────────────
python scripts/train_multi_episode.py \
    data.train_batch_size=192 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    +rllm.env.env_args.inner_env_class=envs.gem_env_adapter.GEMEnvAdapter \
    +rllm.env.env_args.inner_env_kwargs.env_id=$ENV_ID \
    +rllm.env.env_args.inner_env_kwargs.env_kwargs.max_turns=$MAX_TURNS_PER_EPISODE \
    +rllm.env.env_args.total_step_cap=$TOTAL_STEP_CAP \
    +rllm.env.env_args.success_reward=1.0 \
    rllm.agent.max_steps=$TOTAL_STEP_CAP \
    +rllm.env.env_args.episode_header="New episode begins." \
    +rllm.env.env_args.enable_reflection=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ppo_mini_batch_size=384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    rllm.compact_filtering.enable=False \
    rllm.compact_filtering.mask_max_prompt_length_exceeded=True \
    rllm.compact_filtering.mask_max_response_length_exceeded=True \
    rllm.compact_filtering.mask_max_turns_exceeded=False \
    rllm.compact_filtering.mask_timeout=True \
    rllm.rejection_sample.enable=False \
    rllm.rejection_sample.multiplier=1.0 \
    rllm.stepwise_advantage.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rllm-agent' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=10 \
    "$@"
