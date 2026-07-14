#!/usr/bin/env bash
set -e

PROJECT_NAME="agent_vlagent"
EXPERIMENT_NAME="debug_4gpu_qwen3vl_upgraded_env"

# ========== 激活 deepeyes 的克隆环境（只升级了vllm+transformers，其他不变）==========
source /home/zhangyt/miniconda3/etc/profile.d/conda.sh
conda activate /data/zhangyt/conda_envs/deepeyes_qwen3

export SAVE_CHECKPOINT_DIR=/data/zhangyt/verl_checkpoints
export WORLD_SIZE=1
# 换成4卡：3、5、6、7号（这几张卡显存空闲量够，但GPU利用率较高，正被别人用来训练）
# 用前务必先 nvidia-smi 确认这几张卡的显存/利用率情况，若变化记得改这里
export CUDA_VISIBLE_DEVICES=3,5,6,7

export WANDB_MODE=online
export WANDB_PROJECT=${PROJECT_NAME}
export WANDB_NAME=${EXPERIMENT_NAME}
export LLM_AS_A_JUDGE_BASE=${LLM_AS_A_JUDGE_BASE:-http://127.0.0.1:18901/v1}

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

unset PYTORCH_CUDA_ALLOC_CONF

export RAY_TMPDIR=/data/zhangyt/ray_tmp
export TMPDIR=/data/zhangyt/tmp
export HF_HOME=/data/zhangyt/hf_home
export HF_DATASETS_CACHE=/data/zhangyt/hf_datasets_cache

BASEDIR=/data/zhangyt/deepeyes_data

VISUAL_DATASET_TRAIN_0_1_2=${BASEDIR}/data_0.1.2_visual_toolbox_v2.parquet
VISUAL_DATASET_TRAIN_0_8=${BASEDIR}/data_v0.8_visual_toolbox_v2.parquet
EUREKA_DATASET_TRAIN=${BASEDIR}/data_thinklite_reasoning_acc.parquet

REF_MODEL_PATH=/home/zhangyt/models/Qwen3-VL-8B-Instruct

mkdir -p ./logs
mkdir -p ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}
mkdir -p ${SAVE_CHECKPOINT_DIR}/logs/tensorboard
mkdir -p ${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board
mkdir -p ${SAVE_CHECKPOINT_DIR}/logs/wandb
mkdir -p ${RAY_TMPDIR}
mkdir -p ${TMPDIR}
mkdir -p ${HF_HOME}
mkdir -p ${HF_DATASETS_CACHE}

export WANDB_DIR=${SAVE_CHECKPOINT_DIR}/logs/wandb

echo "Active conda env: $(python3 -c 'import sys; print(sys.executable)')"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

nvidia-smi -i 3,5,6,7

python3 - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"visible cuda:{i} name =", torch.cuda.get_device_name(i))

import flash_attn
print("flash_attn version =", flash_attn.__version__)
import vllm
print("vllm version =", vllm.__version__)
import transformers
print("transformers version =", transformers.__version__)
PY

for f in \
    ${VISUAL_DATASET_TRAIN_0_1_2} \
    ${VISUAL_DATASET_TRAIN_0_8} \
    ${EUREKA_DATASET_TRAIN}
do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing data file: $f"
        exit 1
    fi
done

if [ ! -d "${REF_MODEL_PATH}" ]; then
    echo "[ERROR] Missing model dir: ${REF_MODEL_PATH}"
    exit 1
fi

curl --fail --silent --show-error --max-time 10 ${LLM_AS_A_JUDGE_BASE}/models > /dev/null

python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    "data.train_files=[${VISUAL_DATASET_TRAIN_0_1_2},${VISUAL_DATASET_TRAIN_0_8},${EUREKA_DATASET_TRAIN}]" \
    "data.val_files=[${EUREKA_DATASET_TRAIN}]" \
    data.train_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    "actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra']" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=2560 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2560 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=512 \
    actor_rollout_ref.rollout.agent.max_turns=4 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','wandb','rl_logging_board']" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10000 \
    trainer.test_freq=10000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=1 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log

