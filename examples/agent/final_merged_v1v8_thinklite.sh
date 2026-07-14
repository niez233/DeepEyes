#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="agent_vlagent"
EXPERIMENT_NAME="qwen3vl_4gpu"

# ========================================================================
# SMOKE=1 -> 冒烟测试：小 batch 快速跑十几步，只为确认 bbox 修复是否生效
# SMOKE=0 -> 正式训练（默认）
#   用法：SMOKE=1 bash run_4gpu.sh
# ========================================================================
SMOKE="${SMOKE:-0}"

# ========== 激活预装好 flash_attn / vllm / verl 的 conda 环境 ==========
source /home/zhangyt/miniconda3/etc/profile.d/conda.sh
conda activate /data/zhangyt/conda_envs/deepeyes_qwen3

export SAVE_CHECKPOINT_DIR=/data/zhangyt/verl_checkpoints
export WORLD_SIZE=1

# 使用物理 GPU 0、1、2、4。
# 在当前进程内部，它们会映射为 cuda:0、cuda:1、cuda:2、cuda:3。
export CUDA_VISIBLE_DEVICES=0,1,2,4

# ========================================================================
# 【关键】bbox 坐标空间
#   qwen3 : Qwen3-VL，模型输出 0-1000 归一化坐标（本次训练用这个）
#   pixel : Qwen2.5-VL，模型输出原图像素坐标（切回旧模型时改这里）
# 对应 visual_toolbox_v2.py 里的 BBOX_COORD_SPACE
# ========================================================================
export BBOX_COORD_SPACE=qwen3

export WANDB_MODE=online
export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_NAME="${EXPERIMENT_NAME}"

# Judge 使用另行启动的 Qwen3-14B OpenAI 兼容接口。
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-http://127.0.0.1:8000/v1}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

unset PYTORCH_CUDA_ALLOC_CONF
export VLLM_USE_FLASHINFER_SAMPLER=0

export RAY_TMPDIR=/data/zhangyt/ray_tmp
export TMPDIR=/data/zhangyt/tmp

# 将 Hugging Face 缓存放到 /data，避免 /home 空间不足。
export HF_HOME=/data/zhangyt/hf_home
export HF_DATASETS_CACHE=/data/zhangyt/hf_datasets_cache

BASEDIR=/data/zhangyt/deepeyes_data

VISUAL_DATASET_TRAIN_0_1_2=${BASEDIR}/data_0.1.2_visual_toolbox_v2.parquet
VISUAL_DATASET_TRAIN_0_8=${BASEDIR}/data_v0.8_visual_toolbox_v2.parquet
EUREKA_DATASET_TRAIN=${BASEDIR}/data_thinklite_reasoning_acc.parquet

REF_MODEL_PATH=/home/zhangyt/models/Qwen3-VL-8B-Instruct

# ========================================================================
# 训练超参
#
# 官方 8/16 卡配置 -> 4 卡的取舍原则：
#   显存不够时，优先砍 micro_batch（靠梯度累积换时间，数学上等价，不损精度），
#   而不是砍 rollout.n / train_batch_size / 序列长度（这些直接损失训练信号）。
#
#   rollout.n=16        : GRPO 靠组内相对优势估计，n 太小 advantage 方差爆炸。最不能砍。
#   train_batch_size=64 : 官方 256，砍 4 倍，lr 相应从 1e-6 降到 5e-7。
#   ppo_mini_batch=64   : 保持 == train_batch_size，维持完全 on-policy（与官方一致）。
#   micro_batch=1       : 显存全靠这个扛，只影响速度，不影响梯度。
#   TP=1                : 8B 模型 bf16 约 16GB，A100-80G 单卡装得下。
#                         TP=2 会让 vLLM 副本从 4 个变 2 个，rollout 吞吐直接腰斩，
#                         而且没有任何精度收益。这是本次改动性价比最高的一条。
#
#   【2026-07-13 调整】原来 MAX_RESPONSE_LEN=4096 / SINGLE_RESP_MAX_TOKENS=2048
#   跑出来 format_reward 大面积 -1（模型在多轮 zoom-in + 思考里还没写到
#   <answer> 就被截断）。官方 16 卡配置给的是 20480 / 10240，是我们的 5 倍。
#   4 卡显存比官方紧张很多，不直接照抄官方数值，先按比例翻一倍试水：
#     MAX_RESPONSE_LEN        4096  -> 8192
#     SINGLE_RESP_MAX_TOKENS  2048  -> 4096
#     MAX_MODEL_LEN          16384  -> 24576  (必须联动加大，否则
#                                      prompt(8192)+response(8192) 会顶到
#                                      甚至超过 max_model_len，等于白改)
#   如果这次跑完 nvidia-smi / vLLM 日志没有 OOM，且 format_reward=-1 的比例
#   明显下降，再考虑进一步往官方数值靠；如果 OOM，优先调小
#   GPU_MEM_UTIL 或退回 SINGLE_RESP_MAX_TOKENS，不要先动 rollout.n。
#
#   【另一个独立线索，务必留意】即便这次 budget 给够了，如果 format_reward
#   仍然大面积 -1，很可能瓶颈根本不在 token 预算，而是 MAX_TURNS=5 太少——
#   模型（尤其 Qwen2.5-VL-7B-Instruct 这种没见过该协议的原始 checkpoint）
#   反复 zoom-in 同一区域、5 轮工具调用耗尽还没学会收尾给 <answer>。
#   建议跑完之后重点看 batch 里的 tool_cnt 字段分布：如果大量样本
#   tool_cnt 卡在 MAX_TURNS 顶格，说明是轮次上限的问题，跟 token 预算无关，
#   这时候把 MAX_TURNS 临时调小（比如 3）逼模型早点收尾，比继续加大
#   response 长度更有效。MAX_TURNS 已改成可用环境变量覆盖，方便你不改脚本
#   直接测试，例如：MAX_TURNS=3 bash run_4gpu.sh
# ========================================================================
if [ "${SMOKE}" = "1" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_smoke"
    export WANDB_NAME="${EXPERIMENT_NAME}"
    TRAIN_BSZ=4
    MINI_BSZ=4
    ROLLOUT_N=4
    TOTAL_EPOCHS=1
    SAVE_FREQ=10000
else
    TRAIN_BSZ=64
    MINI_BSZ=64
    ROLLOUT_N=16
    TOTAL_EPOCHS=1
    SAVE_FREQ=50
fi

MAX_RESPONSE_LEN=8192
SINGLE_RESP_MAX_TOKENS=4096
MAX_MODEL_LEN=24576
MAX_TURNS="${MAX_TURNS:-5}"
LR=5e-7
GPU_MEM_UTIL=0.7

mkdir -p ./logs
mkdir -p "${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${SAVE_CHECKPOINT_DIR}/logs/tensorboard"
mkdir -p "${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board"
mkdir -p "${SAVE_CHECKPOINT_DIR}/logs/wandb"
mkdir -p "${RAY_TMPDIR}"
mkdir -p "${TMPDIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_DATASETS_CACHE}"

export WANDB_DIR=${SAVE_CHECKPOINT_DIR}/logs/wandb

echo "===================================================="
echo "SMOKE                   = ${SMOKE}"
echo "EXPERIMENT_NAME         = ${EXPERIMENT_NAME}"
echo "BBOX_COORD_SPACE        = ${BBOX_COORD_SPACE}"
echo "CUDA_VISIBLE_DEVICES    = ${CUDA_VISIBLE_DEVICES}"
echo "train_batch_size        = ${TRAIN_BSZ}"
echo "rollout.n                = ${ROLLOUT_N}"
echo "max_response_length     = ${MAX_RESPONSE_LEN}"
echo "single_response_max_tok = ${SINGLE_RESP_MAX_TOKENS}"
echo "max_model_len           = ${MAX_MODEL_LEN}"
echo "max_turns               = ${MAX_TURNS}"
echo "Active conda env        = $(python3 -c 'import sys; print(sys.executable)')"
echo "===================================================="

# ---------- 磁盘检查：ray_tmp 之前一直卡在 95%+，满了 Ray 会直接崩 ----------
RAY_DISK_USE=$(df --output=pcent "${RAY_TMPDIR}" | tail -1 | tr -dc '0-9')
echo "RAY_TMPDIR disk usage: ${RAY_DISK_USE}%"
if [ "${RAY_DISK_USE}" -ge 90 ]; then
    echo "[WARN] ${RAY_TMPDIR} 所在磁盘已用 ${RAY_DISK_USE}%，object spilling 可能失败导致训练崩溃。"
    echo "[WARN] 建议先清理旧 session:  du -sh ${RAY_TMPDIR}/ray/session_* | sort -h"
    echo "[WARN] 5 秒后继续，Ctrl-C 可中止..."
    sleep 5
fi

nvidia-smi -i 0,1,2,4

python3 - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("BBOX_COORD_SPACE =", os.environ.get("BBOX_COORD_SPACE"))
print("torch.cuda.device_count =", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

if torch.cuda.device_count() != 4:
    raise RuntimeError(
        f"Expected 4 visible GPUs, but found {torch.cuda.device_count()}"
    )

for i in range(torch.cuda.device_count()):
    print(f"visible cuda:{i} name =", torch.cuda.get_device_name(i))

import flash_attn
print("flash_attn version =", flash_attn.__version__)
PY

for f in \
    "${VISUAL_DATASET_TRAIN_0_1_2}" \
    "${VISUAL_DATASET_TRAIN_0_8}" \
    "${EUREKA_DATASET_TRAIN}"
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

echo "Checking judge service: ${LLM_AS_A_JUDGE_BASE}/models"

if ! curl \
    --fail \
    --silent \
    --show-error \
    --max-time 10 \
    "${LLM_AS_A_JUDGE_BASE}/models" \
    > /tmp/judge_models.json
then
    echo "[ERROR] Judge service is unavailable: ${LLM_AS_A_JUDGE_BASE}"
    echo "[ERROR] Please start the Qwen3-14B judge service first."
    exit 1
fi

echo "Judge service is ready:"
cat /tmp/judge_models.json
echo

python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    "data.train_files=[${VISUAL_DATASET_TRAIN_0_1_2},${VISUAL_DATASET_TRAIN_0_8},${EUREKA_DATASET_TRAIN}]" \
    "data.val_files=[${EUREKA_DATASET_TRAIN}]" \
    data.train_batch_size=${TRAIN_BSZ} \
    data.max_prompt_length=8192 \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${REF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    "actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra']" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=${SINGLE_RESP_MAX_TOKENS} \
    actor_rollout_ref.rollout.agent.max_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','wandb','rl_logging_board']" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=10000 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    +trainer.tensorboard_dir="${SAVE_CHECKPOINT_DIR}/logs/tensorboard" \
    +trainer.rl_logging_board_dir="${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    2>&1 | tee "./logs/${EXPERIMENT_NAME}.log"

