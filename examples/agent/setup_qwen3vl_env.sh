#!/usr/bin/env bash
set -e
set -o pipefail
trap 'echo "[FATAL] 脚本在第 ${LINENO} 行失败，命令: ${BASH_COMMAND}" >&2' ERR

# ============================================================
# 参考: https://github.com/volcengine/verl/issues/3906
# 这是目前verl官方唯一验证过能训练 Qwen3-VL 系列的组合：
#   vllm==0.11.0 + transformers>=4.57.0 + Megatron-LM(core_v0.13.1) + mbridge
# 用的是 Megatron 后端，不是你 DeepEyes 脚本里用的 FSDP。
#
# 这个脚本只负责把"官方验证过的环境"装出来，装在一个全新的、独立的
# conda env + 独立的verl官方clone目录里，完全不碰你现在能跑的
# deepeyes env 和 /data/zhangyt/DeepEyes 代码。
#
# 装完之后建议的验证顺序：
#   1. 先跑 verl 官方自带的 examples/grpo_trainer/run_qwen3_vl-30b-megatron.sh
#      冒烟测试（可能要把里面30B MoE的模型路径/并行度改成你的8B dense模型），
#      确认这个环境本身能跑通 Qwen3-VL。
#   2. 环境验证没问题后，再考虑DeepEyes那层agent/工具调用rollout代码要怎么
#      迁移过来——这一步官方没有文档，大概率需要手动排查适配。
# ============================================================

ENV_NAME=qwen3vl
ENV_PATH=/data/zhangyt/conda_envs/${ENV_NAME}
VERL_OFFICIAL_DIR=/data/zhangyt/verl_official
PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

# 避免重演之前 /home 分区被pip缓存、编译产物挤满的问题，
# 统一把缓存/临时目录指向空间最富余的 /data 分区
export PIP_CACHE_DIR=/data/zhangyt/pip_cache
export TMPDIR=/data/zhangyt/tmp
mkdir -p ${PIP_CACHE_DIR}
mkdir -p ${TMPDIR}

echo "[STEP 0] 创建独立conda环境: ${ENV_PATH}"
conda create -y -p ${ENV_PATH} python=3.10

source /home/zhangyt/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}

echo "[STEP 1] 安装 uv（比pip快，且能按驱动自动选CUDA变体）"
pip install uv -i ${PIP_MIRROR}

echo "[STEP 2] 安装 torch 2.8.0 (cu128，匹配你机器12.8驱动)"
export UV_INDEX_URL=${PIP_MIRROR}
export UV_EXTRA_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128
uv pip install --python "$(which python3)" torch==2.8.0 --torch-backend=cu128

echo "[STEP 3] 安装 vllm==0.11.0（支持 qwen3_vl rollout 的最低版本）"
pip install vllm==0.11.0 -i ${PIP_MIRROR}

echo "[STEP 4] 安装 transformers>=4.57.0（支持 qwen3_vl 架构识别）"
pip install 'transformers>=4.57.0' -i ${PIP_MIRROR}

echo "[STEP 5] 安装 flash-attn（源码编译，可能要 20-40 分钟，耐心等）"
pip install flash-attn --no-build-isolation -i ${PIP_MIRROR}

echo "[STEP 6] 安装 mbridge（Megatron <-> HF 权重转换桥接）"
pip install -U "git+https://github.com/ISEEKYAN/mbridge.git"

echo "[STEP 7] 安装 Megatron-LM (mcore, 官方issue里验证过的commit: core_v0.13.1)"
pip install --no-deps --no-cache-dir "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1"

echo "[STEP 8] clone verl 官方仓库到独立目录，不覆盖你的 DeepEyes 代码"
if [ ! -d "${VERL_OFFICIAL_DIR}" ]; then
    git clone https://github.com/volcengine/verl.git ${VERL_OFFICIAL_DIR}
fi
cd ${VERL_OFFICIAL_DIR}
git fetch origin
git checkout 2c9e19e   # issue #3906 作者本人测试过能跑通的commit
pip install --no-deps -e .

echo ""
echo "[DONE] qwen3vl 环境安装完成"
echo "  - conda环境路径: ${ENV_PATH}"
echo "  - verl官方代码路径: ${VERL_OFFICIAL_DIR}"
echo "  - 官方Qwen3-VL示例脚本: ${VERL_OFFICIAL_DIR}/examples/grpo_trainer/run_qwen3_vl-30b-megatron.sh"
echo ""
echo "下一步建议: 先用下面命令确认关键包版本都对："
echo "  conda run -p ${ENV_PATH} python -c \"import torch, vllm, transformers, megatron; print('torch', torch.__version__); print('vllm', vllm.__version__); print('transformers', transformers.__version__)\""

