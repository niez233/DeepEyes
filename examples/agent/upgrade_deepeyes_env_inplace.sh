#!/usr/bin/env bash
set -e
set -o pipefail
trap 'echo "[FATAL] 第 ${LINENO} 行失败: ${BASH_COMMAND}" >&2' ERR

# ============================================================
# 直接测试思路：deepeyes环境原地复制一份，只升级vllm+transformers，
# FSDP后端和DeepEyes的agent rollout代码完全不动，
# 直接拿真实训练脚本+Qwen3-VL模型跑一次，看具体卡在哪层。
# ============================================================

SRC_ENV=/data/zhangyt/conda_envs/deepeyes
NEW_ENV=/data/zhangyt/conda_envs/deepeyes_qwen3
PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple

export PIP_CACHE_DIR=/data/zhangyt/pip_cache
export TMPDIR=/data/zhangyt/tmp
mkdir -p ${PIP_CACHE_DIR} ${TMPDIR}

if [ ! -d "${NEW_ENV}" ]; then
    echo "[STEP 1] 复制 deepeyes 环境到 ${NEW_ENV}（原环境不受影响，需要几分钟）..."
    conda create -y -p ${NEW_ENV} --clone ${SRC_ENV}
else
    echo "[STEP 1] ${NEW_ENV} 已存在，跳过复制"
fi

source /home/zhangyt/miniconda3/etc/profile.d/conda.sh
conda activate ${NEW_ENV}

echo "[STEP 2] 只升级 vllm 和 transformers 这两个包..."
pip install -U 'transformers>=4.57.0' 'vllm>=0.11.0' -i ${PIP_MIRROR}

echo ""
echo "===== [诊断] 升级后的关键库版本 ====="
python3 - <<'PY'
import verl
print("verl.__file__            =", verl.__file__, "  （应仍是DeepEyes自己的代码，环境是克隆的，路径没变）")

import torch
print("torch.__version__        =", torch.__version__)

import vllm
print("vllm.__version__         =", vllm.__version__)

import transformers
print("transformers.__version__ =", transformers.__version__)

try:
    import flash_attn
    print("flash_attn.__version__   =", flash_attn.__version__)
except Exception as e:
    print("[WARN] flash_attn import 失败:", repr(e))
    print("       如果torch版本被vllm升级顺带改了，flash_attn可能要重新编译，先看看报错信息")
PY
echo "===== [诊断结束] ====="
echo ""
echo "如果上面torch版本变了（比如从2.6.0变成别的），flash_attn大概率要重新装："
echo "  conda run -p ${NEW_ENV} pip install flash-attn --no-build-isolation -i ${PIP_MIRROR}"
echo ""
echo "确认以上都正常之后，用下面命令跑真正的训练（跟原来deepeyes脚本几乎一样，"
echo "只是conda环境换成了这个升级过的 ${NEW_ENV}）："
echo ""
echo "  source /home/zhangyt/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate ${NEW_ENV}"
echo "  bash /home/zhangyt/DeepEyes/examples/agent/train_2gpu_qwen3vl_experimental.sh"
echo ""
echo "注意：那个脚本里有一行 export PYTHONPATH=/data/zhangyt/DeepEyes:... 和"
echo "conda activate /data/zhangyt/conda_envs/qwen3vl，这两处需要先手动改成"
echo "conda activate ${NEW_ENV}，并删掉PYTHONPATH那行（这次不需要了，因为"
echo "环境本身就是DeepEyes的克隆，verl代码路径没变）。"

