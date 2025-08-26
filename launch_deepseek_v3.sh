#!/bin/bash
set -euxo pipefail

# Direct run script for DeepSeek V3
# This script contains hardcoded values from model_configs/benchmarking/DeepSeek-V3.yaml
# and runtime_configs/benchmarking/runtime.conf and common.conf

# Default values (can be overridden by command line args)
NNODES=1
NODE_RANK=0
NPROC_PER_NODE=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=6000
HOME_DIR="/home/$USER"
DISABLE_WANDB=true

# Parse command-line args
for arg in "$@"; do
  case $arg in
    --nnodes=*)
      NNODES="${arg#*=}"
      shift
      ;;
    --node_rank=*)
      NODE_RANK="${arg#*=}"
      shift
      ;;
    --nproc_per_node=*)
      NPROC_PER_NODE="${arg#*=}"
      shift
      ;;
    --master_addr=*)
      MASTER_ADDR="${arg#*=}"
      shift
      ;;
    --master_port=*)
      MASTER_PORT="${arg#*=}"
      shift
      ;;
    --home=*)
      HOME_DIR="${arg#*=}"
      shift
      ;;
    --mast)
      DISABLE_WANDB=true
      shift
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

# If running under SLURM, auto-fill values if not explicitly set
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  NNODES=${SLURM_NNODES:-$NNODES}
  NODE_RANK=${SLURM_NODEID:-$NODE_RANK}
  if [[ "$MASTER_ADDR" == "127.0.0.1" ]]; then
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
  fi
fi


# Basic configuration
export MODEL="DeepSeek-V3"
# export DATASET="slimpajama_15k"
export DATASET="wikitext"
export WORKSPACE=$(dirname "$(readlink -f "$0")")
export CLUSTER="todoMast"
export MCORE_RELEASE_VERSION="0.14"
export MEGATRON_PATH=$HOME_DIR/"Megatron-LM"


# export DATA_PATH=$HOME_DIR/"datasets/slimpajama_15k/slimpajama_15k_text_document"
export DATA_PATH=$HOME_DIR/"datasets/wikitext/wikitext_text_document"
export TOKENIZER_TYPE="HuggingFaceTokenizer"
export TOKENIZER_MODEL="$HOME_DIR/models/deepseek-v3/tokenizer"   # if load from local
# export TOKENIZER_MODEL="deepseek-ai/DeepSeek-V3"    # if pulling from hugging face

# Parallelism configurations (from runtime.conf DeepSeek-V3 config)
export TP=1
export PP=16
export EP=64
export CP=1
export VPP=1

export PP_FIRST=4
export PP_LAST=1
# Batch size configurations
export MBS=1
export GBS=8192

# Model architecture configurations
export NUM_LAYERS=61

# # MoE configurations
# export MOE_TOKEN_DISPATCHER="alltoall"
# export MOE_GROUPED_GEMM="true"

# MoE configurations
export MOE_TOKEN_DISPATCHER="alltoall"  # [flex, alltoall, allgather]
export MOE_GROUPED_GEMM="true"
export MOE_ENABLE_DEEPEP="true"  # requires flex dispatcher
export MOE_DEEPEP_NUM_SMS=32  # Number of SMs to use for DeepEP


if [[ "${MOE_ENABLE_DEEPEP}" == "true" ]]; then
    MOE_TOKEN_DISPATCHER="flex"
else
    MOE_TOKEN_DISPATCHER="flex"
fi


# Training configurations
# export NNODES=1
export RUN_TIME="00:20:00"
export PRETRAIN=0

# Data configurations
export SEQ_LEN=4096

# Common configurations
export TRAINING_SCRIPT_PATH="${MEGATRON_PATH}/pretrain_gpt.py"
export COMMENT="v${MCORE_RELEASE_VERSION}"
export WANDB_PROJECT="${USER}-moe-benchmarking-v${MCORE_RELEASE_VERSION}"
export PROFILE=0
export PR="bf16"

# Paths
export RUN_NAME="DeepSeek v3"
# export DATA_PATH="${DATA_PATH:-/home/less/datasets/slimpajama_15k/slimpajama_text_document}"

export OUTPUT_PATH="${OUTPUT_PATH:-${WORKSPACE}/outputs}"
export LOAD_PATH="${LOAD_PATH:-}"

# Environment variables (from DeepSeek-V3.yaml ENV_VARS)
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Remove deprecated TORCH_NCCL_AVOID_RECORD_STREAMS (now default)
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1

# Additional environment variables for overlapping
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0

# Build training parameters (from DeepSeek-V3.yaml MODEL_ARGS)
TRAINING_PARAMS="--distributed-timeout-minutes 60"
TRAINING_PARAMS+=" --tensor-model-parallel-size ${TP}"
TRAINING_PARAMS+=" --pipeline-model-parallel-size ${PP}"
TRAINING_PARAMS+=" --expert-model-parallel-size ${EP}"
TRAINING_PARAMS+=" --context-parallel-size ${CP}"
TRAINING_PARAMS+=" --expert-tensor-parallel-size 1"
TRAINING_PARAMS+=" --use-distributed-optimizer"
TRAINING_PARAMS+=" --use-mcore-models"
# Remove sequence-parallel since TP=1 (causes warning)
# TRAINING_PARAMS+=" --sequence-parallel"
TRAINING_PARAMS+=" --use-flash-attn"
TRAINING_PARAMS+=" --disable-bias-linear"
TRAINING_PARAMS+=" --micro-batch-size ${MBS}"
TRAINING_PARAMS+=" --global-batch-size ${GBS}"
TRAINING_PARAMS+=" --train-iters 100"
TRAINING_PARAMS+=" --exit-duration-in-mins 230"
TRAINING_PARAMS+=" --no-bias-swiglu-fusion"
TRAINING_PARAMS+=" --no-check-for-nan-in-loss-and-grad"
TRAINING_PARAMS+=" --no-rope-fusion"
TRAINING_PARAMS+=" --transformer-impl transformer_engine"
TRAINING_PARAMS+=" --seq-length ${SEQ_LEN}"
TRAINING_PARAMS+=" --data-cache-path ${WORKSPACE}/data_cache"
TRAINING_PARAMS+=" --tokenizer-type HuggingFaceTokenizer"
TRAINING_PARAMS+=" --tokenizer-model ${TOKENIZER_MODEL}"
TRAINING_PARAMS+=" --data-path ${DATA_PATH}"
TRAINING_PARAMS+=" --split 99,1,0"
TRAINING_PARAMS+=" --no-mmap-bin-files"
TRAINING_PARAMS+=" --no-create-attention-mask-in-dataloader"
TRAINING_PARAMS+=" --num-workers 6"
TRAINING_PARAMS+=" --num-layers ${NUM_LAYERS}"
TRAINING_PARAMS+=" --hidden-size 2048"
TRAINING_PARAMS+=" --ffn-hidden-size 10944"
TRAINING_PARAMS+=" --num-attention-heads 16"
TRAINING_PARAMS+=" --kv-channels 128"
TRAINING_PARAMS+=" --max-position-embeddings 4096"
TRAINING_PARAMS+=" --position-embedding-type rope"
TRAINING_PARAMS+=" --rotary-base 10000"
TRAINING_PARAMS+=" --make-vocab-size-divisible-by 3200"
TRAINING_PARAMS+=" --normalization RMSNorm"
TRAINING_PARAMS+=" --norm-epsilon 1e-6"
TRAINING_PARAMS+=" --swiglu"
TRAINING_PARAMS+=" --untie-embeddings-and-output-weights"
TRAINING_PARAMS+=" --multi-latent-attention"
TRAINING_PARAMS+=" --attention-dropout 0.0"
TRAINING_PARAMS+=" --hidden-dropout 0.0"
TRAINING_PARAMS+=" --clip-grad 1.0"
TRAINING_PARAMS+=" --weight-decay 0.1"
TRAINING_PARAMS+=" --qk-layernorm"
TRAINING_PARAMS+=" --lr-decay-iters 90"
TRAINING_PARAMS+=" --lr-warmup-iters 10"
TRAINING_PARAMS+=" --lr-warmup-init 1.3e-7"
TRAINING_PARAMS+=" --lr 1.3e-6"
TRAINING_PARAMS+=" --min-lr 1.3e-7"
TRAINING_PARAMS+=" --lr-decay-style cosine"
TRAINING_PARAMS+=" --adam-beta1 0.9"
TRAINING_PARAMS+=" --adam-beta2 0.95"
TRAINING_PARAMS+=" --num-experts 64"
TRAINING_PARAMS+=" --moe-layer-freq [0]+[1]*26"
TRAINING_PARAMS+=" --moe-ffn-hidden-size 1408"
TRAINING_PARAMS+=" --moe-shared-expert-intermediate-size 2816"
TRAINING_PARAMS+=" --moe-router-load-balancing-type seq_aux_loss"
TRAINING_PARAMS+=" --moe-router-topk 6"
TRAINING_PARAMS+=" --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER}"
TRAINING_PARAMS+=" --moe-router-pre-softmax"
# Add moe-grouped-gemm flag if enabled
if [[ "${MOE_GROUPED_GEMM}" == "true" ]]; then
    TRAINING_PARAMS+=" --moe-grouped-gemm"
fi
if [[ "${MOE_ENABLE_DEEPEP}" == "true" ]]; then
    TRAINING_PARAMS+=" --moe-enable-deepep"
    TRAINING_PARAMS+=" --moe-deepep-num-sms ${MOE_DEEPEP_NUM_SMS}"
    export MOE_TOKEN_DISPATCHER="flex"
fi
TRAINING_PARAMS+=" --moe-aux-loss-coeff 1e-3"
TRAINING_PARAMS+=" --moe-router-topk-scaling-factor 1.0"
TRAINING_PARAMS+=" --moe-router-dtype fp32"
TRAINING_PARAMS+=" --moe-permute-fusion"
TRAINING_PARAMS+=" --kv-lora-rank 512"
TRAINING_PARAMS+=" --qk-head-dim 128"
TRAINING_PARAMS+=" --qk-pos-emb-head-dim 64"
TRAINING_PARAMS+=" --v-head-dim 128"
TRAINING_PARAMS+=" --rotary-scaling-factor 40"
TRAINING_PARAMS+=" --mscale 0.707"
TRAINING_PARAMS+=" --mscale-all-dim 0.707"
TRAINING_PARAMS+=" --eval-iters 32"
TRAINING_PARAMS+=" --eval-interval 200"
TRAINING_PARAMS+=" --finetune"
TRAINING_PARAMS+=" --auto-detect-ckpt-format"

# Add load path if specified
if [[ -n "${LOAD_PATH}" ]]; then
    TRAINING_PARAMS+=" --load ${LOAD_PATH}"
fi

TRAINING_PARAMS+=" --save ${OUTPUT_PATH}/checkpoints"
TRAINING_PARAMS+=" --save-interval 500"
TRAINING_PARAMS+=" --dist-ckpt-strictness log_all"
TRAINING_PARAMS+=" --init-method-std 0.02"
TRAINING_PARAMS+=" --log-timers-to-tensorboard"
TRAINING_PARAMS+=" --log-memory-to-tensorboard"
TRAINING_PARAMS+=" --log-num-zeros-in-grad"
TRAINING_PARAMS+=" --log-params-norm"
TRAINING_PARAMS+=" --log-validation-ppl-to-tensorboard"
TRAINING_PARAMS+=" --log-throughput"
TRAINING_PARAMS+=" --log-interval 1"
TRAINING_PARAMS+=" --tensorboard-dir ${OUTPUT_PATH}/tensorboard"
# Add wandb parameters only if not disabled
if [[ "${DISABLE_WANDB}" != "true" ]]; then
  TRAINING_PARAMS+=" --wandb-project ${WANDB_PROJECT}"
  TRAINING_PARAMS+=" --wandb-exp-name DeepSeek-V3-TP${TP}PP${PP}EP${EP}CP${CP}VPP${VPP}-MBS${MBS}GBS${GBS}-${COMMENT}"
fi
TRAINING_PARAMS+=" --bf16"

# Add overlapping parameters (since A2A_OVERLAP is not set, use default overlap settings)
TRAINING_PARAMS+=" --overlap-grad-reduce --overlap-param-gather"

# # Append any command line arguments to TRAINING_PARAMS
# if [[ $# -gt 0 ]]; then
#     TRAINING_PARAMS="${TRAINING_PARAMS} $@"
# fi

# Set default output path if not set
export OUTPUT_PATH=${OUTPUT_PATH:-"${WORKSPACE}/outputs"}

# Export training command with distributed launch
# export TRAINING_CMD="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"
# ---- Build TRAINING_CMD ----
TRAINING_CMD="torchrun --nproc_per_node=${NPROC_PER_NODE}"

# Add distributed args only if set
[[ -n "$NNODES" ]]      && TRAINING_CMD+=" --nnodes=${NNODES}"
[[ -n "$NODE_RANK" ]]   && TRAINING_CMD+=" --node_rank=${NODE_RANK}"
[[ -n "$MASTER_ADDR" ]] && TRAINING_CMD+=" --master_addr=${MASTER_ADDR}"
[[ -n "$MASTER_PORT" ]] && TRAINING_CMD+=" --master_port=${MASTER_PORT}"

TRAINING_CMD+=" ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"

export TRAINING_CMD



# Local execution logs
LOCAL_LOGS="${OUTPUT_PATH}/local_logs"
mkdir -p ${LOCAL_LOGS} || {
    echo "Error: Failed to create local logs directory ${LOCAL_LOGS}"
    exit 1
}


# Generate timestamp for log file
TIMESTAMP=$(date +'%y%m%d_%H%M%S')
LOG_FILE="${LOCAL_LOGS}/${MODEL}-${RUN_NAME// /-}-${TIMESTAMP}.log"

echo "Starting DeepSeek V3 direct run..."
echo "Model: ${MODEL}"
echo "Run Name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Training command: ${TRAINING_CMD}"
echo "Working directory: ${MEGATRON_PATH}"
echo ""
echo "Configuration:"
echo "  TP=${TP}, PP=${PP}, EP=${EP}, CP=${CP}, VPP=${VPP}"
echo "  MBS=${MBS}, GBS=${GBS}"
echo "  SEQ_LEN=${SEQ_LEN}"
echo "  NUM_LAYERS=${NUM_LAYERS}"
echo "  MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER}"
echo "  MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM}"
echo ""

# Change to Megatron directory and run training
cd "${MEGATRON_PATH}"

# Direct execution without container
echo "Running directly on host system"
eval "${TRAINING_CMD}" 2>&1 | tee "${LOG_FILE}"

echo "DeepSeek V3 training completed. Log saved to: ${LOG_FILE}"
