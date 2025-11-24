#!/bin/bash

# Default values
API_MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_TOKENIZER="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_CONFIG="datasets/sharegpt4o_image_caption.jsonl"
SERVER_GPU_COUNT="1"
SEED="41"
TASK_NAME="default_task"

# Help function
usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Parameters:"
  echo "  API_MODEL_NAME     - Model name for API (default: Qwen/Qwen2.5-VL-3B-Instruct)"
  echo "  MODEL_TOKENIZER    - Tokenizer name (default: Qwen/Qwen2.5-VL-3B-Instruct)"
  echo "  DATASET_CONFIG     - Dataset configuration file (default: data/sharegpt4o_image_caption.jsonl)"
  echo "  SERVER_GPU_COUNT   - Number of GPUs (default: 1)"
  echo "  TASK_NAME          - Task name for experiment folder"
  echo ""
  echo "Example:"
  echo "  $0 Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2.5-VL-3B-Instruct data/sharegpt4o_image_caption.jsonl 2"
  exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --api-model-name)
      API_MODEL_NAME="$2"
      shift 2
      ;;
    --model-tokenizer)
      MODEL_TOKENIZER="$2"
      shift 2
      ;;
    --dataset-config)
      DATASET_CONFIG="$2"
      shift 2
      ;;
    --server-gpu-count)
      SERVER_GPU_COUNT="$2"
      shift 2
      ;;
    --task-name)
      TASK_NAME="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

echo "Running benchmark with:"
echo "  API_MODEL_NAME: $API_MODEL_NAME"
echo "  MODEL_TOKENIZER: $MODEL_TOKENIZER"
echo "  DATASET_CONFIG: $DATASET_CONFIG"
echo "  SERVER_GPU_COUNT: $SERVER_GPU_COUNT"
echo "  TASK_NAME: $TASK_NAME"
echo "  SEED: $SEED"
echo ""

source ./genai-bench/.venv/bin/activate

genai-bench benchmark \
  --api-backend vllm \
  --api-base http://localhost:10003 \
  --api-model-name "$API_MODEL_NAME" \
  --api-key "placeholder" \
  --model-tokenizer "$MODEL_TOKENIZER" \
  --task image-text-to-text \
  --max-requests-per-run 500 \
  --max-time-per-run 12 \
  --dataset-config "$DATASET_CONFIG" \
  --experiment-base-dir ./experiments/sharegpt4o_image_caption \
  --experiment-folder-name "$TASK_NAME" \
  --server-engine "vLLM" \
  --server-gpu-type "H100" \
  --server-gpu-count "$SERVER_GPU_COUNT" \
  --seed "$SEED" \
  --poisson-arrival-rate 1 \
  --poisson-arrival-rate 2 \
  --poisson-arrival-rate 4 \
  --trace-file 4 \
  --trace-file 5 \
  --metrics-time-unit s
