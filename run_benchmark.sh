#!/bin/bash

# Default values
API_MODEL_NAME="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_TOKENIZER="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATASET_CONFIG="${3:-datasets/sharegpt4o_image_caption.jsonl}"
SERVER_GPU_COUNT="${4:-1}"

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  echo "Usage: $0 [API_MODEL_NAME] [MODEL_TOKENIZER] [DATASET_CONFIG] [SERVER_GPU_COUNT]"
  echo ""
  echo "Parameters:"
  echo "  API_MODEL_NAME     - Model name for API (default: Qwen/Qwen2.5-VL-7B-Instruct)"
  echo "  MODEL_TOKENIZER    - Tokenizer name (default: Qwen/Qwen2.5-VL-7B-Instruct)"
  echo "  DATASET_CONFIG     - Dataset configuration file (default: data/sharegpt4o_image_caption.jsonl)"
  echo "  SERVER_GPU_COUNT   - Number of GPUs (default: 1)"
  echo ""
  echo "Example:"
  echo "  $0 Qwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-7B-Instruct data/sharegpt4o_image_caption.jsonl 2"
  exit 0
fi

echo "Running benchmark with:"
echo "  API_MODEL_NAME: $API_MODEL_NAME"
echo "  MODEL_TOKENIZER: $MODEL_TOKENIZER"
echo "  DATASET_CONFIG: $DATASET_CONFIG"
echo "  SERVER_GPU_COUNT: $SERVER_GPU_COUNT"
echo ""

source ./genai-bench/.venv/bin/activate

genai-bench benchmark \
  --api-backend vllm \
  --api-base http://localhost:8888 \
  --api-model-name "$API_MODEL_NAME" \
  --api-key "placeholder" \
  --model-tokenizer "$MODEL_TOKENIZER" \
  --task text-to-text \
  --max-requests-per-run 300 \
  --max-time-per-run 10 \
  --dataset-config "$DATASET_CONFIG" \
  --experiment-base-dir ./experiments/sharegpt4o_image_caption \
  --server-engine "vLLM" \
  --server-gpu-type "H100" \
  --server-gpu-count "$SERVER_GPU_COUNT" \
  --poisson-arrival-rate 1 \
  --poisson-arrival-rate 2 \
  --poisson-arrival-rate 4 \
  --poisson-arrival-rate 8 \
  --poisson-arrival-rate 12 \
  --poisson-arrival-rate 16 \
  --poisson-arrival-rate 20 \
  --poisson-arrival-rate 24 \
  --poisson-arrival-rate 28 \
  --poisson-arrival-rate 32 \
  --metrics-time-unit s \