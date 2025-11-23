#!/bin/bash

# Default values
API_MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_TOKENIZER="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_CONFIG="datasets/sharegpt4o_image_caption.jsonl"
SERVER_GPU_COUNT="1"
SEED="-1"
TRACE_FILE=""

# Help function
usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --api-model-name <name>    Model name for API (default: $API_MODEL_NAME)"
  echo "  --model-tokenizer <name>   Tokenizer name (default: $MODEL_TOKENIZER)"
  echo "  --dataset-config <file>    Dataset configuration file (default: $DATASET_CONFIG)"
  echo "  --server-gpu-count <num>   Number of GPUs (default: $SERVER_GPU_COUNT)"
  echo "  --seed <num>               Random seed (default: $SEED)"
  echo "  --trace-file <file>        Trace file (default: \"$TRACE_FILE\")"
  echo "  -h, --help                 Show this help message"
  echo ""
  echo "Example:"
  echo "  $0 --api-model-name Qwen/Qwen2.5-VL-7B-Instruct --server-gpu-count 2"
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
    --seed)
      SEED="$2"
      shift 2
      ;;
    --trace-file)
      TRACE_FILE="$2"
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
echo "  SEED: $SEED"
echo "  TRACE_FILE: $TRACE_FILE"
echo ""

source ./genai-bench/.venv/bin/activate

genai-bench benchmark \
  --api-backend vllm \
  --api-base http://localhost:8888 \
  --api-model-name "$API_MODEL_NAME" \
  --api-key "placeholder" \
  --model-tokenizer "$MODEL_TOKENIZER" \
  --task text-to-text \
  --max-requests-per-run 500 \
  --max-time-per-run 10 \
  --dataset-config "$DATASET_CONFIG" \
  --experiment-base-dir ./experiments/sharegpt4o_image_caption \
  --server-engine "vLLM" \
  --server-gpu-type "H100" \
  --server-gpu-count "$SERVER_GPU_COUNT" \
  --seed "$SEED" \
  --trace-file "$TRACE_FILE" \
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
