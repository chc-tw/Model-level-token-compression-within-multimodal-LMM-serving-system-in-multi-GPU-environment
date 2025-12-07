#!/bin/bash

# --- Configuration ---
# Uses default values if environment variables are not set
PROXY_PORT="${PROXY_PORT:-10002}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
# Assuming git is installed and the file path is correct relative to the git root
# For a simpler test, you might hardcode the file path: FILE="/path/to/your/image.png"
GIT_ROOT="$(pwd)" # Handle case where not in a git repo
FILE="${GIT_ROOT}/profile_images/2700-2700.png"

# --- Request Payload (JSON) ---
# Use printf to create the complex JSON payload correctly, handling nested quotes
# This structure sends both the image URL (file://) and the text prompt.
PAYLOAD=$(printf '{
    "model": "%s",
    "stream": true,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file://%s"}},
            {"type": "text", "text": "What is in this image?"}
        ]}
    ]
}' "${MODEL}" "${FILE}")

# --- TTFT Calculation Logic ---

# 1. Check if the file exists before running the benchmark
if [ ! -f "$FILE" ]; then
    echo "Error: Image file not found at $FILE" >&2
    exit 1
fi

API_URL="http://0.0.0.0:${PROXY_PORT}/v1/chat/completions"

echo "Benchmarking TTFT for Multimodal Request..."
echo "Model: ${MODEL}"
echo "Endpoint: ${API_URL}"
echo "Image File: ${FILE}"

# 2. Record the start time in seconds (with high precision)
START_TIME=$(date +%s.%N)

# 3. Execute curl and pipe the streaming output to awk
# -s: Silent mode
# -N / --no-buffer: Ensures immediate output for accurate timing
curl -sN --no-buffer -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
| awk -v start_time="$START_TIME" '
    # Process each line as it arrives from the stream
    /^{/ || /^data: / {
        # Check if the line contains a non-empty "content" field (the actual first token)
        if ($0 ~ /"content": *"[^"]+"/) {
            # Record the current time with high precision
            # Note: awk/gawk may need different approaches for sub-second time.
            # Using system call for best precision if available, otherwise falling back
            "date +%s.%N" | getline CURRENT_TIME
            
            # Calculate TTFT: Current Time - Start Time
            TTFT = CURRENT_TIME - start_time
            
            # Print the result
            printf "TTFT: %.4f seconds\n", TTFT
            
            # Print the first data chunk for verification
            print $0
            
            # Exit awk after the first token is measured
            exit
        }
    }
    # Optional: print subsequent tokens if you want to see the full response
    # { print } 
'