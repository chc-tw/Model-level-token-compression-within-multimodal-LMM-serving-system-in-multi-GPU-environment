# Title: Efficient Serving of Vision Language Models via Traffic-Aware Token Reduction

Team Members: Chung-En Ho, Hao-Cheng Chang, Min Lu

## Setup Instructions

In this project, we need 3 virtual environments: one for the vLLM serving, one for similarity compute and one for genai-bench because the dependencies are conflicted between them.

To better manage the dependencies, we use uv with pyproject.toml to install the dependencies. First install `uv` via curl:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If running on Georgia Tech PACE ICE cluster, since we have a small disk quota in the home directory, move the default uv cache directory to inside the scratch directory to avoid a Disk Quota Exceeded error during installation:

```bash
export UV_CACHE_DIR=scratch/.cache
```
After setting `uv`, you can clone the repository and install the dependencies:

```bash
git clone --recurse-submodules https://github.gatech.edu/cs8803smr-f25/research3.git
```
### vLLM Serving Virtual Environment
To install and sync the dependencies, run:

```bash
cd research3
MAX_JOBS=2 VLLM_USE_PRECOMPILED=1 uv sync
```

### GenAI Bench Virtual Environment
To install and sync the dependencies, run:

```bash
cd research3/genai-bench
uv sync
```

### Similarity Compute Virtual Environment
To install and sync the dependencies, run:

```bash
cd research3/similarity_compute
uv sync
```

## Run Instruction
### Profiling
A profiling step on the token-latency relation is needed in our method prior to serving. This step is currently done manually, and the token-latency relation is hard-coded into the vLLM API server (see `compute_token_budget_from_ttft_slo` in `vllm/examples/online_serving/disaggregated_encoder_dynamic_sizing/disagg_epd_proxy.py`). We already hard-coded the profiling result for 2xH100 node on PACE ICE cluster into the current code.

To collect the latency profiling data, we need to launch a vLLM server first. Request a GPU node using `slurm`:
```bash
srun --gres=gpu:H100:2 --mem 32768 --cpus-per-task=8 --time=4:00:00 -N 1 --pty /bin/zsh
```
Then launch the vLLM server
```bash
cd research3
export PYTHONPATH="$(pwd)/vllm:$PYTHONPATH"
source .venv/bin/activate
cd profiling
GPU_E=0 GPU_PD=1 PROXY_PORT=10003 bash disagg_1e1pd.sh
```
After the server was successfully launched, at the same session, run
```bash
PROXY_PORT=10003 bash ttft_profile_request.sh
```
We have prepared a few random images under `profile_images`, and you can pass then through the `FILE` environment variable. Following is an output example:
```
Benchmarking latency for Multimodal Request...
Model: Qwen/Qwen2.5-VL-3B-Instruct
Endpoint: http://0.0.0.0:10003/v1/chat/completions
Image File: /storage/ice1/9/1/cho322/research3/profiling/profile_images/1536-1536.png
Latency: 0.4441 seconds
data: {"id":"chatcmpl-7b26c404-75be-4be3-8ea3-3103e15dd184","object":"chat.completion.chunk","created":1765151873,"model":"Qwen/Qwen2.5-VL-3B-Instruct","choices":[{"index":0,"delta":{"content":"The","reasoning_content":null},"logprobs":null,"finish_reason":null,"token_ids":null}]}
```
We can see that the image encoding latency is 0.4441 seconds. The corresponding amount of visual tokens is (1536//28)*(1536//28) = 3481. We benchmark the latency from pictures "1700-1700.png" to "3840-3840.png" (ranging 3721 - 16384 visual tokens).

### Run the Full Experiment
In this project, we need to launch:

1. The vLLM server on the GPU node
2. The benchmark on the CPU node

#### 1. Launch the vLLM server on the GPU node

First, request a GPU node using `slurm`. Our recommended hardware configuration is > 32GB memory, 8 CPUs and 2 GPUs.
```bash
srun --gres=gpu:H100:2 --mem 32768 --cpus-per-task=8 --time=4:00:00 -N 1 --pty /bin/zsh
```
Next, we need to activate the vLLM serving virtual environment and start the vLLM server
```bash
cd research3
export PYTHONPATH="$(pwd)/vllm:$PYTHONPATH"
source .venv/bin/activate
cd vllm/examples/online_serving/disaggregated_encoder_dynamic_sizing
GPU_E=0 GPU_PD=1 PROXY_PORT=10003 bash disagg_1e1pd_dynamic_sizing.sh
```

Wait for the vLLM server launch to complete. Upon completion, you will see message in the console, for example:

```
remove previous ec cache folder
make ec cache folder
Running single request with local image (non-stream)...
{"id":"chatcmpl-c8346702-56fd-41d3-a5c0-b3c1cb1d11ad","object":"chat.completion","created":1765140092,"model":"Qwen/Qwen2.5-VL-3B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"The image depicts an urban scene with several elements:\n\n1. **Foreground**: There are three pigeons on the ground, one of which is prominently in the foreground, facing away from the camera.\n2. **Background**: Two individuals dressed in formal attire (suits and ties) are walking along a sidewalk. The background also includes a brick wall and some trees with autumn leaves.\n3. **Sidewalk**: The sidewalk is made of wooden planks and has scattered leaves on it.\n4. **Buildings**: There are buildings visible in the background, including a multi-story building with balconies.\n\nThe overall setting appears to be a quiet, possibly early morning or late afternoon, urban environment.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":8057,"total_tokens":8198,"completion_tokens":141,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
All services are up!
```

Seeing **All services are up!** indicates we are all set in this stage.

There are three vLLM launch scripts under `vllm/examples/online_serving/disaggregated_encoder_dynamic_sizing`, which correspond to different experimental setups as indicated in our report.
 * `disagg_1e1pd_dynamic_sizing.sh`: Launch a vLLM server that performs traffic-aware dynamic image sizing, the main proposed algorithm (**Dynamic**) in our report.
 * `disagg_1e1pd_static_4k.sh`: Launch a vLLM server that performs constant image resizing with 4K token budget (**Static** Baseline described in the report)
 * `disagg_1e1pd_vanilla.sh`: Launch a vLLM server with no image resizing and exact compute of the target VLM (**Vanilla** Baseline described in the report)
 
Next, because the vLLM server is on GPU node, we need to know the hostname of the server and forward the port to the CPU node. Run this command to get the hostname:

```bash
hostname
```

#### 2. Run the Benchmark on the CPU node

After getting the hostname, **on the CPU node**, we can forward the port on the GPU node to the CPU node:
```bash
ssh -N -f -L 10003:localhost:10003 $USER@<hostname of the GPU node>
```

Once we have the ssh tunnel, we can run the benchmark:
```bash
cd research3/genai-bench
source .venv/bin/activate
cd ..
./run_benchmark.sh --task-name "demo" --trace-file 7 --api-port 10003 --seed 41
```
The `run_benchmark.sh` accept the following arguments:
- `--task-name`: the name of the task
- `--trace-file`: the trace file ID (available trace files are 4, 5, 7 and 10, representing the peak QPS of the trace)
- `--api-port`: the port of vllm server
- `--seed`: the random seed for data sampling. Fix this to ensure consistency among all benchmarks.

**Note:** It is recommended to cold start the vLLM server on the GPU node (i.e., kill all vLLM process and re-launch) to ensure the outcome consistency. Since there is a few cache implemented in vLLM, doing multiple benchmarking on a single vLLM server launch will trigger those caching mechanism and influence reproducibility.

The result will be saved in the `experiments/sharegpt4o_image_caption/demo` folder.
The result will contain the following fields:
- ttft
- tpot
- e2e_latency
- output_latency
- output_inference_speed
- num_input_tokens
- num_output_tokens
- total_tokens
- input_throughput
- output_throughput

To visualize all results for an experiment, run:

```bash
cd research3
python cdf_plot.py "experiments/sharegpt4o_image_caption/demo"
```
The result TTFT CDF figures will be saved in the same path.

### Similarity Compute
To compute the similarity between the predictions and the references on a specific experiment result file, run:

```bash
cd research3/similarity_compute
source .venv/bin/activate
python run.py --file_path "../experiments/sharegpt4o_image_caption/demo/Trace with Peak QPS_<trace file ID>.json"
```
It will print the average similarity score of all requests collected in the experiment in the console.