# research3
Project title: Efficient Serving of Vision Language Models via Traffic-Aware Token Reduction

## Setup Instructions

In this project, we need three virtual environments: one for the vLLM serving, one for similarity compute and one for genai-bench because the dependencies are conflicted between them.

To better manage the dependencies, we use uv with pyproject.toml to install the dependencies. First install `uv` via curl:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For ICE PACE, to avoid Disk Quota Excceeded error, move the default uv cache directory to inside the scratch directory:

```bash
export UV_CACHE_DIR=scratch/.cache
```
After setting uv, you can clone the repository and install the dependencies:

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
<place_holder>

### Run the full experiment
In this project, we need to run:

1. the vLLM server on the gpu node
2. the benchmark on the cpu node

#### 1. Run the vLLM server on the gpu node

First ask for a gpu node with memory > 32GB, 8 CPUs and 2 GPUs
```bash
srun --gres=gpu:H100:2 --mem 32768 --cpus-per-task=8 --time=4:00:00 -N 1 --pty /bin/zsh
```
Next, we need to activate the vLLM serving virtual environment and start the vLLM server
```bash
cd research3
export PYTHONPATH="$(pwd)/vllm:$PYTHONPATH"
source .venv/bin/activate
cd vllm/examples/online_serving/disaggregated_encoder_dynamic_sizing
GPU_E=0 GPU_PD=1 PROXY_PORT=10003 bash disagg_1e1pd_example.sh
```

Next, because the vLLM server is on gpu node, we need to know the hostname of the server and forward the port to the cpu node. Run this command to get the hostname:

```bash
hostname
```

#### 2. Run the benchmark on the cpu node

After getting the hostname, **on the cpu node**, we can forward the port on the gpu node to the cpu node:
```bash
ssh -N -f -L 10003:localhost:10003 $USER@<hostname of the gpu node>
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
- `--trace-file`: the trace file ID (available trace files are 4,5,7 and 10, representing the peak qps of the trace)
- `--api-port`: the port of vllm server
- `--seed`: the random seed for data sampling

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

To visualize all results for a experiment, run:

```bash
cd research3
python cdf_plot.py "experiments/sharegpt4o_image_caption/demo"
```
The result figures will be saved in the same folder.

### Similarity Compute
To compute the similarity between the predictions and the references on a specific result file, run:

```bash
cd research3/similarity_compute
source .venv/bin/activate
python run.py --file_path "../experiments/sharegpt4o_image_caption/demo/Trace with Peak QPS_7.json"
```
It will print the average similarity in the console.