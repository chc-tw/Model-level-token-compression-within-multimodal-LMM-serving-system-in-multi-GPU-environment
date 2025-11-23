# research3

## Setup Instructions

### Virtual Environment

To better manage the dependencies, we use uv with pyproject.toml to install the dependencies. First install `uv` with pip:

```bash
pin install uv
```

If a `.venv` directory already exists (created by `uv`), activate it with the following command:

```bash
source .venv/bin/activate
```

For ICE PACE, to avoid Disk Quota Excceeded error, move the default uv cache directory to inside the scratch directory:

```bash
export UV_CACHE_DIR=scratch/.cache
```

To sync the dependencies, run:

```bash
VLLM_USE_PRECOMPILED=1 uv sync
```

To add a new dependency, run:

```bash
uv add <package-name>
```

To remove a dependency, run:
```bash
uv remove <package-name>
```

### vLLM
>Note: you don't need to use this section. just do the above step. This section is for reference only.

> To install vLLM, we need to set environment variable MAX_JOBS=2 to avoid OOM.

If modifying only the Python code, build and install vLLM without compilation:

```bash
cd vllm
MAX_JOBS=2 VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

If modifying the C++ or CUDA code, the entire vLLM needs to be built from source: 

```bash
cd vllm
uv pip install -e .
```

Here, we fix vLLM version to [v0.11.0](https://github.com/vllm-project/vllm/releases/tag/v0.11.0). After building vLLM, set `PYTHONPATH` to enable vLLM local built module selection.
For example, set the environment variable under the repository path:

```bash
export PYTHONPATH="$(pwd)/vllm:$PYTHONPATH"
``` 

### Dataset

Dataset used in the ModServe paper is the ShareGPT-4o dataset, which includes 50K images of varying resolutions and text prompts from GPT-4o.

```bash
hf auth login
```

### Models

Models used in the ModServe paper are:
- Llama 3.2 Vision 11B
- Llama 3.2 Vision 90B
- LLaVA-OneVision 7B
- LLaVA-OneVision 72B
- InternVL-2.5 26B
- NVLM-D 72B

## Run Disaggregated Encoder VLLM Server

### Install NIXL

Install NIXL for the KV Connector:

```bash
uv pip install nixl
```

### GPU Node

For the disaggregated encoder experiments, obtain at least 2 H100 GPUs for example as follows:

```bash
salloc --nodes=2 --gres=gpu:H100:2 --ntasks-per-node=2 --time=2:00:00
```

### Default Benchmark

Run the default experiment as follows:

```bash
cd disaggregated_encoder
bash disagg_1e1p1d_example.sh # uses Qwen-2.5 3B
```

To run the benchmark for other models, do the following:

```bash
MODEL="Qwen/Qwen2.5-VL-7B-Instruct" bash disagg_1e1p1d_example.sh
```

## Run the GenAI Benchmarks

We first need to ask for gpu machine

```bash
srun --gres=gpu:H100:1 --cpus-per-task=8 --time=9:00:00 -N 1 --pty /bin/bash
```

Next, because the vLLM server is on gpu node, we need to know the hostname of the server. Run this command to get the hostname:

```bash
hostname
```

and start the vLLM server

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct
```

Then, in another terminal, run:

```bash
ssh -N -f -L 8888:localhost:8000 $USER@<hostname>
```

Next, run the benchmark:
>Note because the dependencies are conflicted between vLLM and genai-bench, we need to run the benchmark in the different virtual environment.
>before running the benchmark, we need to create the genai-bench virtual environment:
>```bash
>cd genai-bench
>uv venv --python 3.12.5
>uv sync
>
>```
> 
```bash
source genai-bench/.venv/bin/activate
./run_benchmark.sh
```

### Dataset
The ShareGPT-4o dataset is available on Hugging Face. The repo name is "chc-tw/ShareGPT-4o".
To use this dataset in python, you can run:

```python
from datasets import load_dataset
ds = load_dataset("chc-tw/ShareGPT-4o", "default")
```

The dataset is a DatasetDict object, you can access the train split by:

```python
train_ds = ds["train"]
```

The dataset is a Dataset object, you can access the first example by:

```python
print(train_ds[0])
```
