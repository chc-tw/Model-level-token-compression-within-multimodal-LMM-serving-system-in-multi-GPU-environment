# research3

## Setup Instructions

### Virtual Environment

```bash
source smr/bin/activate
```

Packages are being installed using uv. Commands are of the form 

```bash
uv pip install <package-name>
```

### vLLM

If modifying only the Python code, build and install vLLM without compilation:

```bash
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

If modifying the C++ or CUDA code, the entire vLLM needs to be built from source: 

```bash
cd vllm
uv pip install -e .
```

Here, we fix vLLM version to [v0.11.0](https://github.com/vllm-project/vllm/releases/tag/v0.11.0). After building vLLM, set `PYTHONPATH` to enable vLLM local built module selection.
For example, set the environment variable under the repository path:
```bash
export PYTHONPATH="$pwd/vllm:$PYTHONPATH"
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
