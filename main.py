import os
from datasets import load_dataset
from vllm import LLM

path_var = os.getenv('HF_HUB_CACHE')
if path_var:
    print(f"PATH variable: {path_var}")
else:
    os.environ['HF_HUB_CACHE'] = "scratch/hf_hub"

ds = load_dataset("OpenGVLab/ShareGPT-4o", "image_caption")

models = {
    "qwenvl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llama11": "meta-llama/Llama-3.2-11B-Vision",
    "llama90": "meta-llama/Llama-3.2-90B-Vision",
    "llava7": "lmms-lab/llava-onevision-qwen2-7b-ov",
    "llava72": "llava-hf/llava-onevision-qwen2-72b-ov-hf",
    "intern": "OpenGVLab/InternVL2_5-26B",
    "nvlm": "nvidia/NVLM-D-72B"
}

def main():
    llm = LLM(
        model=models["qwenvl"],
        trust_remote_code=True,
        max_model_len=2048
    )
    prompt = "Testing from Georgia Tech"
    outputs = llm.generate(prompt)
    print(outputs)

if __name__ == "__main__":
    main()
