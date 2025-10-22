from typing import Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

# default processer
processor = AutoProcessor.from_pretrained(model_name)

# default: Load the model on the available device(s)
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

vision_encoder = model.visual
language_model = model.language_model

wte = language_model.get_input_embeddings()

image_output = vision_encoder(inputs['pixel_values'], inputs['image_grid_thw'])
split_sizes = (inputs['image_grid_thw'].prod(-1) // vision_encoder.spatial_merge_size**2).tolist()
image_embeds = torch.split(image_output, split_sizes)

text_output = wte(inputs['input_ids'])

def get_placeholder_mask(
    model,
    input_ids: torch.LongTensor,
    inputs_embeds: torch.FloatTensor,
    image_features: Optional[torch.FloatTensor] = None,
    video_features: Optional[torch.FloatTensor] = None,
):
    """
    Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
    equal to the length of multimodal features. If the lengths are different, an error is raised.
    """
    special_image_mask = input_ids == model.config.image_token_id
    special_video_mask = input_ids == model.config.video_token_id

    n_image_tokens = special_image_mask.sum()
    special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
        )

    n_video_tokens = special_video_mask.sum()
    special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
        raise ValueError(
            f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
        )

    return special_image_mask, special_video_mask


image_embeds = torch.cat(image_embeds, dim=0).to(text_output.device, text_output.dtype)
image_mask, _ = get_placeholder_mask(model, inputs['input_ids'], inputs_embeds=text_output, image_features=image_embeds)
inputs_embeds = text_output.masked_scatter(image_mask, image_embeds)

del inputs['pixel_values']
inputs['inputs_embeds'] = inputs_embeds

print(inputs)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)