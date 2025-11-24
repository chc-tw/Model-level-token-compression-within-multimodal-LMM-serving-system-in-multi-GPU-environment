# import multiprocessing as mp
import torch.multiprocessing as mp
import os
import time

from typing import Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


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

def client(send_q, recv_q, event_to_set, event_to_detect):
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
    send_q.put(messages)
    print(f"Client Send Message: {messages}")
    event_to_detect.wait()
    while True:
        if not recv_q.empty():
            output_text = recv_q.get()
            print(f"Client Receives: {output_text}")
            event_to_set.set()
            break

def ImageProcessor(device, send_q, recv_q, model, processor, event_to_set, event_to_detect):
    vision_encoder = model.visual
    language_model = model.language_model
    while True:
        if not recv_q.empty():
            data = recv_q.get()
            event_to_set.set()
            print(f"IP Receives: {data}")
            # Preparation for inference
            text = processor.apply_chat_template(
                data, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(data)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            wte = language_model.get_input_embeddings()
            inputs = inputs.to(device)
            
            image_output = vision_encoder(inputs['pixel_values'], inputs['image_grid_thw'])
            split_sizes = (inputs['image_grid_thw'].prod(-1) // vision_encoder.spatial_merge_size**2).tolist()
            image_embeds = torch.split(image_output, split_sizes)
            text_output = wte(inputs['input_ids'])
            image_embeds = torch.cat(image_embeds, dim=0).to(text_output.device, text_output.dtype)
            image_mask, _ = get_placeholder_mask(model, inputs['input_ids'], inputs_embeds=text_output, image_features=image_embeds)
            inputs_embeds = text_output.masked_scatter(image_mask, image_embeds)
            del inputs['pixel_values']
            inputs['inputs_embeds'] = inputs_embeds.detach()
            inputs = inputs.to("cpu")

            send_q.put(inputs)
            print(f"IP Sends: {inputs}")
            event_to_detect.wait()
            image_embeds = None
            text_output = None
            torch.cuda.empty_cache()
            break

def LanguageModel(device, send_q, recv_q, model, processor, event_to_set, event_to_detect):
    while True:
        if not recv_q.empty():
            inputs = recv_q.get()
            event_to_set.set()
            print(f"LM Receives: {inputs}")
            inputs = inputs.to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            inputs.clear()
            send_q.put(output_text)
            print(f"LM Sends: {output_text}")
            torch.cuda.empty_cache()
            event_to_detect.wait()
            break


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Setting up model...")
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # default processer
    processor = AutoProcessor.from_pretrained(model_name)
    
    
    print("Starting System...")
    """
    Client -> Image Processor -> Language Model -> Client
    """
    c2ip = mp.Queue()
    ip2lm = mp.Queue()
    lm2c = mp.Queue()

    clientReceivedData = mp.Event()
    ipReceivedData = mp.Event()
    lmReceivedData = mp.Event()

    p_client = mp.Process(target=client, args=(c2ip, lm2c, clientReceivedData, ipReceivedData,))
    p_image = mp.Process(target=ImageProcessor, args=(device, ip2lm, c2ip, model, processor, ipReceivedData, lmReceivedData))
    # p_clientip = mp.Process(target=ClientIP, args=(ip2lm, lm2c, model, processor,))
    p_language = mp.Process(target=LanguageModel, args=(device, lm2c, ip2lm, model, processor, lmReceivedData, clientReceivedData,))

    p_client.start()
    p_image.start()
    # p_clientip.start()
    p_language.start()

    p_client.join()
    p_image.join()
    # p_clientip.join()
    p_language.join()
    print(f"Main process finished.")

