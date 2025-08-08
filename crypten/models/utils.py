import os
import crypten
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from crypten.models.modeling_gemma2 import Gemma2ForCausalLM
from crypten.models.modeling_llama import LlamaForCausalLM
from crypten.models.modeling_qwen2 import Qwen2ForCausalLM
from crypten.models.modeling_mistral import MistralForCausalLM

def get_model_class(model_name):
    if "gemma" in model_name:
        return Gemma2ForCausalLM
    if "Llama" in model_name:
        return LlamaForCausalLM
    if "Qwen" in model_name:
        return Qwen2ForCausalLM
    if "Mistral" in model_name:
        return MistralForCausalLM


def transform_state_dict(state_dict):
    state_dict["model.embed_tokens.wpe.weight"] = state_dict.pop(
        "model.embed_tokens.weight"
    ).T
    keys = list(state_dict.keys())
    for k in keys:
        if "norm" in k:
            state_dict[k + ".data"] = state_dict.pop(k)
    return state_dict


def get_state_dict_from_path(model_path):
    pytorch_model = AutoModelForCausalLM.from_pretrained(model_path)
    return transform_state_dict(pytorch_model.state_dict())


def load_model(model_name):
    model_path = model_name
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_cls = get_model_class(model_path)
    model = model_cls(config)
    model.config._attn_implementation = 'eager'
    state_dict = get_state_dict_from_path(model_path)
    model.load_state_dict(state_dict, strict=False)

    return model, tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=10, device="cuda", vocab_size=None):
    input_ids = tokenizer.encode(prompt)
    prompt_length = len(input_ids)  # in tokens

    with crypten.no_grad():
        for _ in range(max_new_tokens):
            one_hot_input_ids = torch.nn.functional.one_hot(torch.tensor([input_ids]), num_classes=vocab_size if vocab_size is not None else len(tokenizer)).float()
            if device == "cuda":
                one_hot_input_ids = one_hot_input_ids.cuda()
            one_hot_input_ids = crypten.cryptensor(one_hot_input_ids)
            # one_hot_input_ids = encrypt_tensor(one_hot_input_ids)
            encrypted_logits = model(one_hot_input_ids).logits
            logits: torch.tensor = encrypted_logits.get_plain_text()[0][-1]
            generated_token = torch.argmax(logits).item()
            input_ids.append(generated_token)

            if generated_token == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[prompt_length:])

def get_logits(model, input_ids):
    with crypten.no_grad():
        one_hot_input_ids = torch.nn.functional.one_hot(torch.tensor(input_ids), num_classes=model.config.vocab_size).float().cuda()
        one_hot_input_ids = crypten.cryptensor(one_hot_input_ids)
        encrypted_logits = model(one_hot_input_ids).logits
        logits: torch.tensor = encrypted_logits.get_plain_text()

    return logits