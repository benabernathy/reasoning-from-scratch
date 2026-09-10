from pathlib import Path
import re

import torch

from reasoning_from_scratch.ch02 import get_device, generate_text_basic_stream_cache
from reasoning_from_scratch.qwen3 import download_qwen3_small, Qwen3Tokenizer, Qwen3Model, QWEN_CONFIG_06_B

def load_model_and_tokenizer(which_model, device,  use_compile, local_dir="qwen3"):
    if which_model == "base":
        download_qwen3_small(kind="base", tokenizer_only=False, out_dir=local_dir)

        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif which_model == "reasoning":
        download_qwen3_small(kind="reasoning", tokenizer_only=False, out_dir=local_dir)

        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"

        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path,
                                   apply_chat_template=True,
                                   add_generation_prompt=True,
                                   add_thinking=True,)

    else:
        raise ValueError(f"Invalid choice: which_model={which_model}")

    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))

    model.to(device)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer

WHICH_MODEL = "base"
device = get_device()

model, tokenizer = load_model_and_tokenizer(which_model=WHICH_MODEL, device=device, use_compile=False)

prompt = (
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a^2+b^2$?"
)

input_token_ids_tensor = torch.tensor(
    tokenizer.encode(prompt),
    device=device,
).unsqueeze(0)

all_token_ids = []

for token in generate_text_basic_stream_cache(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=2048,
    eos_token_id=tokenizer.eos_token_id,
):
    token_id = token.squeeze(0)
    decode_id = tokenizer.decode(token_id.tolist())
    print(decode_id, end="", flush=True)
    all_token_ids.append(token_id)

all_tokens = tokenizer.decode(all_token_ids)

def generate_text_stream_concat(model, tokenizer, prompt, device, max_new_tokens, verbose=False):
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

    generated_ids = []
    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
        if verbose:
            print(tokenizer.decode(next_token_id.tolist()), end="", flush=True)

    return tokenizer.decode(generated_ids)
generated_text = generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens=2048, verbose=True
)

model_answer = (
    r"""... some explanation...
    ***Final Answer:**
    
    \[
    \boxed{\dfrac{14}{3}}
    \]"""
)

def get_last_boxed(text):
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None
    current_idx = boxed_start_idx + len(r"\boxed")

    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    if brace_depth != 0:
        return None

    return text[content_start_idx:current_idx-1]

extracted_answer = get_last_boxed(model_answer)
print(extracted_answer)

RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

def extract_final_candidate(text, fallback="number_then_full"):
    result = ""

    if text:
        boxed = get_last_boxed(text.strip())

        if boxed:
            result = boxed.strip().strip("$ ")
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                result = m[-1]
            elif fallback == "number_then_full":
                result = text
    return result

print(extract_final_candidate(model_answer))

print(extract_final_candidate(r"\boxed{ 14 / 3. }"))

print(extract_final_candidate("abc < > 14/3 abc"))

LATEX_FIXES = [
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"º", ""),
]


